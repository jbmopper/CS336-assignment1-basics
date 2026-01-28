import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int, Bool
import numpy.typing as npt
from collections.abc import Iterable

from cs336_basics.nn import (
    Linear, Embedding, RMSNorm, SwiGLU, FFNSiLU,
    Rope, Multihead, MultiheadRope,
    softmax, silu, scaled_dot_product_attention
)
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule

__all__ = [
    'SwiGLU', 'softmax', 'silu', 'crossentropy', 'scaled_dot_product_attention',
    'TransformerBlock', 'TransformerLM', 'get_batch',
    'gradient_clipping', 'save_checkpoint', 'load_checkpoint'
]


def crossentropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss.
    
    Args:
        inputs: [batch, seq_length, vocab_size] logits
        targets: [batch, seq_length] target indices
    
    Returns:
        Scalar mean cross-entropy loss
    """
    probs = torch.log_softmax(inputs, dim=-1)
    used_probs = torch.gather(probs, -1, targets.unsqueeze(-1))
    return -torch.mean(used_probs)


class TransformerBlock(nn.Module):
    """Configurable Transformer block for ablations."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        norm_mode: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        ffn_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.eps = 1e-5
        self.norm_mode = norm_mode
        self.use_rope = use_rope

        if self.norm_mode not in {"pre", "post", "none"}:
            raise ValueError(f"Unsupported norm_mode: {self.norm_mode}")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")

        if self.norm_mode == "none":
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
        else:
            self.ln1 = RMSNorm(self.d_model, self.eps)
            self.ln2 = RMSNorm(self.d_model, self.eps)

        if ffn_type == "silu":
            ffn_hidden_dim = ffn_hidden_dim or (4 * d_model)
            self.ffn = FFNSiLU(self.d_model, ffn_hidden_dim)
        else:
            ffn_hidden_dim = ffn_hidden_dim or d_ff
            self.ffn = SwiGLU(self.d_model, ffn_hidden_dim)

        if use_rope:
            self.attn = MultiheadRope(self.num_heads, self.d_model, self.theta, self.max_seq_len)
        else:
            self.attn = Multihead(self.num_heads, self.d_model)

    def forward(self, in_features: Float[Tensor, "... seq d_model"]) -> Float[Tensor, "... seq d_model"]:
        with torch.profiler.record_function("## TRANSFORMER_BLOCK_FORWARD ##"):
            token_positions = torch.arange(in_features.size(-2), device=in_features.device, dtype=torch.long)

        if self.norm_mode == "pre":
            norm1 = self.ln1.forward(in_features)
            if self.use_rope:
                attention_output = self.attn.forward(norm1, token_positions)
            else:
                attention_output = self.attn.forward(norm1)
            in_features = in_features + attention_output
            norm2 = self.ln2.forward(in_features)
            ffn_output = self.ffn.forward(norm2)
            return in_features + ffn_output

        if self.norm_mode == "post":
            if self.use_rope:
                attention_output = self.attn.forward(in_features, token_positions)
            else:
                attention_output = self.attn.forward(in_features)
            in_features = self.ln1.forward(in_features + attention_output)
            ffn_output = self.ffn.forward(in_features)
            return self.ln2.forward(in_features + ffn_output)

        if self.use_rope:
            attention_output = self.attn.forward(in_features, token_positions)
        else:
            attention_output = self.attn.forward(in_features)
        in_features = in_features + attention_output
        ffn_output = self.ffn.forward(in_features)
        return in_features + ffn_output


class TransformerLM(nn.Module):
    """Transformer language model with configurable blocks."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        context_length: int,
        rope_theta: float,
        norm_mode: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        ffn_hidden_dim: int | None = None,
        final_norm: bool | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = context_length
        self.theta = rope_theta
        self.norm_mode = norm_mode
        self.use_rope = use_rope
        self.ffn_type = ffn_type
        self.ffn_hidden_dim = ffn_hidden_dim
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                rope_theta,
                norm_mode=norm_mode,
                use_rope=use_rope,
                ffn_type=ffn_type,
                ffn_hidden_dim=ffn_hidden_dim,
            )
            for _ in range(num_layers)
        ])
        self.eps = 1e-5
        if final_norm is None:
            final_norm = norm_mode == "pre"
        self.ln_final = RMSNorm(self.d_model, self.eps) if final_norm else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size)
        self.token_embeddings = Embedding(vocab_size, d_model)

    def forward(self, in_indices: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch seq vocab"]:
        with torch.profiler.record_function("## TRANSFORMER_LM_FORWARD ##"):
            x = self.token_embeddings.forward(in_indices)
            for layer in self.layers:
                x = layer.forward(x)
            x = self.ln_final.forward(x)
            return self.lm_head.forward(x)


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of input sequences and labels from the dataset."""
    starts = torch.randint(low=0, high=(len(dataset) - context_length), size=(batch_size,))
    indices = starts.unsqueeze(1) + torch.arange(context_length)
    inputs = torch.tensor(dataset[indices.numpy()], dtype=torch.long, device=device)
    labels = torch.tensor(dataset[indices.numpy() + 1], dtype=torch.long, device=device)
    return (inputs, labels)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip gradients to have at most max_l2_norm."""
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    
    gradient = torch.cat([torch.flatten(p.grad) for p in params_with_grad])
    norm = torch.linalg.norm(gradient)
    
    if norm > max_l2_norm:
        eps = 1e-6
        scale = max_l2_norm / (norm + eps)
        for param in params_with_grad:
            param.grad.mul_(scale)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out,    
    config: dict | None = None,
) -> None:
    """Save model, optimizer state, and iteration to a checkpoint file."""
    obj = dict(
        model=model.state_dict(), 
        optimizer=optimizer.state_dict(), 
        iteration=iteration
    )
    if config is not None:
        obj["config"] = config
    torch.save(obj, out)


def load_checkpoint(
    src,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load model and optimizer state from a checkpoint file. Returns iteration."""
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]

def load_model(src) -> tuple[dict, TransformerLM]:
    obj = torch.load(src)
    config = obj["config"]
    if config["device"] == None:
        config["device"] = "cpu"

    model =  TransformerLM(
        config["vocab_size"],
        config["d_model"],
        config["num_heads"],
        config["num_layers"],
        config["d_ff"],
        config["context_length"],
        config["rope_theta"],
        norm_mode=config.get("norm_mode", "pre"),
        use_rope=config.get("use_rope", True),
        ffn_type=config.get("ffn_type", "swiglu"),
        ffn_hidden_dim=config.get("ffn_hidden_dim"),
        final_norm=config.get("final_norm"),
    ).to(config["device"]) # does this happen here?

    model.load_state_dict(obj["model"])
    return config, model
