import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int, Bool
import numpy.typing as npt
from collections.abc import Iterable

from cs336_basics.nn import (
    Linear, Embedding, RMSNorm, SwiGLU,
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
    """Pre-norm Transformer block with RoPE attention and SwiGLU FFN."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.eps = 1e-5
        self.ln1 = RMSNorm(self.d_model, self.eps)
        self.ln2 = RMSNorm(self.d_model, self.eps)
        self.ffn = SwiGLU(self.d_model, self.d_ff)
        self.attn = MultiheadRope(self.num_heads, self.d_model, self.theta, self.max_seq_len)

    def forward(self, in_features: Float[Tensor, "... seq d_model"]) -> Float[Tensor, "... seq d_model"]:
        norm1 = self.ln1.forward(in_features)
        token_positions = torch.arange(in_features.size(-2), dtype=int)
        attention_output = self.attn.forward(norm1, token_positions)
        in_features = in_features + attention_output
        norm2 = self.ln2.forward(in_features)
        ffn_output = self.ffn.forward(norm2)
        return in_features + ffn_output


class TransformerLM(nn.Module):
    """Transformer language model with RoPE."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        context_length: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = context_length
        self.theta = rope_theta
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.eps = 1e-5
        self.ln_final = RMSNorm(self.d_model, self.eps)
        self.lm_head = Linear(d_model, vocab_size)
        self.token_embeddings = Embedding(vocab_size, d_model)

    def forward(self, in_indices: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch seq vocab"]:
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

def load_model(src) -> tuple[dict, torch.nn.Module]:
    config = src["config"]
    if config["device"] == None:
        config["device"] == "cpu"

    model =     model = TransformerLM(
        config["vocab_size"],
        config["d_model"],
        config["num_heads"],
        config["num_layers"],
        config["d_ff"],
        config["context_length"],
        config["rope_theta"]
    ).to(config["device"]) # need better device info?  e.g. cuda:0?

    model.load_state_dict(src["model"])
    return config, model
