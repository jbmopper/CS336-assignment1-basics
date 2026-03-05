"""Neural network modules and functions."""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from jaxtyping import Float, Int, Bool
import einx

__all__ = [
    'Linear', 'Embedding', 'RMSNorm', 'SwiGLU', 'FFNSiLU',
    'Rope', 'Multihead', 'MultiheadRope',
    'softmax', 'silu', 'scaled_dot_product_attention'
]


class Linear(nn.Module):
    """Linear layer without bias."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features)))
        sigma = 2 / (in_features + out_features)
        torch.nn.init.trunc_normal_(self.weight, 0, sigma, -3 * sigma, 3 * sigma)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Tensor:
        return torch.matmul(x, self.weight.T)


class Embedding(nn.Module):
    """Embedding layer."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = Parameter(torch.empty((num_embeddings, embedding_dim)))
        torch.nn.init.trunc_normal_(self.weight, 0, 1.0, -3.0, 3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(
        self,
        d_model: int,
        eps: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(self.eps + torch.mean(x ** 2, dim=-1, keepdim=True))
        result = (x / rms) * self.weight
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))

class FFNSiLU(nn.Module):
    """SiLU feed-forward network with output projection."""
    
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.w2(silu(self.w1(x)))


def softmax(in_features: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax."""
    adjusted_features = in_features - torch.max(in_features, dim=dim, keepdim=True)[0]
    exp_features = torch.exp(adjusted_features)
    return exp_features / torch.sum(exp_features, dim=dim, keepdim=True)


def silu(in_features: Tensor) -> Tensor:
    """SiLU (Swish) activation function."""
    return in_features * torch.sigmoid(in_features)


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... values d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None
) -> Float[Tensor, "... queries d_v"]:
    """Scaled dot-product attention."""
    d_k = Q.size(-1)
    scaled_product = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
    if mask is not None:
        scaled_product = scaled_product.masked_fill_(~mask, float('-inf'))
    output = softmax(scaled_product, -1) @ V
    return output


def _build_mthetas(theta: float, d_k: int, max_seq_len: int) -> Float[Tensor, "max_seq_len d_half"]:
    """Build m*theta matrix for RoPE."""
    assert d_k % 2 == 0
    dk2 = d_k // 2
    thetas = torch.ones(dk2) * theta
    thetas = thetas ** ((-2 * torch.arange(dk2)) / d_k)
    ms = torch.arange(max_seq_len)
    mthetas = einx.multiply("... seq, d -> ... seq d", ms, thetas)
    return mthetas


class Rope(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        mthetas = _build_mthetas(theta, d_k, max_seq_len)
        coses = torch.cos(mthetas)
        sines = torch.sin(mthetas)
        self.register_buffer('coses', coses, persistent=False)
        self.register_buffer('sines', sines, persistent=False)

    def forward(
        self,
        in_query_or_key: Float[Tensor, "... sequence_length d_k"],
        token_positions: Int[Tensor, "... sequence_length"]
    ) -> Float[Tensor, "... sequence_length d_k"]:
        tps = token_positions[..., :self.max_seq_len]
        coses = self.coses[tps]
        sines = self.sines[tps]

        in_pairs = einx.rearrange("... sl (dk2 pair) -> ... sl dk2 pair", in_query_or_key, pair=2)
        arrrs = torch.stack(
            (torch.stack([coses, -sines], dim=-1),
             torch.stack([sines, coses], dim=-1)),
            dim=-2
        )
        rotated = einx.dot("a... sl dk2 row col, b... sl dk2 col -> b... sl dk2 row", arrrs, in_pairs)
        rotated = einx.rearrange("... sl dk pair -> ... sl (dk pair)", rotated)
        return rotated


class Multihead(nn.Module):
    """Multi-head self-attention without RoPE."""
    
    def __init__(self, num_heads: int, d_model: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        sigma = 2 / (d_model + d_model)
        self.q_proj_weights = Parameter(torch.empty((d_model, d_model)))
        self.k_proj_weights = Parameter(torch.empty((d_model, d_model)))
        self.v_proj_weights = Parameter(torch.empty((d_model, d_model)))
        self.o_proj_weights = Parameter(torch.empty((d_model, d_model)))
        torch.nn.init.trunc_normal_(self.q_proj_weights, 0, sigma, -3 * sigma, 3 * sigma)
        torch.nn.init.trunc_normal_(self.k_proj_weights, 0, sigma, -3 * sigma, 3 * sigma)
        torch.nn.init.trunc_normal_(self.v_proj_weights, 0, sigma, -3 * sigma, 3 * sigma)
        torch.nn.init.trunc_normal_(self.o_proj_weights, 0, sigma, -3 * sigma, 3 * sigma)

    def forward(
        self,
        in_features: Float[Tensor, "... sequence_length d_in"],
        k_v_cache=None # or 2ple of Float[Tensor, "... num_heads hist d_head"]
    ) -> tuple[Float[Tensor, "... sequence_length d_out"], tuple]:
        head_dim = self.d_model // self.num_heads

        if k_v_cache is not None:
            # in_features: [... 1 d_model]
            Q     = in_features @ self.q_proj_weights.T                          # [... 1 d_model]
            k_next = in_features[..., -1:, :] @ self.k_proj_weights.T           # [... 1 d_model]
            v_next = in_features[..., -1:, :] @ self.v_proj_weights.T           # [... 1 d_model]
            Q      = einx.rearrange("... sl (h dq) -> ... h sl dq", Q,      h=self.num_heads, dq=head_dim)
            k_next = einx.rearrange("... sl (h dk) -> ... h sl dk", k_next,  h=self.num_heads, dk=head_dim)
            v_next = einx.rearrange("... sl (h dv) -> ... h sl dv", v_next,  h=self.num_heads, dv=head_dim)
            K = torch.cat((k_v_cache[0], k_next), dim=-2)  # [... h hist+1 d_head]
            V = torch.cat((k_v_cache[1], v_next), dim=-2)  # [... h hist+1 d_head]
            mask = None
        else:
            qkv_weights = einx.rearrange(
                "dm dk, dm dk, dm dk -> dm (dk + dk + dk)",
                self.q_proj_weights.T, self.k_proj_weights.T, self.v_proj_weights.T
            )
            QKV = in_features @ qkv_weights
            Q, K, V = einx.rearrange("... sl (dk + dk + dk) -> ... sl dk, ... sl dk, ... sl dk", QKV)
            Q = einx.rearrange("... sl (h dq) -> ... h sl dq", Q, h=self.num_heads, dq=head_dim)
            K = einx.rearrange("... sl (h dk) -> ... h sl dk", K, h=self.num_heads, dk=head_dim)
            V = einx.rearrange("... sl (h dv) -> ... h sl dv", V, h=self.num_heads, dv=head_dim)
            q_len, k_len = Q.size(-2), K.size(-2)
            mask = torch.ones((q_len, k_len), dtype=bool, device=Q.device)
            diagonal=k_len - q_len
            mask = torch.tril(mask, diagonal=diagonal)

        mask = torch.ones((Q.size(-2), K.size(-2)), dtype=bool, device=Q.device)
        mask = torch.tril(mask)

        sdpa = scaled_dot_product_attention(Q, K, V, mask)
        sdpa = einx.rearrange("... h sl d -> ... sl (h d)", sdpa)
        return sdpa @ self.o_proj_weights.T, (K, V)


class MultiheadRope(nn.Module):
    """Multi-head self-attention with RoPE."""
    
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        theta: float,
        max_seq_len: int
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads
        sigma = 2 / (d_model + d_model)
        self.q_proj_weights = Parameter(torch.empty((d_model, d_model)))
        self.k_proj_weights = Parameter(torch.empty((d_model, d_model)))
        self.v_proj_weights = Parameter(torch.empty((d_model, d_model)))
        self.o_proj_weights = Parameter(torch.empty((d_model, d_model)))
        torch.nn.init.trunc_normal_(self.q_proj_weights, 0, sigma, -3 * sigma, 3 * sigma)
        torch.nn.init.trunc_normal_(self.k_proj_weights, 0, sigma, -3 * sigma, 3 * sigma)
        torch.nn.init.trunc_normal_(self.v_proj_weights, 0, sigma, -3 * sigma, 3 * sigma)
        torch.nn.init.trunc_normal_(self.o_proj_weights, 0, sigma, -3 * sigma, 3 * sigma)
        self.rope = Rope(theta, self.d_head, max_seq_len)

    def forward(
        self,
        in_features: Float[Tensor, "... sequence_length d_in"],
        token_positions: Int[Tensor, "... sequence_length"],
        k_v_cache=None # or 2ple of Float[Tensor, "... num_heads hist d_head"]
    ) -> tuple[Float[Tensor, "... sequence_length d_out"], tuple]:

        if k_v_cache is not None:
            # in_features: [... 1 d_model], token_positions: [... 1] (position of new token)
            Q = in_features @ self.q_proj_weights.T                             # [... 1 d_model]
            k_next = (in_features[..., -1:, :] @ self.k_proj_weights.T)        # [... 1 d_model]
            v_next = (in_features[..., -1:, :] @ self.v_proj_weights.T)        # [... 1 d_model]
            Q     = einx.rearrange("... sl (h dq) -> ... h sl dq", Q,     h=self.num_heads, dq=self.d_head)
            k_next = einx.rearrange("... sl (h dk) -> ... h sl dk", k_next, h=self.num_heads, dk=self.d_head)
            v_next = einx.rearrange("... sl (h dv) -> ... h sl dv", v_next, h=self.num_heads, dv=self.d_head)
            # apply RoPE at new token position only, then append to post-RoPE cache
            Q      = self.rope.forward(Q,      token_positions)
            k_next = self.rope.forward(k_next, token_positions)
            K = torch.cat((k_v_cache[0], k_next), dim=-2)  # [... h hist+1 d_head]
            V = torch.cat((k_v_cache[1], v_next), dim=-2)  # [... h hist+1 d_head]
            mask = None
        else:
            qkv_weights = einx.rearrange(
                "dm dk, dm dk, dm dk -> dm (dk + dk + dk)",
                self.q_proj_weights.T, self.k_proj_weights.T, self.v_proj_weights.T
            )
            QKV = in_features @ qkv_weights
            Q, K, V = einx.rearrange("... sl (dk + dk + dk) -> ... sl dk, ... sl dk, ... sl dk", QKV)
            Q = einx.rearrange("... sl (h dq) -> ... h sl dq", Q, h=self.num_heads, dq=self.d_head)
            K = einx.rearrange("... sl (h dk) -> ... h sl dk", K, h=self.num_heads, dk=self.d_head)
            V = einx.rearrange("... sl (h dv) -> ... h sl dv", V, h=self.num_heads, dv=self.d_head)
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)
            q_len, k_len = Q.size(-2), K.size(-2)
            mask = torch.ones((q_len, k_len), dtype=bool, device=Q.device)
            diagonal=k_len - q_len
            mask = torch.tril(mask, diagonal=diagonal)

        sdpa = scaled_dot_product_attention(Q, K, V, mask)
        sdpa = einx.rearrange("... h sl d -> ... sl (h d)", sdpa)
        return sdpa @ self.o_proj_weights.T, (K, V)

