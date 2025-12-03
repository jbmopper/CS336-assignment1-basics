import torch
import torch.nn as nn
from torch.nn.parameter import Parameter, UninitializedParameter
from jaxtyping import Float, Int, Bool
from torch import Tensor
import einx
from collections.abc import Callable, Iterable
from typing import Optional
import math

__all__ =   ['MyLinear', 'MyEmbedding', 'MyRMSNorm', 'Rope', 'Multihead', 'MultiheadRope',
                'MyAdamW', 'get_lr_cosine_schedule']

class MyLinear(nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features # just d_in?  TFA says to match the real API
        self.out_features = out_features # ??
        self.weight = Parameter(torch.empty((out_features, in_features)))
        self._sigma = 2/(in_features+out_features)
        torch.nn.init.trunc_normal_(self.weight, 0, (self._sigma), -3*self._sigma, 3*self._sigma)


    def forward(self, x: Float[Tensor, " ... d_in"]) -> Tensor:
        # W gets added to the test via assighment in the adapter
        return torch.matmul(x, self.weight.T)

class MyEmbedding(nn.Module):
    def __init__(self,
        num_embeddings, 
        embedding_dim, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = Parameter(torch.empty((num_embeddings, embedding_dim)))
        self._sigma = 2/(num_embeddings + embedding_dim)
        torch.nn.init.trunc_normal_(self.weight, 0, (self._sigma), -3*self._sigma, 3*self._sigma) 

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class MyRMSNorm(nn.Module):
    def __init__(self,
        d_model,
        eps, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(torch.ones((d_model))) # noting in TFA about initializing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(self.eps + torch.mean(x ** 2, dim=-1, keepdim=True))
        result = (x/rms) * self.weight
        return result.to(in_dtype)

def build_mthetas(theta: float,
    d_k: int,
    max_seq_len: int
    ) -> Float[Tensor, "... max_seq_len d_half"]:
    assert d_k % 2 == 0
    dk2 = d_k // 2
    thetas = torch.ones(dk2) * theta
    thetas = thetas ** ((-2 * torch.arange(dk2)) / (d_k)) # [d_k/2]

    # ms = token_positions[..., : max_seq_len] # [... max_seq_len]
    ms = torch.arange(max_seq_len) # all possible ms
    # mthetas = ms.unsqueeze(-1) * thetas.unsqueeze(0) # [..., max_seq_len, d_k/2]
    mthetas = einx.multiply("... seq, d -> ... seq d", ms, thetas) # ... not needed now

    return mthetas


class Rope(nn.Module):
    def __init__(self,
    theta: float,
    d_k: int,
    max_seq_len: int,
    device: torch.device | None = None
):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        mthetas = build_mthetas(theta, d_k, max_seq_len)
        coses = torch.cos(mthetas)
        sines = torch.sin(mthetas)
        self.register_buffer('coses', coses, persistent=False)
        self.register_buffer('sines', sines, persistent=False)


    def forward(self,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"], 
        token_positions: Int[Tensor, " ... sequence_length"]
        ) -> Float[Tensor, " ... sequence_length d_k"]:

        tps = token_positions[..., : self.max_seq_len] # really just [... seq_len]
        coses = self.coses[tps] # [... seq_len d_k//2], 
        sines = self.sines[tps] 

        # TODO: replace with non-full matrix implementation
        # rotate pairs of (q_(2k -1), q_(2k)) by theta_ik for k in 1...d_k/2
        # which is the pattern matrix... 

        in_pairs = einx.rearrange("... sl (dk2 pair) -> ... sl dk2 pair", in_query_or_key, pair=2)
        # [... seq_len d_k//2 2]
        arrrs = torch.stack(
            (torch.stack([coses, -sines], dim=-1),
            torch.stack([sines, coses], dim=-1)), 
            dim=-2
        ) # [... seq_len d_k//2 2 2]
        # rotated = einx.dot("... sl dk2 rot1 rot2, ... sl dk2 pair -> ... sl dk2 rot1", arrrs, in_pairs)
        rotated = einx.dot("a... sl dk2 row col, b... sl dk2 col -> b... sl dk2 row", arrrs, in_pairs)
        rotated = einx.rearrange("... sl dk pair -> ... sl (dk pair)", rotated)


        # coses_diag = torch.repeat_interleave(coses, 2, dim=-1)
        # R = torch.diag_embed(coses_diag) # [..., max_seq_len, d, d]
        # indices = torch.arange(d_k//2)
        # R[..., (indices * 2), (indices * 2) + 1] = neg_sines[..., indices] 
        # # zero-indexed rows 0, 2, ... (dim -2); columns 1, 3... (dim -1) 
        # R[..., (indices * 2) + 1, (indices * 2)] = sines[..., indices] 
        # # zero-indexed rows 1, 3, ... (dim -2); columns 0, 2... (dim -1) 
    

        # out = torch.einsum(R, [..., 0, 1, 2], in_query_or_key, [..., 0, 2], [..., 0, 1])
        # return out
        return rotated

def softmax(in_features, dim):
    # adjusted_features = in_features - torch.max(in_features) 
    adjusted_features = in_features - torch.max(in_features, dim=dim, keepdim=True)[0]
    exp_features = torch.exp(adjusted_features)
    return exp_features / torch.sum(exp_features, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... values d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None
) -> Float[Tensor, "... queries d_v"]:

    d_k = Q.size(-1)
    scaled_product = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))
    if mask is not None: 
        scaled_product = scaled_product.masked_fill_(~mask, float('-inf'))
    output = softmax(scaled_product, -1) @ V
    return output

class Multihead(nn.Module):
    def __init__(self,
        num_heads: int,
        d_model: int,
        # q_proj_weight: Float[Tensor, " d_k d_in"],
        # k_proj_weight: Float[Tensor, " d_k d_in"],
        # v_proj_weight: Float[Tensor, " d_v d_in"],
        # o_proj_weight: Float[Tensor, " d_model d_v"],
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # self.d_k = self.d_v = d_model//num_heads # actually no
        # self.d_k = self.d_v = self.d_model # initially dk * heads
        self._sigma = 2/(self.d_model+self.d_model)
        self.q_proj_weights = Parameter(torch.empty((self.d_model, self.d_model)))
        self.k_proj_weights = Parameter(torch.empty((self.d_model, self.d_model)))
        self.v_proj_weights = Parameter(torch.empty((self.d_model, self.d_model)))
        self.o_proj_weights = Parameter(torch.empty((self.d_model, self.d_model)))
        torch.nn.init.trunc_normal_(self.q_proj_weights, 0, (self._sigma), -3*self._sigma, 3*self._sigma)
        torch.nn.init.trunc_normal_(self.k_proj_weights, 0, (self._sigma), -3*self._sigma, 3*self._sigma)
        torch.nn.init.trunc_normal_(self.v_proj_weights, 0, (self._sigma), -3*self._sigma, 3*self._sigma)
        torch.nn.init.trunc_normal_(self.o_proj_weights, 0, (self._sigma), -3*self._sigma, 3*self._sigma)

    def forward(self, in_features: Float[Tensor, " ... sequence_length d_in"]) -> Float[Tensor, " ... sequence_length d_out"]:
        qkv_weights = einx.rearrange( # want to keep dm for in @ QKV [d_seq dm] @ [dm 3*dk]
        # also want to separate out heads
            "dm dk, dm dk, dm dk -> dm (dk + dk + dk)", 
            self.q_proj_weights.T, self.k_proj_weights.T, self.v_proj_weights.T
        )   
        QKV = in_features @ qkv_weights # [seq_length 3*dk]
        # so last time I worked to # [..., num_heads, seq_len, d_k (or v) // num_heads]
        head_dim = self.d_model // self.num_heads
       # Q, K, V = einx.rearrange(
       #     "... sl ((h dk) + (h dk) + (h dk)) -> ... h sl dk, ... h sl dk, ... h sl dk", 
       #     QKV, h=self.num_heads, dk=head_dim)
        Q, K, V = einx.rearrange("... sl (dk + dk + dk) -> ... sl dk, ... sl dk, ... sl dk", QKV)
        Q = einx.rearrange("... sl (h dq) -> ... h sl dq", Q, h=self.num_heads, dq=head_dim)
        K = einx.rearrange("... sl (h dk) -> ... h sl dk", K, h=self.num_heads, dk=head_dim)
        V = einx.rearrange("... sl (h dv) -> ... h sl dv", V, h=self.num_heads, dv=head_dim)

        mask = torch.ones((Q.size(-2), Q.size(-2)), dtype=bool)
        mask = torch.tril(mask)

        sdpa = scaled_dot_product_attention(Q, K, V, mask)
        # [..., num_heads, seq_len, head_dim]
        sdpa = einx.rearrange("... h sl d -> ... sl (h d)", sdpa)
        # [..., seq-lem, d_v]
        # o is [d_model, d_v] so transpose
        return sdpa @ self.o_proj_weights.T

class MultiheadRope(nn.Module):
    def __init__(self,
        num_heads: int,
        d_model: int,
        theta: int,
        max_seq_len: int
        # token_positions: int?
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model//num_heads 
        # self.d_k = self.d_v = self.d_model # initially dk * heads
        self._sigma = 2/(self.d_model+self.d_model)
        self.q_proj_weights = Parameter(torch.empty((self.d_model, self.d_model)))
        self.k_proj_weights = Parameter(torch.empty((self.d_model, self.d_model)))
        self.v_proj_weights = Parameter(torch.empty((self.d_model, self.d_model)))
        self.o_proj_weights = Parameter(torch.empty((self.d_model, self.d_model)))
        torch.nn.init.trunc_normal_(self.q_proj_weights, 0, (self._sigma), -3*self._sigma, 3*self._sigma)
        torch.nn.init.trunc_normal_(self.k_proj_weights, 0, (self._sigma), -3*self._sigma, 3*self._sigma)
        torch.nn.init.trunc_normal_(self.v_proj_weights, 0, (self._sigma), -3*self._sigma, 3*self._sigma)
        torch.nn.init.trunc_normal_(self.o_proj_weights, 0, (self._sigma), -3*self._sigma, 3*self._sigma)
        self.rope = Rope(theta, self.d_head, max_seq_len)

    def forward(self, 
        in_features: Float[Tensor, " ... sequence_length d_in"], 
        token_positions: Int[Tensor, " ... sequence_length"]
        ) -> Float[Tensor, " ... sequence_length d_out"]:
        qkv_weights = einx.rearrange( # want to keep dm for in @ QKV [d_seq dm] @ [dm 3*dk]
        # also want to separate out heads
            "dm dk, dm dk, dm dk -> dm (dk + dk + dk)", 
            self.q_proj_weights.T, self.k_proj_weights.T, self.v_proj_weights.T
        )   
        QKV = in_features @ qkv_weights # [seq_length 3*dk]
        # so last time I worked to # [..., num_heads, seq_len, d_k (or v) // num_heads]
       # Q, K, V = einx.rearrange(
       #     "... sl ((h dk) + (h dk) + (h dk)) -> ... h sl dk, ... h sl dk, ... h sl dk", 
       #     QKV, h=self.num_heads, dk=head_dim)
        Q, K, V = einx.rearrange("... sl (dk + dk + dk) -> ... sl dk, ... sl dk, ... sl dk", QKV)
        Q = einx.rearrange("... sl (h dq) -> ... h sl dq", Q, h=self.num_heads, dq=self.d_head)
        K = einx.rearrange("... sl (h dk) -> ... h sl dk", K, h=self.num_heads, dk=self.d_head)
        V = einx.rearrange("... sl (h dv) -> ... h sl dv", V, h=self.num_heads, dv=self.d_head)

        Q = self.rope.forward(Q, token_positions)
        K = self.rope.forward(K, token_positions)
        # no rope for V
        mask = torch.ones((Q.size(-2), Q.size(-2)), dtype=bool)
        mask = torch.tril(mask)

        sdpa = scaled_dot_product_attention(Q, K, V, mask)
        # [..., num_heads, seq_len, head_dim]
        sdpa = einx.rearrange("... h sl d -> ... sl (h d)", sdpa)
        # [..., seq-lem, d_v]
        # o is [d_model, d_v] so transpose
        return sdpa @ self.o_proj_weights.T

def get_lr_cosine_schedule(    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        # rise over run ... warmup from 0?
        lr = (max_learning_rate / warmup_iters) * it
        
    elif (warmup_iters <= it <= cosine_cycle_iters):
        # recursive approximation from torch.optim.lr_scheduler.CosineAnnealingLR.html
        # eta_next = (min_learning_rate + 
        #     (eta_now - min_learning_rate) * 
        #     ((1 + torch.cos( ((it+1) * pi) / cosine_cycle_iters ) ) / 
        #     (1 + torch.cos((it * pi) / cosine_cycle_iters ))) 
        # )
        # should use the closed form... from the assignment!
        cos_it = it - warmup_iters
        cos_tot_it = cosine_cycle_iters - warmup_iters
        lr = ( min_learning_rate   
            + 0.5 * (max_learning_rate - min_learning_rate) 
            * (1 + math.cos((cos_it * math.pi) / (cos_tot_it))))
    
    else:
        lr = min_learning_rate
    
    return lr

class MyAdamW(torch.optim.Optimizer):
    def __init__(self,
        params,
        lr=1e-03,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-02,) -> None:
            defaults = dict(
                betas=betas,
                eps=eps,
                lr = lr,
                weight_decay= weight_decay
            )
            super().__init__(params, defaults) 

    # bot recommends @toch.no_grad / with torch.enable_grad():
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        # loss = None if closure is None else closure()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        

        for group in self.param_groups:
            # get layer-specific parameters e.g. lr = group["lr"]

            for p in group["params"]:
                # the actual weight tensors
                if p.grad is None:
                    continue # so indented

                state = self.state[p] 
                t = state.get("t", 0)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                betas = group["betas"]                
                lr = group["lr"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                
                t += 1
                state["t"] = t
                m.mul_(betas[0]).add_((1-betas[0]) * p.grad.data)
                state["m"] = m
                v.mul_(betas[1]).add_((1-betas[1]) * (p.grad.data ** 2))
                state["v"] = v
                # don't need to update m or v because in-place
                alpha_t = lr * (((1 - betas[1]**t)**0.5)/(1 - betas[0]**t))
                p.data -=  alpha_t * (m / (torch.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data

        return loss