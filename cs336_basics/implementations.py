import torch
import torch.nn as nn
from torch import Tensor, pi
from jaxtyping import Float, Int, Bool
import numpy.typing as npt
from collections.abc import Iterable
import math
from cs336_basics.implementation2 import *


__all__ =   [ 
                'linear', 'embeddings', 'SwiGLU', 'rmsnorm', 'softmax', 'silu',
                'crossentropy', 'scaled_dot_product_attention', 'rope',
                'multihead_self_attention', 'multihead_self_attention_with_rope',
                'transformer_block', 'transformer_lm', 'get_batch',
                'gradient_clipping', 'get_lr_cosine_schedule'
            ]

def linear(weights, in_features):
    return torch.matmul(in_features, weights.T)

def embeddings(weights, token_ids):
    return weights[token_ids]

class SwiGLU(nn.Module):
    
    def __init__(self, d_model, d_ff) -> None:
        super().__init__()
        self.w1 = MyLinear(d_model, d_ff)
        self.w2 = MyLinear(d_ff, d_model)
        self.w3 = MyLinear(d_model, d_ff)
        # w1_weight, w2_weight, w3_weight
        # get added by the calling function...
        # self.w1 = nn.Linear(d_model, d_ff, bias=False)
        # self.w2 = nn.Linear(d_ff, d_model, bias=False)
        # self.w3 = nn.Linear(d_model, d_ff, bias=False)
        # self.swish = nn.SiLU()
        self.swish = lambda x: x * torch.sigmoid(x)
        

    def forward(self, x):
        return self.w2(self.swish(self.w1(x)) * self.w3(x))

def rmsnorm(eps, weights, in_features):
    """
    rms_norm = nn.RMSNorm(d_model, eps=eps)
    rms_norm.weight.data = weights
    return rms_norm(in_features)
    """
    rms = torch.sqrt(eps + torch.mean(in_features ** 2, dim=-1, keepdim=True))
    return (in_features/rms) * weights

def softmax(in_features, dim):
    # adjusted_features = in_features - torch.max(in_features) 
    adjusted_features = in_features - torch.max(in_features, dim=dim, keepdim=True)[0]
    exp_features = torch.exp(adjusted_features)
    return exp_features / torch.sum(exp_features, dim=dim, keepdim=True)

def silu(in_features):
    return in_features * torch.sigmoid(in_features)

def crossentropy(inputs, targets):
    # for each batch, the target has the index of the correct class
    # so if that is class c, the cross entropy is just -log(input[c])
    # thus select the target index from the input and average -log values
    rows = torch.arange(inputs.shape[0])
    probs = torch.log_softmax(inputs, dim=1) # researched numerical stability
    used_probs = probs[rows, targets] 
    return -torch.mean(used_probs)
    
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

def rope(  d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    theta_i = theta^(-2(i-1)/d) where d in {1...d_k/2} (?)
    originally 1000^(-2(i-1)/d)
    in_query_or_key is W^k@X or W^q@X
    for each x in X (up to sequence length), generate the
    vectors (per section 3.4.2) [cos(m*theta_i)] and [sin(m*theta_i)]
    ...no, this is a whole R_theta approach, apparently...
    so we want to generate a block matrix for the m*theta_i's
    for each m up to max_sequence_length
    """
    assert d_k % 2 == 0
    dk2 = d_k // 2
    thetas = torch.ones(dk2) * theta
    theta_exponents = (-2 * torch.arange(dk2)) / (d_k)
    thetas = thetas ** theta_exponents

    ms = token_positions[..., : max_seq_len]
    mthetas = ms.unsqueeze(-1) * thetas.unsqueeze(0) # [..., max_seq_len, d_k/2]
    # still have inputs in rows (?) and want R tensor [..., d_k, max_seq_len]
    # so each R slice goes across the columns 
    coses = torch.cos(mthetas)
    coses_diag = torch.repeat_interleave(coses, 2, dim=-1)
    sines = torch.sin(mthetas)
    neg_sines = -sines
    R = torch.diag_embed(coses_diag) # [..., max_seq_len, d, d]
    indices = torch.arange(dk2)
    R[..., (indices * 2), (indices * 2) + 1] = neg_sines[..., indices] 
    # zero-indexed rows 0, 2, ... (dim -2); columns 1, 3... (dim -1) 
    R[..., (indices * 2) + 1, (indices * 2)] = sines[..., indices] 
    # zero-indexed rows 1, 3, ... (dim -2); columns 0, 2... (dim -1) 
   

    out = torch.einsum(R, [..., 0, 1, 2], in_query_or_key, [..., 0, 2], [..., 0, 1])
    return out

def multihead_self_attention(
    # d_model: int, # is this needed?
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    # so... I guess we run attention num_heads times,
    # concatenate the results, and put it through O...
    Q = in_features @ q_proj_weight.T # [..., seq_len, d_k]
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T # [..., seq_len, d_v]

    qs = Q.split(Q.size(-1)//num_heads, -1)
    ks = K.split(K.size(-1)//num_heads, -1)
    vs = V.split(V.size(-1)//num_heads, -1)

    qs = torch.stack(qs, -3)
    ks = torch.stack(ks, -3)
    vs = torch.stack(vs, -3)
    # [..., num_heads, seq_len, d_k (or v) // num_heads]

    # mas should be seq_len x seq_len
    mask = torch.ones((qs.size(-2), qs.size(-2)), dtype=bool)
    mask = torch.tril(mask)

    sdpa = scaled_dot_product_attention(qs, ks, vs, mask)
    # [..., num_heads, seq_len, d_v // num_heads]
    
    sdpa = sdpa.transpose(-3, -2)
    # [..., seq_len, num_heads, d_v // num_heads]
    sdpa_concat = sdpa.flatten(start_dim=-2)
    # [..., seq-lem, d_v]
    # o is [d_model, d_v] so transpose
    return sdpa_concat @ o_proj_weight.T

def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    # This implementation should handle the key, query, and value projections for all heads in a single matrix multiply.
    # In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    # because why not? 
    # then need to repeat MHSA logic

    Q = in_features @ q_proj_weight.T # [..., seq_len, d_k]
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T # [..., seq_len, d_v]

    # need d_model // num_heads 
    # rope_dim = o_proj_weight.size(-2) // num_heads # could've just used d_model argument...
    rope_dim = d_model // num_heads
    qs = Q.split(rope_dim, -1)
    ks = K.split(rope_dim, -1)
    # reshape instead? will sizes conform? 

    vs = V.split(V.size(-1)//num_heads, -1)

    qs = torch.stack(qs, -3)
    ks = torch.stack(ks, -3) 
    # [..., num_heads, seq_len, d_model / num_heads]
    vs = torch.stack(vs, -3)

    # d_k // num_heads?
    d_k = qs.size(-1)

    qs = rope(d_k, theta, max_seq_len, qs, token_positions)
    ks = rope(d_k, theta, max_seq_len, ks, token_positions)

    # repeating MHSA 
    mask = torch.ones((qs.size(-2), qs.size(-2)), dtype=bool)   
    mask = torch.tril(mask)

    sdpa = scaled_dot_product_attention(qs, ks, vs, mask)
    # [..., num_heads, seq_len, d_v // num_heads]
    
    sdpa = sdpa.transpose(-3, -2)
    # [..., seq_len, num_heads, d_v // num_heads]
    sdpa_concat = sdpa.flatten(start_dim=-2)
    # [..., seq-lem, d_v]
    # o is [d_model, d_v] so transpose
    return sdpa_concat @ o_proj_weight.T


class transformer_block(nn.Module):
    def __init__(self, 
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        theta
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.eps = 1e-5 # from testing/common practice

    def forward(self, weights, in_features):
        norm1 = rmsnorm(self.eps, weights["ln1.weight"], in_features)
        token_positions = torch.arange(in_features.size(-2), dtype=int) # 0...sequence_length
        attention_output = multihead_self_attention_with_rope(
            self.d_model,
            self.num_heads,
            self.max_seq_len,
            self.theta,
            weights["attn.q_proj.weight"],
            weights["attn.k_proj.weight"],
            weights["attn.v_proj.weight"],
            weights["attn.output_proj.weight"],
            norm1,
            token_positions
        )
        in_features = in_features + attention_output
        norm2 = rmsnorm(self.eps, weights["ln2.weight"], in_features)
        swiglu = SwiGLU(self.d_model, self.d_ff)
        swiglu.w1.weight.data = weights["ffn.w1.weight"]
        swiglu.w2.weight.data = weights["ffn.w2.weight"]
        swiglu.w3.weight.data = weights["ffn.w3.weight"]

        ffn_output = swiglu.forward(norm2)
        return in_features + ffn_output

class transformer_lm(nn.Module):
    def __init__(self, 
        vocab_size,# make argumnent to forward? needed?
        d_model,
        num_heads,
        num_layers,
        d_ff,
        context_length, # assuming samw 
        rope_theta,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size # make argumnent to forward? needed?
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = context_length # assuming samw 
        self.theta = rope_theta
        self.transformers = nn.ModuleList([
            transformer_block(d_model, num_heads, d_ff, context_length, rope_theta)
            for i in range(num_layers)
        ])
        self.eps = 1e-5 # from testing/common practice

    def forward(self, weights, in_indices):
        x = embeddings(weights["token_embeddings.weight"], in_indices)
        for i, l in enumerate(self.transformers): # so it goes
            weight_prefix = f"layers.{i}."
            layer_weights = {
                k.replace(weight_prefix, ""): v
                for k, v in weights.items()
                if k.startswith(weight_prefix) # 
            }
            x = l.forward(layer_weights, x)
            
        x = rmsnorm(self.eps, weights["ln_final.weight"], x) 
        # [... d_model]
        return x @ weights["lm_head.weight"].T # [... vocab_size]

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:

    starts = torch.randint(low=0, high=(len(dataset) - context_length), size=(batch_size,))
    indices = starts.unsqueeze(1) + torch.arange(context_length) # [batch_size, context_length]

    inputs = torch.tensor(dataset[indices.numpy()], dtype=torch.long, device=device)
    labels = torch.tensor(dataset[indices.numpy() +1], dtype=torch.long, device=device)
    # need the whole sequence; randint "high=" is exclusive so +1 is OK
    return (inputs, labels)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    gradient = torch.cat([
        torch.flatten(x.grad) 
        for x in parameters
        if x.grad is not None
    ])
    # this could be made more efficient by sqrt-ing sum({list comprehension that squares not None params})
    # also doing param.grad.mul_() which is more "in place", and precompute the clip coefficient

    if (norm:= torch.linalg.norm(gradient)) > max_l2_norm:
        # map(lambda x: x.grad -> x.grad/(norm/max_l2_norm), parameters )
        for param in parameters:
            if param.grad is not None:
                param.grad = param.grad / (norm/max_l2_norm)

    return None 

class MyAdamW(torch.optim.Optimizer):
    def __init__(self,
    params,
    lr=0.001,
    betas=(0.9, .999),
    eps=1e-08,
    weight_decay=0.01
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    
    def step(self):
        for group in self.param_groups:
            lr = group['lr'] # for example
            for param in group['params']:
                if param.grad is not None:
                    param = - lr * param.grad # for example




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
            * (1 + math.cos((cos_it * pi) / (cos_tot_it))))
    
    else:
        lr = min_learning_rate
    
    return lr
