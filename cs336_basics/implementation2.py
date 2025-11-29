import torch
import torch.nn as nn
from torch.nn.parameter import Parameter, UninitializedParameter
from jaxtyping import Float, Int, Bool
from torch import Tensor

__all__ =   ['MyLinear', 'MyEmbedding', 'MyRMSNorm', 'Rope']

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
        self.embeddings = Parameter(torch.empty((num_embeddings, embedding_dim)))
        self._sigma = 2/(num_embeddings + embedding_dim)
        torch.nn.init.trunc_normal_(self.embeddings, 0, (self._sigma), -3*self._sigma, 3*self._sigma) 

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]

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
        self.weights = Parameter(torch.ones((d_model))) # noting in TFA about initializing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(self.eps + torch.mean(x ** 2, dim=-1, keepdim=True))
        result = (x/rms) * self.weights
        return result.to(in_dtype)

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

    def forward(self,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"], 
        token_positions: Int[Tensor, " ... sequence_length"]
        ) -> Float[Tensor, " ... sequence_length d_k"]:

        assert self.d_k % 2 == 0
        dk2 = self.d_k // 2
        thetas = torch.ones(dk2) * self.theta
        theta_exponents = (-2 * torch.arange(dk2)) / (self.d_k)
        thetas = thetas ** theta_exponents

        ms = token_positions[..., : self.max_seq_len]
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
