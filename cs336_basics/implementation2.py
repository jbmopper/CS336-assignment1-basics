import torch
import torch.nn as nn
from torch.nn.parameter import Parameter, UninitializedParameter
from jaxtyping import Float, Int, Bool
from torch import Tensor

__all__ =   ['MyLinear', 'MyEmbedding', 'MyRMSNorm']

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