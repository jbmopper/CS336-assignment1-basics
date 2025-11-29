import torch
import torch.nn as nn
from torch.nn.parameter import Parameter, UninitializedParameter
from jaxtyping import Float, Int, Bool
from torch import Tensor

__all__ =   ['MyLinear']

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