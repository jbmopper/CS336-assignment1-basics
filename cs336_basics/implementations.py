import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['linear', 'embeddings', "SwiGLU"]

def linear(weights, in_features):
    return torch.matmul(in_features, weights.T)

def embeddings(weights, token_ids):
    return weights[token_ids]

class SwiGLU(nn.Module):
    
    def __init__(self, d_model, d_ff) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        # w1_weight, w2_weight, w3_weight
        # get added by the calling function...
        self.swish = nn.SiLU()

    def forward(self, x):
        return self.w2(self.swish(self.w1(x))*self.w3(x))