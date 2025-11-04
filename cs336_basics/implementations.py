import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['linear', 'embeddings', "SwiGLU", "rmsnorm", "softmax", "silu"]

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

def rmsnorm(eps, weights, in_features):
    """
    rms_norm = nn.RMSNorm(d_model, eps=eps)
    rms_norm.weight.data = weights
    return rms_norm(in_features)
    """
    rms = torch.sqrt(eps + torch.mean(in_features ** 2, dim=-1, keepdim=True))
    return (in_features/rms) * weights

def softmax(in_features, dim):
    adjusted_features = in_features - torch.max(in_features)
    exp_features = torch.exp(adjusted_features)
    return exp_features / torch.sum(exp_features, dim=dim, keepdim=True)

def silu(in_features):
    return in_features * torch.sigmoid(in_features)