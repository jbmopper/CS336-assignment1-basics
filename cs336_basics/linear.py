import torch

def linear(weights, in_features):
    return torch.matmul(in_features, weights.T)
