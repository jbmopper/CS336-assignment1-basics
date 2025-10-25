import torch

__all__ = ['linear', 'embeddings']

def linear(weights, in_features):
    return torch.matmul(in_features, weights.T)

def embeddings(weights, token_ids):
    return weights[token_ids]

