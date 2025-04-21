# import torch.tensor

def embeddings(weights, token_ids):
    return weights[token_ids]