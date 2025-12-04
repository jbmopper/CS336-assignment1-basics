from implementation2 import *
from implementations import *
import json
import torch
from torch import nn, Tensor
import numpy as np
import os

# model parameters:
# from transformer_lm
# - vocab_size
# - d_model,
# - num_heads,
# - num_layers,
# - d_ff,
# - context_length, 
# - rope_theta
# - batch_size
# - device

# paths:
# - memmap
# - hyperparameters
# - data path/mmap
# - checkpoints
# - output? decode later...

# doesn't sound like we want the tokenizer but we will want output...
# will want devices...

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
# import random
# radom.seed(seed)

def get_setup(file):
    directory 


# per lecture:
# - have tokenizer output the integer list representing the tokenized corpus
# - orig_data = no.array([1, 2, ...], dtype=np.int32)
# - orig_data.tofile("data.npy")
# - load data:
# - data = np.memmap("data.npy", dtpye=np.int32)
# then the data is an np.array and can be used by the loader... just werks>

def load_data(path, dtype: np.dtype=np.float32: np.dtype, data_size) -> np.NDArray:
    with 



def create_model(config) -> nn.Module:

