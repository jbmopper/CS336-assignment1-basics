from cs336_basics import (
    Tokenizer, 
    TransformerLM, 
    AdamW, 
    get_lr_cosine_schedule,
    gradient_clipping,
    get_batch,
    crossentropy,
    load_checkpoint,
    save_bpe,
    load_bpe,
    save_checkpoint,
    train_bpe,
)

from tqdm.auto import tqdm
import wandb

import os
import math
import numpy as np
import torch
import time

from datetime import datetime
from torch.profiler import profile, ProfilerActivity

# Setup (paths, device, config)

config = dict(
    # device is set below

    # tokenizer
    training_text = "../data/TinyStoriesV2-GPT4-train.txt",
    validation_text = "../data/TinyStoriesV2-GPT4-valid.txt",
    vocab_size = 10000, # also model parameter
    special_tokens = ["<|endoftext|>"], # takes list, can make file later...
    tokenizer_dir = "../tokenizers/tinystories/",
    train_file="../tokenized/tinystories_train.npy",
    valid_file="../tokenized/tinystories_valid.npy",

    # model # small, ~10M parameters
    d_model=256,
    num_heads=4,
    num_layers=4,
    d_ff=1024,
    context_length=125,
    rope_theta=10000,


    # # Medium model (~50M params) - better results, slower
    # d_model = 512,
    # num_heads = 8,
    # num_layers = 6,
    # d_ff = 2048,
    # context_length = 512,

    # optimizer (adamw)
    optimizer_use_defaults = True,
    optimizer_lr = 0.,
    optimizer_betas = (0., 0.),
    optimizer_eps = 1e-08,
    optimizer_weight_decay = 0.,

    # lr scheduler (cosine with warmup)
    lr_max = 1e-3, # typical 1e-4 to 1e-3, samller models -> larger lr
    lr_min = 1e-4, # 10-100x smaller than max
    warmup_iters = 100, # num_iters // 5 or 10
    cos_iters = 1000, # includes warmup

    # If loss explodes → lower max_lr. If loss plateaus early → try higher max_lr or longer training.

    # wandb/logging
    wandb_entity = "jbmopper-0", 
    log_project = "cs336-a1",
    model = "course rope-transformer",
    optimizer = "course adamw",
    learning_schedule = "course cosine anneal w/warmup",
    dataset = "Tinystories",
    loss_func = "cross-entropy",
    run_name = "one",


    # training loop
    num_iters = 1000,
    batch_size = 32, # "Memory scales with batch_size × context_length × d_model"
    checkpoint_dir = "../checkpoints/",

    # other parameters
    rand_seed = 0,
    gradient_clip = 1.0, # will set in training loop
    save_every = 200,
    eval_every = 20,

)

def setDeviceAndSeeds(config):
    # device check
    if torch.backends.mps.is_available():
        config["device"] = "mps"
    elif torch.cuda.is_available():
        config["device"] = "cuda"
    else:
        config["device"] = "cpu"

    print(f"Using device {config['device']}.")

    # set seed
    # random.seed(config["rand_seed"])
    np.random.seed(config["rand_seed"])
    torch.manual_seed(config["rand_seed"])

    if config["device"] == "mps":
        torch.mps.manual_seed(config["rand_seed"])
    elif config["device"] == "cuda":
        # torch.backends.cuda.deterministic = True # old?
        torch.cuda.manual_seed_all(config["rand_seed"])
    
    print(f"Set seeds for torch/numpy and device {config['device']} to {config['rand_seed']}.")

def _get_or_create_tokenizer(config):
    """Load existing tokenizer or train a new one."""
    vocab_path = os.path.join(config["tokenizer_dir"], "vocab.json")
    merges_path = os.path.join(config["tokenizer_dir"], "merges.pkl")
    
    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        print("Loading existing tokenizer...")
        return load_bpe(config["tokenizer_dir"])
    else:
        print("Training new tokenizer...")
        vocab, merges = train_bpe(config["training_text"], config["vocab_size"], config["special_tokens"])
        save_bpe(config["tokenizer_dir"], vocab, merges)
        return vocab, merges

def _tokenize_and_save(config, text_file, output_file, file_type):
    """Tokenize a text file and save it if the output file doesn't exist."""
    if os.path.exists(output_file) and os.path.isfile(output_file):
        print(f"File {output_file} exists")
        return
    
    vocab, merges = _get_or_create_tokenizer(config)
    tokenizer = Tokenizer(vocab, merges)
    
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Tokenizing {file_type} file...")
    tokens = tokenizer.encode(text)
    
    # Save as contiguous array (memmap-compatible)
    arr = np.array(tokens, dtype=np.uint16)
    np.save(output_file, arr)
    
    print(f"Saved {len(tokens):,} tokens to {output_file}")

def getTokens(config) -> tuple[np.array, np.array]:
    # Tokenize training and validation files if needed
    _tokenize_and_save(config, config["training_text"], config["train_file"], "training")
    _tokenize_and_save(config, config["validation_text"], config["valid_file"], "validation")
    
    # Load tokenized data
    tokens = np.load(config["train_file"], mmap_mode='r')
    print(f"Training tokens loaded from {config['train_file']}, shape {tokens.shape}.")
    
    valid_tokens = np.load(config["valid_file"], mmap_mode='r')
    print(f"Validation tokens loaded from {config['valid_file']}, shape {valid_tokens.shape}.")
    
    return tokens, valid_tokens


