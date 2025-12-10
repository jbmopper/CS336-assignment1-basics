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
    load_model,
)

import torch
import numpy as np


def setDeviceAndSeeds(config):
    """Checks for devices, adds to config, and sets RNG seed values."""
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

def setup_tokenizer(config) -> Tokenizer:
    """Sets up tokenizer from config."""
    vocab, merges = load_bpe(config["tokenizer_dir"])
    tokenizer = Tokenizer(vocab, merges)
    return tokenizer


def main():
    """Main decoding/chat function."""
    setDeviceAndSeeds(config)
    tokens, valid_tokens = getTokens(config)
    train(config, tokens, valid_tokens)

if __name__ == "__main__":
    main()
model = load_model(src)