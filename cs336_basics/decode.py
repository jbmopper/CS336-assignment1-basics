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
    softmax,
)

import torch
import numpy as np
import argparse
import prompt_toolkit



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

# TODO: 
# input/response handler
# forward -> logits -> top p sample -> tokens
# check to see if endoftext has happened and terminate when it happens
# also have a manual advance mode?




def main():
    """Main decoding/chat function."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="Path to checkpoint to load")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p-threshold", type=float, default=0.01)
    args = p.parse_args()

    config, model = load_model(args.ckpt)
    vocab, merges = load_bpe(config["tokenizer_dir"])
    tokenizer = Tokenizer(vocab, merges) # special tokens are already in the vocab at this point
    setDeviceAndSeeds(config)
    model.to(config["device"])

    print("Welcome to tinystories/assignment 1 inference.  Please submit input at the prompt, or type '!q' to quit.")
    quits = ["!q"]
    model.eval()
    with torch.inference_mode():
        while True:
            sub = input("T$ ").strip()
            if sub in quits:
                break
                
            inputs = tokenizer.encode(sub) 
            # turn list into a [1 len(inputs)] tensor of input
            inputs = torch.Tensor(inputs, device=config['device'])
            inputs.unsqueeze(0)
            logits = model.forward(inputs) # Float[Tensor, "batch seq vocab"]... seq?
            logits.divide_(args.temperature)
            probs = softmax(logits)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            top = cdf <= args.top_p_threshold
            
            










if __name__ == "__main__":
    main()
