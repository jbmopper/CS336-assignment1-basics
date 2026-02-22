from cs336_basics import (
    Tokenizer,
    load_bpe,
    load_model,
    softmax,
)

import torch
import numpy as np
import argparse
import signal
import sys


def setDeviceAndSeeds(config):
    """Checks for devices, adds to config, and sets RNG seed values."""
    if torch.backends.mps.is_available():
        config["device"] = "mps"
    elif torch.cuda.is_available():
        config["device"] = "cuda"
    else:
        config["device"] = "cpu"

    print(f"Using device {config['device']}.")

    np.random.seed(config["rand_seed"])
    torch.manual_seed(config["rand_seed"])

    if config["device"] == "mps":
        torch.mps.manual_seed(config["rand_seed"])
    elif config["device"] == "cuda":
        torch.cuda.manual_seed_all(config["rand_seed"])

    print(f"Set seeds for torch/numpy and device {config['device']} to {config['rand_seed']}.")


def sample_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample from logits using top-p (nucleus) sampling.
    
    Args:
        logits: Shape [batch, vocab_size] logits from model
        temperature: Temperature for scaling logits
        top_p: Cumulative probability threshold for nucleus sampling
    
    Returns:
        Tensor of sampled token IDs, shape [batch]
    """
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits, -1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for tokens within top-p threshold
    top_mask = cdf <= top_p
    # Always include at least the first token (highest prob)
    next_elements = top_mask.sum(dim=-1)
    top_mask[0, next_elements[0]] = True
    
    # Zero out tokens outside nucleus and renormalize
    sorted_probs = sorted_probs.masked_fill_(~top_mask, 0.0)
    prob_sum = sorted_probs.sum(dim=-1, keepdim=True)
    sorted_probs = sorted_probs / prob_sum.clamp(min=1e-8)
    
    # Sample and map back to original vocab indices
    selected = torch.multinomial(sorted_probs, 1)
    token_ids = sorted_idx.gather(1, selected).flatten()
    return token_ids


def handle_sigint(signum, frame):
    print("\nexiting")
    sys.exit(0)


def main():
    """Main decoding/chat function."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="Path to checkpoint to load")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p-threshold", type=float, default=0.9)
    args = p.parse_args()

    config, model = load_model(args.ckpt)
    
    # Use defaults for tokenizer if not in config
    tokenizer_dir = config.get("tokenizer_dir", "tokenizers/tinystories")
    special_tokens = config.get("special_tokens", ["<|endoftext|>"])
    
    vocab, merges = load_bpe(tokenizer_dir)
    tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens)
    setDeviceAndSeeds(config)
    model.to(config["device"])

    end_of_text_token = "<|endoftext|>"
    context_length = config.get("context_length")
    if context_length is None:
        model_cfg = config.get("model_settings", config)
        context_length = model_cfg.get("context_length", 256)

    print("Welcome to tinystories/assignment 1 inference.  Please submit input at the prompt, or type '!q' to quit.")
    quits = ["!q"]
    model.eval()

    signal.signal(signal.SIGINT, handle_sigint)

    with torch.inference_mode():
        while True:
            sub = input("T$ ")
            if sub in quits:
                break

            # Accumulated text (prompt + generated) for conditioning; we show prompt then stream the rest
            output = sub
            # Print prompt so the line starts with user input, then we'll append generated tokens
            print(output, end="", flush=True)

            for _ in range(args.max_new_tokens):
                input_ids = tokenizer.encode(output)[-context_length:]
                if not input_ids:
                    input_ids = tokenizer.encode(end_of_text_token)
                inputs = torch.tensor(input_ids, device=config["device"], dtype=torch.long)
                inputs = inputs.unsqueeze(0)

                logits = model.forward(inputs)
                pred_logit = logits[:, -1, :]
                token_ids = sample_top_p(pred_logit, args.temperature, args.top_p_threshold)
                new_text = tokenizer.decode(token_ids.tolist())

                if new_text == end_of_text_token:
                    break
                output = output + new_text
                print(new_text, end="", flush=True)

            print()  # newline after this completion


if __name__ == "__main__":
    main()
