#!/usr/bin/env python3
"""
WandB Sweep Training Script

This script is designed to be called by wandb sweep agents. It receives
hyperparameters from wandb.config and trains the transformer model.

Usage:
    # Initialize sweep (run once):
    wandb sweep cs336_basics/sweep_config.yaml
    
    # Start agent(s) to run training:
    wandb agent <entity>/<project>/<sweep_id>
"""

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
import argparse

from datetime import datetime
from contextlib import contextmanager


# Default configuration (will be overridden by sweep)
DEFAULT_CONFIG = dict(
    # Paths
    training_text="../data/TinyStoriesV2-GPT4-train.txt",
    validation_text="../data/TinyStoriesV2-GPT4-valid.txt",
    tokenizer_dir="../tokenizers/tinystories/",
    train_file="../tokenized/tinystories_train.npy",
    valid_file="../tokenized/tinystories_valid.npy",
    checkpoint_dir="../checkpoints/sweeps/",
    
    # Checkpointing
    save_best=True,       # Save checkpoint when eval_loss improves
    save_final=True,      # Save checkpoint at end of training

    # Tokenizer
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],

    # Model architecture (defaults, will be overridden by sweep)
    d_model=512,
    num_heads=16,
    num_layers=4,
    d_ff=1344,
    context_length=256,
    rope_theta=10000,

    # Training parameters
    batch_size=32,
    num_iters=1000,
    gradient_clip=1.0,

    # Learning rate schedule
    lr_max=1e-3,
    lr_min=1e-4,
    warmup_iters=100,
    cos_iters=1000,

    # Optimizer
    weight_decay=0.01,

    # Evaluation
    eval_every=50,
    eval_batches=5,  # Number of batches for validation

    # Logging
    wandb_entity="jbmopper-0",
    log_project="cs336-a1-sweep",

    # Other
    rand_seed=42,
)


def set_device_and_seeds(seed: int) -> str:
    """Set up device and random seeds. Returns device string."""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == "mps":
        torch.mps.manual_seed(seed)
    elif device == "cuda":
        torch.cuda.manual_seed_all(seed)

    print(f"Using device: {device}")
    return device


def get_tokens(config: dict) -> tuple:
    """Load tokenized data."""
    if not os.path.exists(config["train_file"]):
        raise FileNotFoundError(
            f"Training tokens not found at {config['train_file']}. "
            "Run the main training script first to generate tokenized data."
        )

    tokens = np.load(config["train_file"], mmap_mode="r")
    valid_tokens = np.load(config["valid_file"], mmap_mode="r")

    print(f"Loaded {len(tokens):,} training tokens, {len(valid_tokens):,} validation tokens")
    return tokens, valid_tokens


@contextmanager
def timer(name: str, log_dict: dict = None):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if log_dict is not None:
        log_dict[name] = elapsed


def evaluate(model, valid_tokens, config, device) -> tuple:
    """Run evaluation on validation set. Returns (avg_loss, avg_perplexity)."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(config["eval_batches"]):
            inputs, labels = get_batch(
                valid_tokens,
                config["batch_size"],
                config["context_length"],
                device
            )
            logits = model(inputs)
            loss = crossentropy(logits, labels)
            total_loss += loss.item()

    avg_loss = total_loss / config["eval_batches"]
    avg_perplexity = math.exp(avg_loss)
    return avg_loss, avg_perplexity


def train_sweep():
    """Main training function for sweep runs."""
    # Initialize wandb run - this will get sweep config
    run = wandb.init()

    # Build config from defaults + sweep parameters
    config = DEFAULT_CONFIG.copy()
    
    # Create unique checkpoint directory for this run
    run_checkpoint_dir = os.path.join(config["checkpoint_dir"], run.id)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    config["run_checkpoint_dir"] = run_checkpoint_dir

    # Override with sweep parameters
    sweep_params = dict(wandb.config)
    for key, value in sweep_params.items():
        if key in config:
            config[key] = value
            print(f"Sweep override: {key} = {value}")

    # Ensure cos_iters matches num_iters
    config["cos_iters"] = config["num_iters"]

    # Build model settings dict
    model_settings = dict(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        context_length=config["context_length"],
        rope_theta=config["rope_theta"],
    )

    # Validate model settings
    if model_settings["d_model"] % model_settings["num_heads"] != 0:
        print(f"Invalid config: d_model ({model_settings['d_model']}) must be divisible by num_heads ({model_settings['num_heads']})")
        wandb.finish(exit_code=1)
        return

    # Set up device and seeds
    device = set_device_and_seeds(config["rand_seed"])

    # Load data
    try:
        tokens, valid_tokens = get_tokens(config)
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        wandb.finish(exit_code=1)
        return

    # Create model
    print(f"Creating model with settings: {model_settings}")
    model = TransformerLM(**model_settings).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    wandb.log({"num_parameters": num_params})

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr_max"],
        weight_decay=config["weight_decay"],
    )

    # Training loop
    best_eval_loss = float("inf")

    for step in tqdm(range(config["num_iters"]), desc="Training"):
        log = {}

        # Sync device for accurate timing
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        # Get batch
        inputs, labels = get_batch(
            tokens,
            config["batch_size"],
            config["context_length"],
            device
        )

        # Forward pass
        model.train()
        logits = model(inputs)
        loss = crossentropy(logits, labels)

        log["train_loss"] = loss.item()
        log["train_perplexity"] = math.exp(loss.item())

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf")
        )
        log["grad_norm"] = grad_norm.item()

        gradient_clipping(model.parameters(), config["gradient_clip"])

        # Learning rate schedule
        lr = get_lr_cosine_schedule(
            step,
            config["lr_max"],
            config["lr_min"],
            config["warmup_iters"],
            config["cos_iters"]
        )
        log["learning_rate"] = lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Optimizer step
        optimizer.step()

        # Evaluation
        if step % config["eval_every"] == 0 or step == config["num_iters"] - 1:
            eval_loss, eval_perplexity = evaluate(model, valid_tokens, config, device)
            log["eval_loss"] = eval_loss
            log["eval_perplexity"] = eval_perplexity

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                log["best_eval_loss"] = best_eval_loss
                
                # Save best checkpoint
                if config["save_best"]:
                    best_path = os.path.join(config["run_checkpoint_dir"], "best.pt")
                    save_checkpoint(model, optimizer, step, best_path, config)
                    print(f"Saved best checkpoint (eval_loss={eval_loss:.4f})")

            print(f"Step {step}: train_loss={loss.item():.4f}, eval_loss={eval_loss:.4f}")

        # Log to wandb
        wandb.log(log, step=step)

    # Save final checkpoint
    if config["save_final"]:
        final_path = os.path.join(config["run_checkpoint_dir"], "final.pt")
        save_checkpoint(model, optimizer, config["num_iters"] - 1, final_path, config)
        print(f"Saved final checkpoint")

    # Final summary
    wandb.summary["final_train_loss"] = log.get("train_loss", float("nan"))
    wandb.summary["final_eval_loss"] = best_eval_loss
    wandb.summary["final_eval_perplexity"] = math.exp(best_eval_loss)
    wandb.summary["checkpoint_dir"] = config["run_checkpoint_dir"]

    print(f"Training complete. Best eval loss: {best_eval_loss:.4f}")
    print(f"Checkpoints saved to: {config['run_checkpoint_dir']}")
    wandb.finish()


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="WandB Sweep Training")
    parser.add_argument(
        "--num_iters",
        type=int,
        default=None,
        help="Override number of training iterations"
    )
    args = parser.parse_args()

    # Update default config if args provided
    if args.num_iters is not None:
        DEFAULT_CONFIG["num_iters"] = args.num_iters

    train_sweep()


if __name__ == "__main__":
    main()
