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

from cs336_basics import TransformerLM
from cs336_basics.training import Trainer
import wandb

import os
import numpy as np
import torch
import argparse

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
    scheduler_lr_max=1e-3,
    scheduler_lr_min=1e-4,
    scheduler_warmup_iters=100,
    scheduler_cos_iters=1000,

    # Optimizer
    optimizer_lr=1e-3,
    optimizer_betas=(0.9, 0.999),
    optimizer_eps=1e-8,
    optimizer_weight_decay=1e-2,

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


def train_sweep():
    """Main training function for sweep runs."""
    # Initialize wandb run - this will get sweep config
    run = wandb.init()

    # Build config from defaults + sweep parameters
    config = DEFAULT_CONFIG.copy()
    
    # Create unique checkpoint directory for this run
    run_checkpoint_dir = os.path.join(config["checkpoint_dir"], run.id)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    config["checkpoint_dir"] = run_checkpoint_dir
    config["checkpoint_add_timestamp"] = False

    # Override with sweep parameters
    sweep_params = dict(wandb.config)
    for key, value in sweep_params.items():
        if key in config:
            config[key] = value
            print(f"Sweep override: {key} = {value}")

    # Ensure scheduler_cos_iters matches num_iters
    config["scheduler_cos_iters"] = config["num_iters"]

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
    for key in ("norm_mode", "use_rope", "ffn_type", "ffn_hidden_dim", "final_norm"):
        if key in config:
            model_settings[key] = config[key]

    # Validate model settings
    if model_settings["d_model"] % model_settings["num_heads"] != 0:
        print(f"Invalid config: d_model ({model_settings['d_model']}) must be divisible by num_heads ({model_settings['num_heads']})")
        wandb.finish(exit_code=1)
        return

    # Set up device and seeds
    device = set_device_and_seeds(config["rand_seed"])
    config["device"] = device

    # Load data
    try:
        tokens, valid_tokens = get_tokens(config)
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        wandb.finish(exit_code=1)
        return

    config["model_settings"] = model_settings
    print(f"Creating model with settings: {model_settings}")
    trainer = Trainer(TransformerLM, config, tokens, valid_tokens, wandb_run=run)
    num_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"Model has {num_params:,} parameters")
    run.log({"num_parameters": num_params})

    trainer.train_eval_loop()


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
