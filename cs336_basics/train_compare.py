"""Training comparison script for Model A vs Model B.

Trains the two original model selections from notes.md, logging each to separate
wandb runs for comparison.

Usage:
    # Full training run
    uv run python -m cs336_basics.train_compare

    # Smoke test (small models, few iterations)
    uv run python -m cs336_basics.train_compare --smol

    # Custom checkpoint directory (e.g., external drive)
    uv run python -m cs336_basics.train_compare --checkpoint-dir /Volumes/External/checkpoints

    # Train only one model
    uv run python -m cs336_basics.train_compare --model A
    uv run python -m cs336_basics.train_compare --model B
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch

from cs336_basics import TransformerLM
from cs336_basics.training import Trainer


# =============================================================================
# Model Configurations (from notes.md)
# =============================================================================

MODEL_A_CONFIG = {
    "name": "Model_A_wide_attn",
    "description": "Wide attention, low FFN ratio (1.6x), 48.9M params",
    "batch_size": 64,
    "model_settings": {
        "vocab_size": 10000,
        "d_model": 640,
        "num_heads": 10,  # d_head = 64
        "num_layers": 10,
        "d_ff": 1024,
        "context_length": 256,
        "rope_theta": 10000.0,
    },
}

MODEL_B_CONFIG = {
    "name": "Model_B_standard_ffn",
    "description": "Standard FFN ratio (4.5x), deeper (12L), 38.7M params",
    "batch_size": 48,
    "model_settings": {
        "vocab_size": 10000,
        "d_model": 384,
        "num_heads": 12,  # d_head = 32
        "num_layers": 12,
        "d_ff": 1728,
        "context_length": 256,
        "rope_theta": 10000.0,
    },
}

# Smol versions for smoke testing
SMOL_MODEL_A_CONFIG = {
    "name": "Smol_Model_A",
    "description": "Tiny version of Model A for smoke testing",
    "batch_size": 8,
    "model_settings": {
        "vocab_size": 10000,
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 256,
        "context_length": 64,
        "rope_theta": 10000.0,
    },
}

SMOL_MODEL_B_CONFIG = {
    "name": "Smol_Model_B",
    "description": "Tiny version of Model B for smoke testing",
    "batch_size": 8,
    "model_settings": {
        "vocab_size": 10000,
        "d_model": 96,
        "num_heads": 3,
        "num_layers": 3,
        "d_ff": 384,
        "context_length": 64,
        "rope_theta": 10000.0,
    },
}


# =============================================================================
# Training Configuration
# =============================================================================

def get_base_config(checkpoint_dir: str, smol: bool = False) -> dict:
    """Get base training configuration shared by both models."""
    return {
        # Data paths
        "train_file": "tokenized/tinystories_train_fixed.npy",
        "valid_file": "tokenized/tinystories_valid_fixed.npy",

        # W&B logging
        "wandb_entity": "jbmopper-0",
        "log_project": "cs336-a1-model_comp",

        # Optimizer (AdamW)
        "optimizer_lr": 1e-3,
        "optimizer_betas": (0.9, 0.999),
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 1e-2,

        # LR schedule (cosine with warmup)
        "scheduler_lr_max": 1e-3,
        "scheduler_lr_min": 1e-4,
        "scheduler_warmup_iters": 500 if not smol else 10,
        "scheduler_cos_iters": 5000 if not smol else 50,

        # Training loop
        "num_iters": 5000 if not smol else 50,
        "eval_every": 100 if not smol else 10,
        "eval_batches": 10 if not smol else 2,
        "gradient_clip": 1.0,

        # Checkpointing
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_add_timestamp": False,  # We add timestamp in run_name instead
        "save_every": 500 if not smol else 25,  # Save snapshots
        "save_best": True,
        "save_final": True,

        # Random seed
        "rand_seed": 3,
    }


def estimate_checkpoint_size_mb(model_settings: dict) -> float:
    """Estimate checkpoint size in MB (model + optimizer states)."""
    vocab_size = model_settings["vocab_size"]
    d_model = model_settings["d_model"]
    num_layers = model_settings["num_layers"]
    d_ff = model_settings["d_ff"]

    num_params = (
        2 * vocab_size * d_model +
        d_model +
        num_layers * (
            2 * d_model +
            4 * d_model * d_model +
            3 * d_model * d_ff
        )
    )

    # Model (float32) + Adam states (2x float32 for m and v)
    bytes_per_param = 4 + 4 + 4  # param + m + v
    total_bytes = num_params * bytes_per_param
    return total_bytes / (1024 * 1024)


def setup_device_and_seeds(seed: int) -> str:
    """Set up device and random seeds. Returns device string."""
    if torch.backends.mps.is_available():
        device = "mps"
        torch.mps.manual_seed(seed)
    elif torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed_all(seed)
    else:
        device = "cpu"

    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Using device: {device}")
    print(f"Random seed: {seed}")
    return device


def load_tokens(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load tokenized training and validation data."""
    train_file = config["train_file"]
    valid_file = config["valid_file"]

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Training data not found: {train_file}\n"
            "Run from repository root or check path."
        )
    if not os.path.exists(valid_file):
        raise FileNotFoundError(
            f"Validation data not found: {valid_file}\n"
            "Run from repository root or check path."
        )

    tokens = np.load(train_file, mmap_mode='r')
    valid_tokens = np.load(valid_file, mmap_mode='r')

    print(f"Loaded {len(tokens):,} train tokens, {len(valid_tokens):,} valid tokens")
    return tokens, valid_tokens


def train_model(
    model_config: dict,
    base_config: dict,
    tokens: np.ndarray,
    valid_tokens: np.ndarray,
    device: str,
    timestamp: str,
) -> None:
    """Train a single model configuration."""
    # Merge configs
    config = base_config.copy()
    config["batch_size"] = model_config["batch_size"]
    config["model_settings"] = model_config["model_settings"]
    config["device"] = device

    # Set run name with timestamp
    run_name = f"{model_config['name']}_{timestamp}"
    config["run_name"] = run_name

    # Set checkpoint directory for this model
    model_checkpoint_dir = os.path.join(config["checkpoint_dir"], run_name)
    config["checkpoint_dir"] = model_checkpoint_dir
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    # Calculate and display model info
    num_params = calculate_params(model_config["model_settings"])
    ffn_ratio = model_config["model_settings"]["d_ff"] / model_config["model_settings"]["d_model"]
    checkpoint_size = estimate_checkpoint_size_mb(model_config["model_settings"])

    print("\n" + "=" * 70)
    print(f"Training: {model_config['name']}")
    print(f"Description: {model_config['description']}")
    print(f"Parameters: {num_params / 1e6:.1f}M")
    print(f"FFN ratio: {ffn_ratio:.2f}x")
    print(f"Batch size: {model_config['batch_size']}")
    print(f"Estimated checkpoint size: {checkpoint_size:.0f} MB")
    print(f"Checkpoint directory: {model_checkpoint_dir}")
    print(f"W&B run name: {run_name}")
    print("=" * 70 + "\n")

    # Create trainer and run
    trainer = Trainer(TransformerLM, config, tokens, valid_tokens)
    trainer.train_eval_loop()

    print(f"\nCompleted training: {model_config['name']}")


def calculate_params(model_settings: dict) -> int:
    """Calculate total parameter count."""
    vocab_size = model_settings["vocab_size"]
    d_model = model_settings["d_model"]
    num_layers = model_settings["num_layers"]
    d_ff = model_settings["d_ff"]

    return (
        2 * vocab_size * d_model +
        d_model +
        num_layers * (
            2 * d_model +
            4 * d_model * d_model +
            3 * d_model * d_ff
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train Model A vs Model B for comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--smol",
        action="store_true",
        help="Smoke test mode: tiny models, few iterations",
    )
    parser.add_argument(
        "--model",
        choices=["A", "B", "both"],
        default="both",
        help="Which model(s) to train (default: both)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/model_comp",
        help="Base checkpoint directory (default: checkpoints/model_comp)",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=None,
        help="Override number of iterations (default: 5000, or 50 for smol)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Timestamp for this comparison run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up device and seeds
    device = setup_device_and_seeds(args.seed)

    # Get base config
    base_config = get_base_config(args.checkpoint_dir, smol=args.smol)
    base_config["rand_seed"] = args.seed

    # Override num_iters if specified
    if args.num_iters is not None:
        base_config["num_iters"] = args.num_iters
        # Adjust related settings proportionally
        base_config["scheduler_cos_iters"] = args.num_iters
        base_config["scheduler_warmup_iters"] = max(1, args.num_iters // 10)

    # Select model configs
    if args.smol:
        model_a = SMOL_MODEL_A_CONFIG
        model_b = SMOL_MODEL_B_CONFIG
        print("\n*** SMOL MODE: Using tiny models for smoke testing ***\n")
    else:
        model_a = MODEL_A_CONFIG
        model_b = MODEL_B_CONFIG

    # Load data
    tokens, valid_tokens = load_tokens(base_config)

    # Estimate total disk usage
    models_to_train = []
    if args.model in ("A", "both"):
        models_to_train.append(model_a)
    if args.model in ("B", "both"):
        models_to_train.append(model_b)

    total_checkpoint_mb = 0
    num_snapshots = base_config["num_iters"] // base_config["save_every"]
    for model in models_to_train:
        size = estimate_checkpoint_size_mb(model["model_settings"])
        # latest + best + final + snapshots
        num_checkpoints = 3 + num_snapshots
        total_checkpoint_mb += size * num_checkpoints

    print(f"\nEstimated total disk usage: {total_checkpoint_mb / 1024:.1f} GB")
    print(f"Checkpoint base directory: {args.checkpoint_dir}")

    # Train models
    if args.model in ("A", "both"):
        train_model(model_a, base_config, tokens, valid_tokens, device, timestamp)

    if args.model in ("B", "both"):
        # Reset seeds for fair comparison
        setup_device_and_seeds(args.seed)
        train_model(model_b, base_config, tokens, valid_tokens, device, timestamp)

    print("\n" + "=" * 70)
    print("Training comparison complete!")
    print(f"Results logged to W&B project: cs336-a1-model_comp")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
