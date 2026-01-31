"""Training comparison script for Model A vs Model B.

Trains the two original model selections from notes.md, logging each to separate
wandb runs for comparison.

Usage:
    # Full training run
    uv run python -m cs336_basics.train_compare

    # Smoke test (small models, few iterations)
    uv run python -m cs336_basics.train_compare --smol
    
    # Mixed precision (CUDA only)
    uv run python -m cs336_basics.train_compare --precision bf16
    
    # torch.compile (CUDA only)
    uv run python -m cs336_basics.train_compare --compile --compile-backend aot_eager

    # Custom checkpoint directory (e.g., external drive)
    uv run python -m cs336_basics.train_compare --checkpoint-dir /Volumes/External/checkpoints

    # Train only one model
    uv run python -m cs336_basics.train_compare --model A
    uv run python -m cs336_basics.train_compare --model B
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from cs336_basics import TransformerLM
from cs336_basics.training import (
    Trainer,
    setup_device,
    clear_device_cache,
    load_tokens,
    calculate_params,
    estimate_checkpoint_size_mb,
    print_model_summary,
)


# =============================================================================
# Model Configurations - loaded from configs/models.yaml
# =============================================================================

def load_model_configs() -> dict:
    """Load model configurations from configs/models.yaml."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Training Configuration
# =============================================================================

# Default data paths
DEFAULT_TRAIN_FILE = "tokenized/tinystories_train_fixed.npy"
DEFAULT_VALID_FILE = "tokenized/tinystories_valid_fixed.npy"


def get_base_config(checkpoint_dir: str, smol: bool = False) -> dict:
    """Get base training configuration shared by both models."""
    return {
        # Data paths
        "train_file": DEFAULT_TRAIN_FILE,
        "valid_file": DEFAULT_VALID_FILE,

        # W&B logging
        "wandb_entity": "jbmopper-0",
        "log_project": "cs336-a1-gpu-model_comp",

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
        
        # Precision / compile
        "precision": "fp32",
        "compile_model": False,
        "compile_backend": "aot_eager",

        # Checkpointing
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_add_timestamp": False,  # We add timestamp in run_name instead
        "save_every": 500 if not smol else 25,  # Save snapshots
        "save_best": True,
        "save_final": True,

        # Random seed
        "rand_seed": 3,
    }


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

    # Display model info using shared utility
    print_model_summary(
        name=model_config["name"],
        model_settings=model_config["model_settings"],
        batch_size=model_config["batch_size"],
        description=model_config.get("description"),
    )
    print(f"Checkpoint directory: {model_checkpoint_dir}")
    print(f"W&B run name: {run_name}")

    # Create trainer and run
    trainer = Trainer(TransformerLM, config, tokens, valid_tokens)
    trainer.train_eval_loop()

    print(f"\nCompleted training: {model_config['name']}")


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
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Training precision (CUDA only for fp16/bf16)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (CUDA only)",
    )
    parser.add_argument(
        "--compile-backend",
        type=str,
        default="inductor",
        help="torch.compile backend (default: inductor)",
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

    # Set up device and seeds using shared utility
    device = setup_device(seed=args.seed, prefer_cuda=False)

    # Load model configs from YAML
    model_configs = load_model_configs()

    # Get base config
    base_config = get_base_config(args.checkpoint_dir, smol=args.smol)
    base_config["rand_seed"] = args.seed
    base_config["precision"] = args.precision
    base_config["compile_model"] = args.compile
    base_config["compile_backend"] = args.compile_backend

    # Override num_iters if specified
    if args.num_iters is not None:
        base_config["num_iters"] = args.num_iters
        # Adjust related settings proportionally
        base_config["scheduler_cos_iters"] = args.num_iters
        base_config["scheduler_warmup_iters"] = max(1, args.num_iters // 10)

    # Select model configs from YAML
    if args.smol:
        model_a = model_configs["smol_model_a"]
        model_b = model_configs["smol_model_b"]
        print("\n*** SMOL MODE: Using tiny models for smoke testing ***\n")
    else:
        model_a = model_configs["model_a"]
        model_b = model_configs["model_b"]

    # Load data using shared utility
    tokens, valid_tokens = load_tokens(
        base_config["train_file"],
        base_config["valid_file"],
    )

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
        # Clean up memory before starting Model B to avoid fragmentation issues
        if args.model == "both":
            print("\n" + "-" * 70)
            print("Cleaning up memory before Model B...")
            print("-" * 70)
            clear_device_cache(device)
            print("Memory cleanup complete.\n")
        
        # Reset seeds for fair comparison
        device = setup_device(seed=args.seed, prefer_cuda=False)
        train_model(model_b, base_config, tokens, valid_tokens, device, timestamp)

    print("\n" + "=" * 70)
    print("Training comparison complete!")
    print(f"Results logged to W&B project: cs336-a1-model_comp")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
