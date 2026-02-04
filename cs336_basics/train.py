"""Training script with CLI argument support.

Usage examples:
    # Default training (TinyStories, assignment model)
    uv run python -m cs336_basics.train
    
    # With mixed precision
    uv run python -m cs336_basics.train --precision bf16
    
    # Override data paths (for cloud)
    uv run python -m cs336_basics.train --data-dir /workspace/tokenized
    
    # Override batch size and iterations
    uv run python -m cs336_basics.train --batch-size 64 --num-iters 1000
    
    # Disable W&B logging
    uv run python -m cs336_basics.train --no-wandb
"""

from cs336_basics import (
    Tokenizer,
    TransformerLM,
    save_bpe,
    load_bpe,
    train_bpe,
)
from cs336_basics.training import Trainer

import argparse
import os
import numpy as np
import torch


# Default configuration
DEFAULT_CONFIG = dict(
    # tokenizer (only used if tokenizing from scratch)
    training_text="../data/TinyStoriesV2-GPT4-train.txt",
    validation_text="../data/TinyStoriesV2-GPT4-valid.txt",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
    tokenizer_dir="../tokenizers/tinystories/",
    
    # tokenized data paths
    train_file="../tokenized/tinystories_train.npy",
    valid_file="../tokenized/tinystories_valid.npy",

    # batch size
    batch_size=32,
    
    # model settings (assignment defaults, ~17M parameters)
    model_settings=dict(
        d_model=512,
        num_heads=16,
        num_layers=4,
        d_ff=1344,
        context_length=256,
        rope_theta=10000,
    ),

    # optimizer (adamw)
    optimizer_lr=1e-3,
    optimizer_betas=(0.9, 0.999),
    optimizer_eps=1e-08,
    optimizer_weight_decay=1e-2,

    # lr scheduler (cosine with warmup)
    scheduler_lr_max=1e-3,
    scheduler_lr_min=1e-4,
    scheduler_warmup_iters=100,
    scheduler_cos_iters=1000,

    # wandb/logging
    wandb_entity="jbmopper-0",
    log_project="cs336-a1",
    model="course rope-transformer",
    optimizer="course adamw",
    learning_schedule="course cosine anneal w/warmup",
    dataset="Tinystories",
    loss_func="cross-entropy",
    run_name="cli-train",

    # training loop
    num_iters=10,
    checkpoint_dir="../checkpoints/",

    # other parameters
    rand_seed=0,
    gradient_clip=1.0,
    save_every=200,
    eval_every=20,
    
    # mixed precision (default fp32 for compatibility)
    precision="fp32",
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Transformer LM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("DATA_DIR"),
        help="Directory containing tokenized .npy files (overrides train_file/valid_file)",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        help="Path to tokenized training data (.npy)",
    )
    parser.add_argument(
        "--valid-file",
        type=str,
        help="Path to tokenized validation data (.npy)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.environ.get("CHECKPOINT_DIR", "../checkpoints/"),
        help="Directory to save checkpoints",
    )
    
    # Mixed precision
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Training precision (bf16 recommended for 4090/A100+)",
    )
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--num-iters", type=int, help="Override number of iterations")
    parser.add_argument("--lr", type=float, help="Override learning rate (scheduler max)")
    parser.add_argument("--context-length", type=int, help="Override context length")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    # Model architecture overrides
    parser.add_argument("--d-model", type=int, help="Override d_model")
    parser.add_argument("--num-heads", type=int, help="Override num_heads")
    parser.add_argument("--num-layers", type=int, help="Override num_layers")
    parser.add_argument("--d-ff", type=int, help="Override d_ff")
    
    # Architecture ablation options
    parser.add_argument("--norm-mode", type=str, choices=["pre", "post", "none"], 
                        help="Layer norm mode: pre (default), post, or none")
    parser.add_argument("--no-rope", action="store_true", 
                        help="Disable RoPE position embeddings (NoPE)")
    parser.add_argument("--ffn-type", type=str, choices=["swiglu", "silu"],
                        help="FFN type: swiglu (default) or silu")
    parser.add_argument("--ffn-hidden-dim", type=int,
                        help="FFN hidden dimension (default: 8/3*d_model for swiglu, 4*d_model for silu)")
    
    # Logging
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="W&B run name",
    )
    
    # Evaluation
    parser.add_argument("--eval-every", type=int, help="Evaluate every N iterations")
    parser.add_argument("--eval-batches", type=int, default=1, help="Number of batches for evaluation")
    
    # Checkpointing
    parser.add_argument("--save-best", action="store_true", help="Save best checkpoint by eval loss")
    parser.add_argument("--save-final", action="store_true", help="Save final checkpoint after training")
    parser.add_argument("--save-every", type=int, help="Save snapshot checkpoint every N iterations")
    
    return parser.parse_args()


def build_config(args):
    """Build config dictionary from defaults and CLI arguments."""
    config = DEFAULT_CONFIG.copy()
    config["model_settings"] = DEFAULT_CONFIG["model_settings"].copy()
    
    # Handle data directory
    if args.data_dir:
        config["train_file"] = os.path.join(args.data_dir, "tinystories_train.npy")
        config["valid_file"] = os.path.join(args.data_dir, "tinystories_valid.npy")
    
    # Override specific data files
    if args.train_file:
        config["train_file"] = args.train_file
    if args.valid_file:
        config["valid_file"] = args.valid_file
    
    # Checkpoint directory
    config["checkpoint_dir"] = args.checkpoint_dir
    
    # Precision
    config["precision"] = args.precision
    
    # Training hyperparameters
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_iters is not None:
        config["num_iters"] = args.num_iters
    if args.lr is not None:
        config["scheduler_lr_max"] = args.lr
        config["optimizer_lr"] = args.lr
    if args.seed is not None:
        config["rand_seed"] = args.seed
    
    # Model architecture overrides
    if args.d_model is not None:
        config["model_settings"]["d_model"] = args.d_model
    if args.num_heads is not None:
        config["model_settings"]["num_heads"] = args.num_heads
    if args.num_layers is not None:
        config["model_settings"]["num_layers"] = args.num_layers
    if args.d_ff is not None:
        config["model_settings"]["d_ff"] = args.d_ff
    if args.context_length is not None:
        config["model_settings"]["context_length"] = args.context_length
    
    # Architecture ablation options
    if args.norm_mode is not None:
        config["model_settings"]["norm_mode"] = args.norm_mode
    if args.no_rope:
        config["model_settings"]["use_rope"] = False
    if args.ffn_type is not None:
        config["model_settings"]["ffn_type"] = args.ffn_type
    if args.ffn_hidden_dim is not None:
        config["model_settings"]["ffn_hidden_dim"] = args.ffn_hidden_dim
    
    # Logging
    if args.no_wandb:
        config["wandb_entity"] = None
    if args.run_name:
        config["run_name"] = args.run_name
    
    # Evaluation
    if args.eval_every is not None:
        config["eval_every"] = args.eval_every
    if args.eval_batches is not None:
        config["eval_batches"] = args.eval_batches
    
    # Checkpointing
    if args.save_best:
        config["save_best"] = True
    if args.save_final:
        config["save_final"] = True
    if args.save_every is not None:
        config["save_every"] = args.save_every
    
    # Set vocab_size in model_settings
    config["model_settings"]["vocab_size"] = config["vocab_size"]
    
    return config


def set_device_and_seeds(config):
    """Check for devices, add to config, and set RNG seed values."""
    # Device check (prefer CUDA over MPS for cloud training)
    if torch.cuda.is_available():
        config["device"] = "cuda"
    elif torch.backends.mps.is_available():
        config["device"] = "mps"
    else:
        config["device"] = "cpu"

    print(f"Using device: {config['device']}")
    if config["device"] == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set seeds
    np.random.seed(config["rand_seed"])
    torch.manual_seed(config["rand_seed"])

    if config["device"] == "mps":
        torch.mps.manual_seed(config["rand_seed"])
    elif config["device"] == "cuda":
        torch.cuda.manual_seed_all(config["rand_seed"])
    
    print(f"Random seed: {config['rand_seed']}")


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
    tokenizer = Tokenizer(vocab, merges, special_tokens=config["special_tokens"])
    
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Tokenizing {file_type} file...")
    tokens = tokenizer.encode(text)
    
    # Save as contiguous array (memmap-compatible)
    arr = np.array(tokens, dtype=np.uint16)
    np.save(output_file, arr)
    
    print(f"Saved {len(tokens):,} tokens to {output_file}")


def get_tokens(config) -> tuple[np.ndarray, np.ndarray]:
    """Get/generate memmapped token arrays."""
    # Check if tokenized files exist
    train_exists = os.path.exists(config["train_file"])
    valid_exists = os.path.exists(config["valid_file"])
    
    if not train_exists or not valid_exists:
        # Try to tokenize from source
        if os.path.exists(config["training_text"]) and os.path.exists(config["validation_text"]):
            _tokenize_and_save(config, config["training_text"], config["train_file"], "training")
            _tokenize_and_save(config, config["validation_text"], config["valid_file"], "validation")
        else:
            raise FileNotFoundError(
                f"Tokenized data not found:\n"
                f"  train: {config['train_file']} (exists: {train_exists})\n"
                f"  valid: {config['valid_file']} (exists: {valid_exists})\n"
                f"And source text files not available for tokenization."
            )
    
    # Load tokenized data
    tokens = np.load(config["train_file"], mmap_mode='r')
    print(f"Training tokens: {config['train_file']} ({tokens.shape[0]:,} tokens)")
    
    valid_tokens = np.load(config["valid_file"], mmap_mode='r')
    print(f"Validation tokens: {config['valid_file']} ({valid_tokens.shape[0]:,} tokens)")
    
    return tokens, valid_tokens


def print_config_summary(config):
    """Print a summary of the training configuration."""
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    
    ms = config["model_settings"]
    params = (
        2 * config["vocab_size"] * ms["d_model"] +
        ms["d_model"] +
        ms["num_layers"] * (
            2 * ms["d_model"] +
            4 * ms["d_model"] * ms["d_model"] +
            3 * ms["d_model"] * ms["d_ff"]
        )
    )
    
    print(f"Model: {params / 1e6:.1f}M parameters")
    print(f"  d_model={ms['d_model']}, num_heads={ms['num_heads']}, "
          f"num_layers={ms['num_layers']}, d_ff={ms['d_ff']}")
    print(f"  context_length={ms['context_length']}")
    print(f"Training:")
    print(f"  batch_size={config['batch_size']}, num_iters={config['num_iters']}")
    print(f"  precision={config['precision']}")
    print(f"  lr_max={config['scheduler_lr_max']}, warmup={config['scheduler_warmup_iters']}")
    print(f"Logging: {'W&B enabled' if config.get('wandb_entity') else 'W&B disabled'}")
    print("=" * 60 + "\n")


def main():
    """Main training function."""
    args = parse_args()
    config = build_config(args)
    
    set_device_and_seeds(config)
    print_config_summary(config)
    
    tokens, valid_tokens = get_tokens(config)
    trainer = Trainer(TransformerLM, config, tokens, valid_tokens)
    trainer.train_eval_loop()


if __name__ == "__main__":
    main()
