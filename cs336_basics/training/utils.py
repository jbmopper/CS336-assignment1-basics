"""Shared utilities for training scripts.

This module provides common functionality used across train.py, train_sweep.py,
train_compare.py, and profiling scripts.
"""

import os
from typing import Optional

import numpy as np
import torch


def setup_device(seed: int = 42, prefer_cuda: bool = True, verbose: bool = True) -> str:
    """Set up compute device and random seeds.
    
    Args:
        seed: Random seed for reproducibility.
        prefer_cuda: If True, prefer CUDA over MPS when both available.
        verbose: If True, print device information.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu').
    """
    # Device detection
    if prefer_cuda and torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if device == "mps":
        torch.mps.manual_seed(seed)
    elif device == "cuda":
        torch.cuda.manual_seed_all(seed)
    
    if verbose:
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Random seed: {seed}")
    
    return device


def sync_device(device: str) -> None:
    """Synchronize device for accurate timing.
    
    Args:
        device: Device string ('cuda', 'mps', or 'cpu').
    """
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def clear_device_cache(device: str) -> None:
    """Clear device memory cache.
    
    Args:
        device: Device string ('cuda', 'mps', or 'cpu').
    """
    import gc
    gc.collect()
    
    if device == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()
    
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def load_tokens(
    train_file: str,
    valid_file: str,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load tokenized training and validation data.
    
    Args:
        train_file: Path to tokenized training data (.npy).
        valid_file: Path to tokenized validation data (.npy).
        verbose: If True, print loading information.
    
    Returns:
        Tuple of (train_tokens, valid_tokens) as memory-mapped arrays.
    
    Raises:
        FileNotFoundError: If either file doesn't exist.
    """
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
    
    if verbose:
        print(f"Loaded {len(tokens):,} train tokens, {len(valid_tokens):,} valid tokens")
    
    return tokens, valid_tokens


def calculate_params(model_settings: dict) -> int:
    """Calculate total parameter count for a transformer model.
    
    Args:
        model_settings: Dictionary with vocab_size, d_model, num_layers, d_ff.
    
    Returns:
        Total number of parameters.
    """
    vocab_size = model_settings["vocab_size"]
    d_model = model_settings["d_model"]
    num_layers = model_settings["num_layers"]
    d_ff = model_settings["d_ff"]
    
    # Embedding (input + output tied) + final norm + per-layer params
    return (
        2 * vocab_size * d_model +  # embeddings (input + output)
        d_model +  # final layer norm
        num_layers * (
            2 * d_model +  # attention norm + ffn norm
            4 * d_model * d_model +  # Q, K, V, O projections
            3 * d_model * d_ff  # FFN (up, gate, down for SwiGLU)
        )
    )


def estimate_checkpoint_size_mb(model_settings: dict) -> float:
    """Estimate checkpoint size in MB (model + optimizer states).
    
    Args:
        model_settings: Dictionary with vocab_size, d_model, num_layers, d_ff.
    
    Returns:
        Estimated checkpoint size in MB.
    """
    num_params = calculate_params(model_settings)
    
    # Model (float32) + Adam states (2x float32 for m and v)
    bytes_per_param = 4 + 4 + 4  # param + m + v
    total_bytes = num_params * bytes_per_param
    return total_bytes / (1024 * 1024)


def print_model_summary(
    name: str,
    model_settings: dict,
    batch_size: Optional[int] = None,
    description: Optional[str] = None,
) -> None:
    """Print a summary of model configuration.
    
    Args:
        name: Model name for display.
        model_settings: Model configuration dictionary.
        batch_size: Optional batch size to display.
        description: Optional description string.
    """
    num_params = calculate_params(model_settings)
    d_head = model_settings["d_model"] // model_settings["num_heads"]
    ffn_ratio = model_settings["d_ff"] / model_settings["d_model"]
    checkpoint_size = estimate_checkpoint_size_mb(model_settings)
    
    print("\n" + "=" * 60)
    print(f"Model: {name}")
    if description:
        print(f"Description: {description}")
    print(f"Parameters: {num_params / 1e6:.1f}M")
    print(f"  d_model={model_settings['d_model']}, num_heads={model_settings['num_heads']}, "
          f"num_layers={model_settings['num_layers']}, d_ff={model_settings['d_ff']}")
    print(f"  context_length={model_settings['context_length']}, d_head={d_head}")
    print(f"  FFN ratio: {ffn_ratio:.2f}x")
    if batch_size:
        tokens_per_step = batch_size * model_settings["context_length"]
        print(f"Batch size: {batch_size} ({tokens_per_step:,} tokens/step)")
    print(f"Estimated checkpoint size: {checkpoint_size:.0f} MB")
    print("=" * 60)
