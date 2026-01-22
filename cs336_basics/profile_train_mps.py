#!/usr/bin/env python3
"""
Training profiling harness for macOS Instruments.

Usage:
    # Direct run (for quick testing):
    uv run profile_train.py <checkpoint_path> --num-steps 50

    # With Instruments Time Profiler:
    xcrun xctrace record --template 'Time Profiler' --launch -- \
        python profile_train.py <checkpoint_path> --num-steps 50

    # With Instruments (signposts visible in Instruments > os_signpost):
    xcrun xctrace record --template 'Time Profiler' --launch -- \
        python profile_train.py <checkpoint_path> --num-steps 50

Signpost regions:
    - "get_batch": Data loading and transfer to device
    - "forward": Model forward pass
    - "backward": Loss backpropagation
    - "optimizer_step": Optimizer parameter update
"""

import argparse
import ctypes
import ctypes.util
from contextlib import contextmanager

import numpy as np
import torch

from cs336_basics import (
    TransformerLM,
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
    get_batch,
    crossentropy,
    load_model,
)


# =============================================================================
# macOS Signpost Integration
# =============================================================================

class SignpostLogger:
    """Wrapper for macOS os_signpost API for Instruments profiling."""
    
    def __init__(self, subsystem: str = "com.cs336.profile", category: str = "training"):
        self.enabled = False
        self.log = None
        
        # Try to load macOS signpost library
        try:
            libsystem = ctypes.CDLL(ctypes.util.find_library("System"))
            
            # os_log_create(subsystem, category) -> os_log_t
            libsystem.os_log_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            libsystem.os_log_create.restype = ctypes.c_void_p
            
            # os_signpost_interval_begin / end
            # void os_signpost_interval_begin(os_log_t log, os_signpost_id_t id, const char *name, ...)
            libsystem.os_signpost_interval_begin.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p
            ]
            libsystem.os_signpost_interval_begin.restype = None
            
            libsystem.os_signpost_interval_end.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p
            ]
            libsystem.os_signpost_interval_end.restype = None
            
            self.log = libsystem.os_log_create(
                subsystem.encode('utf-8'),
                category.encode('utf-8')
            )
            self._lib = libsystem
            self.enabled = True
            print(f"Signpost logging enabled: {subsystem}/{category}")
            
        except (OSError, AttributeError) as e:
            print(f"Signpost logging unavailable (not macOS?): {e}")
            self.enabled = False
    
    @contextmanager
    def interval(self, name: str, signpost_id: int = 0):
        """Context manager for signpost intervals."""
        if self.enabled:
            name_bytes = name.encode('utf-8')
            self._lib.os_signpost_interval_begin(self.log, signpost_id, name_bytes)
            try:
                yield
            finally:
                self._lib.os_signpost_interval_end(self.log, signpost_id, name_bytes)
        else:
            yield


# Global signpost logger
_signpost = None

def get_signpost() -> SignpostLogger:
    """Get or create the global signpost logger."""
    global _signpost
    if _signpost is None:
        _signpost = SignpostLogger()
    return _signpost


# =============================================================================
# Device Setup
# =============================================================================

def setup_device_and_seeds(seed: int = 0) -> str:
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


def sync_device(device: str):
    """Synchronize device for accurate timing."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


# =============================================================================
# Profile Training Loop
# =============================================================================

def profile_training(
    checkpoint_path: str,
    num_steps: int = 50,
    seed: int = 0,
):
    """
    Run training steps with signpost instrumentation.
    
    Args:
        checkpoint_path: Path to checkpoint to load (for model config)
        num_steps: Number of training steps to profile
        seed: Random seed
    """
    signpost = get_signpost()
    
    # Setup
    print(f"Loading model from {checkpoint_path}")
    config, model = load_model(checkpoint_path)
    device = setup_device_and_seeds(seed)
    model.to(device)
    
    # Load training data
    tokens = np.load(config["train_file"], mmap_mode='r')
    print(f"Loaded {len(tokens):,} training tokens")
    
    # Create optimizer
    optimizer = AdamW(model.parameters())
    
    # Get config values
    batch_size = config.get("batch_size", 32)
    context_length = config["model_settings"]["context_length"]
    lr_max = config.get("lr_max", 1e-3)
    lr_min = config.get("lr_min", 1e-4)
    warmup_iters = config.get("warmup_iters", 100)
    cos_iters = config.get("cos_iters", 1000)
    gradient_clip = config.get("gradient_clip", 1.0)
    
    print(f"\nStarting profiling run: {num_steps} steps")
    print(f"  batch_size={batch_size}, context_length={context_length}")
    print("-" * 50)
    
    model.train()
    
    for step in range(num_steps):
        sync_device(device)
        
        # === Get batch (signposted) ===
        with signpost.interval("get_batch", signpost_id=step):
            inputs, labels = get_batch(tokens, batch_size, context_length, device)
        
        sync_device(device)
        
        # === Forward pass (signposted) ===
        with signpost.interval("forward", signpost_id=step):
            logits = model.forward(inputs)
            loss = crossentropy(logits, labels)
        
        sync_device(device)
        
        # === Backward pass (signposted) ===
        with signpost.interval("backward", signpost_id=step):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        
        sync_device(device)
        
        # Gradient clipping (not signposted per user request)
        gradient_clipping(model.parameters(), gradient_clip)
        
        # LR schedule and optimizer step
        lr = get_lr_cosine_schedule(step, lr_max, lr_min, warmup_iters, cos_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        sync_device(device)
        
        # === Optimizer step (signposted) ===
        with signpost.interval("optimizer_step", signpost_id=step):
            optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step:4d} | loss={loss.item():.4f}")
    
    print("-" * 50)
    print("Profiling complete.")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Training profiler for macOS Instruments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "checkpoint",
        help="Path to checkpoint file (for model config)"
    )
    parser.add_argument(
        "--num-steps", "-n",
        type=int,
        default=50,
        help="Number of training steps to profile (default: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )
    args = parser.parse_args()
    
    profile_training(
        checkpoint_path=args.checkpoint,
        num_steps=args.num_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
