"""Profile training steps for Model A and Model B.

Captures torch profiler traces, MPS activity, and compile debug output.
Saves all traces to /Volumes/slab_storage/traces.
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.profiler

from cs336_basics import TransformerLM
from cs336_basics.training import Trainer


# =============================================================================
# Model Configurations (same as train_compare.py)
# =============================================================================

MODEL_A_CONFIG = {
    "name": "Model_A_wide_attn",
    "description": "Wide attention, low FFN ratio (1.6x), d_head=64",
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
    "description": "Standard FFN ratio (4.5x), d_head=32 (potential perf issue)",
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


def get_trainer_config(model_config: dict, device: str, checkpoint_dir: str) -> dict:
    """Create a minimal trainer config for profiling (no W&B)."""
    return {
        "model_settings": model_config["model_settings"],
        "device": device,
        "batch_size": model_config["batch_size"],
        "num_iters": 1000,
        "eval_every": 999999,  # Skip eval
        "gradient_clip": 1.0,
        "scheduler_lr_max": 1e-3,
        "scheduler_lr_min": 1e-4,
        "scheduler_warmup_iters": 100,
        "scheduler_cos_iters": 1000,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_add_timestamp": False,
    }


def load_data():
    """Load tokenized training data."""
    train_file = "tokenized/tinystories_train_fixed.npy"
    valid_file = "tokenized/tinystories_valid_fixed.npy"

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found: {train_file}")

    tokens = np.load(train_file, mmap_mode='r')
    valid_tokens = np.load(valid_file, mmap_mode='r')
    return tokens, valid_tokens


def profile_model(
    model_config: dict,
    tokens: np.ndarray,
    valid_tokens: np.ndarray,
    device: str,
    trace_dir: str,
    num_steps: int,
):
    model_name = model_config["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nProfiling {model_name} on {device}...")
    
    # Create trainer
    checkpoint_dir = os.path.join(trace_dir, f"checkpoints_{model_name}_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer_config = get_trainer_config(model_config, device, checkpoint_dir)
    trainer = Trainer(TransformerLM, trainer_config, tokens, valid_tokens)
    
    # Warmup steps
    print("Warmup (5 steps)...")
    for i in range(5):
        trainer._train_step(i)
    
    if device == "mps":
        torch.mps.synchronize()
    
    # Trace paths
    tensorboard_dir = os.path.join(trace_dir, f"{model_name}_{timestamp}_tensorboard")
    
    print(f"Profiling {num_steps} steps...")
    
    # Optional MPS profiling
    mps_profile_started = False
    if device == "mps":
        try:
            print("Starting torch.mps.profiler...")
            # We don't specify output_path here as it can be picky about formats/versions
            torch.mps.profiler.start(mode="interval")
            mps_profile_started = True
        except Exception as e:
            print(f"Warning: Could not start torch.mps.profiler: {e}")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=2, warmup=3, active=num_steps-5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(num_steps):
            # The core training step (includes signposts)
            trainer._train_step(step + 5)
            # The checkpointing (to see overhead)
            trainer._save_latest_checkpoint(step + 5)
            prof.step()
            if step % 5 == 0:
                print(f"  Step {step}/{num_steps} done")

    if mps_profile_started:
        torch.mps.profiler.stop()
    
    print(f"Profiling of {model_name} complete. Results in {tensorboard_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["A", "B", "both"], default="both")
    parser.add_argument("--trace-dir", default="/Volumes/slab_storage/traces")
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    
    os.makedirs(args.trace_dir, exist_ok=True)
    
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    tokens, valid_tokens = load_data()
    
    if args.model in ("A", "both"):
        profile_model(MODEL_A_CONFIG, tokens, valid_tokens, device, args.trace_dir, args.num_steps)
    
    if args.model in ("B", "both"):
        profile_model(MODEL_B_CONFIG, tokens, valid_tokens, device, args.trace_dir, args.num_steps)


if __name__ == "__main__":
    main()
