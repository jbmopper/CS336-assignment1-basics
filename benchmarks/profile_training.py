"""Profile training steps for Model A and Model B.

Captures torch profiler traces, MPS activity, and compile debug output.
Saves all traces to /Volumes/slab_storage/traces.

Usage:
    # Profile Model A only
    uv run python -m benchmarks.profile_training --model A

    # Profile Model B only
    uv run python -m benchmarks.profile_training --model B

    # Profile both models sequentially (default)
    uv run python -m benchmarks.profile_training

    # Custom trace directory
    uv run python -m benchmarks.profile_training --trace-dir /path/to/traces

    # Different number of steps
    uv run python -m benchmarks.profile_training --num-steps 100
"""

import argparse
import os
import subprocess
import sys
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


def get_trainer_config(model_config: dict, device: str, checkpoint_dir: str = "/tmp/profile_checkpoints") -> dict:
    """Create a minimal trainer config for profiling (no W&B)."""
    return {
        "model_settings": model_config["model_settings"],
        "device": device,
        "batch_size": model_config["batch_size"],
        "num_iters": 1,  # Not used - we control the loop
        "eval_every": 999999,  # Skip eval
        "gradient_clip": 1.0,
        # Minimal LR schedule
        "scheduler_lr_max": 1e-3,
        "scheduler_lr_min": 1e-4,
        "scheduler_warmup_iters": 1,
        "scheduler_cos_iters": 1000,
        # Checkpointing
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_add_timestamp": False,
        # No W&B
    }


def load_data():
    """Load tokenized training data."""
    train_file = "tokenized/tinystories_train_fixed.npy"
    valid_file = "tokenized/tinystories_valid_fixed.npy"

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Training data not found: {train_file}\n"
            "Run from repository root."
        )

    tokens = np.load(train_file, mmap_mode='r')
    valid_tokens = np.load(valid_file, mmap_mode='r')
    print(f"Loaded {len(tokens):,} train tokens, {len(valid_tokens):,} valid tokens")
    return tokens, valid_tokens


def profile_model(
    model_config: dict,
    tokens: np.ndarray,
    valid_tokens: np.ndarray,
    device: str,
    trace_dir: str,
    num_steps: int,
    enable_compile_debug: bool = True,
):
    """Profile a model's training steps and save traces."""
    
    model_name = model_config["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 70)
    print(f"Profiling: {model_name}")
    print(f"Description: {model_config['description']}")
    print(f"Steps: {num_steps}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Set up compile debug output (even for non-compiled models, captures some info)
    compile_debug_dir = os.path.join(trace_dir, f"{model_name}_{timestamp}_compile_debug")
    if enable_compile_debug:
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCH_COMPILE_DEBUG_DIR"] = compile_debug_dir
        # Also set inductor config if available
        try:
            import torch._inductor.config as inductor_config
            inductor_config.debug_dir = compile_debug_dir
            inductor_config.trace.enabled = True
        except (ImportError, AttributeError):
            pass
    
    # Create trainer
    trainer_config = get_trainer_config(model_config, device, checkpoint_dir=os.path.join(trace_dir, "checkpoints"))
    trainer = Trainer(TransformerLM, trainer_config, tokens, valid_tokens)
    
    # Calculate model stats
    num_params = sum(p.numel() for p in trainer.model.parameters())
    d_head = model_config["model_settings"]["d_model"] // model_config["model_settings"]["num_heads"]
    
    print(f"Parameters: {num_params / 1e6:.1f}M")
    print(f"d_head: {d_head}")
    print(f"Batch size: {model_config['batch_size']}")
    print(f"Tokens per step: {model_config['batch_size'] * model_config['model_settings']['context_length']:,}")
    
    # Warmup steps (outside profiler)
    print("\nWarming up (5 steps)...")
    for i in range(5):
        trainer._train_step(i)
    
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    
    # Set up profiler output paths
    torch_trace_path = os.path.join(trace_dir, f"{model_name}_{timestamp}_torch_trace.json")
    tensorboard_dir = os.path.join(trace_dir, f"{model_name}_{timestamp}_tensorboard")
    
    print(f"\nProfiling {num_steps} steps...")
    print(f"Torch trace: {torch_trace_path}")
    print(f"TensorBoard dir: {tensorboard_dir}")
    
    # Determine activities based on device
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    # Profile with torch.profiler
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=2,      # Skip first 2 steps
            warmup=3,    # Warmup for 3 steps
            active=num_steps - 5,  # Profile remaining steps
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(num_steps):
            trainer._train_step(step + 5)
            # Include checkpointing in the profile to see real-world overhead
            trainer._save_latest_checkpoint(step + 5)
            prof.step()
    
    # Export chrome trace
    prof.export_chrome_trace(torch_trace_path)
    
    # Print profiler summary
    print("\n" + "-" * 70)
    print("Profiler Summary (top 20 operations by total time):")
    print("-" * 70)
    
    if device == "cuda":
        sort_key = "cuda_time_total"
    else:
        sort_key = "cpu_time_total"
    
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))
    
    # Save text summary
    summary_path = os.path.join(trace_dir, f"{model_name}_{timestamp}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Description: {model_config['description']}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Parameters: {num_params / 1e6:.1f}M\n")
        f.write(f"d_head: {d_head}\n")
        f.write(f"Batch size: {model_config['batch_size']}\n")
        f.write(f"Steps profiled: {num_steps}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Top operations by {sort_key}:\n")
        f.write("=" * 80 + "\n\n")
        f.write(prof.key_averages().table(sort_by=sort_key, row_limit=50))
        
        # Also group by input shape
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Grouped by input shape:\n")
        f.write("=" * 80 + "\n\n")
        f.write(prof.key_averages(group_by_input_shape=True).table(
            sort_by=sort_key, row_limit=30
        ))
    
    print(f"\nSummary saved to: {summary_path}")
    
    # MPS-specific: Try to capture Metal activity if on macOS
    if device == "mps":
        print("\n" + "-" * 70)
        print("MPS Memory Stats:")
        print("-" * 70)
        try:
            current_mem = torch.mps.current_allocated_memory() / 1e9
            print(f"  Current allocated: {current_mem:.2f} GB")
        except (AttributeError, RuntimeError) as e:
            print(f"  Could not get MPS memory stats: {e}")
        
        # Note about Instruments
        print("\nNote: For detailed MPS/Metal profiling, use macOS Instruments:")
        print(f"  xcrun xctrace record --output '{trace_dir}/{model_name}_metal.trace' \\")
        print(f"      --template 'Metal System Trace' \\")
        print(f"      --launch -- python -m benchmarks.profile_training --model {model_name[6]}")
    
    # Cleanup
    del trainer.model
    del trainer.optimizer
    del trainer
    
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    
    print(f"\nCompleted profiling: {model_name}")
    
    return {
        "model_name": model_name,
        "torch_trace": torch_trace_path,
        "tensorboard_dir": tensorboard_dir,
        "summary": summary_path,
        "compile_debug_dir": compile_debug_dir if enable_compile_debug else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile Model A and Model B training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        choices=["A", "B", "both"],
        default="both",
        help="Which model(s) to profile (default: both)",
    )
    parser.add_argument(
        "--trace-dir",
        default="/Volumes/slab_storage/traces",
        help="Directory to save traces (default: /Volumes/slab_storage/traces)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of training steps to profile (default: 50)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (default: auto-detect mps/cuda/cpu)",
    )
    parser.add_argument(
        "--no-compile-debug",
        action="store_true",
        help="Disable torch.compile debug output",
    )
    args = parser.parse_args()
    
    # Validate trace directory
    if not os.path.exists(args.trace_dir):
        print(f"Error: Trace directory does not exist: {args.trace_dir}")
        print("Please ensure the external drive is mounted.")
        sys.exit(1)
    
    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    print(f"Trace directory: {args.trace_dir}")
    
    # Load data
    tokens, valid_tokens = load_data()
    
    # Profile models
    results = []
    
    if args.model in ("A", "both"):
        result = profile_model(
            MODEL_A_CONFIG,
            tokens,
            valid_tokens,
            device,
            args.trace_dir,
            args.num_steps,
            enable_compile_debug=not args.no_compile_debug,
        )
        results.append(result)
    
    if args.model in ("B", "both"):
        result = profile_model(
            MODEL_B_CONFIG,
            tokens,
            valid_tokens,
            device,
            args.trace_dir,
            args.num_steps,
            enable_compile_debug=not args.no_compile_debug,
        )
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)
    
    for result in results:
        print(f"\n{result['model_name']}:")
        print(f"  Chrome trace: {result['torch_trace']}")
        print(f"  TensorBoard:  {result['tensorboard_dir']}")
        print(f"  Summary:      {result['summary']}")
        if result['compile_debug_dir']:
            print(f"  Compile debug: {result['compile_debug_dir']}")
    
    print("\nTo view Chrome traces:")
    print("  1. Open chrome://tracing in Chrome/Chromium")
    print("  2. Load the .json trace file")
    
    print("\nTo view TensorBoard traces:")
    print(f"  tensorboard --logdir {args.trace_dir}")
    
    if device == "mps":
        print("\nFor detailed Metal/MPS profiling, run:")
        print(f"  xcrun xctrace record --output '{args.trace_dir}/metal_trace.trace' \\")
        print(f"      --template 'Metal System Trace' --time-limit 60s \\")
        print(f"      --launch -- uv run python -m benchmarks.profile_training --model A --num-steps 30")


if __name__ == "__main__":
    main()
