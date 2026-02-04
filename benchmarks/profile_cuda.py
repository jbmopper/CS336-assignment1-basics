"""CUDA profiling script for TransformerLM training.

Integrates with torch.profiler for detailed CUDA traces, and provides
wrapper commands for nsys (Nsight Systems) and ncu (Nsight Compute).

Usage:
    # Basic profiling with torch.profiler (outputs Chrome trace)
    uv run python -m benchmarks.profile_cuda --steps 10 --precision bf16
    
    # System-level profiling with nsys (timeline view)
    nsys profile -t cuda,nvtx -o training_trace --stats=true \\
        uv run python -m benchmarks.profile_cuda --steps 10 --precision bf16
    
    # Kernel-level analysis with ncu (detailed metrics)
    ncu --set full --target-processes all -o kernel_analysis \\
        uv run python -m benchmarks.profile_cuda --steps 1 --precision bf16
    
    # View nsys results
    nsys-ui training_trace.nsys-rep
    
    # View ncu results  
    ncu-ui kernel_analysis.ncu-rep

Model presets:
    --model assignment  : Assignment spec (d_model=512, num_layers=4, d_ff=1344)
    --model wide        : Wide attention (d_model=640, num_layers=10, d_ff=1024)
    --model deep        : Deep narrow (d_model=384, num_layers=12, d_ff=1728)
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from cs336_basics import TransformerLM
from cs336_basics.training import Trainer


# Model configuration presets
MODEL_PRESETS = {
    "assignment": {
        "name": "assignment",
        "description": "Assignment spec (~17M params)",
        "batch_size": 64,
        "model_settings": {
            "vocab_size": 10000,
            "d_model": 512,
            "num_heads": 16,
            "num_layers": 4,
            "d_ff": 1344,
            "context_length": 256,
            "rope_theta": 10000.0,
        },
    },
    "wide": {
        "name": "wide_attention",
        "description": "Wide attention, low FFN ratio (~50M params)",
        "batch_size": 64,
        "model_settings": {
            "vocab_size": 10000,
            "d_model": 640,
            "num_heads": 10,
            "num_layers": 10,
            "d_ff": 1024,
            "context_length": 256,
            "rope_theta": 10000.0,
        },
    },
    "deep": {
        "name": "deep_narrow",
        "description": "Deep narrow, high FFN ratio (~28M params)",
        "batch_size": 48,
        "model_settings": {
            "vocab_size": 10000,
            "d_model": 384,
            "num_heads": 12,
            "num_layers": 12,
            "d_ff": 1728,
            "context_length": 256,
            "rope_theta": 10000.0,
        },
    },
}


def get_trainer_config(model_config: dict, device: str, precision: str, checkpoint_dir: str, num_iters: int = 1000) -> dict:
    """Create a trainer config for profiling (no W&B)."""
    return {
        "model_settings": model_config["model_settings"],
        "device": device,
        "precision": precision,
        "batch_size": model_config["batch_size"],
        "num_iters": num_iters,
        "eval_every": 999999,
        "gradient_clip": 1.0,
        "scheduler_lr_max": 1e-3,
        "scheduler_lr_min": 1e-4,
        "scheduler_warmup_iters": 100,
        "scheduler_cos_iters": num_iters,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_add_timestamp": False,
    }


def load_data(data_dir: str):
    """Load tokenized training data."""
    train_file = os.path.join(data_dir, "tinystories_train.npy")
    valid_file = os.path.join(data_dir, "tinystories_valid.npy")
    
    # Try alternative names
    if not os.path.exists(train_file):
        train_file = os.path.join(data_dir, "tinystories_train_fixed.npy")
        valid_file = os.path.join(data_dir, "tinystories_valid_fixed.npy")

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Training data not found in {data_dir}\n"
            f"Looked for: tinystories_train.npy or tinystories_train_fixed.npy"
        )

    tokens = np.load(train_file, mmap_mode='r')
    valid_tokens = np.load(valid_file, mmap_mode='r')
    print(f"Loaded {len(tokens):,} train tokens, {len(valid_tokens):,} valid tokens")
    return tokens, valid_tokens


def profile_with_torch_profiler(
    trainer: Trainer,
    num_steps: int,
    warmup_steps: int,
    output_dir: str,
    model_name: str,
):
    """Profile training steps using torch.profiler."""
    print(f"\nProfiling with torch.profiler ({num_steps} steps, {warmup_steps} warmup)...")
    
    # Warmup
    for i in range(warmup_steps):
        trainer._train_step(i)
    torch.cuda.synchronize()
    
    # Profile
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_dir = os.path.join(output_dir, f"torch_traces_{model_name}_{timestamp}")
    os.makedirs(trace_dir, exist_ok=True)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=1,
            active=num_steps - 1,
            repeat=1,
        ),
        on_trace_ready=tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(num_steps):
            trainer._train_step(step + warmup_steps)
            prof.step()
    
    # Also save Chrome trace
    chrome_trace = os.path.join(output_dir, f"chrome_trace_{model_name}_{timestamp}.json")
    prof.export_chrome_trace(chrome_trace)
    
    print(f"\nTorch profiler traces saved to:")
    print(f"  TensorBoard: {trace_dir}")
    print(f"  Chrome trace: {chrome_trace}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CUDA TIME SUMMARY (sorted by CUDA time)")
    print("=" * 60)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    return prof


def profile_simple_timing(
    trainer: Trainer,
    num_steps: int,
    warmup_steps: int,
    model_name: str,
):
    """Simple timing-based profiling for nsys/ncu integration."""
    print(f"\nSimple timing profiling ({num_steps} steps, {warmup_steps} warmup)...")
    print("(Use with nsys or ncu for detailed GPU analysis)")
    
    # Warmup
    for i in range(warmup_steps):
        trainer._train_step(i)
    torch.cuda.synchronize()
    
    # Profile with NVTX markers for nsys
    step_times = []
    total_start = time.perf_counter()
    
    for step in range(num_steps):
        # NVTX range for nsys visualization
        torch.cuda.nvtx.range_push(f"step_{step}")
        
        step_start = time.perf_counter()
        trainer._train_step(step + warmup_steps)
        torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        
        torch.cuda.nvtx.range_pop()
        
        if (step + 1) % 5 == 0:
            print(f"  Step {step + 1}/{num_steps}: {step_time * 1000:.1f}ms")
    
    total_time = time.perf_counter() - total_start
    
    # Statistics
    avg_ms = sum(step_times) * 1000 / len(step_times)
    min_ms = min(step_times) * 1000
    max_ms = max(step_times) * 1000
    
    tokens_per_step = trainer.config["batch_size"] * trainer.config["model_settings"]["context_length"]
    tokens_per_sec = tokens_per_step / (avg_ms / 1000)
    
    print("\n" + "=" * 60)
    print("TIMING RESULTS")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg step: {avg_ms:.1f}ms")
    print(f"Min step: {min_ms:.1f}ms")
    print(f"Max step: {max_ms:.1f}ms")
    print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")
    
    return step_times


def main():
    parser = argparse.ArgumentParser(
        description="Profile TransformerLM training on CUDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_PRESETS.keys()),
        default="assignment",
        help="Model preset to profile",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Training precision",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of training steps to profile",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "tokenized"),
        help="Directory containing tokenized data",
    )
    parser.add_argument(
        "--output-dir",
        default="profile_results",
        help="Directory for profiling output",
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Use torch.profiler (vs simple timing for nsys/ncu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size",
    )
    # Custom model overrides
    parser.add_argument("--d-model", type=int, help="Override d_model")
    parser.add_argument("--num-layers", type=int, help="Override num_layers")
    parser.add_argument("--num-heads", type=int, help="Override num_heads")
    parser.add_argument("--d-ff", type=int, help="Override d_ff")
    parser.add_argument("--vocab-size", type=int, help="Override vocab_size")
    parser.add_argument("--seq-len", type=int, help="Override context_length")

    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a CUDA GPU.")
        return
    
    print("=" * 60)
    print("CUDA PROFILING")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Precision: {args.precision}")
    print(f"Model Preset: {args.model}")
    
    # Load model config
    model_config = MODEL_PRESETS[args.model].copy()
    model_config["model_settings"] = model_config["model_settings"].copy()
    
    # Apply overrides
    if args.batch_size:
        model_config["batch_size"] = args.batch_size
    
    ms = model_config["model_settings"]
    if args.d_model: ms["d_model"] = args.d_model
    if args.num_layers: ms["num_layers"] = args.num_layers
    if args.num_heads: ms["num_heads"] = args.num_heads
    if args.d_ff: ms["d_ff"] = args.d_ff
    if args.vocab_size: ms["vocab_size"] = args.vocab_size
    if args.seq_len: ms["context_length"] = args.seq_len
    
    print(f"Batch size: {model_config['batch_size']}")
    print(f"Config: d_model={ms['d_model']}, layers={ms['num_layers']}, heads={ms['num_heads']}, ctx={ms['context_length']}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    tokens, valid_tokens = load_data(args.data_dir)
    
    # Create trainer
    checkpoint_dir = os.path.join("/tmp", f"profile_ckpt_{args.model}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    trainer_config = get_trainer_config(
        model_config,
        device="cuda",
        precision=args.precision,
        checkpoint_dir=checkpoint_dir,
        num_iters=args.steps + args.warmup + 10,  # Ensure schedule covers profile duration
    )
    
    trainer = Trainer(TransformerLM, trainer_config, tokens, valid_tokens)
    
    # Print model stats
    num_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"Parameters: {num_params / 1e6:.1f}M")
    
    # Profile
    if args.torch_profiler:
        profile_with_torch_profiler(
            trainer,
            args.steps,
            args.warmup,
            args.output_dir,
            args.model,
        )
    else:
        profile_simple_timing(
            trainer,
            args.steps,
            args.warmup,
            args.model,
        )
    
    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
    
    if not args.torch_profiler:
        print("\nFor detailed GPU analysis, run with nsys or ncu:")
        print(f"  nsys profile -t cuda,nvtx -o trace --stats=true \\")
        print(f"      uv run python -m benchmarks.profile_cuda --model {args.model} --precision {args.precision}")
        print(f"")
        print(f"  ncu --set full -o kernel_analysis \\")
        print(f"      uv run python -m benchmarks.profile_cuda --model {args.model} --precision {args.precision} --steps 1")


if __name__ == "__main__":
    main()
