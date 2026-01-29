"""Sequential benchmark to catch memory fragmentation issues.

Unlike train_benchmark.py which cleans up between configs, this benchmark:
1. Runs multiple configs sequentially WITHOUT cleanup (simulates real training runs)
2. Runs longer per config to catch thermal throttling
3. Tests different execution orders to detect fragmentation sensitivity
4. Reports variance and degradation over time

Usage:
    uv run python -m benchmarks.sequential_benchmark
    uv run python -m benchmarks.sequential_benchmark --configs model_ab  # Just Model A then B
    uv run python -m benchmarks.sequential_benchmark --configs model_ba  # Model B then A
"""

import argparse
import gc
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from cs336_basics import TransformerLM
from cs336_basics.training import Trainer


# =============================================================================
# Model Configurations (same as train_compare.py)
# =============================================================================

MODEL_A_CONFIG = {
    "name": "Model_A_wide_attn",
    "description": "Wide attention, d_head=64, d_ff=1024",
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
}

MODEL_B_CONFIG = {
    "name": "Model_B_standard_ffn",
    "description": "Standard FFN, d_head=32, d_ff=1728",
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
}


def get_trainer_config(model_config: dict, device: str) -> dict:
    """Create trainer config for benchmarking."""
    return {
        "model_settings": model_config["model_settings"],
        "device": device,
        "batch_size": model_config["batch_size"],
        "num_iters": 1000,
        "eval_every": 999999,
        "gradient_clip": 1.0,
        "scheduler_lr_max": 1e-3,
        "scheduler_lr_min": 1e-4,
        "scheduler_warmup_iters": 100,
        "scheduler_cos_iters": 1000,
        "checkpoint_dir": "/tmp/seq_benchmark",
        "checkpoint_add_timestamp": False,
    }


def load_data():
    """Load tokenized data."""
    train_file = "tokenized/tinystories_train_fixed.npy"
    valid_file = "tokenized/tinystories_valid_fixed.npy"
    tokens = np.load(train_file, mmap_mode='r')
    valid_tokens = np.load(valid_file, mmap_mode='r')
    return tokens, valid_tokens


def sync_device(device: str):
    """Synchronize for accurate timing."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def benchmark_config(
    model_config: dict,
    tokens: np.ndarray,
    valid_tokens: np.ndarray,
    device: str,
    num_steps: int,
    warmup_steps: int = 5,
    cleanup_before: bool = False,
) -> dict:
    """Benchmark a single config and return detailed timing data."""
    
    model_name = model_config["name"]
    
    # Optional cleanup before (to test clean vs fragmented)
    if cleanup_before:
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    # Create trainer
    os.makedirs("/tmp/seq_benchmark", exist_ok=True)
    trainer_config = get_trainer_config(model_config, device)
    trainer = Trainer(TransformerLM, trainer_config, tokens, valid_tokens)
    
    tokens_per_step = model_config["batch_size"] * model_config["model_settings"]["context_length"]
    
    # Warmup
    for i in range(warmup_steps):
        trainer._train_step(i)
    sync_device(device)
    
    # Benchmark
    step_times = []
    start_time = time.perf_counter()
    
    for step in range(num_steps):
        step_start = time.perf_counter()
        trainer._train_step(step + warmup_steps)
        sync_device(device)
        step_times.append(time.perf_counter() - step_start)
    
    total_time = time.perf_counter() - start_time
    
    # Statistics
    times_ms = [t * 1000 for t in step_times]
    sorted_times = sorted(times_ms)
    
    # Check for degradation (compare first vs last 10 steps)
    first_10_avg = sum(times_ms[:10]) / 10 if len(times_ms) >= 10 else sum(times_ms) / len(times_ms)
    last_10_avg = sum(times_ms[-10:]) / 10 if len(times_ms) >= 10 else sum(times_ms) / len(times_ms)
    degradation_pct = ((last_10_avg - first_10_avg) / first_10_avg) * 100
    
    result = {
        "model_name": model_name,
        "num_steps": num_steps,
        "tokens_per_step": tokens_per_step,
        "cleanup_before": cleanup_before,
        "total_time_s": total_time,
        "avg_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "p50_ms": sorted_times[len(sorted_times) // 2],
        "p95_ms": sorted_times[int(len(sorted_times) * 0.95)],
        "p99_ms": sorted_times[int(len(sorted_times) * 0.99)],
        "std_ms": (sum((t - sum(times_ms)/len(times_ms))**2 for t in times_ms) / len(times_ms)) ** 0.5,
        "first_10_avg_ms": first_10_avg,
        "last_10_avg_ms": last_10_avg,
        "degradation_pct": degradation_pct,
        "throughput_tokens_per_sec": tokens_per_step / (sum(times_ms) / len(times_ms) / 1000),
        "step_times_ms": times_ms,
    }
    
    # Cleanup trainer but NOT device memory (to test fragmentation)
    del trainer.model
    del trainer.optimizer
    del trainer
    
    return result


def print_result(result: dict):
    """Pretty print a benchmark result."""
    print(f"\n  {result['model_name']}:")
    print(f"    Avg: {result['avg_ms']:.1f}ms  P50: {result['p50_ms']:.1f}ms  P95: {result['p95_ms']:.1f}ms  P99: {result['p99_ms']:.1f}ms")
    print(f"    Min: {result['min_ms']:.1f}ms  Max: {result['max_ms']:.1f}ms  Std: {result['std_ms']:.1f}ms")
    print(f"    Throughput: {result['throughput_tokens_per_sec']:.0f} tok/s")
    print(f"    Degradation (first 10 vs last 10): {result['degradation_pct']:+.1f}%")
    if result['degradation_pct'] > 10:
        print(f"    ⚠️  SIGNIFICANT DEGRADATION DETECTED")


def main():
    parser = argparse.ArgumentParser(description="Sequential benchmark for fragmentation testing")
    parser.add_argument("--configs", default="all", 
                        choices=["all", "model_ab", "model_ba", "model_a", "model_b"],
                        help="Which config sequence to run")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Steps per config (default: 50)")
    parser.add_argument("--device", default=None,
                        help="Device (auto-detect if not specified)")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for results")
    args = parser.parse_args()
    
    # Device
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Device: {device}")
    print(f"Steps per config: {args.num_steps}")
    
    # Load data
    tokens, valid_tokens = load_data()
    print(f"Loaded {len(tokens):,} train tokens")
    
    results = []
    
    if args.configs == "all":
        # Test 1: Each model in isolation (with cleanup)
        print("\n" + "=" * 70)
        print("TEST 1: Models in isolation (clean memory)")
        print("=" * 70)
        
        r = benchmark_config(MODEL_A_CONFIG, tokens, valid_tokens, device, args.num_steps, cleanup_before=True)
        results.append({"test": "isolated_clean", **r})
        print_result(r)
        
        r = benchmark_config(MODEL_B_CONFIG, tokens, valid_tokens, device, args.num_steps, cleanup_before=True)
        results.append({"test": "isolated_clean", **r})
        print_result(r)
        
        # Test 2: A then B (no cleanup between)
        print("\n" + "=" * 70)
        print("TEST 2: Model A -> Model B (NO cleanup between)")
        print("=" * 70)
        
        # Clean before starting sequence
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        
        r = benchmark_config(MODEL_A_CONFIG, tokens, valid_tokens, device, args.num_steps, cleanup_before=False)
        results.append({"test": "A_then_B", **r})
        print_result(r)
        
        r = benchmark_config(MODEL_B_CONFIG, tokens, valid_tokens, device, args.num_steps, cleanup_before=False)
        results.append({"test": "A_then_B", **r})
        print_result(r)
        
        # Test 3: B then A (no cleanup between)
        print("\n" + "=" * 70)
        print("TEST 3: Model B -> Model A (NO cleanup between)")
        print("=" * 70)
        
        # Clean before starting sequence
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        
        r = benchmark_config(MODEL_B_CONFIG, tokens, valid_tokens, device, args.num_steps, cleanup_before=False)
        results.append({"test": "B_then_A", **r})
        print_result(r)
        
        r = benchmark_config(MODEL_A_CONFIG, tokens, valid_tokens, device, args.num_steps, cleanup_before=False)
        results.append({"test": "B_then_A", **r})
        print_result(r)
        
    elif args.configs == "model_ab":
        print("\n" + "=" * 70)
        print("Model A -> Model B (NO cleanup between)")
        print("=" * 70)
        
        r = benchmark_config(MODEL_A_CONFIG, tokens, valid_tokens, device, args.num_steps, cleanup_before=True)
        results.append({"test": "A_then_B", **r})
        print_result(r)
        
        r = benchmark_config(MODEL_B_CONFIG, tokens, valid_tokens, device, args.num_steps, cleanup_before=False)
        results.append({"test": "A_then_B", **r})
        print_result(r)
        
    elif args.configs == "model_ba":
        print("\n" + "=" * 70)
        print("Model B -> Model A (NO cleanup between)")
        print("=" * 70)
        
        r = benchmark_config(MODEL_B_CONFIG, tokens, valid_tokens, device, args.num_steps, cleanup_before=True)
        results.append({"test": "B_then_A", **r})
        print_result(r)
        
        r = benchmark_config(MODEL_A_CONFIG, tokens, valid_tokens, device, args.num_steps, cleanup_before=False)
        results.append({"test": "B_then_A", **r})
        print_result(r)
        
    elif args.configs == "model_a":
        r = benchmark_config(MODEL_A_CONFIG, tokens, valid_tokens, device, args.num_steps, cleanup_before=True)
        results.append({"test": "isolated", **r})
        print_result(r)
        
    elif args.configs == "model_b":
        r = benchmark_config(MODEL_B_CONFIG, tokens, valid_tokens, device, args.num_steps, cleanup_before=True)
        results.append({"test": "isolated", **r})
        print_result(r)
    
    # Summary comparison
    if args.configs == "all":
        print("\n" + "=" * 70)
        print("SUMMARY: Fragmentation Impact")
        print("=" * 70)
        
        # Find results for comparison
        a_isolated = next((r for r in results if r["test"] == "isolated_clean" and "Model_A" in r["model_name"]), None)
        b_isolated = next((r for r in results if r["test"] == "isolated_clean" and "Model_B" in r["model_name"]), None)
        b_after_a = next((r for r in results if r["test"] == "A_then_B" and "Model_B" in r["model_name"]), None)
        a_after_b = next((r for r in results if r["test"] == "B_then_A" and "Model_A" in r["model_name"]), None)
        
        if b_isolated and b_after_a:
            slowdown = (b_after_a["avg_ms"] - b_isolated["avg_ms"]) / b_isolated["avg_ms"] * 100
            print(f"\nModel B slowdown when run after Model A: {slowdown:+.1f}%")
            if slowdown > 10:
                print("  ⚠️  FRAGMENTATION IMPACT DETECTED: Model B is significantly slower after Model A")
        
        if a_isolated and a_after_b:
            slowdown = (a_after_b["avg_ms"] - a_isolated["avg_ms"]) / a_isolated["avg_ms"] * 100
            print(f"Model A slowdown when run after Model B: {slowdown:+.1f}%")
            if slowdown > 10:
                print("  ⚠️  FRAGMENTATION IMPACT DETECTED: Model A is significantly slower after Model B")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"sequential_benchmark_{timestamp}.json")
    
    # Remove step_times_ms for cleaner output (can be huge)
    output_results = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != "step_times_ms"}
        output_results.append(r_copy)
    
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "device": device,
            "num_steps": args.num_steps,
            "results": output_results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
