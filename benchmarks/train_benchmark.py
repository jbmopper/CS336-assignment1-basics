"""Training step benchmark for TransformerLM.

Benchmarks the _train_step method of Trainer across various model configurations.
Uses blocked_autorange for accurate timing measurements.
"""

from cs336_basics.training import Trainer
from cs336_basics import TransformerLM
import numpy as np
import torch.utils.benchmark as benchmark
import torch
from itertools import product
import json
from datetime import datetime
import argparse
import os


def make_config(batch_size, seq_len, d_model, num_heads, num_layers, d_ff, device="mps"):
    """Create a trainer config for the given parameters."""
    return {
        "model_settings": {
            "vocab_size": 10000,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "d_ff": d_ff,
            "context_length": seq_len,
            "rope_theta": 10000.0,
        },
        "device": device,
        "checkpoint_dir": "/tmp/benchmark_checkpoints",
        "checkpoint_add_timestamp": False,
        "batch_size": batch_size,
        "num_iters": 1,  # Not used for direct _train_step benchmark
        "eval_every": 999999,
        "gradient_clip": 1.0,
        # LR schedule - need non-zero values to avoid division by zero
        "scheduler_lr_max": 1e-3,
        "scheduler_lr_min": 1e-4,
        "scheduler_warmup_iters": 1,  # Must be > 0
        "scheduler_cos_iters": 1000,
        # No wandb_entity = no W&B logging
    }


def estimate_memory_gb(batch_size, seq_len, d_model, num_heads, num_layers, d_ff):
    """Rough estimate of peak training memory in GB."""
    # Weights (float32 = 4 bytes)
    V = 10000
    weights = (
        2 * V * d_model +  # embeddings + lm_head
        num_layers * (4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model)
    ) * 4
    
    # Activations per block (rough estimate, ~25 tensors saved for backward)
    d_head = d_model // num_heads
    per_block_activations = (
        10 * batch_size * seq_len * d_model +  # various [B, S, d] tensors
        5 * batch_size * num_heads * seq_len * seq_len +  # attention matrices (S²)
        5 * batch_size * seq_len * d_ff  # SwiGLU intermediates
    ) * 4
    
    total_activations = num_layers * per_block_activations
    
    # Gradients ≈ weights
    gradients = weights
    
    # Peak = weights + activations + gradients
    peak_bytes = weights + total_activations + gradients
    return peak_bytes / 1e9


def main():
    parser = argparse.ArgumentParser(description="Benchmark training step")
    parser.add_argument("--smol", action="store_true", help="Small sweep for quick testing")
    parser.add_argument("--data", default="tinystories", help="Dataset to use")
    parser.add_argument("--device", default="mps", help="Device (mps, cuda, cpu)")
    parser.add_argument("--max-memory-gb", type=float, default=20.0, 
                        help="Skip configs estimated to exceed this memory (GB)")
    parser.add_argument("--min-run-time", type=float, default=2.0,
                        help="Minimum benchmark run time in seconds")
    parser.add_argument("--output-dir", default=".", help="Output directory for results")
    args = parser.parse_args()

    # Define sweep parameters
    # d_head is the grid variable; n_heads = d_model / d_head
    if args.smol:
        batch_sizes = [8, 16, 32]
        seq_lens = [256, 512]
        d_models = [256, 512]
        d_heads = [32, 64]  # d_head as grid variable
        num_layers_list = [4, 8]
        d_ffs = [1024, 2048]
    else:
        batch_sizes = [8, 16, 32, 48, 64, 80]
        seq_lens = [256, 512, 768, 1024]
        d_models = [256, 384, 512, 640, 768]
        d_heads = [32, 64, 96, 128]  # d_head as grid variable
        num_layers_list = [8, 10,12, 14, 16]
        d_ffs = [640, 704, 1024, 1344, 1728, 2048]

    # Load data
    if args.data == "tinystories":
        train_file = "tokenized/tinystories_train_fixed.npy"
        valid_file = "tokenized/tinystories_valid_fixed.npy"
    else:
        raise ValueError(f"Unknown data: {args.data}")

    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found. Run from repository root.")
        return

    tokens = np.load(train_file, mmap_mode='r')
    valid_tokens = np.load(valid_file, mmap_mode='r')
    print(f"Loaded {len(tokens):,} train tokens, {len(valid_tokens):,} valid tokens")

    # Generate all configs
    all_configs = []
    for B, S, d, dh, L, dff in product(
        batch_sizes, seq_lens, d_models, d_heads, num_layers_list, d_ffs
    ):
        # Compute n_heads from d_model and d_head
        if d % dh != 0:
            continue  # d_model must be divisible by d_head
        n_heads = d // dh
        if n_heads < 1:
            continue  # need at least 1 head
        
        # Estimate memory and skip if too large
        est_mem = estimate_memory_gb(B, S, d, n_heads, L, dff)
        if est_mem > args.max_memory_gb:
            continue
        
        all_configs.append({
            "batch_size": B,
            "seq_len": S,
            "d_model": d,
            "d_head": dh,
            "num_heads": n_heads,  # computed from d_model / d_head
            "num_layers": L,
            "d_ff": dff,
            "est_memory_gb": est_mem,
        })

    print(f"Running {len(all_configs)} configs (filtered by memory < {args.max_memory_gb} GB)")

    results = []
    oom_configs = []
    skipped_configs = []

    for i, cfg in enumerate(all_configs):
        label = "train_step"
        sublabel = f"B={cfg['batch_size']}, S={cfg['seq_len']}, d={cfg['d_model']}, dh={cfg['d_head']}, h={cfg['num_heads']}, L={cfg['num_layers']}, dff={cfg['d_ff']}"
        
        print(f"\n[{i+1}/{len(all_configs)}] {sublabel} (est. {cfg['est_memory_gb']:.1f} GB)")
        
        trainer = None
        try:
            # Create trainer config
            trainer_config = make_config(
                batch_size=cfg["batch_size"],
                seq_len=cfg["seq_len"],
                d_model=cfg["d_model"],
                num_heads=cfg["num_heads"],  # pass computed num_heads
                num_layers=cfg["num_layers"],
                d_ff=cfg["d_ff"],
                device=args.device,
            )
            
            # Create trainer (no W&B logging)
            trainer = Trainer(TransformerLM, trainer_config, tokens, valid_tokens)
            
            # Warmup run
            if args.device == "mps":
                torch.mps.synchronize()
            elif args.device == "cuda":
                torch.cuda.synchronize()
            
            trainer._train_step(0)
            
            if args.device == "mps":
                torch.mps.synchronize()
            elif args.device == "cuda":
                torch.cuda.synchronize()
            
            # Benchmark
            def bench_fn():
                trainer._train_step(0)
                if args.device == "mps":
                    torch.mps.synchronize()
                elif args.device == "cuda":
                    torch.cuda.synchronize()
            
            result = benchmark.Timer(
                stmt="bench_fn()",
                globals={"bench_fn": bench_fn},
                label=label,
                sub_label=sublabel,
                description=f"L={cfg['num_layers']}",
            ).blocked_autorange(min_run_time=args.min_run_time)
            
            results.append(result)
            cfg["median_s"] = result.median
            cfg["mean_s"] = result.mean
            cfg["iqr_s"] = result.iqr
            cfg["num_runs"] = len(result.times)
            
            # Calculate throughput
            tokens_per_step = cfg["batch_size"] * cfg["seq_len"]
            cfg["tokens_per_sec"] = tokens_per_step / result.median
            cfg["steps_per_sec"] = 1.0 / result.median
            
            print(f"  -> {result.median*1000:.1f} ms/step, {cfg['tokens_per_sec']:.0f} tok/s")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  -> OOM, skipping")
                oom_configs.append(cfg)
                if args.device == "mps":
                    torch.mps.empty_cache()
                elif args.device == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise
        except Exception as e:
            print(f"  -> Error: {e}")
            skipped_configs.append({"config": cfg, "error": str(e)})
        finally:
            # Cleanup
            if trainer is not None:
                del trainer.model
                del trainer.optimizer
                del trainer
            if args.device == "mps":
                torch.mps.empty_cache()
            elif args.device == "cuda":
                torch.cuda.empty_cache()

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Detailed results
    output = {
        "timestamp": timestamp,
        "device": args.device,
        "max_memory_gb": args.max_memory_gb,
        "min_run_time": args.min_run_time,
        "results": [cfg for cfg in all_configs if "median_s" in cfg],
        "oom_configs": oom_configs,
        "skipped_configs": skipped_configs,
    }
    
    filename = os.path.join(args.output_dir, f"train_benchmark_{timestamp}.json")
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved detailed results to {filename}")

    # Print comparison table
    if results:
        print("\n" + "="*80)
        compare = benchmark.Compare(results)
        compare.colorize()
        compare.print()

    # Summary
    print(f"\n{'='*80}")
    print(f"Summary: {len(results)} successful, {len(oom_configs)} OOM, {len(skipped_configs)} errors")
    
    if results:
        successful = [cfg for cfg in all_configs if "median_s" in cfg]
        fastest = min(successful, key=lambda x: x["median_s"])
        most_throughput = max(successful, key=lambda x: x["tokens_per_sec"])
        
        print(f"\nFastest config ({fastest['median_s']*1000:.1f} ms/step):")
        print(f"  B={fastest['batch_size']}, S={fastest['seq_len']}, d={fastest['d_model']}, "
              f"dh={fastest['d_head']}, h={fastest['num_heads']}, L={fastest['num_layers']}, dff={fastest['d_ff']}")
        
        print(f"\nHighest throughput ({most_throughput['tokens_per_sec']:.0f} tok/s):")
        print(f"  B={most_throughput['batch_size']}, S={most_throughput['seq_len']}, d={most_throughput['d_model']}, "
              f"dh={most_throughput['d_head']}, h={most_throughput['num_heads']}, L={most_throughput['num_layers']}, dff={most_throughput['d_ff']}")


if __name__ == "__main__":
    main()
