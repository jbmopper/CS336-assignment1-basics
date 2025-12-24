# WandB Hyperparameter Sweeps

This document explains how to run hyperparameter sweeps using Weights & Biases (wandb) for the transformer language model.

## Overview

Hyperparameter sweeps allow you to systematically search for the best model configuration by training multiple models with different hyperparameter combinations. This implementation uses Bayesian optimization to efficiently explore the hyperparameter space.

## Prerequisites

1. **WandB Account**: Sign up at [wandb.ai](https://wandb.ai)

2. **Login to WandB**:
   ```bash
   wandb login
   ```

3. **Tokenized Data**: Ensure you have pre-tokenized training and validation data:
   ```
   ../tokenized/tinystories_train.npy
   ../tokenized/tinystories_valid.npy
   ```
   
   If not, run the main training script first to generate tokenized data.

## Quick Start

### 1. Initialize the Sweep

From the repository root, create a new sweep:

```bash
wandb sweep cs336_basics/sweep_config.yaml
```

This will output a sweep ID like:
```
Created sweep with ID: abc123
Run sweep agent with: wandb agent jbmopper-0/cs336-a1-sweep/abc123
```

### 2. Start Sweep Agent(s)

Run one or more agents to execute training runs:

```bash
# Single agent
wandb agent jbmopper-0/cs336-a1-sweep/<sweep_id>

# Or specify a count (number of runs)
wandb agent --count 10 jbmopper-0/cs336-a1-sweep/<sweep_id>
```

You can run multiple agents in parallel on different machines to speed up the sweep.

## Sweep Configuration

The sweep is configured in `cs336_basics/sweep_config.yaml`. Here are the key settings:

### Search Method

```yaml
method: bayes  # Bayesian optimization
```

Options:
- `bayes`: Bayesian optimization (recommended for expensive runs)
- `random`: Random search
- `grid`: Grid search (exhaustive)

### Optimization Metric

```yaml
metric:
  name: eval_loss
  goal: minimize
```

The sweep optimizes to minimize validation loss.

### Hyperparameters

| Parameter | Type | Range/Values | Description |
|-----------|------|--------------|-------------|
| `lr_max` | log_uniform | 1e-4 to 1e-2 | Maximum learning rate |
| `lr_min` | log_uniform | 1e-6 to 1e-4 | Minimum learning rate |
| `d_model` | categorical | [256, 384, 512, 768] | Model dimension |
| `num_heads` | categorical | [4, 8, 16] | Number of attention heads |
| `num_layers` | categorical | [2, 4, 6, 8] | Number of transformer layers |
| `d_ff` | int_uniform | 512 to 2048 | Feed-forward dimension |
| `batch_size` | categorical | [16, 32, 64] | Batch size |
| `context_length` | categorical | [128, 256, 512] | Context window length |
| `warmup_iters` | int_uniform | 50 to 500 | Warmup iterations |
| `weight_decay` | log_uniform | 1e-5 to 1e-1 | Weight decay coefficient |
| `gradient_clip` | uniform | 0.5 to 2.0 | Gradient clipping threshold |

### Early Termination

Poor-performing runs are automatically stopped using Hyperband:

```yaml
early_terminate:
  type: hyperband
  min_iter: 100
  eta: 3
  s: 2
```

## Customization

### Modifying the Sweep

Edit `cs336_basics/sweep_config.yaml` to:
- Add/remove hyperparameters
- Change parameter ranges
- Modify the search method
- Adjust early termination settings

### Modifying Training

Edit `cs336_basics/train_sweep.py` to:
- Change the training loop
- Add new metrics
- Modify data loading
- Adjust model architecture options

### Running Shorter Experiments

Override the number of iterations:

```bash
# Modify DEFAULT_CONFIG in train_sweep.py, or use the --num_iters flag
python cs336_basics/train_sweep.py --num_iters 100
```

## Viewing Results

1. **Dashboard**: Visit your project page on wandb.ai to see:
   - Parallel coordinates plot of hyperparameters
   - Parameter importance analysis
   - Best performing runs

2. **Best Parameters**: The wandb UI shows which hyperparameter combinations work best.

3. **Export Results**: Use the wandb API or download CSV from the dashboard.

## Tips

1. **Start Small**: Run a few short experiments first to verify everything works.

2. **Resource Planning**: Each sweep agent runs one training job at a time. Plan GPU resources accordingly.

3. **Parallelization**: Run multiple agents on different GPUs/machines for faster sweeps.

4. **Checkpoint Resume**: The sweep script can be extended to save and resume from checkpoints.

## Troubleshooting

### "Data not found" Error
Run the main training script first to generate tokenized data, or manually run the tokenization.

### Invalid Model Configuration
The sweep validates that `d_model` is divisible by `num_heads`. Incompatible configurations are skipped.

### OOM Errors
Reduce `batch_size` or `context_length` in the sweep config if runs are running out of memory.
