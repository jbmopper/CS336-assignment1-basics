#!/usr/bin/env python3
"""
WandB Optimizer Sweep Training Script

Sweeps optimizer hyperparameters (lr_max, beta2, weight_decay, warmup_iters)
using a composite metric M that Hyperband minimizes for early stopping.

Composite metric M (logged every evaluation step):

    M = Σ f(ℓ_ι, L_ι) · κ^ι

where ℓ_ι is the rolling average of METRIC_WINDOW eval-loss deltas,
L_ι is the current eval loss, ρ is the LR delta between eval points, and:

    f = -ℓ   when ρ ≥ 0  (warmup: reward loss increase)
    f =  L   when ρ < 0  (decay:  penalize high absolute loss)

With eval_every=5 over 1000 steps, there are 200 eval points for Hyperband.

Usage:
    # Initialize sweep (run once):
    wandb sweep cs336_basics/optimizer_sweep_config.yaml

    # Start agent(s) to run training:
    wandb agent <entity>/<project>/<sweep_id>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import wandb

from cs336_basics import TransformerLM
from cs336_basics.training import Trainer, setup_device, load_tokens

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Metric hyper-parameters ──────────────────────────────────────────────────
METRIC_KAPPA = 1.015   # exponential time-weighting
METRIC_WINDOW = 5      # rolling-average window (in eval points)

# ── Default training config ──────────────────────────────────────────────────
DEFAULT_CONFIG = dict(
    # Paths (overridden by DATA_DIR env on pod)
    train_file=str(BASE_DIR / "tokenized/tinystories_train.npy"),
    valid_file=str(BASE_DIR / "tokenized/tinystories_valid.npy"),
    checkpoint_dir=str(BASE_DIR / "checkpoints/optimizer_sweeps"),

    save_best=False,
    save_final=False,
    save_latest=False,

    # Tokenizer
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],

    # Model architecture (fixed for this sweep)
    d_model=512,
    num_heads=16,
    num_layers=4,
    d_ff=1344,
    context_length=256,
    rope_theta=10000,

    # Training
    batch_size=32,
    num_iters=1000,
    gradient_clip=1.0,

    # LR schedule (lr_max and warmup_iters are swept)
    scheduler_lr_max=1e-3,
    scheduler_lr_min=1e-4,
    scheduler_warmup_iters=100,
    scheduler_cos_iters=1000,

    # Optimizer (beta2 and weight_decay are swept)
    optimizer_lr=1e-3,
    optimizer_betas=(0.9, 0.999),
    optimizer_eps=1e-8,
    optimizer_weight_decay=1e-2,

    # Evaluation cadence drives sweep metric logging/Hyperband iterations
    eval_every=5,
    eval_batches=2,

    # W&B
    wandb_entity="jbmopper-0",
    log_project="cs336-optimizer-sweep-high",

    rand_seed=42,
)


# ── Composite metric tracker ────────────────────────────────────────────────

class SweepMetricTracker:
    """Accumulates the composite sweep metric M over training.

    Called as a step_callback by Trainer.train_eval_loop.
    """

    def __init__(
        self,
        kappa: float = METRIC_KAPPA,
        window: int = METRIC_WINDOW,
    ):
        self.kappa = kappa
        self.window = window
        self.eval_losses: list[float] = []
        self.lrs: list[float] = []
        self.M = 0.0
        self.iota = 0

    def __call__(self, step: int, log: dict) -> dict | None:
        # Only update/log sweep metric on evaluation steps
        eval_loss = log.get("Eval Loss")
        lr = log.get("LR")
        if eval_loss is None or lr is None:
            return None

        self.eval_losses.append(eval_loss)
        self.lrs.append(lr)

        # Update M once rolling window is full (need window+1 points for `window` deltas)
        if len(self.eval_losses) >= self.window + 1:
            recent = self.eval_losses[-(self.window + 1):]
            deltas = [recent[i + 1] - recent[i] for i in range(self.window)]
            ell = sum(deltas) / self.window

            rho = self.lrs[-1] - self.lrs[-2]

            if rho >= 0:
                f_val = -ell    # warmup: reward loss increase
            else:
                f_val = eval_loss  # decay: penalize high absolute eval loss

            self.M += f_val * (self.kappa ** self.iota)
            self.iota += 1

        # Always log M on eval points so Hyperband iteration count matches eval count
        return {"Sweep Metric M": self.M}


# ── Main sweep entry point ──────────────────────────────────────────────────

def train_sweep():
    run = wandb.init()

    config = DEFAULT_CONFIG.copy()

    def resolve_tokenized_pair(data_dir: str) -> tuple[str, str] | None:
        candidates = (
            ("tinystories_train.npy", "tinystories_valid.npy"),
            ("tinystories_train_fixed.npy", "tinystories_valid_fixed.npy"),
        )
        for train_name, valid_name in candidates:
            train_path = os.path.join(data_dir, train_name)
            valid_path = os.path.join(data_dir, valid_name)
            if os.path.exists(train_path) and os.path.exists(valid_path):
                return train_path, valid_path
        return None

    # Prefer DATA_DIR on pod if provided; support normal and *_fixed naming.
    data_dir = os.environ.get("DATA_DIR")
    selected_pair = resolve_tokenized_pair(data_dir) if data_dir else None
    if selected_pair is None:
        selected_pair = resolve_tokenized_pair(str(BASE_DIR / "tokenized"))
    if selected_pair is not None:
        config["train_file"], config["valid_file"] = selected_pair

    # Per-run checkpoint directory
    run_checkpoint_dir = os.path.join(config["checkpoint_dir"], run.id)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    config["checkpoint_dir"] = run_checkpoint_dir
    config["checkpoint_add_timestamp"] = False

    # ── Apply sweep overrides ────────────────────────────────────────────
    sweep_params = dict(wandb.config)

    # beta2 is swept as a scalar; reconstruct the betas tuple
    if "optimizer_beta2" in sweep_params:
        beta2 = sweep_params.pop("optimizer_beta2")
        config["optimizer_betas"] = (0.9, beta2)
        print(f"Sweep override: optimizer_beta2 = {beta2}")

    for key, value in sweep_params.items():
        if key in config:
            config[key] = value
            print(f"Sweep override: {key} = {value}")

    # Tie lr_min to lr_max (1/10 ratio) unless explicitly swept
    if "scheduler_lr_min" not in sweep_params:
        config["scheduler_lr_min"] = config["scheduler_lr_max"] / 10
    if config["scheduler_lr_min"] >= config["scheduler_lr_max"]:
        config["scheduler_lr_min"] = config["scheduler_lr_max"] / 10

    config["optimizer_lr"] = config["scheduler_lr_max"]
    config["scheduler_cos_iters"] = config["num_iters"]

    # ── Build model ──────────────────────────────────────────────────────
    model_settings = dict(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        context_length=config["context_length"],
        rope_theta=config["rope_theta"],
    )
    for key in ("norm_mode", "use_rope", "ffn_type", "ffn_hidden_dim", "final_norm"):
        if key in config:
            model_settings[key] = config[key]

    if model_settings["d_model"] % model_settings["num_heads"] != 0:
        print(
            f"Invalid config: d_model ({model_settings['d_model']}) "
            f"not divisible by num_heads ({model_settings['num_heads']})"
        )
        wandb.finish(exit_code=1)
        return

    prefer_cuda = torch.cuda.is_available()
    device = setup_device(seed=config["rand_seed"], prefer_cuda=prefer_cuda)
    config["device"] = device

    try:
        tokens, valid_tokens = load_tokens(
            config["train_file"], config["valid_file"]
        )
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        wandb.finish(exit_code=1)
        return

    config["model_settings"] = model_settings
    print(f"Creating model with settings: {model_settings}")

    run.config.update({
        "metric_source": "Eval Loss",
        "metric_eval_every": config["eval_every"],
        "metric_kappa": METRIC_KAPPA,
        "metric_window": METRIC_WINDOW,
    })

    tracker = SweepMetricTracker()
    trainer = Trainer(TransformerLM, config, tokens, valid_tokens, wandb_run=run)

    num_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"Model has {num_params:,} parameters")
    run.log({"num_parameters": num_params})

    trainer.train_eval_loop(step_callback=tracker)


def main():
    parser = argparse.ArgumentParser(description="WandB Optimizer Sweep")
    parser.add_argument(
        "--num_iters", type=int, default=None,
        help="Override number of training iterations",
    )
    args, _unknown = parser.parse_known_args()

    if args.num_iters is not None:
        DEFAULT_CONFIG["num_iters"] = args.num_iters

    train_sweep()


if __name__ == "__main__":
    main()
