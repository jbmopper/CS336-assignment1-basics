"""Trainer utilities for transformer training loops."""

from __future__ import annotations

import math
import os
from datetime import datetime

import torch
import wandb
from tqdm.auto import tqdm

from ..implementations import crossentropy, get_batch, gradient_clipping, save_checkpoint
from ..optimizer import AdamW, get_lr_cosine_schedule
from .timer import timer


class Trainer:
    """Manages model training, optimization, and logging."""

    def __init__(self, model_class, config, tokens, valid_tokens):
        """Initialize the Trainer.

        Args:
            model_class: The model class to instantiate (e.g., TransformerLM)
            config: Configuration dictionary containing model_settings and other params
            tokens: Training token array
            valid_tokens: Validation token array
        """
        self.model = model_class(**config["model_settings"]).to(config["device"])
        if config["optimizer_use_defaults"]:
            self.optimizer = AdamW(self.model.parameters())  # using defaults
        else:
            # TODO: optimizer call with parameters
            pass

        # hardcoding the wand B for now
        wandb.login()
        self.wandb_run = wandb.init(
            entity=config["wandb_entity"],
            project=config["log_project"],
            name=config["run_name"],
            config=config,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        config["checkpoint_dir"] = config["checkpoint_dir"] + "_" + timestamp
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        self.config = config
        self.tokens = tokens
        self.valid_tokens = valid_tokens

    def train_eval_loop(self):
        """Main training loop that orchestrates training and evaluation."""
        for i in tqdm(range(self.config["num_iters"]), desc="Training Progress"):
            # Training step
            train_log = self._train_step(i)

            # Evaluation (if needed)
            if i % self.config["eval_every"] == 0:
                eval_log = self._eval_step(i)
                train_log.update(eval_log)

            # Log all metrics
            self.wandb_run.log(train_log, step=i)

            # Save latest checkpoint every iteration (for crash recovery)
            self._save_latest_checkpoint(i)

            # Save snapshot checkpoint at save_every intervals
            if i % self.config["save_every"] == 0:
                self._save_snapshot_checkpoint(i)

        self.wandb_run.finish()

    def _train_step(self, step):
        """Perform a single training step. Returns dict of metrics to log."""
        log = {}

        # Device sync for benchmarking
        if self.config["device"] == "mps":
            torch.mps.synchronize()
        elif self.config["device"] == "cuda":
            torch.cuda.synchronize()

        # Get batch
        with timer("Batch getting time", log):
            inputs, labels = get_batch(
                self.tokens,
                self.config["batch_size"],
                self.config["model_settings"]["context_length"],
                self.config["device"],
            )

        # Forward pass
        with timer("Forward pass time", log):
            self.model.train()
            out_logits = self.model.forward(inputs)

        # Loss calculation
        with timer("Loss calc time", log):
            loss = crossentropy(out_logits, labels)
            perplexity = math.exp(loss.item())

        log["Loss"] = loss.item()
        log["Perplexity"] = perplexity

        # Backward pass
        with timer("Backwards pass time", log):
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

        # Gradient clipping
        with timer("Grad calc time", log):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float("inf"),
            )
        log["Grad norm"] = grad_norm.item()

        with timer("Grad clip time", log):
            gradient_clipping(
                self.model.parameters(),
                self.config["gradient_clip"],
            )

        # Optimizer step with LR schedule
        lr = get_lr_cosine_schedule(
            step,
            self.config["lr_max"],
            self.config["lr_min"],
            self.config["warmup_iters"],
            self.config["cos_iters"],
        )
        log["LR"] = lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        with timer("Optimizer step time", log):
            self.optimizer.step()

        return log

    def _eval_step(self, step):
        """Perform evaluation on validation set. Returns dict of metrics to log."""
        log = {}

        with timer("Eval batch getting time", log):
            self.model.eval()
            eval_inputs, eval_labels = get_batch(
                self.valid_tokens,
                self.config["batch_size"],
                self.config["model_settings"]["context_length"],
                self.config["device"],
            )

        with timer("Eval forward pass time", log):
            with torch.no_grad():
                eval_logits = self.model.forward(eval_inputs)
                eval_loss = crossentropy(eval_logits, eval_labels)
                eval_perplexity = math.exp(eval_loss.item())

        log["Eval loss"] = eval_loss.item()
        log["Eval perplexity"] = eval_perplexity

        return log

    def _save_latest_checkpoint(self, step):
        """Save latest checkpoint (for crash recovery)."""
        log = {}
        with timer("Checkpoint save time", log):
            save_checkpoint(
                self.model,
                self.optimizer,
                step,
                f"{self.config['checkpoint_dir']}/latest.pt",
                self.config,
            )
        # Note: checkpoint save time will be logged in the next iteration's train_step

    def _save_snapshot_checkpoint(self, step):
        """Save snapshot checkpoint at save_every intervals."""
        log = {}
        with timer("Snapshot checkpoint save time", log):
            save_checkpoint(
                self.model,
                self.optimizer,
                step,
                f"{self.config['checkpoint_dir']}/checkpoint_{step}.pt",
                self.config,
            )
        # Log snapshot save time
        self.wandb_run.log(log, step=step)
