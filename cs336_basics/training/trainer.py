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

    def __init__(self, model_class, config, tokens, valid_tokens, wandb_run=None):
        """Initialize the Trainer.

        Args:
            model_class: The model class to instantiate (e.g., TransformerLM)
            config: Configuration dictionary containing model_settings and other params
            tokens: Training token array
            valid_tokens: Validation token array
            wandb_run: Optional existing Weights & Biases run
        """

        self.model = model_class(**config["model_settings"]).to(config["device"])
        weight_decay = config.get("optimizer_weight_decay", 1e-2)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("optimizer_lr", 1e-3),
            betas=config.get("optimizer_betas", (0.9, 0.999)),
            eps=config.get("optimizer_eps", 1e-8),
            weight_decay=weight_decay,
        )

        # Allow caller (e.g., sweep) to provide an existing run.
        if wandb_run is None:
            wandb.login()
            self.wandb_run = wandb.init(
                entity=config["wandb_entity"],
                project=config["log_project"],
                name=config["run_name"],
                config=config,
            )
        else:
            self.wandb_run = wandb_run

        if config.get("checkpoint_add_timestamp", True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            config["checkpoint_dir"] = config["checkpoint_dir"] + "_" + timestamp
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        self.config = config
        self.tokens = tokens
        self.valid_tokens = valid_tokens
        self.best_eval_loss = float("inf")

    def train_eval_loop(self):
        """Main training loop that orchestrates training and evaluation."""
        save_every = self.config.get("save_every")
        save_best = self.config.get("save_best", False)
        save_final = self.config.get("save_final", False)

        for i in tqdm(range(self.config["num_iters"]), desc="Training Progress"):
            # Training step
            train_log = self._train_step(i)

            # Evaluation (if needed)
            if i % self.config["eval_every"] == 0:
                eval_log = self._eval_step(i)
                train_log.update(eval_log)

                eval_loss = eval_log.get("Eval loss")
                if eval_loss is not None and eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    train_log["Best eval loss"] = eval_loss
                    if save_best:
                        self._save_best_checkpoint(i)

            # Log all metrics
            self.wandb_run.log(train_log, step=i)

            # Save latest checkpoint every iteration (for crash recovery)
            self._save_latest_checkpoint(i)

            # Save snapshot checkpoint at save_every intervals
            if save_every is not None and i % save_every == 0:
                self._save_snapshot_checkpoint(i)

        if save_final:
            self._save_final_checkpoint(self.config["num_iters"] - 1)
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
            self.config.get("scheduler_lr_max", 1e-3),
            self.config.get("scheduler_lr_min", 1e-4),
            self.config.get("scheduler_warmup_iters", 0),
            self.config.get("scheduler_cos_iters", 0),
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

        eval_batches = self.config.get("eval_batches", 1)
        total_loss = 0.0

        with timer("Eval batch getting time", log):
            self.model.eval()
            eval_batches = max(1, int(eval_batches))
            eval_inputs = []
            eval_labels = []
            for _ in range(eval_batches):
                batch_inputs, batch_labels = get_batch(
                    self.valid_tokens,
                    self.config["batch_size"],
                    self.config["model_settings"]["context_length"],
                    self.config["device"],
                )
                eval_inputs.append(batch_inputs)
                eval_labels.append(batch_labels)

        with timer("Eval forward pass time", log):
            with torch.no_grad():
                for batch_inputs, batch_labels in zip(eval_inputs, eval_labels, strict=True):
                    eval_logits = self.model.forward(batch_inputs)
                    batch_loss = crossentropy(eval_logits, batch_labels)
                    total_loss += batch_loss.item()

        avg_loss = total_loss / eval_batches
        log["Eval loss"] = avg_loss
        log["Eval perplexity"] = math.exp(avg_loss)

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

    def _save_best_checkpoint(self, step):
        """Save best checkpoint based on eval loss."""
        log = {}
        with timer("Best checkpoint save time", log):
            save_checkpoint(
                self.model,
                self.optimizer,
                step,
                f"{self.config['checkpoint_dir']}/best.pt",
                self.config,
            )
        self.wandb_run.log(log, step=step)

    def _save_final_checkpoint(self, step):
        """Save final checkpoint after training."""
        log = {}
        with timer("Final checkpoint save time", log):
            save_checkpoint(
                self.model,
                self.optimizer,
                step,
                f"{self.config['checkpoint_dir']}/final.pt",
                self.config,
            )
        self.wandb_run.log(log, step=step)
