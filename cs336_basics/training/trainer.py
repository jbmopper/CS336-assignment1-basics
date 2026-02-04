"""Trainer utilities for transformer training loops."""

from __future__ import annotations

import math
import os
import time
from datetime import datetime

import torch
import wandb
from tqdm.auto import tqdm

from ..implementations import crossentropy, get_batch, gradient_clipping, save_checkpoint
from ..optimizer import AdamW, get_lr_cosine_schedule
from .timer import timer


# Precision mode to dtype mapping
PRECISION_DTYPES = {
    "fp32": None,  # No autocast
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class Trainer:
    """Manages model training, optimization, and logging.
    
    Expected config dictionary structure:
    
    Required keys:
        model_settings (dict): Kwargs passed to model_class.__init__. Must include:
            - vocab_size (int): Vocabulary size
            - d_model (int): Model dimension
            - num_heads (int): Number of attention heads
            - num_layers (int): Number of transformer blocks
            - d_ff (int): Feed-forward hidden dimension
            - context_length (int): Maximum sequence length
            - rope_theta (float): RoPE theta parameter
        device (str): Device to train on ("cpu", "cuda", "mps")
        batch_size (int): Training batch size
        num_iters (int): Total training iterations
        eval_every (int): Evaluate every N iterations
        gradient_clip (float): Max gradient L2 norm for clipping
        checkpoint_dir (str): Directory to save checkpoints
    
    Optional keys (with defaults):
        # W&B logging (all optional - if wandb_entity not provided, logging is disabled)
        wandb_entity (str): W&B entity/username (default: None, disables W&B)
        log_project (str): W&B project name (required if wandb_entity set)
        run_name (str): W&B run name (required if wandb_entity set)
        
        # Mixed Precision
        precision (str): Training precision - "fp32", "fp16", or "bf16" (default: "fp32")
            - fp32: Full precision, no autocast
            - fp16: Half precision with GradScaler (requires loss scaling)
            - bf16: BFloat16, recommended for 4090/A100+ (no scaling needed)
        
        # Torch Compile (CUDA only)
        compile_model (bool): Whether to torch.compile the model (default: False)
        compile_backend (str): torch.compile backend (default: "aot_eager")
        
        # Optimizer
        optimizer_lr (float): Learning rate (default: 1e-3)
        optimizer_betas (tuple): Adam betas (default: (0.9, 0.999))
        optimizer_eps (float): Adam epsilon (default: 1e-8)
        optimizer_weight_decay (float): Weight decay (default: 1e-2)
        
        # LR Schedule (cosine with warmup)
        scheduler_lr_max (float): Max LR after warmup (default: 1e-3)
        scheduler_lr_min (float): Min LR at end of cosine (default: 1e-4)
        scheduler_warmup_iters (int): Warmup iterations (default: 0)
        scheduler_cos_iters (int): Cosine decay iterations (default: 0)
        
        # Checkpointing
        checkpoint_add_timestamp (bool): Append timestamp to checkpoint_dir (default: True)
        save_every (int): Save snapshot every N iters (default: None)
        save_best (bool): Save best checkpoint by eval loss (default: False)
        save_final (bool): Save final checkpoint (default: False)
        
        # Evaluation
        eval_batches (int): Number of batches for evaluation (default: 1)
    
    Memory considerations:
        Peak training memory ≈ weights + activations + gradients
        - Activations scale as O(batch_size × context_length × d_model × num_layers)
        - Attention matrices scale as O(batch_size × num_heads × context_length²)
        - For 24GB RAM, typical safe configs: batch_size≤32, context_length≤512
    """

    def __init__(self, model_class, config, tokens, valid_tokens, wandb_run=None):
        """Initialize the Trainer.

        Args:
            model_class: The model class to instantiate (e.g., TransformerLM)
            config: Configuration dictionary (see class docstring for required/optional keys)
            tokens: Training token array (numpy array of token IDs)
            valid_tokens: Validation token array (numpy array of token IDs)
            wandb_run: Optional existing Weights & Biases run (for sweep integration)
        """

        self.model = model_class(**config["model_settings"]).to(config["device"])
        if config["device"] == "cuda":
            torch.set_float32_matmul_precision('high')
        
        if config.get("compile_model"):
            if config["device"] == "cuda":
                backend = config.get("compile_backend", "aot_eager")
                self.model = torch.compile(self.model, backend=backend)
                print(f"Compiled model with torch.compile (backend={backend})")
            else:
                print("Skipping torch.compile: CUDA device not available")
        weight_decay = config.get("optimizer_weight_decay", 1e-2)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("optimizer_lr", 1e-3),
            betas=config.get("optimizer_betas", (0.9, 0.999)),
            eps=config.get("optimizer_eps", 1e-8),
            weight_decay=weight_decay,
        )

        # Mixed precision setup
        self.precision = config.get("precision", "fp32")
        self.amp_dtype = PRECISION_DTYPES.get(self.precision)
        self.use_amp = self.amp_dtype is not None and config["device"] == "cuda"
        
        # GradScaler only needed for FP16 (BF16 doesn't need loss scaling)
        if self.use_amp and self.precision == "fp16":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # W&B logging is optional - only initialize if wandb_entity is provided
        self.use_wandb = config.get("wandb_entity") is not None
        if wandb_run is not None:
            self.wandb_run = wandb_run
            self.use_wandb = True
        elif self.use_wandb:
            wandb.login()
            self.wandb_run = wandb.init(
                entity=config["wandb_entity"],
                project=config["log_project"],
                name=config["run_name"],
                config=config,
            )
        else:
            self.wandb_run = None

        if config.get("checkpoint_add_timestamp", True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            config["checkpoint_dir"] = config["checkpoint_dir"] + "_" + timestamp
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        self.config = config
        self.tokens = tokens
        self.valid_tokens = valid_tokens
        self.best_eval_loss = float("inf")
        self.start_time = time.perf_counter()
        
        # Determine device for timers (sync only if profiling is requested)
        # Default to False to avoid training slowdown
        self.sync_timers = config.get("profile_timers", False)
        self.timer_device = config["device"] if self.sync_timers else None

    def _log(self, metrics, step):
        """Log metrics to W&B if enabled."""
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def _finish(self):
        """Finish W&B run if enabled."""
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()

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

                eval_loss = eval_log.get("Eval Loss")
                if eval_loss is not None and eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    train_log["Eval Best loss"] = eval_loss
                    if save_best:
                        self._save_best_checkpoint(i)

            # Save latest checkpoint every iteration (for crash recovery)
            checkpoint_log = self._save_latest_checkpoint(i)
            train_log.update(checkpoint_log)

            # Log all metrics
            self._log(train_log, step=i)

            # Save snapshot checkpoint at save_every intervals
            if save_every is not None and i % save_every == 0:
                self._save_snapshot_checkpoint(i)

        if save_final:
            self._save_final_checkpoint(self.config["num_iters"] - 1)
        self._finish()

    def _train_step(self, step):
        """Perform a single training step. Returns dict of metrics to log."""
        log = {}
        step_start = time.perf_counter()

        # Get batch
        with torch.profiler.record_function("## BATCH_GET ##"):
            with timer("Time/Batch getting", log, device=self.timer_device):
                inputs, labels = get_batch(
                    self.tokens,
                    self.config["batch_size"],
                    self.config["model_settings"]["context_length"],
                    self.config["device"],
                )

        # Forward pass (with optional autocast for mixed precision)
        with torch.profiler.record_function("## FORWARD ##"):
            with timer("Time/Forward", log, device=self.timer_device):
                self.model.train()
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        out_logits = self.model.forward(inputs)
                else:
                    out_logits = self.model.forward(inputs)

        # Loss calculation (with optional autocast for mixed precision)
        with torch.profiler.record_function("## LOSS_CALC ##"):
            with timer("Time/Loss calc", log, device=self.timer_device):
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        loss = crossentropy(out_logits, labels)
                else:
                    loss = crossentropy(out_logits, labels)
                perplexity = math.exp(loss.item())

        log["Loss"] = loss.item()
        log["Perplexity"] = perplexity

        # Backward pass (with GradScaler for FP16)
        with torch.profiler.record_function("## BACKWARD ##"):
            with timer("Time/Backward", log, device=self.timer_device):
                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

        # Gradient clipping - unscale first if using GradScaler
        with torch.profiler.record_function("## GRAD_NORM ##"):
            with timer("Time/Grad norm calc", log, device=self.timer_device):
                if self.scaler is not None:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float("inf"),
                )
        
        log["Grad/Norm (unclipped)"] = grad_norm.item()
        log["Grad/Norm (clipped)"] = min(grad_norm.item(), self.config["gradient_clip"])

        with torch.profiler.record_function("## GRAD_CLIP ##"):
            with timer("Time/Grad clip", log, device=self.timer_device):
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

        with torch.profiler.record_function("## OPTIMIZER_STEP ##"):
            with timer("Time/Optimizer step", log, device=self.timer_device):
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

        # Efficient timing: one synchronization at the end of the step
        # Note: We now sync inside timer() blocks for accurate profiling, 
        # but this final sync ensures total_step_time is correct even if timers were disabled
        if self.config["device"] == "mps":
            torch.mps.synchronize()
        elif self.config["device"] == "cuda":
            torch.cuda.synchronize()
            
        total_step_time = time.perf_counter() - step_start
        log["Time/Total step"] = total_step_time
        
        tokens_per_step = self.config["batch_size"] * self.config["model_settings"]["context_length"]
        log["Throughput/Tokens per sec"] = tokens_per_step / total_step_time
        log["Time/Cumulative (min)"] = (time.perf_counter() - self.start_time) / 60.0

        if self.config["device"] == "cuda":
            log["Memory/Max allocated (GB)"] = torch.cuda.max_memory_allocated() / 1e9
        elif self.config["device"] == "mps":
            try:
                log["Memory/Current allocated (GB)"] = torch.mps.current_allocated_memory() / 1e9
            except (AttributeError, RuntimeError):
                pass

        return log

    def _eval_step(self, step):
        """Perform evaluation on validation set. Returns dict of metrics to log."""
        log = {}

        eval_batches = self.config.get("eval_batches", 1)
        total_loss = 0.0

        with timer("Time/Eval batch getting", log, device=self.timer_device):
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

        with timer("Time/Eval forward pass", log, device=self.timer_device):
            with torch.no_grad():
                for batch_inputs, batch_labels in zip(eval_inputs, eval_labels, strict=True):
                    if self.use_amp:
                        with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                            eval_logits = self.model.forward(batch_inputs)
                            batch_loss = crossentropy(eval_logits, batch_labels)
                    else:
                        eval_logits = self.model.forward(batch_inputs)
                        batch_loss = crossentropy(eval_logits, batch_labels)
                    total_loss += batch_loss.item()

        avg_loss = total_loss / eval_batches
        log["Eval Loss"] = avg_loss
        log["Eval Perplexity"] = math.exp(avg_loss)

        return log

    def _save_latest_checkpoint(self, step):
        """Save latest checkpoint (for crash recovery)."""
        log = {}
        with timer("Time/Checkpoint save (latest)", log, device=self.timer_device):
            save_checkpoint(
                self.model,
                self.optimizer,
                step,
                f"{self.config['checkpoint_dir']}/latest.pt",
                self.config,
            )
        return log

    def _save_snapshot_checkpoint(self, step):
        """Save snapshot checkpoint at save_every intervals."""
        log = {}
        with timer("Time/Checkpoint save (snapshot)", log, device=self.timer_device):
            save_checkpoint(
                self.model,
                self.optimizer,
                step,
                f"{self.config['checkpoint_dir']}/checkpoint_{step}.pt",
                self.config,
            )
        # Log snapshot save time
        self._log(log, step=step)

    def _save_best_checkpoint(self, step):
        """Save best checkpoint based on eval loss."""
        log = {}
        with timer("Time/Checkpoint save (best)", log, device=self.timer_device):
            save_checkpoint(
                self.model,
                self.optimizer,
                step,
                f"{self.config['checkpoint_dir']}/best.pt",
                self.config,
            )
        self._log(log, step=step)

    def _save_final_checkpoint(self, step):
        """Save final checkpoint after training."""
        log = {}
        with timer("Time/Checkpoint save (final)", log, device=self.timer_device):
            save_checkpoint(
                self.model,
                self.optimizer,
                step,
                f"{self.config['checkpoint_dir']}/final.pt",
                self.config,
            )
        self._log(log, step=step)
