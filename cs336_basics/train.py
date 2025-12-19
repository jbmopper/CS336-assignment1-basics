from cs336_basics import (
    Tokenizer, 
    TransformerLM, 
    AdamW, 
    get_lr_cosine_schedule,
    gradient_clipping,
    get_batch,
    crossentropy,
    load_checkpoint,
    save_bpe,
    load_bpe,
    save_checkpoint,
    train_bpe,
)

from tqdm.auto import tqdm
import wandb

import os
import math
import numpy as np
import torch
import time

from datetime import datetime
from contextlib import contextmanager

# Setup (paths, device, config)

config = dict(
    # device is set below

    # tokenizer
    training_text = "../data/TinyStoriesV2-GPT4-train.txt",
    validation_text = "../data/TinyStoriesV2-GPT4-valid.txt",
    vocab_size = 10000, # also model parameter
    special_tokens = ["<|endoftext|>"], # takes list, can make file later...
    tokenizer_dir = "../tokenizers/tinystories/",
    train_file="../tokenized/tinystories_train.npy",
    valid_file="../tokenized/tinystories_valid.npy",

    # # model # small, ~10M parameters
    # d_model=256,
    # num_heads=4,
    # num_layers=4,
    # d_ff=1024,
    # context_length=125,
    # rope_theta=10000,


    # # Medium model (~50M params) - better results, slower
    # d_model = 512,
    # num_heads = 8,
    # num_layers = 6,
    # d_ff = 2048,
    # context_length = 512,
    # rope_theta=10000



    batch_size = 32, # "Memory scales with batch_size × context_length^2 × d_model"
    model_settings = dict(
        # Assignment settings, ~17M parameters
        d_model = 512,
        num_heads = 16,
        num_layers = 4,
        d_ff = 1344,
        # context_length = 256,
        context_length = 256,
        rope_theta = 10000,
    ),
    # will add vocab size...


    # optimizer (adamw)
    optimizer_use_defaults = True,
    optimizer_lr = 0.,
    optimizer_betas = (0., 0.),
    optimizer_eps = 1e-08,
    optimizer_weight_decay = 0.,

    # lr scheduler (cosine with warmup)
    lr_max = 1e-3, # typical 1e-4 to 1e-3, samller models -> larger lr
    lr_min = 1e-4, # 10-100x smaller than max
    warmup_iters = 100, # num_iters // 5 or 10
    cos_iters = 1000, # includes warmup

    # If loss explodes → lower max_lr. If loss plateaus early → try higher max_lr or longer training.

    # wandb/logging
    wandb_entity = "jbmopper-0", 
    log_project = "cs336-a1",
    model = "course rope-transformer",
    optimizer = "course adamw",
    learning_schedule = "course cosine anneal w/warmup",
    dataset = "Tinystories",
    loss_func = "cross-entropy",
    run_name = "refactor",


    # training loop
    num_iters = 10,
    checkpoint_dir = "../checkpoints/",

    # other parameters
    rand_seed = 0,
    gradient_clip = 1.0, # will set in training loop
    save_every = 200,
    eval_every = 20,

)
config["model_settings"]["vocab_size"] = config["vocab_size"]

def setDeviceAndSeeds(config):
    """Checks for devices, adds to config, and sets RNG seed values."""
    # device check
    if torch.backends.mps.is_available():
        config["device"] = "mps"
    elif torch.cuda.is_available():
        config["device"] = "cuda"
    else:
        config["device"] = "cpu"

    print(f"Using device {config['device']}.")

    # set seed
    # random.seed(config["rand_seed"])
    np.random.seed(config["rand_seed"])
    torch.manual_seed(config["rand_seed"])

    if config["device"] == "mps":
        torch.mps.manual_seed(config["rand_seed"])
    elif config["device"] == "cuda":
        # torch.backends.cuda.deterministic = True # old?
        torch.cuda.manual_seed_all(config["rand_seed"])
    
    print(f"Set seeds for torch/numpy and device {config['device']} to {config['rand_seed']}.")

def _get_or_create_tokenizer(config):
    """Load existing tokenizer or train a new one."""
    vocab_path = os.path.join(config["tokenizer_dir"], "vocab.json")
    merges_path = os.path.join(config["tokenizer_dir"], "merges.pkl")
    
    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        print("Loading existing tokenizer...")
        return load_bpe(config["tokenizer_dir"])
    else:
        print("Training new tokenizer...")
        vocab, merges = train_bpe(config["training_text"], config["vocab_size"], config["special_tokens"])
        save_bpe(config["tokenizer_dir"], vocab, merges)
        return vocab, merges

def _tokenize_and_save(config, text_file, output_file, file_type):
    """Tokenize a text file and save it if the output file doesn't exist."""
    if os.path.exists(output_file) and os.path.isfile(output_file):
        print(f"File {output_file} exists")
        return
    
    vocab, merges = _get_or_create_tokenizer(config)
    tokenizer = Tokenizer(vocab, merges)
    
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Tokenizing {file_type} file...")
    tokens = tokenizer.encode(text)
    
    # Save as contiguous array (memmap-compatible)
    arr = np.array(tokens, dtype=np.uint16)
    np.save(output_file, arr)
    
    print(f"Saved {len(tokens):,} tokens to {output_file}")

def getTokens(config) -> tuple[np.array, np.array]:
    """Get/generate memmapped token arrays"""
    # Tokenize training and validation files if needed
    _tokenize_and_save(config, config["training_text"], config["train_file"], "training")
    _tokenize_and_save(config, config["validation_text"], config["valid_file"], "validation")
    
    # Load tokenized data
    tokens = np.load(config["train_file"], mmap_mode='r')
    print(f"Training tokens loaded from {config['train_file']}, shape {tokens.shape}.")
    
    valid_tokens = np.load(config["valid_file"], mmap_mode='r')
    print(f"Validation tokens loaded from {config['valid_file']}, shape {valid_tokens.shape}.")
    
    return tokens, valid_tokens

@contextmanager
def timer(name, log_dict=None):
    """Context manager for timing code blocks.
    
    Usage:
        with timer("Forward pass", timings):
            out_logits = model.forward(inputs)
        # timings["Forward pass"] now contains the elapsed time
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if log_dict is not None:
        log_dict[name] = elapsed

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
            config=config
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
                self.config["device"]
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
                max_norm=float('inf')
            )
        log["Grad norm"] = grad_norm.item()
        
        with timer("Grad clip time", log):
            gradient_clipping(
                self.model.parameters(), 
                self.config["gradient_clip"]
            )
        
        # Optimizer step with LR schedule
        lr = get_lr_cosine_schedule(
            step,
            self.config["lr_max"],
            self.config["lr_min"],
            self.config["warmup_iters"],
            self.config["cos_iters"]
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
                self.config["device"]
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
                self.config
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
                self.config
            )
        # Log snapshot save time
        self.wandb_run.log(log, step=step)





def main():
    """Main training function."""
    setDeviceAndSeeds(config)
    tokens, valid_tokens = getTokens(config)
    trainer = Trainer(TransformerLM, config, tokens, valid_tokens)
    trainer.train_eval_loop()

if __name__ == "__main__":
    main()