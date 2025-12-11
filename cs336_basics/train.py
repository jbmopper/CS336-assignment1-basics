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
from torch.profiler import profile, ProfilerActivity

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

    # Assignment settings, ~17M parameters
    d_model = 512,
    num_heads = 16,
    num_layers = 4,
    d_ff = 1344,
    # context_length = 256,
    context_length = 256,
    rope_theta = 10000,
    batch_size = 32, # "Memory scales with batch_size × context_length^2 × d_model"


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
    run_name = "creating loadable checkpoint for inference",


    # training loop
    num_iters = 1000,
    checkpoint_dir = "../checkpoints/",

    # other parameters
    rand_seed = 0,
    gradient_clip = 1.0, # will set in training loop
    save_every = 200,
    eval_every = 20,

)

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

def train(config, tokens, valid_tokens):
    """Set up model, optimizer, and run training loop"""
    # Model
    model = TransformerLM(
        config["vocab_size"],
        config["d_model"],
        config["num_heads"],
        config["num_layers"],
        config["d_ff"],
        config["context_length"],
        config["rope_theta"]
    ).to(config["device"]) # need better device info?  e.g. cuda:0?


    # Optimizer
    if config["optimizer_use_defaults"]:
        optimizer = AdamW(model.parameters()) # using defaults
    else:
        # TODO: optimizer call with parameters
        pass

    # Wandb 
    wandb.login()
    run = wandb.init(
        entity = config["wandb_entity"],
        project = config["log_project"],
        name = config["run_name"],
        config = config
    ) #...

    # checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    config["checkpoint_dir"] = config["checkpoint_dir"] + "_" + timestamp
    os.makedirs(config["checkpoint_dir"], exist_ok=True)


   # Training loop

    # with tqdm(total=config["num_iters"], desc="Training Progress") as pbar:
    #    for i in range(config["num_iters"]):
    for i in tqdm(range(config["num_iters"]), desc="Training Progress"):

        # set device
        if config["device"] == "mps":
            torch.mps.synchronize()
        elif config["device"] == "cuda":
            torch.cuda.synchronize()


        if i % config["eval_every"] == 0:

            # with torch.mps.profiler.profile(mode="interval,event", wait_until_completed=False): 
        # train
            # get batch
            start = time.perf_counter()
            inputs, labels = get_batch(
                tokens, 
                config["batch_size"], 
                config["context_length"],
                config["device"]
            )
            elapsed = time.perf_counter() - start
            wandb.log({"Batch getting time": elapsed}, step=i)

            # forward pass
            start = time.perf_counter() 
            model.train()
            out_logits = model.forward(inputs) 
            elapsed = time.perf_counter() - start
            wandb.log({"Forward pass time": elapsed}, step=i)

            # loss caclulation
            start = time.perf_counter() 
            loss = crossentropy(out_logits, labels)
            elapsed = time.perf_counter() - start 
            perplexity = math.exp(loss)
            wandb.log({"Loss calc time": elapsed,
                "Loss": loss, "Perplexity": perplexity}, step=i)

            # backward pass
            start = time.perf_counter()  
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            elapsed = time.perf_counter() - start 
            wandb.log({"Backwards pass time": elapsed}, step=i)

            # gradient clipping
            start = time.perf_counter() 
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            # print(f"Grad norm: {grad_norm:.2f}")
            elapsed = time.perf_counter() - start
            wandb.log({"Grad calc time": elapsed, "Grad norm": grad_norm}, step=i)

            start = time.perf_counter() 
            gradient_clipping(model.parameters(), config["gradient_clip"])
            elapsed = time.perf_counter() - start
            wandb.log({"Grad clip time": elapsed}, step=i) 

            # optimize
            lr = get_lr_cosine_schedule(i,
                config["lr_max"],
                config["lr_min"],
                config["warmup_iters"],
                config["cos_iters"]
            )
            wandb.log({"LR": lr}, step=i)

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            start = time.perf_counter()  
            optimizer.step() 
            elapsed = time.perf_counter() - start
            wandb.log({"Optimizer step time": elapsed}, step=i)
            

            # most recent checkpoint
            start = time.perf_counter() 
            save_checkpoint(model, optimizer, i, f"{config['checkpoint_dir']}/latest.pt", config)
            elapsed = time.perf_counter() - start
            wandb.log({"Checkpoint save time": elapsed}, step=i) # lol 

        # eval
            model.eval() 
            eval_inputs, eval_labels = get_batch(
                valid_tokens,
                config["batch_size"], 
                config["context_length"],
                config["device"]
            )
            with torch.no_grad():
                eval_logits = model.forward(eval_inputs)
                eval_loss = crossentropy(eval_logits, eval_labels)
                eval_perplexity = math.exp(eval_loss)
            
            wandb.log({"Eval loss": eval_loss, "Eval perplexity": eval_perplexity}, step=i)

        else:

            # get batch
            start = time.perf_counter()
            inputs, labels = get_batch(
                tokens, 
                config["batch_size"], 
                config["context_length"],
                config["device"]
            )
            elapsed = time.perf_counter() - start
            wandb.log({"Batch getting time": elapsed}, step=i)

            # forward pass
            start = time.perf_counter() 
            model.train()
            out_logits = model.forward(inputs) 
            elapsed = time.perf_counter() - start
            wandb.log({"Forward pass time": elapsed}, step=i)

            # loss caclulation
            start = time.perf_counter() 
            loss = crossentropy(out_logits, labels)
            elapsed = time.perf_counter() - start 
            perplexity = math.exp(loss)
            wandb.log({"Loss calc time": elapsed,
                "Loss": loss, "Perplexity": perplexity}, step=i)

            # backward pass
            start = time.perf_counter()  
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            elapsed = time.perf_counter() - start 
            wandb.log({"Backwards pass time": elapsed}, step=i)

            # gradient clipping
            start = time.perf_counter() 
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            # print(f"Grad norm: {grad_norm:.2f}")
            elapsed = time.perf_counter() - start
            wandb.log({"Grad calc time": elapsed, "Grad norm": grad_norm}, step=i)

            start = time.perf_counter() 
            gradient_clipping(model.parameters(), config["gradient_clip"])
            elapsed = time.perf_counter() - start
            wandb.log({"Grad clip time": elapsed}, step=i) 

            # optimize
            lr = get_lr_cosine_schedule(i,
                config["lr_max"],
                config["lr_min"],
                config["warmup_iters"],
                config["cos_iters"]
            )
            wandb.log({"LR": lr}, step=i)

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            start = time.perf_counter()  
            optimizer.step() 
            elapsed = time.perf_counter() - start
            wandb.log({"Optimizer step time": elapsed}, step=i)
            

            # most recent checkpoint
            start = time.perf_counter() 
            save_checkpoint(model, optimizer, i, f"{config['checkpoint_dir']}/latest.pt", config)
            elapsed = time.perf_counter() - start
            wandb.log({"Checkpoint save time": elapsed}, step=i) # lol




        # archive checkpoint
        if i % config["save_every"] == 0: # checkpoint every 4 iters?  
            save_checkpoint(model, optimizer, i, f"{config['checkpoint_dir']}/checkpoint_{i}.pt", config)

        #    pbar.update(1)
        
    run.finish() 


def main():
    """Main training function."""
    setDeviceAndSeeds(config)
    tokens, valid_tokens = getTokens(config)
    train(config, tokens, valid_tokens)

if __name__ == "__main__":
    main()