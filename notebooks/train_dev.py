import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.md("""
    # Training Development
    
    Run with: `marimo run notebooks/train_dev.py` or `marimo edit notebooks/train_dev.py`
    """)
    return (mo,)


@app.cell
def _():
    # Imports
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
    return (
        AdamW,
        Tokenizer,
        TransformerLM,
        crossentropy,
        datetime,
        get_batch,
        get_lr_cosine_schedule,
        gradient_clipping,
        load_bpe,
        load_checkpoint,
        math,
        np,
        os,
        save_bpe,
        save_checkpoint,
        time,
        torch,
        tqdm,
        train_bpe,
        wandb,
    )


@app.cell
def _(torch):
    # Setup (paths, device, config)

    config = dict(
        # device is set below

        # tokenizer
        training_text="../data/TinyStoriesV2-GPT4-train.txt",
        validation_text="../data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,  # also model parameter
        special_tokens=["<|endoftext|>"],  # takes list, can make file later...
        tokenizer_dir="../tokenizers/tinystories/",
        train_file="../tokenized/tinystories_train.npy",
        valid_file="../tokenized/tinystories_valid.npy",

        # model # small, ~10M parameters
        d_model=256,
        num_heads=4,
        num_layers=4,
        d_ff=1024,
        context_length=125,
        rope_theta=10000,

        # optimizer (adamw)
        optimizer_use_defaults=True,
        optimizer_lr=0.0,
        optimizer_betas=(0.0, 0.0),
        optimizer_eps=1e-08,
        optimizer_weight_decay=0.0,

        # lr scheduler (cosine with warmup)
        lr_max=1e-3,  # typical 1e-4 to 1e-3, smaller models -> larger lr
        lr_min=1e-4,  # 10-100x smaller than max
        warmup_iters=100,  # num_iters // 5 or 10
        cos_iters=1000,  # includes warmup

        # wandb/logging
        wandb_entity="jbmopper-0",
        log_project="cs336-a1",
        model="course rope-transformer",
        optimizer="course adamw",
        learning_schedule="course cosine anneal w/warmup",
        dataset="Tinystories",
        loss_func="cross-entropy",
        run_name="one",

        # training loop
        num_iters=1000,
        batch_size=32,
        checkpoint_dir="../checkpoints/",

        # other parameters
        rand_seed=0,
        gradient_clip=1.0,
        save_every=200,
        eval_every=20,
    )

    # device check
    if torch.backends.mps.is_available():
        config["device"] = "mps"
    elif torch.cuda.is_available():
        config["device"] = "cuda"
    else:
        config["device"] = "cpu"

    print(f"Using device {config['device']}.")
    return (config,)


@app.cell
def _(config, np, torch):
    # Set seeds
    np.random.seed(config["rand_seed"])
    torch.manual_seed(config["rand_seed"])

    if config["device"] == "mps":
        torch.mps.manual_seed(config["rand_seed"])
    elif config["device"] == "cuda":
        torch.cuda.manual_seed_all(config["rand_seed"])
    return


@app.cell
def _(Tokenizer, config, load_bpe, np, os, save_bpe, train_bpe):
    # Get training tokens
    if os.path.exists(config["train_file"]) and os.path.isfile(config["train_file"]):
        print(f"File {config['train_file']} exists")
    else:
        # check if tokenizer files are present
        vocab_path = os.path.join(config["tokenizer_dir"], "vocab.json")
        merges_path = os.path.join(config["tokenizer_dir"], "merges.pkl")
        if os.path.exists(vocab_path) and os.path.exists(merges_path):
            print("Loading existing tokenizer...")
            vocab, merges = load_bpe(config["tokenizer_dir"])
        else:
            print("Training new tokenizer...")
            vocab, merges = train_bpe(
                config["training_text"], config["vocab_size"], config["special_tokens"]
            )
            save_bpe(config["tokenizer_dir"], vocab, merges)

        tokenizer = Tokenizer(vocab, merges, special_tokens=config["special_tokens"])
        with open(config["training_text"], "r", encoding="utf-8") as f:
            text = f.read()

        print("Tokenizing training file...")
        tokens = tokenizer.encode(text)

        arr = np.array(tokens, dtype=np.uint16)
        np.save(config["train_file"], arr)

        print(f"Saved {len(tokens):,} tokens to {config['train_file']}")
    return


@app.cell
def _(Tokenizer, config, load_bpe, np, os, save_bpe, train_bpe):
    # Get validation tokens
    if os.path.exists(config["valid_file"]) and os.path.isfile(config["valid_file"]):
        print(f"File {config['valid_file']} exists")
    else:
        vocab_path = os.path.join(config["tokenizer_dir"], "vocab.json")
        merges_path = os.path.join(config["tokenizer_dir"], "merges.pkl")
        if os.path.exists(vocab_path) and os.path.exists(merges_path):
            print("Loading existing tokenizer...")
            vocab, merges = load_bpe(config["tokenizer_dir"])
        else:
            print("Training new tokenizer...")
            vocab, merges = train_bpe(
                config["training_text"], config["vocab_size"], config["special_tokens"]
            )
            save_bpe(config["tokenizer_dir"], vocab, merges)

        tokenizer = Tokenizer(vocab, merges, special_tokens=config["special_tokens"])
        with open(config["validation_text"], "r", encoding="utf-8") as f:
            text = f.read()

        print("Tokenizing validation file...")
        tokens = tokenizer.encode(text)

        arr = np.array(tokens, dtype=np.uint16)
        np.save(config["valid_file"], arr)

        print(f"Saved {len(tokens):,} tokens to {config['valid_file']}")
    return


@app.cell
def _(config, np):
    # Data preparation
    tokens = np.load(config["train_file"], mmap_mode="r")
    print(f"Training tokens loaded from {config['train_file']}, shape {tokens.shape}.")

    valid_tokens = np.load(config["valid_file"], mmap_mode="r")
    print(f"Validation tokens loaded from {config['valid_file']}, shape {valid_tokens.shape}.")
    return tokens, valid_tokens


@app.cell
def _(TransformerLM, config):
    # Model
    model = TransformerLM(
        config["vocab_size"],
        config["d_model"],
        config["num_heads"],
        config["num_layers"],
        config["d_ff"],
        config["context_length"],
        config["rope_theta"],
    ).to(config["device"])
    return (model,)


@app.cell
def _(AdamW, config, model):
    # Optimizer
    if config["optimizer_use_defaults"]:
        optimizer = AdamW(model.parameters())
    else:
        # TODO: optimizer call with parameters
        optimizer = None
    return (optimizer,)


@app.cell
def _(config, wandb):
    # Wandb
    wandb.login()
    run = wandb.init(
        entity=config["wandb_entity"],
        project=config["log_project"],
        name=config["run_name"],
        config=config,
    )
    return (run,)


@app.cell
def _(config, datetime, os):
    # Checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    checkpoint_dir = config["checkpoint_dir"] + "_" + timestamp
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir, timestamp


@app.cell
def _(
    checkpoint_dir,
    config,
    crossentropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    math,
    model,
    optimizer,
    run,
    save_checkpoint,
    time,
    tokens,
    torch,
    tqdm,
    valid_tokens,
    wandb,
):
    # Training loop
    with tqdm(total=config["num_iters"], desc="Training Progress") as pbar:
        for i in range(config["num_iters"]):
            # sync device
            if config["device"] == "mps":
                torch.mps.synchronize()
            elif config["device"] == "cuda":
                torch.cuda.synchronize()

            # get batch
            start = time.perf_counter()
            inputs, labels = get_batch(
                tokens,
                config["batch_size"],
                config["context_length"],
                config["device"],
            )
            elapsed = time.perf_counter() - start
            wandb.log({"Batch getting time": elapsed}, step=i)

            # forward pass
            start = time.perf_counter()
            model.train()
            out_logits = model.forward(inputs)
            elapsed = time.perf_counter() - start
            wandb.log({"Forward pass time": elapsed}, step=i)

            # loss calculation
            start = time.perf_counter()
            loss = crossentropy(out_logits, labels)
            elapsed = time.perf_counter() - start
            perplexity = math.exp(loss)
            wandb.log(
                {"Loss calc time": elapsed, "Loss": loss, "Perplexity": perplexity},
                step=i,
            )

            # backward pass
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            elapsed = time.perf_counter() - start
            wandb.log({"Backwards pass time": elapsed}, step=i)

            # gradient clipping
            start = time.perf_counter()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            )
            elapsed = time.perf_counter() - start
            wandb.log({"Grad calc time": elapsed, "Grad norm": grad_norm}, step=i)

            start = time.perf_counter()
            gradient_clipping(model.parameters(), config["gradient_clip"])
            elapsed = time.perf_counter() - start
            wandb.log({"Grad clip time": elapsed}, step=i)

            # optimize
            lr = get_lr_cosine_schedule(
                i,
                config["lr_max"],
                config["lr_min"],
                config["warmup_iters"],
                config["cos_iters"],
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
            save_checkpoint(model, optimizer, i, f"{checkpoint_dir}/latest.pt", config)
            elapsed = time.perf_counter() - start
            wandb.log({"Checkpoint save time": elapsed}, step=i)

            # eval
            if i % config["eval_every"] == 0:
                model.eval()
                eval_inputs, eval_labels = get_batch(
                    valid_tokens,
                    config["batch_size"],
                    config["context_length"],
                    config["device"],
                )
                with torch.no_grad():
                    eval_logits = model.forward(eval_inputs)
                    eval_loss = crossentropy(eval_logits, eval_labels)
                    eval_perplexity = math.exp(eval_loss)

                wandb.log(
                    {"Eval loss": eval_loss, "Eval perplexity": eval_perplexity}, step=i
                )

            # archive checkpoint
            if i % config["save_every"] == 0:
                save_checkpoint(
                    model, optimizer, i, f"{checkpoint_dir}/checkpoint_{i}.pt", config
                )

            pbar.update(1)

        run.finish()
    return


if __name__ == "__main__":
    app.run()
