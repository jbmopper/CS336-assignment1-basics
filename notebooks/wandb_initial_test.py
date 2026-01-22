import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.md("""
    # WandB Initial Test
    
    Simple test to verify wandb integration is working.
    """)
    return (mo,)


@app.cell
def _():
    import random
    import wandb

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged
        entity="jbmopper-0",
        # Set the wandb project where this run will be logged
        project="initial-test",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
    )

    # Simulate training
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset

        # Log metrics to wandb
        run.log({"acc": acc, "loss": loss})

    # Finish the run and upload any remaining data
    run.finish()
    return acc, epoch, epochs, loss, offset, random, run, wandb


if __name__ == "__main__":
    app.run()
