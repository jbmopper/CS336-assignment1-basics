import torch
from cs336_basics import TransformerLM
from typing import Any
from pathlib import Path


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Explicitly disable weights_only to support checkpoints that also store config.
    return torch.load(path, map_location="cpu", weights_only=False)

def _build_model_from_checkpoint(checkpoint)