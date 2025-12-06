"""CS336 Basics - Transformer implementation from scratch."""

import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from cs336_basics.nn import (
    Linear, Embedding, RMSNorm, SwiGLU,
    Rope, Multihead, MultiheadRope,
    softmax, silu, scaled_dot_product_attention
)
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.implementations import (
    TransformerBlock, TransformerLM, crossentropy,
    get_batch, gradient_clipping, save_checkpoint, load_checkpoint
)
from cs336_basics.bpe import train_bpe, Tokenizer, save_bpe, load_bpe

__all__ = [
    # Neural network modules
    'Linear', 'Embedding', 'RMSNorm', 'SwiGLU',
    'Rope', 'Multihead', 'MultiheadRope',
    'softmax', 'silu', 'scaled_dot_product_attention',
    # Transformer
    'TransformerBlock', 'TransformerLM', 'crossentropy',
    # Training utilities
    'get_batch', 'gradient_clipping', 'save_checkpoint', 'load_checkpoint',
    'AdamW', 'get_lr_cosine_schedule',
    # Tokenizer
    'train_bpe', 'Tokenizer', 'save_bpe', 'load_bpe'
]
