"""Optimizer implementations and learning rate schedules."""

import math
import torch
from collections.abc import Callable
from typing import Optional

__all__ = ['AdamW', 'get_lr_cosine_schedule']


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer with decoupled weight decay."""
    
    def __init__(
        self,
        params,
        lr: float = 1e-03,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 1e-02,
    ) -> None:
        defaults = dict(betas=betas, eps=eps, lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                betas = group["betas"]
                lr = group["lr"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                t += 1
                state["t"] = t
                m.mul_(betas[0]).add_((1 - betas[0]) * p.grad.data)
                state["m"] = m
                v.mul_(betas[1]).add_((1 - betas[1]) * (p.grad.data ** 2))
                state["v"] = v

                alpha_t = lr * (((1 - betas[1] ** t) ** 0.5) / (1 - betas[0] ** t))
                p.data -= alpha_t * (m / (torch.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning rate schedule with linear warmup.
    
    Args:
        it: Current iteration
        max_learning_rate: Maximum (peak) learning rate
        min_learning_rate: Minimum (final) learning rate
        warmup_iters: Number of warmup iterations
        cosine_cycle_iters: Total iterations for cosine annealing
    
    Returns:
        Learning rate for the given iteration
    """
    if it < warmup_iters:
        lr = (max_learning_rate / warmup_iters) * it
    elif warmup_iters <= it <= cosine_cycle_iters:
        cos_it = it - warmup_iters
        cos_tot_it = cosine_cycle_iters - warmup_iters
        lr = (
            min_learning_rate
            + 0.5 * (max_learning_rate - min_learning_rate)
            * (1 + math.cos((cos_it * math.pi) / cos_tot_it))
        )
    else:
        lr = min_learning_rate
    return lr

