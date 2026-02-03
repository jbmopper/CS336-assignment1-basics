"""Timing utilities for training workflows."""

from contextlib import contextmanager
import time


import torch

@contextmanager
def timer(name: str, log_dict: dict | None = None, device: str | None = None):
    """Context manager for timing code blocks.
    
    Args:
        name: Name of the timer (key for log_dict)
        log_dict: Dictionary to log the elapsed time to
        device: Device string ("cuda", "mps", "cpu") to synchronize. 
                If None, no synchronization is performed (CPU timing only).
    """
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
        
    start = time.perf_counter()
    yield
    
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
        
    elapsed = time.perf_counter() - start
    if log_dict is not None:
        log_dict[name] = elapsed
