"""Timing utilities for training workflows."""

from contextlib import contextmanager
import time


@contextmanager
def timer(name: str, log_dict: dict | None = None):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if log_dict is not None:
        log_dict[name] = elapsed
