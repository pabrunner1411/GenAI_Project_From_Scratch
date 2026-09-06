"""Small helpers for reporting and releasing GPU memory.

Shared by the unload() methods on the local model wrappers and by the
notebook's explicit "free GPU memory" cell, so the user can see the
effect of unloading rather than have it happen silently.
"""

import gc


def describe_gpu_memory() -> str:
    """One-line summary of current CUDA memory usage, or a CPU note."""
    import torch

    if not torch.cuda.is_available():
        return "Running on CPU, no GPU memory to report."

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved."


def release_gpu_memory() -> None:
    """Forces garbage collection and releases cached CUDA memory back to the driver."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
