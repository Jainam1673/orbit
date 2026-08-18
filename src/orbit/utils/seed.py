"""Reproducibility utilities for deterministic random state management in ORBIT."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Sets the random seed across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed to initialize PRNGs.
        deterministic: If True, configures PyTorch and cuDNN to use deterministic
            algorithms where available, which may impact performance but guarantees
            reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Note: Some operations might not have deterministic implementations in CUDA.
        # Set environment variable for cuBLAS determinism if needed.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, ValueError, AttributeError):
            # Not all PyTorch backends support deterministic algorithms
            pass
        if torch.cuda.is_available() and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
