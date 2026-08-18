"""Experiment provenance and audit metadata extraction for ORBIT."""

from __future__ import annotations

import platform
import subprocess
from typing import Any

import torch

from orbit.data.trajectory import Provenance


def get_git_info() -> tuple[str, bool]:
    """Retrieves current git commit hash and dirty status."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        is_dirty = len(status) > 0
        return commit, is_dirty
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown", False


def get_hardware_info() -> dict[str, Any]:
    """Collects system, OS, and GPU hardware metrics for audit logs."""
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def capture_provenance(
    model_version: str = "unknown",
    env_version: str = "unknown",
    seed: int = 0,
) -> Provenance:
    """Captures full experiment provenance snapshot."""
    commit, is_dirty = get_git_info()
    hardware = get_hardware_info()
    return Provenance(
        git_commit=commit,
        git_dirty=is_dirty,
        model_version=model_version,
        env_version=env_version,
        seed=seed,
        hardware_info=hardware,
    )
