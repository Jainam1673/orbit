"""Distributed model wrapping abstractions for DDP and FSDP."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from orbit.distributed.context import DistributedContext


def wrap_model_distributed(
    model: nn.Module,
    context: DistributedContext,
    use_fsdp: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Wraps PyTorch model in DDP or FSDP if running in a distributed environment."""
    if not context.is_distributed:
        return model

    if use_fsdp:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            return FSDP(model, **kwargs)
        except ImportError:
            pass

    device_ids = None
    if torch.cuda.is_available() and "cuda" in context.device:
        device_ids = [context.local_rank]

    return DDP(model, device_ids=device_ids, **kwargs)
