"""Distributed communication primitives and tensor reductions for ORBIT."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from orbit.distributed.context import DistributedContext


def all_reduce_mean(
    tensor: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    """Computes all-reduce mean across all distributed ranks."""
    if not context.is_distributed or not dist.is_initialized():
        return tensor

    cloned = tensor.clone().detach()
    dist.all_reduce(cloned, op=dist.ReduceOp.SUM)
    cloned /= context.world_size
    return cloned


def all_gather_object(
    obj: Any,
    context: DistributedContext,
) -> list[Any]:
    """Gathers arbitrary Python objects (e.g. trajectories) across all ranks."""
    if not context.is_distributed or not dist.is_initialized():
        return [obj]

    gathered_objects: list[Any] = [None for _ in range(context.world_size)]
    dist.all_gather_object(gathered_objects, obj)
    return gathered_objects


def broadcast_object(
    obj: Any,
    src_rank: int = 0,
    context: DistributedContext | None = None,
) -> Any:
    """Broadcasts a Python object from src_rank to all other ranks."""
    if context is None or not context.is_distributed or not dist.is_initialized():
        return obj

    object_container = [obj] if context.rank == src_rank else [None]
    dist.broadcast_object_list(object_container, src=src_rank)
    return object_container[0]
