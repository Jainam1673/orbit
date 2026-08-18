"""Distributed training context, rank initialization, and multi-worker seeding."""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from orbit.utils.seed import set_seed


@dataclass(frozen=True)
class DistributedContext:
    """Encapsulates process rank, cluster size, and device placement."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cpu"
    is_distributed: bool = False

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def init_distributed(
    backend: str | None = None,
    timeout_seconds: int = 60,
) -> DistributedContext:
    """Initializes torch.distributed if launched via torchrun or multi-process cluster."""
    # Check if distributed environment variables exist
    rank_str = os.environ.get("RANK")
    world_size_str = os.environ.get("WORLD_SIZE")
    local_rank_str = os.environ.get("LOCAL_RANK", "0")

    if rank_str is not None and world_size_str is not None:
        rank = int(rank_str)
        world_size = int(world_size_str)
        local_rank = int(local_rank_str)

        if not dist.is_initialized():
            if backend is None:
                backend = "nccl" if torch.cuda.is_available() else "gloo"

            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.default_pg_timeout,
            )

        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        return DistributedContext(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            is_distributed=world_size > 1,
        )

    # Single process fallback
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return DistributedContext(
        rank=0,
        world_size=1,
        local_rank=0,
        device=device,
        is_distributed=False,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    """Shuts down process group if active."""
    if context.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def set_distributed_seed(
    base_seed: int,
    rank: int,
    deterministic: bool = True,
) -> int:
    """Applies rank-offset deterministic seeding to ensure independent rollouts."""
    effective_seed = base_seed * 1000 + rank
    set_seed(effective_seed, deterministic=deterministic)
    return effective_seed
