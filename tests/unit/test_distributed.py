import torch
from torch import nn

from orbit.agents.base import AgentConfig
from orbit.agents.reasoning import ReasoningAgent
from orbit.distributed import (
    DistributedContext,
    DistributedWorkerPool,
    all_gather_object,
    all_reduce_mean,
    broadcast_object,
    init_distributed,
    set_distributed_seed,
    wrap_model_distributed,
)
from orbit.environments.math.environment import MathEnvironment
from orbit.environments.math.generator import MathTaskGenerator
from orbit.models.mock import MockModelClient


def test_distributed_context_and_seeding():
    ctx = init_distributed()
    assert ctx.rank == 0
    assert ctx.world_size == 1
    assert ctx.is_main_process is True
    assert ctx.is_distributed is False

    seed0 = set_distributed_seed(base_seed=42, rank=0)
    seed1 = set_distributed_seed(base_seed=42, rank=1)
    assert seed0 == 42000
    assert seed1 == 42001
    assert seed0 != seed1


def test_distributed_communication_primitives_fallback():
    ctx = DistributedContext(rank=0, world_size=1, is_distributed=False)

    tensor = torch.tensor([1.0, 2.0, 3.0])
    reduced = all_reduce_mean(tensor, ctx)
    assert torch.equal(reduced, tensor)

    obj = {"test": 123}
    gathered = all_gather_object(obj, ctx)
    assert gathered == [obj]

    broadcasted = broadcast_object(obj, src_rank=0, context=ctx)
    assert broadcasted == obj


def test_wrap_model_distributed():
    ctx = DistributedContext(rank=0, world_size=1, is_distributed=False)
    model = nn.Linear(5, 2)

    wrapped = wrap_model_distributed(model, ctx)
    assert wrapped is model


def test_distributed_worker_pool_parallel_rollouts():
    client = MockModelClient(default_response="\\boxed{42}")
    agent = ReasoningAgent(AgentConfig(), model_client=client)
    env = MathEnvironment()

    gen = MathTaskGenerator(seed=42)
    tasks = [gen.generate_task(difficulty=d) for d in [0.1, 0.3, 0.5, 0.7]]

    pool = DistributedWorkerPool(num_workers=2, run_id="test_dist_pool")
    trajectories, stats = pool.collect_parallel_rollouts(
        agent=agent,
        env=env,
        tasks=tasks,
        base_seed=100,
    )

    assert len(trajectories) == 4
    assert stats.total_episodes == 4
    assert stats.mean_reward is not None
    assert all(t.episode_id.startswith("dist_ep_") for t in trajectories)
