import os

import torch
from torch import nn
from torch.optim import Adam

from orbit.agents.base import AgentConfig
from orbit.agents.reasoning import ReasoningAgent
from orbit.algorithms.orbit import OrbitAlgorithm
from orbit.config import ExperimentConfig
from orbit.curriculum.strategies import (
    AdaptiveFrontierCurriculum,
    FixedDistributionCurriculum,
)
from orbit.environments.math.environment import MathEnvironment
from orbit.environments.math.generator import MathTaskGenerator
from orbit.models.mock import MockModelClient
from orbit.training.runner import run_experiment
from orbit.training.trainer import TrainingLoop


def test_training_loop_end_to_end():
    client = MockModelClient(default_response="\\boxed{42}")
    agent = ReasoningAgent(AgentConfig(), model_client=client)
    env = MathEnvironment()
    curriculum = AdaptiveFrontierCurriculum(seed=42)
    eval_gen = MathTaskGenerator(seed=999)
    eval_tasks = [eval_gen.generate_task(difficulty=0.2), eval_gen.generate_task(difficulty=0.5)]

    loop = TrainingLoop(
        agent=agent,
        env=env,
        curriculum=curriculum,
        eval_tasks=eval_tasks,
        run_id="test_integration_loop",
    )

    summary = loop.run(num_steps=5, eval_interval=2, base_seed=42)

    assert summary["total_steps"] == 5
    assert len(summary["training_history"]) == 5
    assert len(summary["eval_history"]) >= 2
    assert "overall_success_rate" in summary


def test_baseline_vs_orbit_adaptive_curriculum_comparison():
    client = MockModelClient(default_response="\\boxed{42}")
    agent = ReasoningAgent(AgentConfig(), model_client=client)
    env = MathEnvironment()

    # 1. Baseline: Fixed distribution
    fixed_curr = FixedDistributionCurriculum(seed=100)
    loop_fixed = TrainingLoop(agent=agent, env=env, curriculum=fixed_curr)
    summary_fixed = loop_fixed.run(num_steps=6, eval_interval=3)

    # 2. Treatment: Adaptive Frontier
    adaptive_curr = AdaptiveFrontierCurriculum(seed=100)
    loop_adaptive = TrainingLoop(agent=agent, env=env, curriculum=adaptive_curr)
    summary_adaptive = loop_adaptive.run(num_steps=6, eval_interval=3)

    assert summary_fixed["total_steps"] == 6
    assert summary_adaptive["total_steps"] == 6

    # Verify both tracked step telemetry properly
    assert all("task_difficulty" in s for s in summary_fixed["training_history"])
    assert all("frontier_difficulty" in s for s in summary_adaptive["training_history"])


def test_run_experiment_manifest_persistence(tmp_path):
    cfg = ExperimentConfig(
        name="test_experiment",
        seed=1337,
        output_dir=str(tmp_path / "experiments"),
    )
    cfg.model.provider = "mock"

    result = run_experiment(config=cfg, num_steps=4, eval_interval=2)

    assert result.experiment_id.startswith("exp_test_experiment_")
    assert result.duration_sec >= 0.0
    assert os.path.exists(result.output_dir)

    manifest_file = os.path.join(result.output_dir, "manifest.json")
    assert os.path.isfile(manifest_file)

    manifest_dict = result.to_dict()
    assert manifest_dict["config"]["seed"] == 1337
    assert manifest_dict["provenance"]["seed"] == 1337
    assert "summary" in manifest_dict


def test_orbit_algorithm_step():
    model = nn.Linear(4, 2)
    optimizer = Adam(model.parameters(), lr=1e-2)
    algo = OrbitAlgorithm(model=model, optimizer=optimizer)

    # Sample task from curriculum
    task = algo.sample_task()
    assert task is not None

    # Forward mock training step
    x = torch.randn(4, 4)
    logits = model(x)
    logprobs = torch.log_softmax(logits, dim=-1)[:, 0:1]
    old_logprobs = logprobs.detach().clone()
    ref_logprobs = old_logprobs.clone()
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.ones(4, 1)

    step_result = algo.train_step(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        rewards=rewards,
        group_size=4,
        mask=mask,
    )

    assert step_result.frontier_difficulty >= 0.1
    assert step_result.mean_reward == 0.5
    assert step_result.loss_output.loss is not None
