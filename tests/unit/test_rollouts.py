import pytest

from orbit.agents.base import AgentConfig
from orbit.agents.reasoning import ReasoningAgent
from orbit.environments.base import TaskSpec
from orbit.environments.math.environment import MathEnvironment
from orbit.models.mock import MockModelClient
from orbit.rollouts.collector import RolloutCollector


def test_rollout_collector_single_episode_success():
    task = TaskSpec(
        task_id="task_easy_1",
        family="math_algebra",
        prompt="Solve 2x = 6",
        ground_truth="3",
    )

    client = MockModelClient(default_response="The answer is \\boxed{3}.")
    agent = ReasoningAgent(AgentConfig(), model_client=client)
    env = MathEnvironment()
    collector = RolloutCollector(default_run_id="test_run")

    traj = collector.collect_episode(agent=agent, env=env, task=task, seed=42)

    assert traj.run_id == "test_run"
    assert traj.task_id == "task_easy_1"
    assert traj.num_steps == 1
    assert traj.success is True
    assert pytest.approx(traj.total_reward) == 1.0
    assert traj.provenance.model_version == "mock-model-v1"
    assert traj.provenance.seed == 42


def test_rollout_collector_batch_aggregation():
    # Model returns \boxed{42}
    client = MockModelClient(default_response="\\boxed{42}")
    agent = ReasoningAgent(AgentConfig(), model_client=client)
    env = MathEnvironment()
    collector = RolloutCollector()

    tasks = [
        TaskSpec("t1", "math", "What is 42?", ground_truth="42"),
        TaskSpec("t2", "math", "What is 42?", ground_truth="42"),
        TaskSpec("t3", "math", "What is 0?", ground_truth="0"),  # Fail
    ]

    trajectories, stats = collector.collect_batch(agent=agent, env=env, tasks=tasks, base_seed=100)

    assert len(trajectories) == 3
    assert stats.total_episodes == 3
    assert pytest.approx(stats.success_rate) == 2 / 3
    assert pytest.approx(stats.mean_reward) == (1.0 + 1.0 + 0.0) / 3
    assert stats.total_steps == 3
    assert stats.mean_steps == 1.0
    assert "verifier_reward" in stats.reward_breakdown_mean
