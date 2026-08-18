"""Parallel rollout worker pool for asynchronous trajectory generation."""

from concurrent.futures import ThreadPoolExecutor, as_completed

from orbit.agents.base import BaseAgent
from orbit.data.trajectory import Trajectory
from orbit.environments.base import BaseEnvironment, TaskSpec
from orbit.rollouts.collector import RolloutCollector, RolloutStats


class DistributedWorkerPool:
    """Coordinates parallel trajectory collection across worker threads or nodes."""

    def __init__(self, num_workers: int = 4, run_id: str = "dist_pool"):
        self.num_workers = num_workers
        self.run_id = run_id
        self.collector = RolloutCollector(default_run_id=run_id)

    def collect_parallel_rollouts(
        self,
        agent: BaseAgent,
        env: BaseEnvironment,
        tasks: list[TaskSpec],
        base_seed: int = 42,
    ) -> tuple[list[Trajectory], RolloutStats]:
        """Dispatches rollouts in parallel across worker pool."""
        trajectories: list[Trajectory] = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_task = {
                executor.submit(
                    self.collector.collect_episode,
                    agent=agent,
                    env=env,
                    task=task,
                    run_id=self.run_id,
                    episode_id=f"dist_ep_{i:04d}",
                    seed=base_seed + i,
                ): task
                for i, task in enumerate(tasks)
            }

            for future in as_completed(future_to_task):
                traj = future.result()
                trajectories.append(traj)

        # Sort trajectories by episode_id to maintain deterministic ordering
        trajectories.sort(key=lambda t: t.episode_id)

        # Compute aggregate batch statistics
        total = len(trajectories)
        if total == 0:
            stats = RolloutStats(
                total_episodes=0,
                success_rate=0.0,
                mean_reward=0.0,
                mean_steps=0.0,
                total_steps=0,
            )
        else:
            successes = sum(1 for t in trajectories if t.success)
            rewards = [t.total_reward for t in trajectories]
            steps = [t.num_steps for t in trajectories]
            stats = RolloutStats(
                total_episodes=total,
                success_rate=successes / total,
                mean_reward=sum(rewards) / total,
                mean_steps=sum(steps) / total,
                total_steps=sum(steps),
            )

        return trajectories, stats
