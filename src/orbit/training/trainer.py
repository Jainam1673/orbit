"""End-to-end training loop orchestrator for ORBIT."""

from __future__ import annotations

from typing import Any

from orbit.agents.base import BaseAgent
from orbit.curriculum.base import BaseCurriculum
from orbit.environments.base import BaseEnvironment, TaskSpec
from orbit.rollouts.collector import RolloutCollector


class TrainingLoop:
    """Orchestrates the iterative training loop across environments, agents, curricula, and evaluation."""

    def __init__(
        self,
        agent: BaseAgent,
        env: BaseEnvironment,
        curriculum: BaseCurriculum,
        eval_tasks: list[TaskSpec] | None = None,
        eval_env: BaseEnvironment | None = None,
        collector: RolloutCollector | None = None,
        run_id: str = "train_run",
    ):
        self.agent = agent
        self.env = env
        self.curriculum = curriculum
        self.eval_tasks = eval_tasks or []
        self.eval_env = eval_env or env
        self.collector = collector or RolloutCollector(default_run_id=run_id)
        self.run_id = run_id

        self.history: list[dict[str, Any]] = []
        self.eval_history: list[dict[str, Any]] = []

    def evaluate(self, step: int) -> dict[str, Any]:
        """Runs evaluation over held-out evaluation tasks."""
        if not self.eval_tasks:
            eval_result = {
                "step": step,
                "eval_total_episodes": 0,
                "eval_success_rate": 0.0,
                "eval_mean_reward": 0.0,
                "eval_mean_steps": 0.0,
            }
            self.eval_history.append(eval_result)
            return eval_result

        _trajectories, stats = self.collector.collect_batch(
            agent=self.agent,
            env=self.eval_env,
            tasks=self.eval_tasks,
            run_id=f"{self.run_id}_eval_step_{step}",
        )

        eval_result = {
            "step": step,
            "eval_total_episodes": stats.total_episodes,
            "eval_success_rate": stats.success_rate,
            "eval_mean_reward": stats.mean_reward,
            "eval_mean_steps": stats.mean_steps,
        }
        self.eval_history.append(eval_result)
        return eval_result

    def run(
        self,
        num_steps: int = 10,
        eval_interval: int = 5,
        base_seed: int = 42,
    ) -> dict[str, Any]:
        """Executes training loop for specified number of steps."""
        for step_idx in range(num_steps):
            step_seed = base_seed + step_idx

            # 1. Sample task from curriculum
            task = self.curriculum.sample_task()

            # 2. Collect interaction episode
            trajectory = self.collector.collect_episode(
                agent=self.agent,
                env=self.env,
                task=task,
                run_id=self.run_id,
                episode_id=f"ep_{step_idx:05d}",
                seed=step_seed,
            )

            # 3. Update curriculum state
            self.curriculum.update(trajectory)

            # 4. Log step metrics
            curriculum_metrics = self.curriculum.get_metrics()
            step_metrics: dict[str, Any] = {
                "step": step_idx,
                "task_id": task.task_id,
                "task_difficulty": task.difficulty,
                "success": trajectory.success,
                "total_reward": trajectory.total_reward,
                "num_steps": trajectory.num_steps,
                "frontier_difficulty": curriculum_metrics.get(
                    "current_frontier_difficulty", task.difficulty
                ),
                "recent_success_rate": curriculum_metrics.get(
                    "recent_success_rate", 0.0
                ),
            }
            self.history.append(step_metrics)

            # 5. Periodic evaluation
            if (step_idx + 1) % eval_interval == 0 or (step_idx + 1) == num_steps:
                self.evaluate(step=step_idx + 1)

        summary: dict[str, Any] = {
            "total_steps": num_steps,
            "mean_reward": sum(h["total_reward"] for h in self.history) / max(1, len(self.history)),
            "overall_success_rate": sum(1 for h in self.history if h["success"]) / max(1, len(self.history)),
            "training_history": self.history,
            "eval_history": self.eval_history,
        }
        return summary
