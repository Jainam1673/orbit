"""Curriculum strategies implementing baselines and adaptive learning frontier."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from orbit.curriculum.base import BaseCurriculum
from orbit.curriculum.difficulty import (
    DifficultyTracker,
    LearningFrontierEstimator,
)
from orbit.data.trajectory import Trajectory
from orbit.environments.base import TaskSpec
from orbit.environments.math.generator import MathTaskGenerator


class FixedDistributionCurriculum(BaseCurriculum):
    """Baseline A / B: Samples tasks from a static, fixed difficulty distribution."""

    def __init__(
        self,
        min_difficulty: float = 0.1,
        max_difficulty: float = 1.0,
        generator: MathTaskGenerator | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.generator = generator or MathTaskGenerator(seed=seed)
        self.rng = random.Random(seed)

    def sample_task(self) -> TaskSpec:
        diff = self.rng.uniform(self.min_difficulty, self.max_difficulty)
        task = self.generator.generate_task(difficulty=diff)
        self.state.total_tasks_generated += 1
        return task

    def update(self, trajectory: Trajectory) -> None:
        self.state.total_tasks_evaluated += 1
        self.state.history.append(
            {
                "task_id": trajectory.task_id,
                "success": trajectory.success,
                "reward": trajectory.total_reward,
            }
        )


class StaticCurriculum(BaseCurriculum):
    """Baseline C: Advances difficulty monotonically after fixed step thresholds."""

    def __init__(
        self,
        stages: list[tuple[int, float]] | None = None,
        generator: MathTaskGenerator | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        # List of (evaluation_count_threshold, difficulty)
        self.stages = stages or [
            (10, 0.2),
            (25, 0.45),
            (45, 0.7),
            (70, 0.95),
        ]
        self.generator = generator or MathTaskGenerator(seed=seed)
        self.current_stage_idx = 0

    @property
    def current_difficulty(self) -> float:
        for threshold, diff in self.stages:
            if self.state.total_tasks_evaluated < threshold:
                return diff
        return self.stages[-1][1]

    def sample_task(self) -> TaskSpec:
        diff = self.current_difficulty
        self.state.current_frontier_difficulty = diff
        task = self.generator.generate_task(difficulty=diff)
        self.state.total_tasks_generated += 1
        return task

    def update(self, trajectory: Trajectory) -> None:
        self.state.total_tasks_evaluated += 1
        self.state.history.append(
            {
                "task_id": trajectory.task_id,
                "success": trajectory.success,
                "difficulty": self.current_difficulty,
            }
        )


class AdaptiveFrontierCurriculum(BaseCurriculum):
    """ORBIT Core: Dynamically centers task generation around the empirical learning frontier."""

    def __init__(
        self,
        target_success_rate: float = 0.6,
        learning_rate: float = 0.05,
        min_difficulty: float = 0.1,
        max_difficulty: float = 1.0,
        difficulty_std: float = 0.08,
        window_size: int = 20,
        generator: MathTaskGenerator | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.tracker = DifficultyTracker(window_size=window_size)
        self.estimator = LearningFrontierEstimator(
            tracker=self.tracker,
            target_success_rate=target_success_rate,
            learning_rate=learning_rate,
            min_difficulty=min_difficulty,
            max_difficulty=max_difficulty,
        )
        self.difficulty_std = difficulty_std
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.generator = generator or MathTaskGenerator(seed=seed)
        self.rng = np.random.default_rng(seed)

    def sample_task(self) -> TaskSpec:
        frontier = self.estimator.current_frontier
        # Sample around current frontier with Gaussian perturbation
        sampled_diff = float(
            np.clip(
                self.rng.normal(loc=frontier, scale=self.difficulty_std),
                self.min_difficulty,
                self.max_difficulty,
            )
        )
        self.state.current_frontier_difficulty = frontier
        self.state.recent_success_rate = self.tracker.get_overall_success_rate()
        task = self.generator.generate_task(difficulty=sampled_diff)
        self.state.total_tasks_generated += 1
        return task

    def update(self, trajectory: Trajectory) -> None:
        self.state.total_tasks_evaluated += 1

        # Extract task difficulty from metadata or default to current frontier
        diff = trajectory.metadata.get(
            "difficulty", self.estimator.current_frontier
        )

        self.tracker.record_outcome(
            difficulty=diff,
            success=trajectory.success,
            reward=trajectory.total_reward,
            num_steps=trajectory.num_steps,
        )

        # Update frontier
        new_frontier = self.estimator.update_frontier(
            recent_success=trajectory.success
        )
        self.state.current_frontier_difficulty = new_frontier
        self.state.recent_success_rate = self.tracker.get_overall_success_rate()

        self.state.history.append(
            {
                "task_id": trajectory.task_id,
                "success": trajectory.success,
                "difficulty": diff,
                "new_frontier": new_frontier,
            }
        )

    def get_metrics(self) -> dict[str, Any]:
        metrics = super().get_metrics()
        metrics["bin_success_rates"] = self.tracker.get_bin_success_rates()
        return metrics
