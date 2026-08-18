"""Self-generated task curriculum executing generation, validation, and admission."""

from __future__ import annotations

from typing import Any

from orbit.curriculum.base import BaseCurriculum
from orbit.curriculum.difficulty import DifficultyTracker, LearningFrontierEstimator
from orbit.curriculum.task_generator import LLMTaskGenerator
from orbit.curriculum.validator import TaskPipelineValidator, TaskValidationResult
from orbit.data.trajectory import Trajectory
from orbit.environments.base import TaskSpec
from orbit.environments.math.generator import MathTaskGenerator
from orbit.models.base import BaseModelClient


class SelfGeneratedCurriculum(BaseCurriculum):
    """Curriculum engine driving self-generated tasks through a rigorous validation pipeline."""

    def __init__(
        self,
        model_client: BaseModelClient | None = None,
        target_success_rate: float = 0.6,
        learning_rate: float = 0.05,
        min_difficulty: float = 0.1,
        max_difficulty: float = 1.0,
        window_size: int = 20,
        max_generation_retries: int = 5,
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
        self.llm_generator = (
            LLMTaskGenerator(model_client) if model_client is not None else None
        )
        self.fallback_generator = MathTaskGenerator(seed=seed)
        self.validator = TaskPipelineValidator()
        self.max_retries = max_generation_retries

        self.admitted_pool: list[TaskSpec] = []
        self.validation_history: list[TaskValidationResult] = []

    def sample_task(self) -> TaskSpec:
        """Generates, validates, and admits the next task for training."""
        frontier = self.estimator.current_frontier
        self.state.current_frontier_difficulty = frontier
        self.state.recent_success_rate = self.tracker.get_overall_success_rate()

        admitted_task: TaskSpec | None = None

        for _ in range(self.max_retries):
            # 1. Generate candidate
            if self.llm_generator is not None:
                candidate = self.llm_generator.generate_candidate(
                    target_difficulty=frontier
                )
            else:
                candidate = self.fallback_generator.generate_task(
                    difficulty=frontier
                )

            self.state.total_tasks_generated += 1

            # 2. Validate candidate through pipeline
            val_result = self.validator.process_candidate(candidate)
            self.validation_history.append(val_result)

            if val_result.is_admitted:
                admitted_task = candidate
                self.admitted_pool.append(candidate)
                break

        # If LLM generation retries were exhausted, use verified procedural generator
        if admitted_task is None:
            fallback = self.fallback_generator.generate_task(difficulty=frontier)
            val_result = self.validator.process_candidate(fallback)
            self.validation_history.append(val_result)
            admitted_task = fallback
            self.admitted_pool.append(fallback)

        return admitted_task

    def update(self, trajectory: Trajectory) -> None:
        self.state.total_tasks_evaluated += 1

        diff = trajectory.metadata.get(
            "difficulty", self.estimator.current_frontier
        )

        self.tracker.record_outcome(
            difficulty=diff,
            success=trajectory.success,
            reward=trajectory.total_reward,
            num_steps=trajectory.num_steps,
        )

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
        metrics["total_admitted_tasks"] = self.validator.total_admitted
        metrics["total_processed_tasks"] = self.validator.total_processed
        metrics["admission_rate"] = (
            self.validator.total_admitted / max(1, self.validator.total_processed)
        )
        metrics["rejection_distribution"] = dict(self.validator.rejection_counts)
        return metrics
