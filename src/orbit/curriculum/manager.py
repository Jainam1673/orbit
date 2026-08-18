"""Curriculum Manager and decision audit logging for ORBIT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from orbit.curriculum.base import BaseCurriculum
from orbit.curriculum.strategies import (
    AdaptiveFrontierCurriculum,
    FixedDistributionCurriculum,
    StaticCurriculum,
)
from orbit.data.trajectory import Trajectory
from orbit.environments.base import TaskSpec
from orbit.environments.math.generator import MathTaskGenerator


@dataclass(frozen=True)
class CurriculumDecision:
    """Audit log entry capturing the rationale behind a task sampling decision."""

    decision_id: int
    strategy: str
    sampled_task_id: str
    difficulty: float
    frontier_difficulty: float
    recent_success_rate: float
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CurriculumManager:
    """Coordinates task selection and logs curriculum decisions."""

    def __init__(
        self,
        strategy: str = "adaptive",
        generator: MathTaskGenerator | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ):
        self.strategy_name = strategy.lower()
        self.generator = generator or MathTaskGenerator(seed=seed)
        self.decision_history: list[CurriculumDecision] = []

        if self.strategy_name == "fixed":
            self.curriculum: BaseCurriculum = FixedDistributionCurriculum(
                generator=self.generator, seed=seed, **kwargs
            )
        elif self.strategy_name == "static":
            self.curriculum = StaticCurriculum(
                generator=self.generator, seed=seed, **kwargs
            )
        elif self.strategy_name == "adaptive":
            self.curriculum = AdaptiveFrontierCurriculum(
                generator=self.generator, seed=seed, **kwargs
            )
        elif self.strategy_name in ("self_generated", "generated"):
            from orbit.curriculum.self_generated import SelfGeneratedCurriculum

            self.curriculum = SelfGeneratedCurriculum(seed=seed, **kwargs)
        else:
            raise ValueError(
                f"Unknown curriculum strategy '{strategy}'. Supported: ['fixed', 'static', 'adaptive', 'self_generated']"
            )

    def sample_task(self) -> TaskSpec:
        """Samples the next task and records the decision audit record."""
        task = self.curriculum.sample_task()
        decision_id = len(self.decision_history)

        metrics = self.curriculum.get_metrics()
        decision = CurriculumDecision(
            decision_id=decision_id,
            strategy=self.strategy_name,
            sampled_task_id=task.task_id,
            difficulty=task.difficulty,
            frontier_difficulty=metrics.get("current_frontier_difficulty", task.difficulty),
            recent_success_rate=metrics.get("recent_success_rate", 0.0),
            reason=f"Sampled via {self.strategy_name} curriculum at difficulty {task.difficulty:.3f}",
        )
        self.decision_history.append(decision)
        return task

    def update(self, trajectory: Trajectory) -> None:
        """Updates curriculum internal state with completed trajectory results."""
        self.curriculum.update(trajectory)

    def get_metrics(self) -> dict[str, Any]:
        """Returns observable telemetry on curriculum health."""
        metrics = self.curriculum.get_metrics()
        metrics["total_decisions_logged"] = len(self.decision_history)
        metrics["strategy"] = self.strategy_name
        return metrics
