"""Abstract curriculum engine contracts and state definitions for ORBIT."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from orbit.data.trajectory import Trajectory
from orbit.environments.base import TaskSpec


@dataclass
class CurriculumState:
    """State of the adaptive curriculum engine."""

    total_tasks_generated: int = 0
    total_tasks_evaluated: int = 0
    current_frontier_difficulty: float = 0.5
    recent_success_rate: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseCurriculum(ABC):
    """Abstract curriculum engine for task selection, generation, and pacing."""

    def __init__(self) -> None:
        self.state = CurriculumState()

    @abstractmethod
    def sample_task(self) -> TaskSpec:
        """Samples or generates the next task for training or evaluation.

        Returns:
            TaskSpec describing the selected task.
        """
        raise NotImplementedError("Subclasses must implement sample_task()")

    @abstractmethod
    def update(self, trajectory: Trajectory) -> None:
        """Updates curriculum internal state and difficulty estimates with trajectory results.

        Args:
            trajectory: Completed episode trajectory.
        """
        raise NotImplementedError("Subclasses must implement update()")

    def get_metrics(self) -> dict[str, Any]:
        """Returns observable telemetry on curriculum health and state."""
        return {
            "total_tasks_generated": self.state.total_tasks_generated,
            "total_tasks_evaluated": self.state.total_tasks_evaluated,
            "current_frontier_difficulty": self.state.current_frontier_difficulty,
            "recent_success_rate": self.state.recent_success_rate,
        }
