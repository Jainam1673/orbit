"""Abstract verification and reward computation contracts for ORBIT."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from orbit.data.trajectory import RewardBreakdown, Step, Trajectory
from orbit.environments.base import TaskSpec


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of objective task solution verification."""

    is_correct: bool
    score: float
    feedback: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationResult:
        return cls(
            is_correct=data["is_correct"],
            score=float(data["score"]),
            feedback=data.get("feedback", ""),
            metrics=data.get("metrics", {}),
        )


class BaseVerifier(ABC):
    """Abstract verifier for objective validation of agent answers and code."""

    @abstractmethod
    def verify(
        self, task: TaskSpec, trajectory: Trajectory
    ) -> VerificationResult:
        """Evaluates a completed trajectory against the task ground truth.

        Args:
            task: The original task specification.
            trajectory: The full episode trajectory.

        Returns:
            VerificationResult containing objective success score and metadata.
        """
        raise NotImplementedError("Subclasses must implement verify()")


class BaseRewardFunction(ABC):
    """Abstract reward computer decomposing reward signals into explicit components."""

    @abstractmethod
    def compute_reward(
        self,
        step: Step,
        task: TaskSpec,
        trajectory: Trajectory | None = None,
    ) -> RewardBreakdown:
        """Computes explicit reward breakdown for a given step.

        Args:
            step: The interaction step.
            task: The active task specification.
            trajectory: The trajectory context so far, if available.

        Returns:
            RewardBreakdown with explicitly separated reward sources.
        """
        raise NotImplementedError("Subclasses must implement compute_reward()")
