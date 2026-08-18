"""Abstract base environment and task specification interfaces for ORBIT.

Environments provide objective, Gymnasium-inspired interfaces with machine-verifiable
task specifications and decomposed reward signals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from orbit.data.trajectory import Action, Observation, RewardBreakdown


@dataclass(frozen=True)
class TaskSpec:
    """Specification of a task provided to an environment."""

    task_id: str
    family: str
    prompt: str
    ground_truth: Any = None
    difficulty: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        return cls(
            task_id=data["task_id"],
            family=data["family"],
            prompt=data["prompt"],
            ground_truth=data.get("ground_truth"),
            difficulty=float(data.get("difficulty", 0.5)),
            metadata=data.get("metadata", {}),
        )


class BaseEnvironment(ABC):
    """Gymnasium-inspired abstract base class for ORBIT environments."""

    def __init__(self, env_id: str, version: str = "0.1.0"):
        self.env_id = env_id
        self.version = version
        self.current_task: TaskSpec | None = None

    @abstractmethod
    def reset(
        self,
        task: TaskSpec | None = None,
        seed: int | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Resets the environment for a new episode.

        Args:
            task: Task specification to initialize. If None, environment samples a default task.
            seed: Seed for environment-level PRNG.

        Returns:
            Tuple of (initial Observation, auxiliary info dict).
        """
        raise NotImplementedError("Subclasses must implement reset()")

    @abstractmethod
    def step(
        self, action: Action
    ) -> tuple[Observation, RewardBreakdown, bool, bool, dict[str, Any]]:
        """Executes an action within the environment.

        Args:
            action: Action executed by the agent.

        Returns:
            Tuple of (Observation, RewardBreakdown, terminated, truncated, info dict).
        """
        raise NotImplementedError("Subclasses must implement step()")

    def render(self) -> Any:
        """Optional render hook for visualization or debugging."""
        return None

    def close(self) -> None:
        """Closes and releases any environment resources."""

