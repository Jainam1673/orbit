"""Core trajectory data structures and serialization for ORBIT.

Provides unambiguous, type-safe data representations for agent-environment
interactions, decomposed reward metrics, and experiment provenance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class Observation:
    """Observation received by an agent from the environment."""

    text: str
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(
            text=data["text"],
            state=data.get("state", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class Action:
    """Action executed by an agent in the environment."""

    raw_text: str
    tool_call: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        return cls(
            raw_text=data["raw_text"],
            tool_call=data.get("tool_call"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class RewardBreakdown:
    """Decomposed reward structure to detect reward hacking and isolate signals.

    Total reward is strictly the sum of all individual components:
    total = env_reward + verifier_reward + shaping_reward + critic_reward - penalties
    """

    env_reward: float = 0.0
    verifier_reward: float = 0.0
    shaping_reward: float = 0.0
    critic_reward: float = 0.0
    penalties: float = 0.0

    @property
    def total(self) -> float:
        """Computes unambiguous scalar total reward."""
        return (
            self.env_reward
            + self.verifier_reward
            + self.shaping_reward
            + self.critic_reward
            - self.penalties
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total"] = self.total
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewardBreakdown:
        return cls(
            env_reward=float(data.get("env_reward", 0.0)),
            verifier_reward=float(data.get("verifier_reward", 0.0)),
            shaping_reward=float(data.get("shaping_reward", 0.0)),
            critic_reward=float(data.get("critic_reward", 0.0)),
            penalties=float(data.get("penalties", 0.0)),
        )


@dataclass(frozen=True)
class Step:
    """Single transition step within an interaction episode."""

    step_index: int
    observation: Observation
    action: Action
    reward: RewardBreakdown
    done: bool
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "observation": self.observation.to_dict(),
            "action": self.action.to_dict(),
            "reward": self.reward.to_dict(),
            "done": self.done,
            "truncated": self.truncated,
            "info": self.info,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        return cls(
            step_index=data["step_index"],
            observation=Observation.from_dict(data["observation"]),
            action=Action.from_dict(data["action"]),
            reward=RewardBreakdown.from_dict(data["reward"]),
            done=data["done"],
            truncated=data.get("truncated", False),
            info=data.get("info", {}),
            timestamp=data.get(
                "timestamp", datetime.now(UTC).isoformat()
            ),
        )


@dataclass(frozen=True)
class Provenance:
    """Audit metadata ensuring full experimental reproducibility."""

    git_commit: str = "unknown"
    git_dirty: bool = False
    model_version: str = "unknown"
    env_version: str = "unknown"
    seed: int = 0
    hardware_info: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        return cls(
            git_commit=data.get("git_commit", "unknown"),
            git_dirty=data.get("git_dirty", False),
            model_version=data.get("model_version", "unknown"),
            env_version=data.get("env_version", "unknown"),
            seed=data.get("seed", 0),
            hardware_info=data.get("hardware_info", {}),
            timestamp=data.get(
                "timestamp", datetime.now(UTC).isoformat()
            ),
        )


@dataclass
class Trajectory:
    """Complete episodic trajectory of an agent interacting with an environment."""

    run_id: str
    episode_id: str
    task_id: str
    steps: list[Step] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = False

    @property
    def total_reward(self) -> float:
        """Cumulative sum of total rewards across all steps."""
        return sum(s.reward.total for s in self.steps)

    @property
    def reward_breakdown_sum(self) -> RewardBreakdown:
        """Aggregated reward breakdown across all steps."""
        return RewardBreakdown(
            env_reward=sum(s.reward.env_reward for s in self.steps),
            verifier_reward=sum(s.reward.verifier_reward for s in self.steps),
            shaping_reward=sum(s.reward.shaping_reward for s in self.steps),
            critic_reward=sum(s.reward.critic_reward for s in self.steps),
            penalties=sum(s.reward.penalties for s in self.steps),
        )

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def add_step(self, step: Step) -> None:
        """Appends a validated transition step."""
        if step.step_index != len(self.steps):
            raise ValueError(
                f"Step index mismatch: expected {len(self.steps)}, got {step.step_index}"
            )
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "steps": [s.to_dict() for s in self.steps],
            "provenance": self.provenance.to_dict(),
            "metadata": self.metadata,
            "success": self.success,
            "total_reward": self.total_reward,
            "num_steps": self.num_steps,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        steps = [Step.from_dict(s) for s in data.get("steps", [])]
        traj = cls(
            run_id=data["run_id"],
            episode_id=data["episode_id"],
            task_id=data["task_id"],
            steps=steps,
            provenance=Provenance.from_dict(data.get("provenance", {})),
            metadata=data.get("metadata", {}),
            success=data.get("success", False),
        )
        return traj

    @classmethod
    def from_json(cls, json_str: str) -> Trajectory:
        return cls.from_dict(json.loads(json_str))
