"""Abstract agent architecture and policy runtime for ORBIT."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from orbit.data.trajectory import Action, Observation, Trajectory
from orbit.models.base import GenerationConfig


@dataclass
class AgentConfig:
    """Configuration for an ORBIT agent."""

    agent_id: str = "default_agent"
    model_name: str = "mock"
    system_prompt: str = ""
    max_steps: int = 10
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract agent runtime responsible for policy execution and decision-making."""

    def __init__(self, config: AgentConfig):
        self.config = config

    @abstractmethod
    def act(
        self,
        observation: Observation,
        trajectory: Trajectory | None = None,
    ) -> Action:
        """Selects an action given the current observation and past trajectory.

        Args:
            observation: Current environment observation.
            trajectory: History of the current episode so far.

        Returns:
            Action to be executed in the environment.
        """
        raise NotImplementedError("Subclasses must implement act()")

    def reset(self) -> None:
        """Resets any internal agent state between episodes."""

