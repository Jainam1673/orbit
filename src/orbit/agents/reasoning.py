"""Reasoning agent implementation with Chain-of-Thought prompting and trajectory context."""

from __future__ import annotations

from typing import Any

from orbit.agents.base import AgentConfig, BaseAgent
from orbit.data.trajectory import Action, Observation, Trajectory
from orbit.models.base import BaseModelClient


class ReasoningAgent(BaseAgent):
    """Baseline reasoning agent using Chain-of-Thought and multi-step conversation context."""

    def __init__(
        self,
        config: AgentConfig,
        model_client: BaseModelClient,
    ):
        super().__init__(config=config)
        self.model_client = model_client

    def build_prompt(
        self,
        observation: Observation,
        trajectory: Trajectory | None = None,
    ) -> str:
        """Constructs the prompt including system instructions, history, and current observation."""
        sections: list[str] = []

        if self.config.system_prompt:
            sections.append(f"System: {self.config.system_prompt}")

        if trajectory is not None and trajectory.steps:
            sections.append("Interaction History:")
            for s in trajectory.steps:
                sections.append(f"Observation: {s.observation.text}")
                sections.append(f"Action: {s.action.raw_text}")

        sections.append(f"Current Observation: {observation.text}")
        sections.append("Assistant:")

        return "\n\n".join(sections)

    def act(
        self,
        observation: Observation,
        trajectory: Trajectory | None = None,
    ) -> Action:
        """Generates an action given observation and prior episodic history."""
        prompt = self.build_prompt(observation, trajectory)
        output = self.model_client.generate(
            prompt=prompt,
            config=self.config.generation_config,
        )

        metadata: dict[str, Any] = {
            "prompt_tokens": output.prompt_tokens,
            "completion_tokens": output.completion_tokens,
            "latency_ms": output.latency_ms,
            "finish_reason": output.finish_reason,
            "model_id": self.model_client.model_id,
        }

        return Action(
            raw_text=output.text,
            metadata=metadata,
        )
