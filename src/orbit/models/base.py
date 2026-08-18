"""Abstract foundation model and inference interfaces for ORBIT."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration parameters for language model generation."""

    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    stop_sequences: list[str] = field(default_factory=list)
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelOutput:
    """Standardized output from a model generation request."""

    text: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseModelClient(ABC):
    """Abstract client interface for foundation models (HuggingFace, vLLM, API)."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> ModelOutput:
        """Generates text from a given prompt.

        Args:
            prompt: Text prompt string.
            config: Optional generation parameters.

        Returns:
            ModelOutput with response text and generation telemetry.
        """
        raise NotImplementedError("Subclasses must implement generate()")

    @abstractmethod
    def get_logprobs(
        self,
        prompt: str,
        completion: str,
    ) -> list[float]:
        """Calculates per-token log-probabilities for a given prompt-completion pair.

        Args:
            prompt: Base prompt.
            completion: Model completion text.

        Returns:
            List of float log probabilities per completion token.
        """
        raise NotImplementedError("Subclasses must implement get_logprobs()")
