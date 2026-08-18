"""Deterministic mock model client for fast unit tests and reproducible baselines."""

from __future__ import annotations

import time
from collections.abc import Callable

from orbit.models.base import BaseModelClient, GenerationConfig, ModelOutput


class MockModelClient(BaseModelClient):
    """Deterministic model client that returns predefined or rule-based responses."""

    def __init__(
        self,
        model_id: str = "mock-model-v1",
        default_response: str = "The answer is \\boxed{42}.",
        response_fn: Callable[[str], str] | None = None,
    ):
        super().__init__(model_id=model_id)
        self.default_response = default_response
        self.response_fn = response_fn
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> ModelOutput:
        start_time = time.perf_counter()
        self.call_count += 1

        if self.response_fn is not None:
            text = self.response_fn(prompt)
        else:
            text = self.default_response

        latency = (time.perf_counter() - start_time) * 1000.0

        # Mock tokenization (whitespace based for simplicity)
        tokens = text.split()
        prompt_tokens = len(prompt.split())
        completion_tokens = len(tokens)
        token_ids = [hash(t) % 32000 for t in tokens]
        logprobs = [-0.1] * completion_tokens

        return ModelOutput(
            text=text,
            token_ids=token_ids,
            logprobs=logprobs,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency,
            metadata={"model_id": self.model_id, "mock": True},
        )

    def get_logprobs(
        self,
        prompt: str,
        completion: str,
    ) -> list[float]:
        tokens = completion.split()
        return [-0.1] * len(tokens)
