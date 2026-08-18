"""LLM-based task generation and structured parsing for ORBIT."""

from __future__ import annotations

import json
import re
import uuid

from orbit.environments.base import TaskSpec
from orbit.models.base import BaseModelClient, GenerationConfig


class LLMTaskGenerator:
    """Generates candidate reasoning tasks using a foundation language model."""

    def __init__(
        self,
        model_client: BaseModelClient,
        system_prompt: str | None = None,
    ):
        self.model_client = model_client
        self.system_prompt = system_prompt or (
            "You are a mathematical curriculum generator. Generate verified, rigorous "
            "math reasoning tasks. Always respond in valid JSON format:\n"
            '{"prompt": "...", "ground_truth": "...", "family": "math", "difficulty": 0.5}'
        )

    def generate_candidate(
        self,
        target_difficulty: float = 0.5,
        family: str = "math_reasoning",
    ) -> TaskSpec:
        """Prompts LLM to generate a candidate task matching target difficulty."""
        prompt = (
            f"System: {self.system_prompt}\n\n"
            f"User: Generate a new {family} task at difficulty level {target_difficulty:.2f} (scale 0.0 to 1.0).\n"
            f"Return ONLY valid JSON with keys 'prompt', 'ground_truth', 'family', 'difficulty'.\n\n"
            "Assistant:"
        )

        output = self.model_client.generate(
            prompt=prompt,
            config=GenerationConfig(temperature=0.7, max_tokens=512),
        )

        # Parse JSON from model output
        task_data = self._parse_json_response(output.text, target_difficulty, family)
        return TaskSpec(
            task_id=f"gen_{uuid.uuid4().hex[:8]}",
            family=task_data.get("family", family),
            prompt=task_data.get("prompt", ""),
            ground_truth=task_data.get("ground_truth", ""),
            difficulty=float(task_data.get("difficulty", target_difficulty)),
            metadata={"generated_by": self.model_client.model_id},
        )

    def _parse_json_response(
        self, text: str, default_diff: float, default_family: str
    ) -> dict[str, str | float]:
        """Extracts JSON object from model generation text."""
        # Try direct json loads
        clean = text.strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # Try finding markdown code block ```json ... ```
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding raw { ... }
        brace_match = re.search(r"(\{.*\})", clean, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(1))
            except json.JSONDecodeError:
                pass

        # Fallback if model output is unstructured
        return {
            "prompt": clean,
            "ground_truth": "",
            "family": default_family,
            "difficulty": default_diff,
        }
