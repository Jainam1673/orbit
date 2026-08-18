"""Mathematics reasoning environment implementation for ORBIT."""

from __future__ import annotations

from typing import Any

from orbit.data.trajectory import Action, Observation, RewardBreakdown, Step
from orbit.environments.base import BaseEnvironment, TaskSpec
from orbit.environments.math.generator import MathTaskGenerator
from orbit.rewards.math_verifier import (
    MathRewardFunction,
    MathVerifier,
    extract_math_answer,
)


class MathEnvironment(BaseEnvironment):
    """Gymnasium-style environment for evaluating mathematical reasoning."""

    def __init__(
        self,
        env_id: str = "math_reasoning",
        version: str = "0.1.0",
        max_steps: int = 10,
        difficulty: float = 0.5,
        verifier: MathVerifier | None = None,
        reward_fn: MathRewardFunction | None = None,
    ):
        super().__init__(env_id=env_id, version=version)
        self.max_steps = max_steps
        self.default_difficulty = difficulty
        self.generator = MathTaskGenerator()
        self.verifier = verifier or MathVerifier()
        self.reward_fn = reward_fn or MathRewardFunction(verifier=self.verifier)

        self.current_task: TaskSpec | None = None
        self.current_step_index: int = 0
        self.history: list[dict[str, Any]] = []

    def reset(
        self,
        task: TaskSpec | None = None,
        seed: int | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Resets the environment and returns the initial math problem observation."""
        if seed is not None:
            self.generator.seed(seed)

        if task is None:
            self.current_task = self.generator.generate_task(
                difficulty=self.default_difficulty
            )
        else:
            self.current_task = task

        self.current_step_index = 0
        self.history = []

        obs = Observation(
            text=self.current_task.prompt,
            state={"task_id": self.current_task.task_id, "step": 0},
            metadata={
                "difficulty": self.current_task.difficulty,
                "family": self.current_task.family,
            },
        )
        info: dict[str, Any] = {
            "task_id": self.current_task.task_id,
            "difficulty": self.current_task.difficulty,
            "family": self.current_task.family,
        }
        return obs, info

    def step(
        self, action: Action
    ) -> tuple[Observation, RewardBreakdown, bool, bool, dict[str, Any]]:
        """Applies agent action, evaluates step, and returns feedback."""
        if self.current_task is None:
            raise RuntimeError("Cannot step an uninitialized environment. Call reset() first.")

        self.history.append(
            {"step": self.current_step_index, "action": action.raw_text}
        )

        extracted_answer = extract_math_answer(action.raw_text)
        # If an answer was extracted or max_steps reached, we terminate
        has_answer = extracted_answer is not None
        step_limit_reached = (self.current_step_index + 1) >= self.max_steps

        terminated = has_answer
        truncated = (not terminated) and step_limit_reached

        done = terminated or truncated

        # Create step model to evaluate reward
        current_obs = Observation(
            text=f"Scratchpad step {self.current_step_index}",
            state={"step": self.current_step_index},
        )
        step_model = Step(
            step_index=self.current_step_index,
            observation=current_obs,
            action=action,
            reward=RewardBreakdown(),
            done=done,
            truncated=truncated,
        )

        reward_breakdown = self.reward_fn.compute_reward(
            step=step_model,
            task=self.current_task,
        )

        self.current_step_index += 1

        if done:
            next_obs_text = "Episode finished."
        else:
            next_obs_text = f"Step {self.current_step_index} recorded. Continue reasoning or provide final answer."

        next_obs = Observation(
            text=next_obs_text,
            state={"step": self.current_step_index},
            metadata={"extracted_answer": extracted_answer},
        )

        info: dict[str, Any] = {
            "extracted_answer": extracted_answer,
            "ground_truth": self.current_task.ground_truth,
            "is_correct": reward_breakdown.verifier_reward > 0,
            "step_index": self.current_step_index,
        }

        return next_obs, reward_breakdown, terminated, truncated, info
