"""Process-supervised step-level reward function decomposing credit across intermediate sub-goals."""

from __future__ import annotations

from typing import Any

from orbit.data.trajectory import RewardBreakdown, Step, Trajectory
from orbit.environments.base import TaskSpec
from orbit.rewards.base import BaseRewardFunction
from orbit.rewards.math_verifier import MathVerifier


class StepProcessRewardFunction(BaseRewardFunction):
    """Assigns dense intermediate process rewards for valid sub-goals and tool interactions."""

    def __init__(
        self,
        verifier: MathVerifier | None = None,
        success_reward: float = 1.0,
        subgoal_reward: float = 0.1,
        invalid_step_penalty: float = 0.05,
    ):
        super().__init__()
        self.verifier = verifier or MathVerifier()
        self.success_reward = success_reward
        self.subgoal_reward = subgoal_reward
        self.invalid_step_penalty = invalid_step_penalty

    def compute_reward(
        self,
        step: Step,
        task: TaskSpec,
        trajectory: Trajectory | None = None,
    ) -> RewardBreakdown:
        verifier_rew = 0.0
        shaping_rew = 0.0
        penalties = 0.0
        meta: dict[str, Any] = {}

        # 1. Process / Sub-goal reward for intermediate non-terminal steps
        if not step.done:
            # Check if step executed a successful tool call or meaningful calculation
            if step.action.tool_call is not None:
                tool_out = step.info.get("tool_result", {})
                if tool_out.get("success", True):
                    shaping_rew += self.subgoal_reward
                    meta["subgoal_status"] = "valid_tool_execution"
                else:
                    penalties += self.invalid_step_penalty
                    meta["subgoal_status"] = "failed_tool_execution"
            else:
                # Normal reasoning step: award small progress credit if substantial reasoning occurred
                if len(step.action.raw_text.strip()) > 10:
                    shaping_rew += self.subgoal_reward * 0.5
                    meta["subgoal_status"] = "valid_reasoning_step"

        # 2. Terminal verification reward
        if step.done:
            if trajectory is not None and trajectory.steps:
                ver_res = self.verifier.verify(task, trajectory)
            else:
                ver_res = self.verifier.verify_step(task, step)

            if ver_res.is_correct:
                verifier_rew = self.success_reward
                meta["terminal_status"] = "correct"
            else:
                meta["terminal_status"] = "incorrect"

        return RewardBreakdown(
            env_reward=0.0,
            verifier_reward=verifier_rew,
            shaping_reward=shaping_rew,
            critic_reward=0.0,
            penalties=penalties,
        )
