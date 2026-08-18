"""Symbolic and numerical verifier and reward functions for mathematical reasoning."""

import math
import re
from fractions import Fraction
from typing import Any

from orbit.data.trajectory import RewardBreakdown, Step, Trajectory
from orbit.environments.base import TaskSpec
from orbit.rewards.base import BaseRewardFunction, BaseVerifier, VerificationResult


def extract_boxed_answer(text: str) -> str | None:
    """Extracts answer content from LaTeX \\boxed{...} tags."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    idx += len("\\boxed{")
    depth = 1
    extracted: list[str] = []
    for i in range(idx, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(extracted).strip()
        extracted.append(char)
    return None


def extract_math_answer(text: str) -> str | None:
    """Extracts the final mathematical answer from agent response text.

    Supports:
    1. LaTeX \\boxed{answer}
    2. 'Final Answer: answer'
    3. 'The answer is answer'
    4. '#### answer' (GSM8K style)
    5. Pure numeric / fraction strings
    """
    # 1. Boxed
    boxed = extract_boxed_answer(text)
    if boxed is not None and boxed.strip():
        return boxed.strip()

    # 2. GSM8K format #### <answer>
    gsm_match = re.search(r"####\s*([^\n]+)", text)
    if gsm_match:
        return gsm_match.group(1).strip()

    # 3. Final Answer / The answer is
    match = re.search(
        r"(?:final\s+answer|the\s+answer\s+is)[:\s]+([^\n]+)",
        text,
        re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip().rstrip(".").rstrip(",")
        # Take first token/expression if trailing text exists
        tokens = candidate.split()
        if tokens:
            return tokens[0].rstrip(".").rstrip(",")

    # 4. Direct numerical value check
    clean = text.strip()
    if normalize_numeric_string(clean) is not None:
        return clean

    return None


def normalize_numeric_string(val_str: str) -> float | int | Fraction | None:
    """Parses a string into a numerical value (int, float, or Fraction)."""
    clean = val_str.strip().replace("$", "").replace(",", "").strip()

    # Fraction representation e.g. "3/4" or "-1/2"
    if "/" in clean:
        parts = clean.split("/")
        if len(parts) == 2:
            try:
                num = int(parts[0].strip())
                den = int(parts[1].strip())
                if den != 0:
                    return Fraction(num, den)
            except ValueError:
                pass

    # Standard integer
    try:
        if clean.isdigit() or (clean.startswith("-") and clean[1:].isdigit()):
            return int(clean)
    except ValueError:
        pass

    # Float
    try:
        return float(clean)
    except ValueError:
        pass

    return None


def are_math_answers_equivalent(
    pred: str | None,
    target: Any,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-4,
) -> bool:
    """Checks mathematical equivalence between predicted and ground-truth answers."""
    if pred is None:
        return False

    pred_str = str(pred).strip()
    target_str = str(target).strip()

    # Exact string match
    if pred_str.lower() == target_str.lower():
        return True

    # Numeric conversion & comparison
    pred_num = normalize_numeric_string(pred_str)
    target_num = normalize_numeric_string(target_str)

    if pred_num is not None and target_num is not None:
        # Both fractions or ints
        if (
            isinstance(pred_num, (int, Fraction))
            and isinstance(target_num, (int, Fraction))
            and Fraction(pred_num) == Fraction(target_num)
        ):
            return True

        # Float comparison
        pred_float = float(pred_num)
        target_float = float(target_num)
        if math.isclose(
            pred_float, target_float, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            return True

    return False


class MathVerifier(BaseVerifier):
    """Objective verifier for mathematical solutions."""

    def __init__(self, rel_tol: float = 1e-4, abs_tol: float = 1e-4):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def verify_step(
        self, task: TaskSpec, step: Step
    ) -> VerificationResult:
        """Evaluates a single step containing a potential answer against ground truth."""
        extracted_answer = extract_math_answer(step.action.raw_text)

        is_correct = are_math_answers_equivalent(
            pred=extracted_answer,
            target=task.ground_truth,
            rel_tol=self.rel_tol,
            abs_tol=self.abs_tol,
        )

        score = 1.0 if is_correct else 0.0
        feedback = (
            "Correct answer."
            if is_correct
            else f"Extracted '{extracted_answer}', expected '{task.ground_truth}'."
        )

        return VerificationResult(
            is_correct=is_correct,
            score=score,
            feedback=feedback,
            metrics={
                "extracted_answer": extracted_answer,
                "ground_truth": task.ground_truth,
            },
        )

    def verify(
        self, task: TaskSpec, trajectory: Trajectory
    ) -> VerificationResult:
        if not trajectory.steps:
            return VerificationResult(
                is_correct=False,
                score=0.0,
                feedback="Trajectory contains no steps.",
            )

        return self.verify_step(task, trajectory.steps[-1])


class MathRewardFunction(BaseRewardFunction):
    """Computes decomposed rewards for mathematical reasoning episodes."""

    def __init__(
        self,
        verifier: MathVerifier | None = None,
        success_reward: float = 1.0,
        step_penalty: float = 0.0,
    ):
        self.verifier = verifier or MathVerifier()
        self.success_reward = success_reward
        self.step_penalty = step_penalty

    def compute_reward(
        self,
        step: Step,
        task: TaskSpec,
        trajectory: Trajectory | None = None,
    ) -> RewardBreakdown:
        verifier_rew = 0.0
        if step.done:
            if trajectory is not None and trajectory.steps:
                res = self.verifier.verify(task, trajectory)
            else:
                res = self.verifier.verify_step(task, step)

            if res.is_correct:
                verifier_rew = self.success_reward

        return RewardBreakdown(
            env_reward=0.0,
            verifier_reward=verifier_rew,
            shaping_reward=0.0,
            critic_reward=0.0,
            penalties=self.step_penalty,
        )
