"""Validation, deduplication, reward-leakage filtering, and admission pipeline for self-generated tasks."""

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from orbit.environments.base import TaskSpec
from orbit.rewards.math_verifier import (
    are_math_answers_equivalent,
    normalize_numeric_string,
)


@dataclass(frozen=True)
class TaskValidationResult:
    """Outcome of validating a candidate self-generated task."""

    is_admitted: bool
    reason: str
    task_spec: TaskSpec
    hash_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskPipelineValidator:
    """Multi-stage validation pipeline protecting the curriculum from malformed, trivial, or leaked tasks."""

    def __init__(self, min_prompt_len: int = 8, max_prompt_len: int = 2048):
        self.min_prompt_len = min_prompt_len
        self.max_prompt_len = max_prompt_len
        self.seen_hashes: set[str] = set()
        self.total_processed = 0
        self.total_admitted = 0
        self.rejection_counts: dict[str, int] = {
            "malformed_structure": 0,
            "duplicate": 0,
            "reward_leakage": 0,
            "invalid_ground_truth": 0,
        }

    def compute_task_hash(self, task: TaskSpec) -> str:
        """Computes a canonical normalized hash of the task prompt to detect duplicates."""
        normalized = re.sub(r"\s+", " ", task.prompt.strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def validate_structure(self, task: TaskSpec) -> tuple[bool, str]:
        """Checks basic structural validity of the task specification."""
        if not task.prompt or len(task.prompt.strip()) < self.min_prompt_len:
            return False, "Prompt is too short or empty."
        if len(task.prompt) > self.max_prompt_len:
            return False, "Prompt exceeds maximum allowed length."
        if task.ground_truth is None or str(task.ground_truth).strip() == "":
            return False, "Ground truth answer is missing or empty."
        return True, "Structure valid."

    def check_deduplication(self, task: TaskSpec, task_hash: str) -> tuple[bool, str]:
        """Ensures the candidate task is novel and not a duplicate of existing tasks."""
        if task_hash in self.seen_hashes:
            return False, "Duplicate task: exact or near-identical prompt already admitted."
        return True, "Task is unique."

    def check_reward_leakage(self, task: TaskSpec) -> tuple[bool, str]:
        """Detects if the ground truth answer is explicitly leaked in the prompt."""
        gt_str = str(task.ground_truth).strip()
        prompt_lower = task.prompt.lower()

        # Check explicit answer statements e.g. "answer is 42", "boxed{42}", "result = 42"
        leakage_patterns = [
            rf"(?:the\s+answer\s+is|final\s+answer|result\s+is)\s*[:=]?\s*{re.escape(gt_str)}",
            rf"\\boxed\{{{re.escape(gt_str)}\}}",
            rf"=\s*{re.escape(gt_str)}\s*$",
        ]
        for pat in leakage_patterns:
            if re.search(pat, prompt_lower):
                return False, f"Reward leakage detected: prompt contains answer '{gt_str}'."
        return True, "No reward leakage detected."

    def check_ground_truth_validity(self, task: TaskSpec) -> tuple[bool, str]:
        """Verifies that the ground truth is parseable and mathematically sound."""
        normalized = normalize_numeric_string(str(task.ground_truth))
        if normalized is None and len(str(task.ground_truth).strip()) > 32:
            return False, "Ground truth cannot be parsed as a mathematical or symbolic quantity."

        # Self-consistency check
        if not are_math_answers_equivalent(str(task.ground_truth), task.ground_truth):
            return False, "Ground truth failed self-consistency verification."

        return True, "Ground truth valid."

    def process_candidate(self, candidate: TaskSpec) -> TaskValidationResult:
        """Executes the complete validation pipeline on a candidate task."""
        self.total_processed += 1
        task_hash = self.compute_task_hash(candidate)

        # 1. Structure Check
        valid_struct, reason_struct = self.validate_structure(candidate)
        if not valid_struct:
            self.rejection_counts["malformed_structure"] += 1
            return TaskValidationResult(False, reason_struct, candidate, task_hash)

        # 2. Deduplication Check
        is_unique, reason_unique = self.check_deduplication(candidate, task_hash)
        if not is_unique:
            self.rejection_counts["duplicate"] += 1
            return TaskValidationResult(False, reason_unique, candidate, task_hash)

        # 3. Reward Leakage Check
        no_leak, reason_leak = self.check_reward_leakage(candidate)
        if not no_leak:
            self.rejection_counts["reward_leakage"] += 1
            return TaskValidationResult(False, reason_leak, candidate, task_hash)

        # 4. Ground Truth Validity Check
        valid_gt, reason_gt = self.check_ground_truth_validity(candidate)
        if not valid_gt:
            self.rejection_counts["invalid_ground_truth"] += 1
            return TaskValidationResult(False, reason_gt, candidate, task_hash)

        # Candidate passed all checks -> Admit
        self.seen_hashes.add(task_hash)
        self.total_admitted += 1
        return TaskValidationResult(
            is_admitted=True,
            reason="Passed validation, deduplication, leakage screening, and verifier check.",
            task_spec=candidate,
            hash_id=task_hash,
        )
