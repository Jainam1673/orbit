"""Empirical difficulty tracking and learning frontier estimation for ORBIT."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DifficultyRecord:
    """Historical outcome recorded for a task difficulty."""

    difficulty: float
    success: bool
    reward: float
    num_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)


class DifficultyTracker:
    """Tracks rolling window task outcomes and computes empirical success rates."""

    def __init__(
        self,
        window_size: int = 20,
        num_bins: int = 4,
    ):
        self.window_size = window_size
        self.num_bins = num_bins
        self.records: deque[DifficultyRecord] = deque(maxlen=window_size)
        self.bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        self.bin_records: list[deque[DifficultyRecord]] = [
            deque(maxlen=window_size) for _ in range(num_bins)
        ]

    def record_outcome(
        self,
        difficulty: float,
        success: bool,
        reward: float,
        num_steps: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Records an episode outcome."""
        rec = DifficultyRecord(
            difficulty=difficulty,
            success=success,
            reward=reward,
            num_steps=num_steps,
            metadata=metadata or {},
        )
        self.records.append(rec)

        # Assign to difficulty bin
        bin_idx = min(
            int(np.digitize(difficulty, self.bin_edges) - 1),
            self.num_bins - 1,
        )
        bin_idx = max(0, bin_idx)
        self.bin_records[bin_idx].append(rec)

    def get_overall_success_rate(self) -> float:
        """Computes rolling success rate across all difficulties."""
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.success) / len(self.records)

    def get_bin_success_rates(self) -> list[float | None]:
        """Computes success rates per difficulty bin, returning None if bin is empty."""
        rates: list[float | None] = []
        for b in self.bin_records:
            if not b:
                rates.append(None)
            else:
                rates.append(sum(1 for r in b if r.success) / len(b))
        return rates


class LearningFrontierEstimator:
    """Estimates the agent's active learning frontier difficulty d*."""

    def __init__(
        self,
        tracker: DifficultyTracker,
        target_success_rate: float = 0.6,
        learning_rate: float = 0.05,
        min_difficulty: float = 0.1,
        max_difficulty: float = 1.0,
    ):
        self.tracker = tracker
        self.target_success_rate = target_success_rate
        self.learning_rate = learning_rate
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.current_frontier: float = min_difficulty

    def update_frontier(self, recent_success: bool) -> float:
        """Dynamically adjusts frontier difficulty based on recent success/failure.

        If the agent succeeds, difficulty increases towards target challenge;
        If the agent fails, difficulty eases down.
        """
        # Directional update: success -> +lr * (1 - target), failure -> -lr * target
        if recent_success:
            delta = self.learning_rate * (1.0 - self.target_success_rate)
        else:
            delta = -self.learning_rate * self.target_success_rate

        self.current_frontier = float(
            np.clip(
                self.current_frontier + delta,
                self.min_difficulty,
                self.max_difficulty,
            )
        )
        return self.current_frontier
