"""Standardized benchmark evaluation and stratified metric computation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from orbit.agents.base import BaseAgent
from orbit.environments.base import BaseEnvironment, TaskSpec
from orbit.evaluation.statistics import compute_bootstrap_ci
from orbit.rollouts.collector import RolloutCollector


@dataclass(frozen=True)
class EvaluationResult:
    """Standardized benchmark evaluation report."""

    total_tasks: int
    pass_at_1: float
    mean_reward: float
    ci_95: tuple[float, float]
    mean_steps: float
    difficulty_stratified: dict[str, float] = field(default_factory=dict)
    category_stratified: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StandardEvaluator:
    """Standardized evaluator computing difficulty-stratified metrics and confidence intervals."""

    def __init__(self, run_id: str = "eval_standard"):
        self.run_id = run_id
        self.collector = RolloutCollector(default_run_id=run_id)

    def evaluate_agent(
        self,
        agent: BaseAgent,
        env: BaseEnvironment,
        tasks: list[TaskSpec],
        base_seed: int = 42,
    ) -> EvaluationResult:
        """Executes evaluation across benchmark task suite and computes stratified statistics."""
        trajectories, _stats = self.collector.collect_batch(
            agent=agent,
            env=env,
            tasks=tasks,
            run_id=self.run_id,
        )

        if not trajectories:
            return EvaluationResult(
                total_tasks=0,
                pass_at_1=0.0,
                mean_reward=0.0,
                ci_95=(0.0, 0.0),
                mean_steps=0.0,
            )

        total_tasks = len(trajectories)
        successes = [1.0 if t.success else 0.0 for t in trajectories]
        rewards = [t.total_reward for t in trajectories]
        steps = [t.num_steps for t in trajectories]

        pass_at_1 = sum(successes) / total_tasks
        mean_reward = sum(rewards) / total_tasks
        mean_steps = sum(steps) / total_tasks
        ci_95 = compute_bootstrap_ci(rewards, num_bootstraps=1000, seed=base_seed)

        # 1. Difficulty Tier Stratification
        # Bins: [0.0-0.25], [0.25-0.50], [0.50-0.75], [0.75-1.00]
        diff_bins: dict[str, list[float]] = {
            "tier_1_easy": [],
            "tier_2_medium": [],
            "tier_3_hard": [],
            "tier_4_expert": [],
        }

        # 2. Category Stratification
        cat_bins: dict[str, list[float]] = {}

        for task, traj in zip(tasks, trajectories, strict=False):
            diff = task.difficulty
            succ = 1.0 if traj.success else 0.0

            if diff <= 0.25:
                diff_bins["tier_1_easy"].append(succ)
            elif diff <= 0.50:
                diff_bins["tier_2_medium"].append(succ)
            elif diff <= 0.75:
                diff_bins["tier_3_hard"].append(succ)
            else:
                diff_bins["tier_4_expert"].append(succ)

            category = task.metadata.get("category", task.family)
            if category not in cat_bins:
                cat_bins[category] = []
            cat_bins[category].append(succ)

        stratified_diff: dict[str, float] = {}
        for k, v in diff_bins.items():
            stratified_diff[k] = float(np.mean(v)) if v else 0.0

        stratified_cat: dict[str, float] = {}
        for k, v in cat_bins.items():
            stratified_cat[k] = float(np.mean(v)) if v else 0.0

        return EvaluationResult(
            total_tasks=total_tasks,
            pass_at_1=pass_at_1,
            mean_reward=mean_reward,
            ci_95=ci_95,
            mean_steps=mean_steps,
            difficulty_stratified=stratified_diff,
            category_stratified=stratified_cat,
        )
