"""Evaluation harness, statistical significance testing, and ablation runners."""

from orbit.evaluation.ablations import (
    AblationComparison,
    AblationCondition,
    AblationRunner,
)
from orbit.evaluation.evaluator import EvaluationResult, StandardEvaluator
from orbit.evaluation.statistics import (
    compute_bootstrap_ci,
    compute_cohens_d,
    compute_pass_at_k,
    compute_welch_t_test,
)

__all__ = [
    "AblationComparison",
    "AblationCondition",
    "AblationRunner",
    "EvaluationResult",
    "StandardEvaluator",
    "compute_bootstrap_ci",
    "compute_cohens_d",
    "compute_pass_at_k",
    "compute_welch_t_test",
]
