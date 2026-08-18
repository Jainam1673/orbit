"""Standardized model card and dataset card generators conforming to scientific release standards."""

from __future__ import annotations

from typing import Any


def generate_model_card(
    model_name: str,
    run_id: str,
    provenance: dict[str, Any],
    eval_metrics: dict[str, Any],
) -> str:
    """Generates a standardized Model Card."""
    lines = [
        f"# Model Card: {model_name}",
        "",
        "## Model Overview",
        f"- **Model Identifier**: `{model_name}`",
        f"- **Experiment Run ID**: `{run_id}`",
        "- **Training Framework**: ORBIT (Reinforcement Learning & Adaptive Curriculum)",
        f"- **Base Model**: `{provenance.get('model_version', 'N/A')}`",
        f"- **Date**: `{provenance.get('timestamp', 'N/A')}`",
        "",
        "## Intended Use",
        "- **Primary Intended Uses**: Multi-step mathematical reasoning, algorithmic task execution.",
        "- **Out-of-Scope Uses**: High-stakes decisions without human verification.",
        "",
        "## Training Details",
        f"- **Random Seed**: `{provenance.get('seed', 'N/A')}`",
        f"- **Git Commit**: `{provenance.get('git_commit', 'N/A')}`",
        f"- **Hardware Used**: `{provenance.get('gpu_name', 'CPU')}`",
        "",
        "## Evaluation Results",
        f"- **Pass@1**: {eval_metrics.get('pass_at_1', 0.0) * 100:.1f}%",
        f"- **Mean Reward**: {eval_metrics.get('mean_reward', 0.0):.3f}",
        f"- **95% Bootstrap CI**: `{eval_metrics.get('ci_95', 'N/A')}`",
        "",
    ]
    return "\n".join(lines)


def generate_dataset_card(
    dataset_name: str,
    total_tasks: int,
    categories: list[str],
    difficulty_range: tuple[float, float] = (0.0, 1.0),
) -> str:
    """Generates a standardized Dataset Card."""
    cats_str = ", ".join(f"`{c}`" for c in categories)
    lines = [
        f"# Dataset Card: {dataset_name}",
        "",
        "## Dataset Summary",
        f"- **Dataset Identifier**: `{dataset_name}`",
        f"- **Total Tasks**: {total_tasks}",
        f"- **Difficulty Range**: `[{difficulty_range[0]:.2f}, {difficulty_range[1]:.2f}]`",
        f"- **Task Categories**: {cats_str}",
        "",
        "## Dataset Structure",
        "- `task_id`: Unique canonical task identifier.",
        "- `family`: Task problem domain.",
        "- `prompt`: Textual mathematical question or reasoning goal.",
        "- `ground_truth`: Exact mathematical answer verified by symbolic/numerical equivalence.",
        "- `difficulty`: Continuous challenge rating in $[0.0, 1.0]$.",
        "",
        "## Validation & Integrity",
        (
            "All tasks are verified through multi-stage structural validation, SHA256 deduplication, "
            "and reward-leakage screening."
        ),
    ]
    return "\n".join(lines)
