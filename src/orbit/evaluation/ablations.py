"""Systematic ablation runner and comparative statistical matrix generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from orbit.agents.base import AgentConfig
from orbit.agents.reasoning import ReasoningAgent
from orbit.environments.base import BaseEnvironment, TaskSpec
from orbit.evaluation.evaluator import StandardEvaluator
from orbit.evaluation.statistics import compute_cohens_d, compute_welch_t_test
from orbit.models.base import BaseModelClient


@dataclass(frozen=True)
class AblationCondition:
    """Configuration specification for an ablation variant."""

    name: str
    description: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class AblationComparison:
    """Statistical comparison of an ablation condition against the control baseline."""

    condition_name: str
    pass_at_1: float
    mean_reward: float
    ci_95: tuple[float, float]
    cohens_d_vs_baseline: float
    welch_t_stat: float
    is_baseline: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AblationRunner:
    """Executes multi-condition ablation matrices and calculates effect sizes."""

    def __init__(self, evaluator: StandardEvaluator | None = None):
        self.evaluator = evaluator or StandardEvaluator(run_id="ablation_study")

    def run_ablation_matrix(
        self,
        conditions: list[AblationCondition],
        model_client: BaseModelClient,
        env: BaseEnvironment,
        tasks: list[TaskSpec],
    ) -> list[AblationComparison]:
        """Runs evaluation over all ablation conditions and computes statistical differences."""
        if not conditions:
            return []

        results: list[tuple[str, list[float], float, tuple[float, float], float]] = []

        for cond in conditions:
            agent_cfg = AgentConfig(
                agent_id=f"agent_{cond.name}",
                **cond.overrides.get("agent_config", {}),
            )
            agent = ReasoningAgent(config=agent_cfg, model_client=model_client)

            eval_res = self.evaluator.evaluate_agent(
                agent=agent,
                env=env,
                tasks=tasks,
            )

            # Collect reward sample for statistical testing
            trajectories, _ = self.evaluator.collector.collect_batch(
                agent=agent,
                env=env,
                tasks=tasks,
                run_id=f"ablation_{cond.name}",
            )
            sample_rewards = [t.total_reward for t in trajectories]

            results.append(
                (
                    cond.name,
                    sample_rewards,
                    eval_res.pass_at_1,
                    eval_res.ci_95,
                    eval_res.mean_reward,
                )
            )

        baseline_name, baseline_rewards, base_pass, base_ci, base_reward = results[0]
        comparisons: list[AblationComparison] = [
            AblationComparison(
                condition_name=baseline_name,
                pass_at_1=base_pass,
                mean_reward=base_reward,
                ci_95=base_ci,
                cohens_d_vs_baseline=0.0,
                welch_t_stat=0.0,
                is_baseline=True,
            )
        ]

        for name, rewards, p1, ci, mean_r in results[1:]:
            d = compute_cohens_d(rewards, baseline_rewards)
            welch = compute_welch_t_test(rewards, baseline_rewards)
            comparisons.append(
                AblationComparison(
                    condition_name=name,
                    pass_at_1=p1,
                    mean_reward=mean_r,
                    ci_95=ci,
                    cohens_d_vs_baseline=d,
                    welch_t_stat=welch["t_stat"],
                    is_baseline=False,
                )
            )

        return comparisons

    def format_markdown_table(self, comparisons: list[AblationComparison]) -> str:
        """Formats ablation results into a clean markdown table for research documentation."""
        lines = [
            "| Condition | Pass@1 | Mean Reward | 95% CI | Cohen's d (vs. Base) | Welch's t |",
            "|---|---|---|---|---|---|",
        ]
        for c in comparisons:
            base_tag = " (Baseline)" if c.is_baseline else ""
            ci_str = f"[{c.ci_95[0]:.2f}, {c.ci_95[1]:.2f}]"
            d_str = f"{c.cohens_d_vs_baseline:+.2f}" if not c.is_baseline else "-"
            t_str = f"{c.welch_t_stat:+.2f}" if not c.is_baseline else "-"
            lines.append(
                f"| {c.condition_name}{base_tag} | {c.pass_at_1 * 100:.1f}% | {c.mean_reward:.3f} | {ci_str} | {d_str} | {t_str} |"
            )
        return "\n".join(lines)
