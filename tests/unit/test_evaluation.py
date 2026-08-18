import pytest

from orbit.environments.math.environment import MathEnvironment
from orbit.environments.math.generator import MathTaskGenerator
from orbit.evaluation import (
    AblationCondition,
    AblationRunner,
    StandardEvaluator,
    compute_bootstrap_ci,
    compute_cohens_d,
    compute_pass_at_k,
    compute_welch_t_test,
)
from orbit.models.mock import MockModelClient


def test_compute_pass_at_k():
    assert compute_pass_at_k(n=10, c=10, k=1) == 1.0
    assert compute_pass_at_k(n=10, c=0, k=1) == 0.0
    assert pytest.approx(compute_pass_at_k(n=10, c=5, k=1)) == 0.5
    assert compute_pass_at_k(n=10, c=1, k=10) == 1.0

    with pytest.raises(ValueError, match="Total samples n=2 must be >= k=5"):
        compute_pass_at_k(n=2, c=1, k=5)


def test_compute_bootstrap_ci():
    data = [1.0, 1.0, 1.0, 0.0, 0.0]
    low, high = compute_bootstrap_ci(data, num_bootstraps=500, seed=42)
    assert 0.0 <= low <= 0.6 <= high <= 1.0


def test_cohens_d_and_welch_t_test():
    sample_a = [1.0, 1.0, 1.0, 1.0, 0.9]
    sample_b = [0.1, 0.2, 0.0, 0.1, 0.2]

    d = compute_cohens_d(sample_a, sample_b)
    assert d > 2.0  # Large positive effect size

    welch = compute_welch_t_test(sample_a, sample_b)
    assert welch["t_stat"] > 5.0
    assert welch["mean_diff"] > 0.7


def test_standard_evaluator_stratification():
    client = MockModelClient(default_response="\\boxed{42}")
    env = MathEnvironment()
    gen = MathTaskGenerator(seed=42)

    tasks = [
        gen.generate_task(difficulty=0.1, task_id="t1"),
        gen.generate_task(difficulty=0.4, task_id="t2"),
        gen.generate_task(difficulty=0.6, task_id="t3"),
        gen.generate_task(difficulty=0.9, task_id="t4"),
    ]

    from orbit.agents.base import AgentConfig
    from orbit.agents.reasoning import ReasoningAgent

    agent = ReasoningAgent(AgentConfig(), model_client=client)
    evaluator = StandardEvaluator(run_id="test_strat_eval")

    res = evaluator.evaluate_agent(agent=agent, env=env, tasks=tasks)

    assert res.total_tasks == 4
    assert "tier_1_easy" in res.difficulty_stratified
    assert "tier_4_expert" in res.difficulty_stratified
    assert res.ci_95 is not None


def test_ablation_runner_and_markdown_table():
    client = MockModelClient(default_response="\\boxed{42}")
    env = MathEnvironment()
    gen = MathTaskGenerator(seed=42)
    tasks = [gen.generate_task(difficulty=0.2), gen.generate_task(difficulty=0.5)]

    conditions = [
        AblationCondition("Baseline_Full", "Full agent with CoT", {}),
        AblationCondition("Ablation_No_Sys", "No system prompt", {"agent_config": {"system_prompt": ""}}),
    ]

    runner = AblationRunner()
    comparisons = runner.run_ablation_matrix(
        conditions=conditions,
        model_client=client,
        env=env,
        tasks=tasks,
    )

    assert len(comparisons) == 2
    assert comparisons[0].is_baseline is True
    assert comparisons[1].is_baseline is False

    md_table = runner.format_markdown_table(comparisons)
    assert "| Baseline_Full (Baseline) |" in md_table
    assert "| Ablation_No_Sys |" in md_table
