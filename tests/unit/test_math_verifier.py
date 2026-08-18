import pytest

from orbit.data.trajectory import Action, Observation, RewardBreakdown, Step, Trajectory
from orbit.environments.base import TaskSpec
from orbit.rewards.math_verifier import (
    MathRewardFunction,
    MathVerifier,
    are_math_answers_equivalent,
    extract_boxed_answer,
    extract_math_answer,
    normalize_numeric_string,
)


def test_extract_boxed_answer():
    assert extract_boxed_answer("The result is \\boxed{42}.") == "42"
    assert extract_boxed_answer("Nested: \\boxed{\\frac{1}{2}} here") == "\\frac{1}{2}"
    assert extract_boxed_answer("No boxed content here") is None
    assert extract_boxed_answer("Multiple \\boxed{1} and \\boxed{2}") == "2"


def test_extract_math_answer_formats():
    assert extract_math_answer("Final Answer: 15") == "15"
    assert extract_math_answer("Therefore, the answer is: -7.5") == "-7.5"
    assert extract_math_answer("Step by step...\n#### 128\n") == "128"
    assert extract_math_answer("Let's calculate.\n\\boxed{3/4}") == "3/4"
    assert extract_math_answer("42") == "42"


def test_normalize_numeric_string():
    assert normalize_numeric_string("100") == 100
    assert normalize_numeric_string("-42") == -42
    assert normalize_numeric_string("3.1415") == 3.1415
    assert normalize_numeric_string("$1,000") == 1000
    assert str(normalize_numeric_string("3/4")) == "3/4"
    assert normalize_numeric_string("invalid_symbol") is None


def test_are_math_answers_equivalent():
    # Exact / integer
    assert are_math_answers_equivalent("42", "42") is True
    assert are_math_answers_equivalent("42", 42) is True
    assert are_math_answers_equivalent("-5", -5) is True

    # Fractions
    assert are_math_answers_equivalent("2/4", "1/2") is True
    assert are_math_answers_equivalent("0.5", "1/2") is True
    assert are_math_answers_equivalent("1/2", 0.5) is True

    # Float tolerances
    assert are_math_answers_equivalent("3.14159", "3.14158") is True
    assert are_math_answers_equivalent("10.0", "15.0") is False

    # None / invalid
    assert are_math_answers_equivalent(None, "42") is False


def test_math_verifier_and_reward_function():
    verifier = MathVerifier()
    reward_fn = MathRewardFunction(verifier=verifier, success_reward=2.0, step_penalty=0.05)

    task = TaskSpec(
        task_id="test_math_1",
        family="math",
        prompt="Solve 2x = 8",
        ground_truth="4",
    )

    traj_correct = Trajectory(run_id="r1", episode_id="e1", task_id="test_math_1")
    step_correct = Step(
        step_index=0,
        observation=Observation(text="Solve 2x = 8"),
        action=Action(raw_text="The solution is \\boxed{4}."),
        reward=RewardBreakdown(),
        done=True,
    )
    traj_correct.add_step(step_correct)

    res = verifier.verify(task, traj_correct)
    assert res.is_correct is True
    assert res.score == 1.0

    rb = reward_fn.compute_reward(step_correct, task, traj_correct)
    assert rb.verifier_reward == 2.0
    assert rb.penalties == 0.05
    assert pytest.approx(rb.total) == 1.95
