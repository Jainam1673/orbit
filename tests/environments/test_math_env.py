import pytest

from orbit.data.trajectory import Action
from orbit.environments.base import TaskSpec
from orbit.environments.math.environment import MathEnvironment
from orbit.environments.math.generator import MathTaskGenerator
from orbit.environments.registry import list_environments, make_environment


def test_math_task_generator_difficulty_tiers():
    gen = MathTaskGenerator(seed=42)

    # Tier 1: Arithmetic [0.0 - 0.25)
    t1 = gen.generate_task(difficulty=0.1)
    assert t1.family == "math_arithmetic"
    assert t1.ground_truth is not None

    # Tier 2: Linear Algebra [0.25 - 0.50)
    t2 = gen.generate_task(difficulty=0.35)
    assert t2.family == "math_algebra"
    assert "x" in t2.prompt

    # Tier 3: Quadratic [0.50 - 0.75)
    t3 = gen.generate_task(difficulty=0.6)
    assert t3.family == "math_quadratic"
    assert "quadratic" in t3.prompt

    # Tier 4: Discrete / Modular [0.75 - 1.0]
    t4 = gen.generate_task(difficulty=0.85)
    assert t4.family == "math_discrete"
    assert t4.ground_truth is not None


def test_math_task_generator_reproducibility():
    gen1 = MathTaskGenerator(seed=12345)
    gen2 = MathTaskGenerator(seed=12345)

    tasks1 = [gen1.generate_task(difficulty=d).prompt for d in [0.1, 0.4, 0.7, 0.9]]
    tasks2 = [gen2.generate_task(difficulty=d).prompt for d in [0.1, 0.4, 0.7, 0.9]]

    assert tasks1 == tasks2


def test_math_environment_step_lifecycle():
    env = MathEnvironment(max_steps=3)
    task = TaskSpec(
        task_id="custom_task",
        family="math_algebra",
        prompt="Solve x + 5 = 10",
        ground_truth="5",
        difficulty=0.3,
    )

    obs, info = env.reset(task=task)
    assert obs.text == "Solve x + 5 = 10"
    assert info["task_id"] == "custom_task"

    # Step 1: Intermediate reasoning step (no answer yet)
    obs, reward, terminated, truncated, step_info = env.step(
        Action(raw_text="Let's subtract 5 from both sides.")
    )
    assert terminated is False
    assert truncated is False
    assert reward.verifier_reward == 0.0
    assert step_info["extracted_answer"] is None

    # Step 2: Final answer step
    obs, reward, terminated, truncated, step_info = env.step(
        Action(raw_text="The solution is \\boxed{5}.")
    )
    assert terminated is True
    assert truncated is False
    assert reward.verifier_reward == 1.0
    assert step_info["is_correct"] is True


def test_math_environment_truncation_on_max_steps():
    env = MathEnvironment(max_steps=2)
    task = TaskSpec(
        task_id="trunc_task",
        family="math",
        prompt="Find answer",
        ground_truth="100",
    )

    env.reset(task=task)
    _, _, term1, trunc1, _ = env.step(Action(raw_text="Step 1 thinking..."))
    assert term1 is False and trunc1 is False

    _, _, term2, trunc2, _ = env.step(Action(raw_text="Step 2 still thinking..."))
    assert term2 is False
    assert trunc2 is True  # Truncated after exceeding max_steps


def test_environment_registry():
    registered = list_environments()
    assert "math_reasoning" in registered

    env = make_environment("math_reasoning", max_steps=5)
    assert isinstance(env, MathEnvironment)
    assert env.max_steps == 5

    with pytest.raises(ValueError, match="Unknown environment"):
        make_environment("non_existent_env")
