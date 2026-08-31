import pytest

from orbit.agents.tools.repl import StatefulSymbolicREPLTool
from orbit.curriculum.difficulty import DifficultyTracker, LearningFrontierEstimator
from orbit.curriculum.strategies import RegretCurriculum
from orbit.data.trajectory import Action, Observation, RewardBreakdown, Step
from orbit.environments.base import TaskSpec
from orbit.rewards.process import StepProcessRewardFunction
from orbit.rewards.safety import RewardAnomalyDetector


def test_stateful_symbolic_repl_persistence_and_math():
    repl = StatefulSymbolicREPLTool()

    # 1. Define variable in turn 1
    res1 = repl.execute("x, y = symbols('x y')\na = 25\nb = 4")
    assert res1.success is True

    # 2. Use variable in turn 2
    res2 = repl.execute("a * b")
    assert res2.success is True
    assert "100" in res2.output

    # 3. Solve quadratic equation
    res3 = repl.execute("eq = x**2 - 7*x + 12\nsolve(eq, x)")
    assert res3.success is True
    assert "[3, 4]" in res3.output or "3, 4" in res3.output

    # 4. Matrix operations
    res4 = repl.execute("M = Matrix([[1, 2], [3, 4]])\nM.det()")
    assert res4.success is True
    assert "-2" in res4.output

    # 5. Security blocks
    res_sec = repl.execute("import os")
    assert res_sec.success is False
    assert "Security Violation" in res_sec.error

    # 6. Reset workspace
    repl.reset()
    res_after_reset = repl.execute("a")
    assert res_after_reset.success is False
    assert "Runtime Error" in res_after_reset.error


def test_learning_progress_derivative_and_frontier_pacing():
    tracker = DifficultyTracker(window_size=10, num_bins=2)

    # First half: all failures (0% success)
    for _ in range(4):
        tracker.record_outcome(difficulty=0.2, success=False, reward=0.0, num_steps=1)

    # Second half: all successes (100% success)
    for _ in range(4):
        tracker.record_outcome(difficulty=0.2, success=True, reward=1.0, num_steps=1)

    # Learning progress should be +1.0 (mean second half 1.0 - mean first half 0.0)
    lp = tracker.get_learning_progress(tracker.bin_records[0])
    assert lp == 1.0

    # Estimator in learning_progress mode
    estimator = LearningFrontierEstimator(tracker=tracker, mode="learning_progress")
    frontier = estimator.update_frontier(recent_success=True)
    assert frontier > 0.1


def test_information_theoretic_compression_anomaly_detector():
    detector = RewardAnomalyDetector(min_compression_ratio=0.25)

    # Normal reasoning string (natural high entropy)
    natural_text = "To find the root of the equation x^2 - 5x + 6 = 0, we factor it into (x - 2)(x - 3) = 0, giving x = 2 and x = 3."
    rep_nat = detector.analyze(natural_text)
    assert rep_nat.is_anomalous is False
    assert rep_nat.metadata["compression_ratio"] > 0.40

    # Low-entropy bloated repeating string
    bloated_text = "step step step step step step step step step step step step step step step step step step step step step step step step step step step step step step step step step step step step step step step step "
    rep_bloat = detector.analyze(bloated_text)
    assert rep_bloat.is_anomalous is True
    assert any("Information-theoretic anomaly" in v or "Repetition loop" in v for v in rep_bloat.violations)
    assert rep_bloat.metadata["compression_ratio"] < 0.25


def test_step_process_reward_function():
    proc_rf = StepProcessRewardFunction(subgoal_reward=0.1, success_reward=1.0)
    task = TaskSpec(task_id="t_proc", family="math", prompt="Solve 2x = 8", ground_truth="4")

    # 1. Non-terminal intermediate step with successful tool
    step_intermediate = Step(
        step_index=0,
        observation=Observation(text="Solve 2x = 8"),
        action=Action(raw_text="Calling tool", tool_call={"tool": "calculator", "args": {"expr": "8/2"}}),
        reward=RewardBreakdown(),
        done=False,
        info={"tool_result": {"success": True, "output": "4"}},
    )
    rb_step = proc_rf.compute_reward(step_intermediate, task)
    assert rb_step.shaping_reward == pytest.approx(0.1)
    assert rb_step.verifier_reward == 0.0

    # 2. Terminal step with correct answer
    step_term = Step(
        step_index=1,
        observation=Observation(text="Next step"),
        action=Action(raw_text="The solution is \\boxed{4}."),
        reward=RewardBreakdown(),
        done=True,
    )
    rb_term = proc_rf.compute_reward(step_term, task)
    assert rb_term.verifier_reward == pytest.approx(1.0)


def test_regret_curriculum_sampling():
    regret_curric = RegretCurriculum(seed=42)

    task = regret_curric.sample_task()
    assert task is not None
    assert task.difficulty >= 0.1
    assert regret_curric.state.total_tasks_generated == 1
