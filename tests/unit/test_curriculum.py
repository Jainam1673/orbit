import pytest

from orbit.curriculum import (
    AdaptiveFrontierCurriculum,
    CurriculumManager,
    DifficultyTracker,
    FixedDistributionCurriculum,
    LearningFrontierEstimator,
    StaticCurriculum,
)
from orbit.data.trajectory import Trajectory


def test_difficulty_tracker():
    tracker = DifficultyTracker(window_size=10, num_bins=4)

    # Record 2 easy successes (diff=0.1 -> bin 0)
    tracker.record_outcome(difficulty=0.1, success=True, reward=1.0, num_steps=1)
    tracker.record_outcome(difficulty=0.1, success=True, reward=1.0, num_steps=1)

    # Record 2 hard failures (diff=0.9 -> bin 3)
    tracker.record_outcome(difficulty=0.9, success=False, reward=0.0, num_steps=5)
    tracker.record_outcome(difficulty=0.9, success=False, reward=0.0, num_steps=5)

    assert pytest.approx(tracker.get_overall_success_rate()) == 0.5

    bin_rates = tracker.get_bin_success_rates()
    assert bin_rates[0] == 1.0
    assert bin_rates[1] is None  # empty bin
    assert bin_rates[2] is None  # empty bin
    assert bin_rates[3] == 0.0


def test_learning_frontier_estimator():
    tracker = DifficultyTracker()
    estimator = LearningFrontierEstimator(
        tracker=tracker,
        target_success_rate=0.6,
        learning_rate=0.1,
        min_difficulty=0.1,
        max_difficulty=1.0,
    )

    initial_frontier = estimator.current_frontier
    assert initial_frontier == 0.1

    # On success: delta = 0.1 * (1 - 0.6) = +0.04
    f1 = estimator.update_frontier(recent_success=True)
    assert pytest.approx(f1) == 0.14

    # On failure: delta = -0.1 * 0.6 = -0.06 -> clips to min_difficulty 0.1
    f2 = estimator.update_frontier(recent_success=False)
    assert pytest.approx(f2) == 0.1


def test_fixed_and_static_curricula():
    # Fixed Distribution
    fixed = FixedDistributionCurriculum(min_difficulty=0.2, max_difficulty=0.8, seed=42)
    t_fixed = fixed.sample_task()
    assert 0.2 <= t_fixed.difficulty <= 0.8
    assert fixed.state.total_tasks_generated == 1

    # Static Curriculum with stages
    stages = [(2, 0.2), (4, 0.6), (6, 1.0)]
    static = StaticCurriculum(stages=stages, seed=42)

    # Stage 1: count < 2 -> difficulty 0.2
    t1 = static.sample_task()
    assert pytest.approx(t1.difficulty) == 0.2
    traj1 = Trajectory("r1", "e1", t1.task_id, success=True)
    static.update(traj1)

    t2 = static.sample_task()
    assert pytest.approx(t2.difficulty) == 0.2
    static.update(Trajectory("r1", "e2", t2.task_id, success=True))

    # Stage 2: count >= 2 -> difficulty 0.6
    t3 = static.sample_task()
    assert pytest.approx(t3.difficulty) == 0.6


def test_adaptive_frontier_curriculum_progression():
    curriculum = AdaptiveFrontierCurriculum(
        target_success_rate=0.6,
        learning_rate=0.1,
        min_difficulty=0.1,
        max_difficulty=1.0,
        difficulty_std=0.01,
        seed=42,
    )

    # Sequence of 10 successes should increase difficulty frontier
    for i in range(10):
        task = curriculum.sample_task()
        traj = Trajectory("run", f"ep_{i}", task.task_id, success=True)
        curriculum.update(traj)

    metrics = curriculum.get_metrics()
    assert metrics["current_frontier_difficulty"] > 0.1
    assert metrics["recent_success_rate"] == 1.0


def test_curriculum_manager_and_decision_logging():
    manager = CurriculumManager(strategy="adaptive", seed=42)
    task1 = manager.sample_task()

    assert len(manager.decision_history) == 1
    decision = manager.decision_history[0]
    assert decision.sampled_task_id == task1.task_id
    assert decision.strategy == "adaptive"
    assert "difficulty" in decision.to_dict()

    manager.update(Trajectory("run", "ep_0", task1.task_id, success=True))
    metrics = manager.get_metrics()
    assert metrics["total_decisions_logged"] == 1
    assert metrics["strategy"] == "adaptive"
