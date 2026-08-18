import pytest

from orbit.data.trajectory import (
    Action,
    Observation,
    Provenance,
    RewardBreakdown,
    Step,
    Trajectory,
)


def test_reward_breakdown_total():
    rb = RewardBreakdown(
        env_reward=1.0,
        verifier_reward=2.5,
        shaping_reward=0.2,
        critic_reward=0.0,
        penalties=0.1,
    )
    # 1.0 + 2.5 + 0.2 + 0.0 - 0.1 = 3.6
    assert pytest.approx(rb.total) == 3.6

    rb_dict = rb.to_dict()
    assert rb_dict["total"] == pytest.approx(3.6)

    reconstructed = RewardBreakdown.from_dict(rb_dict)
    assert reconstructed == rb


def test_step_lifecycle_and_serialization():
    obs = Observation(text="Calculate 2 + 2", state={"val": 4}, metadata={"env": "math"})
    action = Action(raw_text="Answer: 4", metadata={"tokens": 3})
    reward = RewardBreakdown(env_reward=1.0, verifier_reward=1.0)

    step = Step(
        step_index=0,
        observation=obs,
        action=action,
        reward=reward,
        done=True,
        truncated=False,
        info={"correct": True},
    )

    d = step.to_dict()
    assert d["step_index"] == 0
    assert d["observation"]["text"] == "Calculate 2 + 2"
    assert d["action"]["raw_text"] == "Answer: 4"
    assert d["reward"]["total"] == pytest.approx(2.0)
    assert d["done"] is True

    reconstructed = Step.from_dict(d)
    assert reconstructed.step_index == step.step_index
    assert reconstructed.observation.text == step.observation.text
    assert reconstructed.action.raw_text == step.action.raw_text
    assert reconstructed.reward.total == step.reward.total
    assert reconstructed.done == step.done


def test_trajectory_accumulation_and_json_roundtrip():
    prov = Provenance(
        git_commit="abc1234",
        git_dirty=False,
        model_version="test-model-v1",
        env_version="0.1.0",
        seed=42,
    )

    traj = Trajectory(
        run_id="run_001",
        episode_id="ep_001",
        task_id="task_math_01",
        provenance=prov,
        metadata={"split": "train"},
    )

    step0 = Step(
        step_index=0,
        observation=Observation(text="Step 1: Simplify 3x = 9"),
        action=Action(raw_text="Divide by 3"),
        reward=RewardBreakdown(shaping_reward=0.1),
        done=False,
    )
    step1 = Step(
        step_index=1,
        observation=Observation(text="Step 2: x = 3"),
        action=Action(raw_text="Final Answer: 3"),
        reward=RewardBreakdown(verifier_reward=1.0),
        done=True,
    )

    traj.add_step(step0)
    traj.add_step(step1)
    traj.success = True

    assert traj.num_steps == 2
    assert pytest.approx(traj.total_reward) == 1.1

    summed_rb = traj.reward_breakdown_sum
    assert pytest.approx(summed_rb.shaping_reward) == 0.1
    assert pytest.approx(summed_rb.verifier_reward) == 1.0
    assert pytest.approx(summed_rb.total) == 1.1

    # Invariant: Step index mismatch must raise ValueError
    with pytest.raises(ValueError, match="Step index mismatch"):
        traj.add_step(step0)

    # JSON roundtrip
    json_str = traj.to_json()
    reconstructed = Trajectory.from_json(json_str)

    assert reconstructed.run_id == traj.run_id
    assert reconstructed.episode_id == traj.episode_id
    assert reconstructed.task_id == traj.task_id
    assert reconstructed.num_steps == 2
    assert reconstructed.success is True
    assert pytest.approx(reconstructed.total_reward) == 1.1
    assert reconstructed.provenance.git_commit == "abc1234"
