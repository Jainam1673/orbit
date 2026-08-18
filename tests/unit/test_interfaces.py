import pytest

from orbit.agents.base import AgentConfig, BaseAgent
from orbit.curriculum.base import BaseCurriculum
from orbit.data.trajectory import Action, Observation, RewardBreakdown, Trajectory
from orbit.environments.base import BaseEnvironment, TaskSpec
from orbit.rewards.base import (
    BaseRewardFunction,
    BaseVerifier,
    VerificationResult,
)


def test_base_environment_abc():
    # Instantiating abstract class directly should raise TypeError
    with pytest.raises(TypeError):
        BaseEnvironment("test_env")  # type: ignore

    # Concrete implementation
    class MockEnvironment(BaseEnvironment):
        def reset(self, task=None, seed=None):
            return Observation(text="test observation"), {}

        def step(self, action):
            return (
                Observation(text="next observation"),
                RewardBreakdown(env_reward=1.0),
                True,
                False,
                {},
            )

    env = MockEnvironment("mock_env")
    obs, _info = env.reset()
    assert obs.text == "test observation"

    next_obs, reward, done, _trunc, _step_info = env.step(Action(raw_text="act"))
    assert next_obs.text == "next observation"
    assert reward.env_reward == 1.0
    assert done is True


def test_base_agent_abc():
    with pytest.raises(TypeError):
        BaseAgent(AgentConfig())  # type: ignore

    class MockAgent(BaseAgent):
        def act(self, observation, trajectory=None):
            return Action(raw_text=f"Response to: {observation.text}")

    agent = MockAgent(AgentConfig(agent_id="test_agent"))
    action = agent.act(Observation(text="hello"))
    assert action.raw_text == "Response to: hello"


def test_base_verifier_and_reward_abc():
    with pytest.raises(TypeError):
        BaseVerifier()  # type: ignore

    with pytest.raises(TypeError):
        BaseRewardFunction()  # type: ignore

    class ExactVerifier(BaseVerifier):
        def verify(self, task, trajectory):
            is_correct = (
                trajectory.steps[-1].action.raw_text == task.ground_truth
            )
            return VerificationResult(
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
            )

    verifier = ExactVerifier()
    task = TaskSpec(
        task_id="t1",
        family="math",
        prompt="1+1",
        ground_truth="2",
    )
    traj = Trajectory(run_id="r1", episode_id="e1", task_id="t1")
    from orbit.data.trajectory import Step

    traj.add_step(
        Step(
            step_index=0,
            observation=Observation(text="1+1"),
            action=Action(raw_text="2"),
            reward=RewardBreakdown(),
            done=True,
        )
    )

    res = verifier.verify(task, traj)
    assert res.is_correct is True
    assert res.score == 1.0


def test_base_curriculum_abc():
    with pytest.raises(TypeError):
        BaseCurriculum()  # type: ignore

    class StaticCurriculum(BaseCurriculum):
        def sample_task(self):
            return TaskSpec(task_id="task_0", family="toy", prompt="toy task")

        def update(self, trajectory):
            self.state.total_tasks_evaluated += 1

    curriculum = StaticCurriculum()
    task = curriculum.sample_task()
    assert task.task_id == "task_0"
    metrics = curriculum.get_metrics()
    assert metrics["total_tasks_evaluated"] == 0
