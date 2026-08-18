from orbit.agents.base import AgentConfig
from orbit.agents.reasoning import ReasoningAgent
from orbit.data.trajectory import Action, Observation, RewardBreakdown, Step, Trajectory
from orbit.models.mock import MockModelClient


def test_reasoning_agent_build_prompt():
    config = AgentConfig(
        agent_id="test_reasoner",
        system_prompt="You are a helpful math solver.",
    )
    client = MockModelClient()
    agent = ReasoningAgent(config=config, model_client=client)

    traj = Trajectory(run_id="r1", episode_id="e1", task_id="t1")
    traj.add_step(
        Step(
            step_index=0,
            observation=Observation(text="Solve 3x = 9"),
            action=Action(raw_text="Let's divide by 3."),
            reward=RewardBreakdown(),
            done=False,
        )
    )

    prompt = agent.build_prompt(Observation(text="x = ?"), trajectory=traj)
    assert "System: You are a helpful math solver." in prompt
    assert "Interaction History:" in prompt
    assert "Observation: Solve 3x = 9" in prompt
    assert "Action: Let's divide by 3." in prompt
    assert "Current Observation: x = ?" in prompt


def test_reasoning_agent_act():
    config = AgentConfig(agent_id="test_reasoner")
    client = MockModelClient(default_response="\\boxed{3}")
    agent = ReasoningAgent(config=config, model_client=client)

    obs = Observation(text="What is 9 / 3?")
    action = agent.act(obs)

    assert action.raw_text == "\\boxed{3}"
    assert action.metadata["model_id"] == "mock-model-v1"
    assert "latency_ms" in action.metadata
