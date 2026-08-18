from orbit.agents.base import AgentConfig
from orbit.agents.long_horizon import LongHorizonAgent
from orbit.agents.memory import EpisodicMemory, MemoryEntry, WorkingMemory
from orbit.agents.tools.base import ToolRegistry
from orbit.agents.tools.calculator import PythonCalculatorTool
from orbit.data.trajectory import Observation
from orbit.models.mock import MockModelClient


def test_episodic_and_working_memory():
    # Episodic Memory
    ep_mem = EpisodicMemory()
    ep_mem.add("user", "What is 10 + 10?")
    ep_mem.add("assistant", "Thinking...")
    ep_mem.add("tool", "Result: 20")

    assert len(ep_mem) == 3
    tool_entries = ep_mem.get_by_role("tool")
    assert len(tool_entries) == 1
    assert tool_entries[0].content == "Result: 20"

    # Working Memory with capacity 2
    work_mem = WorkingMemory(max_entries=2)
    work_mem.add(MemoryEntry(role="user", content="Step 1"))
    work_mem.add(MemoryEntry(role="assistant", content="Step 2"))
    work_mem.add(MemoryEntry(role="user", content="Step 3"))

    assert len(work_mem) == 2
    ctx = work_mem.format_context()
    assert "Step 1" not in ctx
    assert "Step 2" in ctx
    assert "Step 3" in ctx


def test_python_calculator_tool_execution():
    tool = PythonCalculatorTool()

    # Basic arithmetic
    res1 = tool.execute({"expression": "15 * 12 + 4"})
    assert res1.success is True
    assert res1.output == "184"

    # Advanced functions
    res2 = tool.execute({"expression": "comb(10, 3) + gcd(48, 18)"})
    assert res2.success is True
    assert res2.output == "126"

    # Square root
    res3 = tool.execute({"expression": "sqrt(144)"})
    assert res3.success is True
    assert float(res3.output) == 12.0

    # Error handling on invalid syntax / div zero
    res_err = tool.execute({"expression": "10 / 0"})
    assert res_err.success is False
    assert "ZeroDivisionError" in str(res_err.error)


def test_python_calculator_tool_security_restrictions():
    tool = PythonCalculatorTool()

    # Attempt import
    res1 = tool.execute({"expression": "__import__('os').system('ls')"})
    assert res1.success is False
    assert "Evaluation error" in str(res1.error)

    # Attempt dunder attribute inspection
    res2 = tool.execute({"expression": "().___class___"})
    assert res2.success is False


def test_long_horizon_agent_tool_calling_and_reflection():
    # Sequence of mock responses:
    # 1. Calls calculator: [TOOL: calculator(expression="14 * 14")]
    # 2. Receives tool output 196, produces final answer \boxed{196}
    responses = [
        '[TOOL: calculator(expression="14 * 14")]',
        "The calculated value is 196. Final Answer: \\boxed{196}",
    ]
    call_idx = 0

    def custom_response(_prompt: str) -> str:
        nonlocal call_idx
        resp = responses[min(call_idx, len(responses) - 1)]
        call_idx += 1
        return resp

    client = MockModelClient(response_fn=custom_response)
    registry = ToolRegistry()
    registry.register(PythonCalculatorTool())

    agent = LongHorizonAgent(
        config=AgentConfig(agent_id="lh_agent"),
        model_client=client,
        tool_registry=registry,
    )

    obs = Observation(text="What is 14 squared?")
    action = agent.act(obs)

    assert "\\boxed{196}" in action.raw_text
    assert action.metadata["num_tool_calls"] == 1
    assert len(agent.episodic_memory) >= 3
