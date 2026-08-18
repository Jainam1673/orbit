"""Long-horizon reasoning agent with modular memory, tool execution, and error reflection."""

from __future__ import annotations

import json
import re
from typing import Any

from orbit.agents.base import AgentConfig, BaseAgent
from orbit.agents.memory import EpisodicMemory, WorkingMemory
from orbit.agents.tools.base import ToolRegistry
from orbit.agents.tools.calculator import PythonCalculatorTool
from orbit.data.trajectory import Action, Observation, Trajectory
from orbit.models.base import BaseModelClient


class LongHorizonAgent(BaseAgent):
    """Long-horizon agent capable of multi-step planning, tool invocation, and reflection."""

    def __init__(
        self,
        config: AgentConfig,
        model_client: BaseModelClient,
        tool_registry: ToolRegistry | None = None,
        max_internal_tool_steps: int = 3,
    ):
        super().__init__(config=config)
        self.model_client = model_client
        self.episodic_memory = EpisodicMemory()
        self.working_memory = WorkingMemory(max_entries=config.max_steps * 2)
        self.max_tool_steps = max_internal_tool_steps

        # Initialize default tools if not provided
        self.tools = tool_registry or ToolRegistry()
        if not self.tools.list_tools():
            self.tools.register(PythonCalculatorTool())

    def reset(self) -> None:
        """Clears working and episodic memory between interaction episodes."""
        self.episodic_memory.clear()
        self.working_memory.clear()

    def build_prompt(
        self,
        observation: Observation,
        trajectory: Trajectory | None = None,
    ) -> str:
        """Constructs prompt with system instructions, tool schemas, history, and observation."""
        sections: list[str] = []

        system_msg = self.config.system_prompt or (
            "You are an expert long-horizon reasoning agent with tool access.\n"
            "To use a tool, output: [TOOL: tool_name({\"arg\": \"val\"})]\n"
            "When you have the final solution, provide it in \\boxed{answer}."
        )
        sections.append(f"System: {system_msg}")

        # Available tools
        schemas = self.tools.get_schemas()
        if schemas:
            sections.append(f"Available Tools:\n{json.dumps(schemas, indent=2)}")

        # Interaction History from memory or trajectory
        if len(self.episodic_memory) > 0:
            sections.append(f"Episode History:\n{self.working_memory.format_context()}")
        elif trajectory is not None and trajectory.steps:
            sections.append("Interaction History:")
            for s in trajectory.steps:
                sections.append(f"Observation: {s.observation.text}")
                sections.append(f"Action: {s.action.raw_text}")

        sections.append(f"Current Observation: {observation.text}")
        sections.append("Assistant:")
        return "\n\n".join(sections)

    def extract_tool_call(self, text: str) -> tuple[str | None, dict[str, Any] | None]:
        """Parses [TOOL: name({"arg": "val"})] from agent output."""
        pattern = r"\[TOOL:\s*([a-zA-Z0-9_]+)\s*\((.*?)\)\s*\]"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None, None

        tool_name = match.group(1).strip()
        raw_args = match.group(2).strip()

        # Parse arguments (JSON format or key="value")
        if raw_args.startswith("{") and raw_args.endswith("}"):
            try:
                args = json.loads(raw_args)
                return tool_name, args
            except json.JSONDecodeError:
                pass

        # Fallback to key="value" or expression argument
        if "=" in raw_args:
            parts = raw_args.split("=", 1)
            key = parts[0].strip()
            val = parts[1].strip().strip('"').strip("'")
            return tool_name, {key: val}

        return tool_name, {"expression": raw_args.strip('"').strip("'")}

    def act(
        self,
        observation: Observation,
        trajectory: Trajectory | None = None,
    ) -> Action:
        """Executes reasoning, tool calls with reflection, and returns final environment action."""
        self.episodic_memory.add("user", observation.text)

        current_prompt = self.build_prompt(observation, trajectory)
        last_action_text = ""
        tool_invocations: list[dict[str, Any]] = []

        for _ in range(self.max_tool_steps):
            output = self.model_client.generate(
                prompt=current_prompt,
                config=self.config.generation_config,
            )
            response_text = output.text.strip()
            last_action_text = response_text

            tool_name, tool_args = self.extract_tool_call(response_text)
            if tool_name is None:
                # No tool call, output is final response
                break

            # Execute tool
            tool = self.tools.get(tool_name)
            if tool is None:
                tool_output_str = f"Error: Tool '{tool_name}' not found."
                is_success = False
            else:
                res = tool.execute(tool_args or {})
                tool_output_str = res.output if res.success else f"Error: {res.error}"
                is_success = res.success

            tool_invocations.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "success": is_success,
                    "result": tool_output_str,
                }
            )

            # Record in memory
            self.episodic_memory.add("assistant", response_text)
            self.episodic_memory.add("tool", f"Tool '{tool_name}' returned: {tool_output_str}")

            # If tool execution failed, add reflection guidance to prompt
            reflection_note = ""
            if not is_success:
                reflection_note = (
                    "\nReflection: The previous tool call failed. Carefully inspect the error "
                    "and self-correct your reasoning."
                )

            current_prompt += (
                f"\n\nAssistant: {response_text}\n"
                f"Tool Result: {tool_output_str}"
                f"{reflection_note}\n\n"
                "Assistant:"
            )

        metadata: dict[str, Any] = {
            "tool_invocations": tool_invocations,
            "num_tool_calls": len(tool_invocations),
            "model_id": self.model_client.model_id,
        }

        return Action(
            raw_text=last_action_text,
            tool_call=tool_invocations[0] if tool_invocations else None,
            metadata=metadata,
        )
