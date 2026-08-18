"""Abstract base classes and registry for isolated agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """Standardized output returned by a tool invocation."""

    success: bool
    output: str
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseTool(ABC):
    """Abstract interface for agent-executable tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema specification of accepted arguments."""
        raise NotImplementedError("Subclasses must declare parameters schema")

    @abstractmethod
    def execute(self, args: dict[str, Any]) -> ToolResult:
        """Executes the tool with validated arguments."""
        raise NotImplementedError("Subclasses must implement execute()")


class ToolRegistry:
    """Registry managing available tools and schemas for agents."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Registers a tool instance."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Retrieves a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Lists all registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Returns JSON schemas for all registered tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]
