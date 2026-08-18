"""Episodic and working memory systems for long-horizon agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class MemoryEntry:
    """Single item stored in agent memory (observation, thought, tool result, reflection)."""

    role: str  # 'system', 'user', 'assistant', 'tool', 'reflection'
    content: str
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EpisodicMemory:
    """Persistent episodic log storing full chronological interaction history."""

    def __init__(self) -> None:
        self.entries: list[MemoryEntry] = []

    def add(
        self,
        role: str,
        content: str,
        step_index: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Appends a new memory entry."""
        entry = MemoryEntry(
            role=role,
            content=content,
            step_index=step_index,
            metadata=metadata or {},
        )
        self.entries.append(entry)

    def get_by_role(self, role: str) -> list[MemoryEntry]:
        """Filters memory entries by role."""
        return [e for e in self.entries if e.role == role]

    def clear(self) -> None:
        """Clears all entries."""
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)


class WorkingMemory:
    """Sliding-window short-term memory buffer managing active prompt context."""

    def __init__(self, max_entries: int = 20):
        self.max_entries = max_entries
        self.entries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        """Adds entry, evicting oldest entries if capacity is exceeded."""
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def format_context(self) -> str:
        """Formats working memory entries into a prompt string."""
        formatted: list[str] = []
        for e in self.entries:
            prefix = e.role.capitalize()
            formatted.append(f"{prefix}: {e.content}")
        return "\n\n".join(formatted)

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)
