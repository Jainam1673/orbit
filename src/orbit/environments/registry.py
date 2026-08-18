"""Environment registry and factory for ORBIT environments."""

from __future__ import annotations

from typing import Any

from orbit.environments.base import BaseEnvironment
from orbit.environments.math.environment import MathEnvironment

_REGISTRY: dict[str, type[BaseEnvironment]] = {}


def register_environment(env_id: str, env_cls: type[BaseEnvironment]) -> None:
    """Registers an environment class under a unique identifier."""
    if env_id in _REGISTRY:
        raise ValueError(f"Environment '{env_id}' is already registered.")
    _REGISTRY[env_id] = env_cls


def make_environment(env_id: str, **kwargs: Any) -> BaseEnvironment:
    """Instantiates a registered environment by ID."""
    if env_id not in _REGISTRY:
        raise ValueError(
            f"Unknown environment '{env_id}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[env_id](**kwargs)


def list_environments() -> list[str]:
    """Returns a list of all registered environment IDs."""
    return list(_REGISTRY.keys())


# Default built-in registrations
register_environment("math_reasoning", MathEnvironment)
