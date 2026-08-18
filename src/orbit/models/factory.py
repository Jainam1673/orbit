"""Model client factory and discovery for ORBIT."""

from __future__ import annotations

from typing import Any

from orbit.models.base import BaseModelClient
from orbit.models.mock import MockModelClient


def get_model_client(
    provider: str,
    model_id: str = "default_model",
    **kwargs: Any,
) -> BaseModelClient:
    """Factory function to instantiate model clients by provider name.

    Supported providers:
    - 'mock': MockModelClient
    - 'huggingface' / 'hf': HuggingFaceModelClient
    """
    prov = provider.lower()
    if prov == "mock":
        return MockModelClient(model_id=model_id, **kwargs)
    elif prov in ("huggingface", "hf"):
        from orbit.models.hf import HuggingFaceModelClient

        return HuggingFaceModelClient(model_id=model_id, **kwargs)
    else:
        raise ValueError(
            f"Unsupported model provider '{provider}'. Supported: ['mock', 'huggingface']"
        )
