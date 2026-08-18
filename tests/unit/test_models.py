import pytest

from orbit.models.base import GenerationConfig
from orbit.models.factory import get_model_client
from orbit.models.mock import MockModelClient


def test_mock_model_client_default_response():
    client = MockModelClient(model_id="test_mock", default_response="\\boxed{100}")
    output = client.generate("Calculate 10*10", GenerationConfig(max_tokens=64))

    assert output.text == "\\boxed{100}"
    assert output.completion_tokens == 1
    assert output.prompt_tokens == 2
    assert output.finish_reason == "stop"
    assert output.latency_ms >= 0.0
    assert len(output.logprobs) == 1

    logprobs = client.get_logprobs("Calculate 10*10", "\\boxed{100}")
    assert len(logprobs) == 1
    assert client.call_count == 1


def test_mock_model_client_response_fn():
    def custom_responder(prompt: str) -> str:
        if "2+2" in prompt:
            return "\\boxed{4}"
        return "\\boxed{0}"

    client = MockModelClient(response_fn=custom_responder)
    out1 = client.generate("What is 2+2?")
    assert out1.text == "\\boxed{4}"

    out2 = client.generate("What is 5*5?")
    assert out2.text == "\\boxed{0}"


def test_model_factory():
    client = get_model_client("mock", model_id="mock_v1", default_response="test")
    assert isinstance(client, MockModelClient)
    assert client.model_id == "mock_v1"

    with pytest.raises(ValueError, match="Unsupported model provider"):
        get_model_client("unsupported_provider", model_id="foo")
