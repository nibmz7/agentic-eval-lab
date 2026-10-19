"""What AnthropicModel sends to the Messages API, checked against a mock HTTP
transport. No network and no API key needed."""

import json

import anthropic
import httpx2
import pytest

from support_lab.agent.model import DEFAULT_MODEL, FALLBACK_BETA, AnthropicModel, Intent

pytestmark = pytest.mark.anyio

THINKING = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
TOOL_USE = {"type": "tool_use", "id": "toolu_1", "name": "check_balance",
            "input": {"user_id": "u_alice"}}  # fmt: skip


def fake_api(content: list[dict], stop_reason: str):
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": DEFAULT_MODEL,
                "content": content,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    client = anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicModel(client), requests


async def test_respond_request_shape_and_blocks_round_trip_unchanged():
    model, requests = fake_api([THINKING, TOOL_USE], "tool_use")
    tools = [{"name": "check_balance", "description": "d", "input_schema": {"type": "object"}}]

    turn = await model.respond(
        system="sys", messages=[{"role": "user", "content": "hi"}], tools=tools
    )

    (request,) = requests
    body = json.loads(request.content)
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    assert body["model"] == DEFAULT_MODEL
    assert body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "medium"}
    assert "tool_choice" not in body  # forced tool choice is rejected on this model
    assert "thinking" not in body  # adaptive thinking is always on; nothing to set
    assert body["tools"] == tools
    # Thinking blocks must go back to the API exactly as they came.
    assert turn.content == [THINKING, TOOL_USE]
    assert turn.stop_reason == "tool_use"


async def test_classify_uses_structured_output():
    model, requests = fake_api(
        [{"type": "text", "text": '{"intent": "refund_request"}'}], "end_turn"
    )

    intent = await model.classify("user: I want my money back")

    body = json.loads(requests[0].content)
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["fallbacks"] == "default"
    assert intent is Intent.REFUND_REQUEST


async def test_classify_falls_back_to_general_on_refusal():
    model, _ = fake_api([], "refusal")
    assert await model.classify("...") is Intent.GENERAL
