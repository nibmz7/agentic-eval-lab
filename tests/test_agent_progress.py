"""Streaming a turn step by step, and the progress lines the CLI prints."""

import pytest
from mcp import Client

from support_lab.agent import AgentSettings, McpToolbox, SupportAgent
from support_lab.agent.progress import describe_step
from support_lab.agent.runner import StepFinished, StepStarted, TurnResult
from support_lab.agent.scripted import ScriptedModel, text, tool_use
from support_lab.mcp_server import build_server

pytestmark = pytest.mark.anyio


async def stream_turn(store, model, message: str) -> list:
    async with Client(build_server(store)) as client:
        agent = await SupportAgent.create(
            model, McpToolbox(client), settings=AgentSettings(retry_delays=())
        )
        return [event async for event in agent.stream("t1", "u_bob", message)]


async def test_steps_stream_in_order_then_the_result(store):
    model = ScriptedModel(
        [
            [tool_use("process_refund", user_id="u_bob", amount=900)],
            [tool_use("escalate_ticket", user_id="u_bob", reason="over limit")],
            [text("A human agent will follow up.")],
        ]
    )

    events = await stream_turn(store, model, "Refund 900")

    steps = [
        (type(e).__name__, e.node) for e in events if isinstance(e, StepStarted | StepFinished)
    ]
    nodes = ["planner", "agent", "guardrail", "tools", "agent", "guardrail", "tools", "agent"]
    assert steps == [(kind, n) for n in nodes for kind in ("StepStarted", "StepFinished")]
    assert isinstance(events[-1], TurnResult)
    assert events[-1].reply == "A human agent will follow up."


async def test_progress_lines_for_a_blocked_refund(store):
    model = ScriptedModel(
        [
            [text("Let me check that."), tool_use("process_refund", user_id="u_bob", amount=900)],
            [tool_use("escalate_ticket", user_id="u_bob", reason="over limit")],
            [text("Escalated.")],
        ]
    )

    events = await stream_turn(store, model, "Refund 900")

    lines = [line for e in events if isinstance(e, StepFinished) for line in describe_step(e)]
    assert lines == [
        "planner   intent: general",
        'agent     says "Let me check that."',
        "agent     wants process_refund(user_id='u_bob', amount=900)",
        "guardrail checked check_balance(user_id='u_bob') → ok: active, balance 1500.00",
        "guardrail blocked refund: requires_escalation: blocked by guardrail, "
        "900 exceeds the 500.00 auto-approve limit",
        "tools     process_refund(user_id='u_bob', amount=900) → "
        "not run, blocked by guardrail (requires_escalation)",
        "agent     wants escalate_ticket(user_id='u_bob', reason='over limit')",
        "tools     escalate_ticket(user_id='u_bob', reason='over limit') → ok: ticket #1 opened",
        "agent     replied",
    ]


async def test_progress_line_for_an_allowed_refund(store):
    model = ScriptedModel(
        [[tool_use("process_refund", user_id="u_bob", amount=20)], [text("Done")]]
    )

    events = await stream_turn(store, model, "Refund 20")

    lines = [line for e in events if isinstance(e, StepFinished) for line in describe_step(e)]
    assert "guardrail refund allowed" in lines
    assert "tools     process_refund(user_id='u_bob', amount=20) → ok: new balance 1480.00" in lines
