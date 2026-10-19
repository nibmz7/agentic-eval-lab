"""The LangGraph agent, driven by a scripted model against the real MCP server.

The model is fake; everything else is real: graph, guardrail, retries, MCP
calls and the SQLite store. Assertions check the store's end state, not just
what the agent said.
"""

from contextlib import AsyncExitStack
from decimal import Decimal

import pytest
from mcp import Client

from support_lab.agent import (
    AgentSettings,
    AgentStepLimitError,
    Intent,
    McpToolbox,
    ModelTurn,
    SupportAgent,
)
from support_lab.agent.graph import DECLINE_REPLY, REFUSAL_REPLY
from support_lab.agent.scripted import ScriptedModel, last_tool_results, text, tool_use
from support_lab.backend import SupportStore
from support_lab.backend.faults import FlakyStore
from support_lab.backend.store import REFUND_AUTO_APPROVE_LIMIT_CENTS
from support_lab.mcp_server import build_server

pytestmark = pytest.mark.anyio

FAST = AgentSettings(retry_delays=())


@pytest.fixture
def flaky_store():
    s = FlakyStore.seeded()
    yield s
    s.close()


@pytest.fixture
async def make_agent(store):
    async with AsyncExitStack() as stack:

        async def factory(model, *, backend: SupportStore = store, step_limit: int = 30):
            client = await stack.enter_async_context(Client(build_server(backend)))
            return await SupportAgent.create(
                model, McpToolbox(client), settings=FAST, step_limit=step_limit
            )

        yield factory


def names(result) -> list[tuple[str, str]]:
    return [(c["source"], c["name"]) for c in result.tool_calls]


async def test_balance_inquiry(make_agent):
    model = ScriptedModel(
        [[tool_use("check_balance", user_id="u_alice")], [text("Your balance is 120.00.")]],
        intent=Intent.BALANCE_INQUIRY,
    )
    agent = await make_agent(model)

    result = await agent.send("t1", "u_alice", "What's my balance?")

    assert result.reply == "Your balance is 120.00."
    assert result.intent == "balance_inquiry"
    assert names(result) == [("model", "check_balance")]
    (tool_result,) = last_tool_results(model.calls[1])
    assert '"balance": "120.00"' in tool_result["content"]


async def test_refund_runs_with_an_idempotency_key(make_agent, store):
    model = ScriptedModel(
        [[tool_use("process_refund", user_id="u_alice", amount=20)], [text("Refunded 20.00.")]]
    )
    agent = await make_agent(model)

    result = await agent.send("t1", "u_alice", "Please refund 20")

    assert store.get_account("u_alice").balance_cents == 100_00
    assert names(result) == [("guardrail", "check_balance"), ("model", "process_refund")]
    refund = result.tool_calls[-1]
    assert refund["input"]["idempotency_key"] == f"agent-{refund['tool_use_id']}"


async def test_guardrail_blocks_refund_over_limit_and_agent_escalates(make_agent, store):
    model = ScriptedModel(
        [
            [tool_use("process_refund", user_id="u_bob", amount=900)],
            [tool_use("escalate_ticket", user_id="u_bob", reason="refund of 900 over limit")],
            [text("I've passed this to a human agent.")],
        ]
    )
    agent = await make_agent(model)

    result = await agent.send("t1", "u_bob", "Refund 900 please")

    (blocked,) = last_tool_results(model.calls[1])
    assert blocked["is_error"] and blocked["content"].startswith("requires_escalation:")
    assert result.escalated
    assert store.snapshot()["refunds"] == []
    assert store.snapshot()["tickets"] == [{"user_id": "u_bob", "status": "open"}]
    # The refund never reached the backend at all.
    assert all(e["action"] != "process_refund" for e in store.audit_log())


@pytest.mark.parametrize(
    ("customer", "amount", "code"),
    [("u_alice", 200, "insufficient_balance"), ("u_carol", 10, "account_not_active")],
)
async def test_guardrail_blocks_refunds_policy_would_refuse(
    make_agent, store, customer, amount, code
):
    model = ScriptedModel(
        [[tool_use("process_refund", user_id=customer, amount=amount)], [text("Sorry.")]]
    )
    agent = await make_agent(model)
    before = store.snapshot()

    result = await agent.send("t1", customer, f"Refund {amount}")

    refund = result.tool_calls[-1]
    assert refund["blocked"] and refund["output"].startswith(f"{code}:")
    assert store.snapshot() == before


async def test_transient_failure_is_retried(make_agent, flaky_store):
    flaky_store.inject("process_refund", "before")
    model = ScriptedModel(
        [[tool_use("process_refund", user_id="u_alice", amount=20)], [text("Done.")]]
    )
    agent = await make_agent(model, backend=flaky_store)

    result = await agent.send("t1", "u_alice", "Refund 20")

    assert result.tool_calls[-1]["attempts"] == 2
    assert not result.tool_calls[-1]["is_error"]
    assert flaky_store.get_account("u_alice").balance_cents == 100_00


async def test_retry_after_lost_response_does_not_refund_twice(make_agent, flaky_store):
    # The first attempt commits, then the response is lost. The retry reuses the
    # idempotency key, so the backend replays the refund instead of repeating it.
    flaky_store.inject("process_refund", "after")
    model = ScriptedModel(
        [[tool_use("process_refund", user_id="u_alice", amount=20)], [text("Done.")]]
    )
    agent = await make_agent(model, backend=flaky_store)

    result = await agent.send("t1", "u_alice", "Refund 20")

    assert result.tool_calls[-1]["attempts"] == 2
    assert '"replayed": true' in result.tool_calls[-1]["output"]
    assert flaky_store.snapshot()["refunds"] == [{"user_id": "u_alice", "amount_cents": 20_00}]
    assert flaky_store.get_account("u_alice").balance_cents == 100_00


async def test_persistent_failure_escalates_without_the_model(make_agent, flaky_store):
    flaky_store.inject("process_refund", "before", "before", "before")
    model = ScriptedModel([[tool_use("process_refund", user_id="u_alice", amount=20)]])
    agent = await make_agent(model, backend=flaky_store)

    result = await agent.send("t1", "u_alice", "Refund 20")

    assert result.escalated
    assert "ticket #1" in result.reply
    assert result.tool_calls[-2]["attempts"] == FAST.max_retries + 1
    assert names(result)[-1] == ("system", "escalate_ticket")
    assert flaky_store.snapshot()["refunds"] == []
    ticket = flaky_store.audit_log()[-1]
    assert ticket["payload"]["reason"].startswith("Automatic escalation: process_refund failed")
    assert len(model.calls) == 1  # the model was not asked to recover


async def test_policy_errors_are_not_retried(make_agent):
    model = ScriptedModel(
        [[tool_use("process_refund", user_id="u_alice", amount="1.005")], [text("Invalid.")]]
    )
    agent = await make_agent(model)

    result = await agent.send("t1", "u_alice", "Refund 1.005")

    refund = result.tool_calls[-1]
    assert (refund["code"], refund["attempts"]) == ("invalid_amount", 1)


async def test_out_of_scope_is_declined_without_calling_the_model(make_agent):
    model = ScriptedModel([], intent=Intent.OUT_OF_SCOPE)
    agent = await make_agent(model)

    result = await agent.send("t1", "u_alice", "Write me a poem")

    assert result.reply == DECLINE_REPLY
    assert model.calls == []


async def test_conversation_history_is_kept_per_thread(make_agent):
    model = ScriptedModel([[text("Hi Alice.")], [text("Still here.")], [text("New chat.")]])
    agent = await make_agent(model)

    await agent.send("t1", "u_alice", "Hello")
    await agent.send("t1", "u_alice", "Are you there?")
    await agent.send("t2", "u_alice", "Hello again")

    assert [m["role"] for m in model.calls[1]] == ["user", "assistant", "user"]
    assert model.calls[1][0]["content"] == "Hello"
    assert len(model.calls[2]) == 1


async def test_planner_sees_recent_text_but_not_tool_traffic(make_agent):
    seen: list[str] = []

    def classify(transcript: str) -> Intent:
        seen.append(transcript)
        return Intent.GENERAL

    model = ScriptedModel(
        [[tool_use("check_balance", user_id="u_alice")], [text("120.00")], [text("Sure.")]],
        intent=classify,
    )
    agent = await make_agent(model)

    await agent.send("t1", "u_alice", "Balance?")
    await agent.send("t1", "u_alice", "Thanks")

    assert seen[1] == "user: Balance?\nassistant: 120.00\nuser: Thanks"


async def test_refusal_gets_a_safe_reply(make_agent):
    class RefusingModel(ScriptedModel):
        async def respond(self, *, system, messages, tools):
            return ModelTurn(content=[], stop_reason="refusal")

    agent = await make_agent(RefusingModel([]))

    result = await agent.send("t1", "u_alice", "...")

    assert result.reply == REFUSAL_REPLY


async def test_tool_loop_hits_the_step_limit(make_agent):
    loop = [tool_use("check_balance", user_id="u_alice")]
    agent = await make_agent(ScriptedModel([loop] * 50), step_limit=10)

    with pytest.raises(AgentStepLimitError):
        await agent.send("t1", "u_alice", "Balance?")


def test_guardrail_limit_matches_backend_policy():
    assert AgentSettings().refund_limit == Decimal(REFUND_AUTO_APPROVE_LIMIT_CENTS) / 100
