"""The support agent as a LangGraph state machine.

    START → planner ─┬─ out_of_scope ─→ decline → END
                     └─→ agent ─┬─ no tool calls ─→ END
                                └─→ guardrail → tools ─┬─→ agent (loop)
                                                       └─ transient failure ─→ escalate → END

planner    classifies the customer's intent (recorded for evaluation; only
           out_of_scope changes the route).
agent      one model call. Its tool_use blocks become `pending`.
guardrail  checks every process_refund against the account *before* it runs and
           blocks the ones policy would refuse, so the model gets a clear error
           instead of the backend having to catch it.
tools      runs the approved calls through MCP. Transient failures are retried;
           policy errors are not. Refunds get an idempotency key so a retry can
           never refund twice.
escalate   if a call still fails after retries, opens a ticket deterministically
           rather than trusting the model to recover.
"""

from __future__ import annotations

import asyncio
import json
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Required, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from .model import Block, Intent, Message, ModelClient
from .tools import McpToolbox, ToolOutcome

DECLINE_REPLY = (
    "Sorry, I can only help with account balances, refunds, and passing issues to a human agent."
)
REFUSAL_REPLY = "Sorry, I can't help with that request."


@dataclass(frozen=True)
class AgentSettings:
    # Must match the backend's REFUND_AUTO_APPROVE_LIMIT_CENTS (a test checks it).
    # The agent doesn't import the backend: it only talks to it through MCP.
    refund_limit: Decimal = Decimal("500.00")
    max_retries: int = 2
    retry_delays: Sequence[float] = (0.5, 2.0)


DEFAULT_SETTINGS = AgentSettings()


class AgentState(TypedDict, total=False):
    customer_id: Required[str]
    messages: Required[Annotated[list[Message], operator.add]]
    turn: int
    intent: str
    tool_log: Annotated[list[dict[str, Any]], operator.add]
    pending: list[Block]  # tool_use blocks from the latest model turn
    blocked: dict[str, str]  # tool_use id -> guardrail error
    tool_failure: str | None
    escalated: bool


def system_prompt(customer_id: str, policy: str) -> str:
    return f"""\
You are a customer support agent. You are helping customer {customer_id} and \
may only act on their account.

Use the tools to look up balances, process refunds, and escalate to a human. \
Check the balance before refunding. If a tool returns an error code, tell the \
customer what happened in plain language; when the code is requires_escalation, \
or you can't resolve the request yourself, open a ticket with escalate_ticket. \
Only say an action succeeded if the tool result confirms it. Keep replies brief.

{policy}"""


def transcript(messages: list[Message], last: int = 6) -> str:
    """The text of recent turns, without tool traffic, for the intent classifier."""
    lines = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            parts = [content]
        else:
            parts = [b["text"] for b in content if b.get("type") == "text" and b.get("text")]
        if parts:
            lines.append(f"{message['role']}: {' '.join(parts)}")
    return "\n".join(lines[-last:])


def _record(state: AgentState, source: str, call_id: str | None, name: str, args: dict,
            outcome: ToolOutcome | None, *, attempts: int = 1, blocked: str | None = None
            ) -> dict[str, Any]:  # fmt: skip
    return {
        "turn": state.get("turn", 0),
        "source": source,  # model | guardrail | system
        "tool_use_id": call_id,
        "name": name,
        "input": args,
        "attempts": attempts,
        "blocked": blocked is not None,
        "is_error": blocked is not None or (outcome is not None and outcome.is_error),
        "code": outcome.code if outcome else None,
        "output": blocked if blocked is not None else (outcome.text if outcome else None),
    }


def _tool_result(call_id: str, content: str, is_error: bool) -> Block:
    block: Block = {"type": "tool_result", "tool_use_id": call_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def build_graph(
    model: ModelClient,
    toolbox: McpToolbox,
    *,
    tool_specs: list[Block],
    policy: str,
    settings: AgentSettings = DEFAULT_SETTINGS,
    checkpointer: Checkpointer = None,
) -> CompiledStateGraph:
    async def planner(state: AgentState) -> dict[str, Any]:
        intent = await model.classify(transcript(state["messages"]))
        return {"turn": state.get("turn", 0) + 1, "intent": intent.value}

    async def decline(state: AgentState) -> dict[str, Any]:
        reply = {"role": "assistant", "content": [{"type": "text", "text": DECLINE_REPLY}]}
        return {"messages": [reply]}

    async def agent(state: AgentState) -> dict[str, Any]:
        turn = await model.respond(
            system=system_prompt(state["customer_id"], policy),
            messages=state["messages"],
            tools=tool_specs,
        )
        content = turn.content
        if turn.stop_reason == "refusal" or not content:
            content = [{"type": "text", "text": REFUSAL_REPLY}]
        pending = [b for b in content if b.get("type") == "tool_use"]
        return {
            "messages": [{"role": "assistant", "content": content}],
            "pending": pending,
            "blocked": {},
            "tool_failure": None,
        }

    async def check_refund(state: AgentState, args: dict, log: list) -> str | None:
        """The guardrail's error for this refund, or None to let it through.

        Anything it can't evaluate (bad input, unknown user) goes through to the
        backend, which validates it again. The guardrail only blocks clear cases.
        """
        try:
            amount = Decimal(str(args["amount"]))
            user_id = str(args["user_id"])
        except (KeyError, InvalidOperation):
            return None
        lookup = await toolbox.call("check_balance", {"user_id": user_id})
        log.append(_record(state, "guardrail", None, "check_balance", {"user_id": user_id}, lookup))
        if lookup.is_error:
            return None
        account = json.loads(lookup.text)
        balance = Decimal(account["balance"])
        if account["status"] != "active":
            return f"account_not_active: blocked by guardrail, account is {account['status']}"
        if amount > settings.refund_limit:
            return (
                f"requires_escalation: blocked by guardrail, {amount} exceeds the "
                f"{settings.refund_limit} auto-approve limit"
            )
        if amount > balance:
            return f"insufficient_balance: blocked by guardrail, {amount} exceeds balance {balance}"
        return None

    async def guardrail(state: AgentState) -> dict[str, Any]:
        blocked: dict[str, str] = {}
        log: list[dict[str, Any]] = []
        for call in state.get("pending", []):
            if call["name"] == "process_refund":
                error = await check_refund(state, call["input"], log)
                if error:
                    blocked[call["id"]] = error
        return {"blocked": blocked, "tool_log": log}

    async def call_with_retries(name: str, args: dict) -> tuple[ToolOutcome, int]:
        attempt = 1
        outcome = await toolbox.call(name, args)
        while outcome.transient and attempt <= settings.max_retries:
            delays = settings.retry_delays
            await asyncio.sleep(delays[min(attempt - 1, len(delays) - 1)] if delays else 0)
            attempt += 1
            outcome = await toolbox.call(name, args)
        return outcome, attempt

    async def tools(state: AgentState) -> dict[str, Any]:
        results: list[Block] = []
        log: list[dict[str, Any]] = []
        failure: str | None = None
        escalated = state.get("escalated", False)
        for call in state.get("pending", []):
            call_id, name, args = call["id"], call["name"], dict(call["input"])
            error = state.get("blocked", {}).get(call_id)
            if error is not None:
                log.append(_record(state, "model", call_id, name, args, None, blocked=error))
                results.append(_tool_result(call_id, error, is_error=True))
                continue
            if name == "process_refund":
                args.setdefault("idempotency_key", f"agent-{call_id}")
            outcome, attempts = await call_with_retries(name, args)
            log.append(_record(state, "model", call_id, name, args, outcome, attempts=attempts))
            results.append(_tool_result(call_id, outcome.text, outcome.is_error))
            if outcome.transient:
                failure = f"{name} failed after {attempts} attempts"
            elif name == "escalate_ticket" and not outcome.is_error:
                escalated = True
        return {
            "messages": [{"role": "user", "content": results}],
            "tool_log": log,
            "pending": [],
            "blocked": {},
            "tool_failure": failure,
            "escalated": escalated,
        }

    async def escalate(state: AgentState) -> dict[str, Any]:
        args = {
            "user_id": state["customer_id"],
            "reason": f"Automatic escalation: {state.get('tool_failure')}",
        }
        outcome = await toolbox.call("escalate_ticket", args)
        if outcome.is_error:
            reply = (
                "Sorry, I ran into a technical problem and couldn't finish this, and I "
                "couldn't open a ticket for a human agent either. Please try again later."
            )
        else:
            ticket = json.loads(outcome.text)["ticket_id"]
            reply = (
                "Sorry, I ran into a technical problem and couldn't finish this. I've passed "
                f"it to a human agent (ticket #{ticket}), who will follow up with you."
            )
        return {
            "messages": [{"role": "assistant", "content": [{"type": "text", "text": reply}]}],
            "tool_log": [_record(state, "system", None, "escalate_ticket", args, outcome)],
            "escalated": state.get("escalated", False) or not outcome.is_error,
        }

    def after_planner(state: AgentState) -> str:
        return "decline" if state.get("intent") == Intent.OUT_OF_SCOPE else "agent"

    def after_agent(state: AgentState) -> str:
        return "guardrail" if state.get("pending", []) else END

    def after_tools(state: AgentState) -> str:
        return "escalate" if state.get("tool_failure") else "agent"

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("decline", decline)
    graph.add_node("agent", agent)
    graph.add_node("guardrail", guardrail)
    graph.add_node("tools", tools)
    graph.add_node("escalate", escalate)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", after_planner, ["decline", "agent"])
    graph.add_edge("decline", END)
    graph.add_conditional_edges("agent", after_agent, ["guardrail", END])
    graph.add_edge("guardrail", "tools")
    graph.add_conditional_edges("tools", after_tools, ["escalate", "agent"])
    graph.add_edge("escalate", END)
    return graph.compile(checkpointer=checkpointer)
