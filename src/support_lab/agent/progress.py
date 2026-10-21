"""Turn graph steps into short progress lines for the terminal."""

from __future__ import annotations

import json
from typing import Any

from .runner import StepFinished

# Steps slow enough to show a "working…" line while they run.
SLOW_STEPS = {"planner": "understanding the request…", "agent": "Claude is thinking…"}


def _args(values: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in values.items() if k != "idempotency_key")


def _summary(name: str, output: str) -> str:
    """A few words from a successful tool result."""
    try:
        data = json.loads(output)
    except (TypeError, ValueError):
        return ""
    if name == "check_balance":
        return f"{data['status']}, balance {data['balance']}"
    if name == "process_refund":
        replayed = " (replayed, not repeated)" if data.get("replayed") else ""
        return f"new balance {data['new_balance']}{replayed}"
    if name == "escalate_ticket":
        return f"ticket #{data['ticket_id']} opened"
    return ""


def describe_call(call: dict[str, Any]) -> str:
    """One logged tool call, e.g. `process_refund(user_id='u_bob', amount=20) → ok`."""
    if call["blocked"]:
        status = f"not run, blocked by guardrail ({call['output'].split(':')[0]})"
    elif call["is_error"]:
        status = f"error: {call['code'] or 'temporary failure'}"
    else:
        detail = _summary(call["name"], call["output"])
        status = f"ok: {detail}" if detail else "ok"
    retries = f" after {call['attempts']} attempts" if call["attempts"] > 1 else ""
    return f"{call['name']}({_args(call['input'])}) → {status}{retries}"


def describe_step(event: StepFinished) -> list[str]:
    """Progress lines for a finished step, each prefixed with the step's name."""
    node, update = event.node, event.update
    lines: list[str] = []
    if node == "planner":
        lines.append(f"intent: {update.get('intent')}")
    elif node == "decline":
        lines.append("out of scope, declining")
    elif node == "agent":
        content = update["messages"][0]["content"]
        calls = [b for b in content if b.get("type") == "tool_use"]
        said = " ".join(b["text"] for b in content if b.get("type") == "text" and b["text"].strip())
        if calls:
            if said:
                lines.append(f'says "{said}"')
            lines += [f"wants {b['name']}({_args(b['input'])})" for b in calls]
        else:
            lines.append("replied")
    elif node == "guardrail":
        for lookup in update.get("tool_log", []):
            lines.append(f"checked {describe_call(lookup)}")
        for reason in update.get("blocked", {}).values():
            lines.append(f"blocked refund: {reason}")
        if update.get("tool_log") and not update.get("blocked"):
            lines.append("refund allowed")
    elif node == "tools":
        lines += [describe_call(c) for c in update.get("tool_log", [])]
    elif node == "escalate":
        lines += [f"repeated failures, {describe_call(c)}" for c in update.get("tool_log", [])]
    return [f"{node:<9} {line}" for line in lines]
