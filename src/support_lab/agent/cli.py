"""Chat with the support agent in the terminal.

    uv run support-agent --customer u_alice

The agent talks to the MCP server over stdio, as a separate process, so it can
only reach the database through the server's tools. Needs Claude credentials:
ANTHROPIC_API_KEY, or `ant auth login`.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import uuid

import anthropic
import anyio
from mcp import Client, StdioServerParameters

from .model import DEFAULT_MODEL, AnthropicModel
from .runner import AgentStepLimitError, SupportAgent
from .tools import McpToolbox


def _describe(call: dict) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in call["input"].items() if k != "idempotency_key")
    if call["blocked"]:
        status = f"blocked by guardrail ({call['output'].split(':')[0]})"
    elif call["is_error"]:
        status = f"error ({call['code'] or 'transient'})"
    else:
        status = "ok"
    retries = f", {call['attempts']} attempts" if call["attempts"] > 1 else ""
    return f"    [{call['source']}] {call['name']}({args}) → {status}{retries}"


async def chat(customer_id: str, db: str, model_name: str, effort: str) -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "support_lab.mcp_server"],
        env={**os.environ, "SUPPORT_DB": db, "SUPPORT_TRANSPORT": "stdio"},
    )
    model = AnthropicModel(model=model_name, effort=effort)
    async with Client(server) as client:
        agent = await SupportAgent.create(model, McpToolbox(client))
        thread = uuid.uuid4().hex
        print(f"Chatting as {customer_id} with {model_name}. Ctrl+D to quit.\n")
        while True:
            try:
                text = await anyio.to_thread.run_sync(input, "you> ")
            except EOFError:
                print()
                return
            if not text.strip():
                continue
            try:
                result = await agent.send(thread, customer_id, text)
            except AgentStepLimitError as exc:
                print(f"agent> [stopped: {exc}]")
                return
            except anthropic.AuthenticationError:
                print("Claude rejected the credentials. Set ANTHROPIC_API_KEY or `ant auth login`.")
                return
            for call in result.tool_calls:
                print(_describe(call))
            flags = f"intent={result.intent}" + (", escalated" if result.escalated else "")
            print(f"agent> {result.reply}\n    ({flags})\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the support agent in the terminal.")
    parser.add_argument("--customer", default="u_alice", help="customer id (default: u_alice)")
    parser.add_argument("--db", default=os.environ.get("SUPPORT_DB", ":memory:"))
    parser.add_argument("--model", default=os.environ.get("SUPPORT_AGENT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--effort", default=os.environ.get("SUPPORT_AGENT_EFFORT", "medium"))
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        anyio.run(chat, args.customer, args.db, args.model, args.effort)
