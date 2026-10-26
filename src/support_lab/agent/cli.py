"""Chat with the support agent in the terminal.

    uv run support-agent --customer u_alice           # local server over stdio
    uv run support-agent --http http://127.0.0.1:8000/mcp   # sign in through the browser

Over stdio the agent starts the MCP server as a separate process and you say
which customer you are. Over HTTP you sign in at the demo bank (run
`support-auth` and `SUPPORT_TRANSPORT=http support-mcp` first), and the server
decides who you are from the token. Needs Claude credentials:
ANTHROPIC_API_KEY, or `ant auth login`.

With LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY set, each turn is traced to
Langfuse, and the conversation is one Langfuse session.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import uuid

import anthropic
import anyio
import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.auth import OAuthFlowError
from mcp.client.streamable_http import streamable_http_client

from support_lab import tracing

from .login import oauth_provider
from .model import DEFAULT_MODEL, AnthropicModel
from .progress import SLOW_STEPS, describe_step
from .runner import AgentStepLimitError, StepFinished, StepStarted, SupportAgent, TurnResult
from .tools import McpToolbox


def server_params(db: str) -> StdioServerParameters:
    """How to launch the MCP server as a subprocess.

    Only the server's own settings are passed. The MCP SDK adds a small
    allowlist (PATH, HOME, ...) and nothing else from our environment, so the
    server never sees ANTHROPIC_API_KEY or other secrets it doesn't need.
    """
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "support_lab.mcp_server"],
        env={"SUPPORT_DB": db, "SUPPORT_TRANSPORT": "stdio"},
    )


async def chat_over_stdio(customer_id: str, db: str, model: AnthropicModel) -> None:
    async with Client(server_params(db)) as client:
        await chat(McpToolbox(client), customer_id, model)


async def chat_over_http(url: str, callback_port: int, model: AnthropicModel) -> None:
    auth = oauth_provider(url, callback_port=callback_port)
    try:
        async with (
            httpx2.AsyncClient(auth=auth, timeout=60) as http,
            Client(streamable_http_client(url, http_client=http)) as client,
        ):
            toolbox = McpToolbox(client)
            customer_id = await toolbox.read_customer()
            if customer_id is None:
                print("The server didn't say who is signed in.")
                return
            print(f"Signed in as {customer_id}.")
            await chat(toolbox, customer_id, model)
    except* OAuthFlowError as group:
        print(f"Sign-in failed: {group.exceptions[0]}")
    except* httpx2.ConnectError:
        print(f"Couldn't reach {url}. Is `SUPPORT_TRANSPORT=http uv run support-mcp` running?")


async def chat(toolbox: McpToolbox, customer_id: str, model: AnthropicModel) -> None:
    agent = await SupportAgent.create(model, toolbox)
    thread = uuid.uuid4().hex
    print(f"Tools: {', '.join(agent.tool_names)}. Policy loaded from policy://refunds.")
    if tracing.enabled():
        print(f"Tracing to Langfuse, session {thread}.")
    print(f"Chatting as {customer_id} with {model.model}. Ctrl+D to quit.\n")
    while True:
        try:
            text = await anyio.to_thread.run_sync(input, "you> ")
        except EOFError:
            print()
            return
        if not text.strip():
            continue
        try:
            with tracing.attributes(session_id=thread, user_id=customer_id, trace_name="chat"):
                result = await run_turn(agent, thread, customer_id, text)
        except AgentStepLimitError as exc:
            print(f"agent> [stopped: {exc}]")
            return
        except anthropic.AuthenticationError:
            print("Claude rejected the credentials. Set ANTHROPIC_API_KEY or `ant auth login`.")
            return
        flags = f"intent={result.intent}" + (", escalated" if result.escalated else "")
        print(f"agent> {result.reply}\n    ({flags})\n")


async def run_turn(agent: SupportAgent, thread: str, customer_id: str, text: str) -> TurnResult:
    """Run a turn, printing each step as it happens."""
    live = sys.stdout.isatty()
    waiting = False  # a "working…" line is on screen, to be replaced
    async for event in agent.stream(thread, customer_id, text):
        if isinstance(event, StepStarted) and live and event.node in SLOW_STEPS:
            print(f"  {event.node:<9} {SLOW_STEPS[event.node]}", end="", flush=True)
            waiting = True
        elif isinstance(event, StepFinished):
            if waiting:
                print("\r\033[K", end="")  # clear the "working…" line
                waiting = False
            for line in describe_step(event):
                print(f"  {line}")
        elif isinstance(event, TurnResult):
            return event
    raise AssertionError("stream ended without a result")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the support agent in the terminal.")
    parser.add_argument("--customer", default="u_alice", help="customer id, stdio only")
    parser.add_argument("--http", metavar="URL", help="MCP server URL; sign in through the browser")
    parser.add_argument("--callback-port", type=int, default=8765, help="local port for sign-in")
    parser.add_argument("--db", default=os.environ.get("SUPPORT_DB", ":memory:"))
    parser.add_argument("--model", default=os.environ.get("SUPPORT_AGENT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--effort", default=os.environ.get("SUPPORT_AGENT_EFFORT", "medium"))
    args = parser.parse_args()
    model = AnthropicModel(model=args.model, effort=args.effort)
    if problem := tracing.connect():
        print(problem)
    try:
        with contextlib.suppress(KeyboardInterrupt):
            if args.http:
                anyio.run(chat_over_http, args.http, args.callback_port, model)
            else:
                anyio.run(chat_over_stdio, args.customer, args.db, model)
    finally:
        tracing.flush()
