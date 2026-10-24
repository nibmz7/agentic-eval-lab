"""Play scenarios against the agent and record what happened.

Every run gets a fresh database and its own in-process MCP server, so runs
can't affect each other and can run concurrently. The agent sees only the
customer's messages, never the scenario's expectations.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import anyio
from mcp import Client

from support_lab.agent import AgentSettings, McpToolbox, ModelClient, SupportAgent
from support_lab.backend.faults import FlakyStore
from support_lab.mcp_server import build_server

from .scenario import Scenario

# Builds a fresh model for one run. Scripted models keep state, so each run needs its own.
ModelFactory = Callable[[Scenario, int], ModelClient]


@dataclass
class TurnRecord:
    say: str
    reply: str
    intent: str
    escalated: bool
    tool_calls: list[dict[str, Any]]
    snapshot: dict[str, Any]  # database state right after this turn


@dataclass
class RunRecord:
    scenario_id: str
    run: int
    customer: str
    before: dict[str, Any]
    turns: list[TurnRecord] = field(default_factory=list)
    error: str | None = None  # set when the run crashed or hit the step limit
    seconds: float = 0.0
    model: str | None = None  # the agent's model, when it reports one
    usage: list[dict[str, int]] = field(default_factory=list)  # per API call
    judge_model: str | None = None
    judge_usage: list[dict[str, int]] = field(default_factory=list)

    @property
    def after(self) -> dict[str, Any]:
        return self.turns[-1].snapshot if self.turns else self.before

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return [call for turn in self.turns for call in turn.tool_calls]

    @property
    def escalated(self) -> bool:
        return bool(self.turns) and self.turns[-1].escalated


def describe_error(exc: BaseException) -> str:
    """The underlying errors, unwrapped from any task-group exception groups."""
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(describe_error(inner) for inner in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


async def run_once(
    scenario: Scenario, run: int, model: ModelClient, settings: AgentSettings
) -> RunRecord:
    store = FlakyStore.seeded()
    for method, modes in scenario.faults.items():
        store.inject(method, *modes)  # type: ignore[arg-type]
    record = RunRecord(scenario.id, run, scenario.customer, before=store.snapshot())
    started = time.monotonic()
    try:
        async with Client(build_server(store)) as client:
            agent = await SupportAgent.create(model, McpToolbox(client), settings=settings)
            thread = uuid.uuid4().hex
            for turn in scenario.turns:
                result = await agent.send(thread, scenario.customer, turn.say)
                record.turns.append(
                    TurnRecord(
                        say=turn.say,
                        reply=result.reply,
                        intent=result.intent,
                        escalated=result.escalated,
                        tool_calls=result.tool_calls,
                        snapshot=store.snapshot(),
                    )
                )
    except Exception as exc:  # a crash is a failed run, never a skipped one
        record.error = describe_error(exc)
    finally:
        record.seconds = time.monotonic() - started
        record.model = getattr(model, "model", None)
        record.usage = list(getattr(model, "usage", []))
        store.close()
    return record


async def run_scenarios(
    scenarios: list[Scenario],
    make_model: ModelFactory,
    *,
    runs: int = 1,
    concurrency: int = 4,
    settings: AgentSettings | None = None,
    on_done: Callable[[RunRecord], None] | None = None,
) -> list[RunRecord]:
    """Run every scenario `runs` times. Records come back in scenario order, then run order."""
    settings = settings or AgentSettings()
    limiter = anyio.CapacityLimiter(concurrency)
    results: dict[tuple[str, int], RunRecord] = {}

    async def one(scenario: Scenario, run: int) -> None:
        async with limiter:
            record = await run_once(scenario, run, make_model(scenario, run), settings)
        results[(scenario.id, run)] = record
        if on_done:
            on_done(record)

    async with anyio.create_task_group() as group:
        for scenario in scenarios:
            for run in range(1, runs + 1):
                group.start_soon(one, scenario, run)
    return [results[(s.id, r)] for s in scenarios for r in range(1, runs + 1)]
