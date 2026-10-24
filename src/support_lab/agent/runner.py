"""`SupportAgent`: one conversation turn in, one reply out."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from support_lab import tracing

from .graph import DEFAULT_SETTINGS, AgentSettings, build_graph
from .model import ModelClient
from .tools import McpToolbox


class AgentStepLimitError(RuntimeError):
    """The turn hit the step limit (usually a tool-calling loop). The thread's
    history ends mid-turn and can't be continued."""


@dataclass(frozen=True)
class TurnResult:
    """One turn's reply. `escalated` covers the whole conversation so far;
    `tool_calls` covers this turn only."""

    reply: str
    intent: str
    escalated: bool
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class StepStarted:
    """A graph node began running (e.g. `agent` is about to call Claude)."""

    node: str


@dataclass(frozen=True)
class StepFinished:
    """A graph node finished. `update` is what it wrote to the state."""

    node: str
    update: dict[str, Any]


AgentEvent = StepStarted | StepFinished | TurnResult


class SupportAgent:
    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        step_limit: int = 30,
        tool_names: list[str] | None = None,
    ):
        self.graph = graph
        self.step_limit = step_limit
        self.tool_names = tool_names or []

    @classmethod
    async def create(
        cls,
        model: ModelClient,
        toolbox: McpToolbox,
        *,
        settings: AgentSettings = DEFAULT_SETTINGS,
        checkpointer: Checkpointer = None,
        step_limit: int = 30,
    ) -> SupportAgent:
        tool_specs = await toolbox.tool_specs()
        graph = build_graph(
            model,
            toolbox,
            tool_specs=tool_specs,
            policy=await toolbox.read_policy(),
            settings=settings,
            checkpointer=checkpointer or InMemorySaver(),
        )
        return cls(graph, step_limit=step_limit, tool_names=[t["name"] for t in tool_specs])

    async def send(self, thread_id: str, customer_id: str, text: str) -> TurnResult:
        """Run one turn and return the result."""
        async for event in self.stream(thread_id, customer_id, text):
            if isinstance(event, TurnResult):
                return event
        raise AssertionError("stream ended without a result")

    async def stream(
        self, thread_id: str, customer_id: str, text: str
    ) -> AsyncIterator[AgentEvent]:
        """Run one turn, yielding each node as it starts and finishes, then the result.

        With tracing on, the turn is one observation, nested under whatever the
        caller has open (an eval run), or the root of its own trace (a chat turn).
        """
        config: Any = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.step_limit,
            "callbacks": tracing.langgraph_callbacks(),
        }
        turn_input = {"customer_id": customer_id, "messages": [{"role": "user", "content": text}]}
        with tracing.observe("support turn", as_type="agent", input=text) as observation:
            try:
                async for task in self.graph.astream(turn_input, config, stream_mode="tasks"):
                    if "result" in task:
                        yield StepFinished(task["name"], dict(task["result"] or {}))
                    else:
                        yield StepStarted(task["name"])
            except GraphRecursionError as exc:
                observation.update(level="ERROR", status_message="step limit")
                raise AgentStepLimitError(f"turn exceeded {self.step_limit} steps") from exc
            result = self._result((await self.graph.aget_state(config)).values)
            observation.update(
                output=result.reply,
                metadata={"intent": result.intent, "escalated": result.escalated},
            )
        # Yielded after the observation ends: callers usually stop reading here, and
        # an observation left open across that would be closed later, out of context.
        yield result

    @staticmethod
    def _result(state: dict[str, Any]) -> TurnResult:
        reply_blocks = state["messages"][-1]["content"]
        reply = "\n".join(b["text"] for b in reply_blocks if b.get("type") == "text")
        turn = state["turn"]
        return TurnResult(
            reply=reply,
            intent=state["intent"],
            escalated=state.get("escalated", False),
            tool_calls=[c for c in state.get("tool_log", []) if c["turn"] == turn],
        )
