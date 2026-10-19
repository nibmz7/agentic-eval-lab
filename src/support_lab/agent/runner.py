"""`SupportAgent`: one conversation turn in, one reply out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

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


class SupportAgent:
    def __init__(self, graph: CompiledStateGraph, *, step_limit: int = 30):
        self.graph = graph
        self.step_limit = step_limit

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
        graph = build_graph(
            model,
            toolbox,
            tool_specs=await toolbox.tool_specs(),
            policy=await toolbox.read_policy(),
            settings=settings,
            checkpointer=checkpointer or InMemorySaver(),
        )
        return cls(graph, step_limit=step_limit)

    async def send(self, thread_id: str, customer_id: str, text: str) -> TurnResult:
        config: Any = {"configurable": {"thread_id": thread_id}, "recursion_limit": self.step_limit}
        try:
            state = await self.graph.ainvoke(
                {"customer_id": customer_id, "messages": [{"role": "user", "content": text}]},
                config,
            )
        except GraphRecursionError as exc:
            raise AgentStepLimitError(f"turn exceeded {self.step_limit} steps") from exc
        reply_blocks = state["messages"][-1]["content"]
        reply = "\n".join(b["text"] for b in reply_blocks if b.get("type") == "text")
        turn = state["turn"]
        return TurnResult(
            reply=reply,
            intent=state["intent"],
            escalated=state.get("escalated", False),
            tool_calls=[c for c in state.get("tool_log", []) if c["turn"] == turn],
        )
