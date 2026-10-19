"""A model that replays a fixed script instead of calling an LLM.

Used by tests, and later by evals to build deliberately broken agents: if a
scripted "always refunds" agent passes the eval suite, the suite is wrong.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .model import Block, Intent, Message, ModelTurn

_ids = itertools.count(1)


def text(value: str) -> Block:
    return {"type": "text", "text": value}


def tool_use(name: str, **input: Any) -> Block:
    return {"type": "tool_use", "id": f"toolu_scripted_{next(_ids)}", "name": name, "input": input}


@dataclass
class ScriptedModel:
    """Each `respond` call returns the next turn. A turn is a list of blocks, or a
    function of the messages so far (to react to tool results)."""

    turns: Sequence[list[Block] | Callable[[list[Message]], list[Block]]]
    intent: Intent | Callable[[str], Intent] = Intent.GENERAL
    calls: list[list[Message]] = field(default_factory=list)

    async def classify(self, transcript: str) -> Intent:
        return self.intent(transcript) if callable(self.intent) else self.intent

    async def respond(
        self, *, system: str, messages: list[Message], tools: list[Block]
    ) -> ModelTurn:
        if len(self.calls) >= len(self.turns):
            raise AssertionError(f"script exhausted after {len(self.turns)} turns")
        turn = self.turns[len(self.calls)]
        self.calls.append(list(messages))
        content = turn(messages) if callable(turn) else turn
        stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
        return ModelTurn(content=content, stop_reason=stop)


def last_tool_results(messages: list[Message]) -> list[Block]:
    """The tool_result blocks the model is about to see."""
    content = messages[-1]["content"]
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
