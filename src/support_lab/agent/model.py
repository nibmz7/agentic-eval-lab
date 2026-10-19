"""Model clients.

The graph depends only on the `ModelClient` protocol. `AnthropicModel` calls
Claude; `ScriptedModel` (scripted.py) replays fixed turns so tests and
"broken agent" experiments are deterministic and free.

Messages and content blocks are plain dicts in the Messages API shape. The
model's blocks (including thinking and fallback blocks) are stored exactly as
returned and sent back unchanged on the next request, as the API requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import anthropic
from pydantic import BaseModel

Block = dict[str, Any]
Message = dict[str, Any]

DEFAULT_MODEL = "claude-opus-5-5"
# Re-runs a request on another model server-side if the first one declines it.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Intent(StrEnum):
    BALANCE_INQUIRY = "balance_inquiry"
    REFUND_REQUEST = "refund_request"
    NEEDS_HUMAN = "needs_human"
    GENERAL = "general"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class ModelTurn:
    content: list[Block]
    stop_reason: str


class ModelClient(Protocol):
    async def classify(self, transcript: str) -> Intent: ...

    async def respond(
        self, *, system: str, messages: list[Message], tools: list[Block]
    ) -> ModelTurn: ...


CLASSIFIER_SYSTEM = """\
You route messages for a customer support desk that handles account balances, \
refunds, and handing issues to a human agent. Classify the customer's latest \
message, using the earlier conversation for context:
- balance_inquiry: wants to know their balance or account status
- refund_request: wants money refunded, or is continuing a refund conversation
- needs_human: asks for a person, complains, or has a problem the desk can't fix itself
- general: greetings, thanks, or follow-ups that fit none of the above
- out_of_scope: unrelated to this desk (trivia, coding help, other companies)"""


class _IntentChoice(BaseModel):
    intent: Intent


class AnthropicModel:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_tokens: int = 16000,
    ):
        self._client = client or anthropic.AsyncAnthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    async def classify(self, transcript: str) -> Intent:
        response = await self._client.beta.messages.parse(
            model=self.model,
            max_tokens=4096,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            output_config={"effort": "low"},
            system=CLASSIFIER_SYSTEM,
            messages=[{"role": "user", "content": transcript}],
            output_format=_IntentChoice,
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            return Intent.GENERAL
        return response.parsed_output.intent

    async def respond(
        self, *, system: str, messages: list[Message], tools: list[Block]
    ) -> ModelTurn:
        response = await self._client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            output_config={"effort": self.effort},  # type: ignore[typeddict-item]
            system=system,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
        )
        return ModelTurn(
            content=[block.to_dict() for block in response.content],
            stop_reason=response.stop_reason or "end_turn",
        )
