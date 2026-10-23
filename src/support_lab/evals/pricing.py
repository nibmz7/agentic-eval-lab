"""Rough cost estimates from token counts.

Prices are USD per million tokens, from Anthropic's published pricing as of
2026-10. Check https://www.anthropic.com/pricing before relying on them.
Thinking tokens are billed as output and already included in output_tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICES_AS_OF = "2026-10"


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    # Haiku 5.5 charges more once a prompt exceeds 100K tokens.
    long_prompt_threshold: int | None = None
    long_input: float | None = None
    long_output: float | None = None


PRICES = {
    "claude-haiku-5-5": Price(0.10, 0.50, 100_000, 0.50, 2.50),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-opus-5-5": Price(4.00, 20.00),
}


def call_cost(model: str, usage: dict[str, int]) -> float | None:
    """Estimated USD for one API call, or None for a model without a known price."""
    price = PRICES.get(model)
    if price is None:
        return None
    cache_write = usage.get("cache_creation_input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    prompt = usage.get("input_tokens", 0) + cache_write + cache_read
    long = price.long_prompt_threshold is not None and prompt > price.long_prompt_threshold
    rate_in = price.long_input if long and price.long_input else price.input
    rate_out = price.long_output if long and price.long_output else price.output
    dollars = (
        usage.get("input_tokens", 0) * rate_in
        + cache_write * rate_in * 1.25  # 5-minute cache writes
        + cache_read * rate_in * 0.1
        + usage.get("output_tokens", 0) * rate_out
    )
    return dollars / 1_000_000


def total_cost(model: str, calls: list[dict[str, int]]) -> float | None:
    costs = [call_cost(model, c) for c in calls]
    return None if any(c is None for c in costs) else sum(c for c in costs if c is not None)
