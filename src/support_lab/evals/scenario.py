"""Scenarios: a customer conversation and what should happen.

Scenarios live in YAML under evals/scenarios/. Each file holds a list:

    - id: refund-over-limit
      description: Refunds above the auto-approve limit must be escalated.
      customer: u_bob
      turns:
        - say: "I'd like a refund of $900."
          expect: {intent: refund_request}      # optional, checked after this turn
      faults: {process_refund: [before]}        # optional, see backend/faults.py
      expect:                                   # checked after the last turn
        refunds: []                             # new refunds, as amounts
        tickets: 1                              # new tickets: a number, or {min: 1}
        escalated: true                         # true, false, or null for "either"
        tools:
          required: [escalate_ticket]           # must run successfully
          forbidden: [process_refund]           # the model must not even ask for it
          order: [[check_balance, process_refund]]   # if both are called, in this order
          max_calls: 3                          # tool calls the model asks for, in total

Amounts are written in dollars ("20.00") and compared in cents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

KNOWN_TOOLS = {"check_balance", "process_refund", "escalate_ticket"}
KNOWN_FAULT_TARGETS = {"process_refund", "escalate_ticket"}
INTENTS = {"balance_inquiry", "refund_request", "needs_human", "general", "out_of_scope"}


class ScenarioError(ValueError):
    pass


@dataclass(frozen=True)
class Range:
    min: int
    max: int | None

    def contains(self, value: int) -> bool:
        return value >= self.min and (self.max is None or value <= self.max)

    def __str__(self) -> str:
        if self.max == self.min:
            return str(self.min)
        return f"at least {self.min}" if self.max is None else f"{self.min}–{self.max}"


@dataclass(frozen=True)
class ToolExpectations:
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    order: tuple[tuple[str, str], ...] = ()
    max_calls: int | None = None


@dataclass(frozen=True)
class Expectations:
    refunds: tuple[int, ...] | None = None  # cents, None = not checked
    tickets: Range | None = None
    escalated: bool | None = None  # None = either is fine
    escalation_checked: bool = False
    intent: str | None = None
    tools: ToolExpectations = field(default_factory=ToolExpectations)


@dataclass(frozen=True)
class Turn:
    say: str
    expect: Expectations | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    customer: str
    turns: tuple[Turn, ...]
    expect: Expectations
    faults: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source: str = ""


def _cents(value: Any, where: str) -> int:
    try:
        amount = Decimal(str(value))
    except Exception:
        raise ScenarioError(f"{where}: {value!r} is not an amount") from None
    if amount != amount.quantize(Decimal("0.01")) or amount <= 0:
        raise ScenarioError(f"{where}: {value!r} must be positive with at most 2 decimals")
    return int(amount * 100)


def _range(value: Any, where: str) -> Range:
    if isinstance(value, int):
        return Range(value, value)
    if isinstance(value, dict) and set(value) <= {"min", "max"}:
        return Range(int(value.get("min", 0)), value.get("max"))
    raise ScenarioError(f"{where}: expected a number or {{min, max}}, got {value!r}")


def _tools(raw: dict[str, Any], where: str) -> ToolExpectations:
    unknown_keys = set(raw) - {"required", "forbidden", "order", "max_calls"}
    if unknown_keys:
        raise ScenarioError(f"{where}: unknown keys {sorted(unknown_keys)}")
    names = list(raw.get("required", [])) + list(raw.get("forbidden", []))
    pairs = [tuple(pair) for pair in raw.get("order", [])]
    for pair in pairs:
        if len(pair) != 2:
            raise ScenarioError(f"{where}: order entries are [first, then], got {list(pair)}")
        names += list(pair)
    if unknown := set(names) - KNOWN_TOOLS:
        raise ScenarioError(f"{where}: unknown tools {sorted(unknown)}")
    return ToolExpectations(
        required=tuple(raw.get("required", [])),
        forbidden=tuple(raw.get("forbidden", [])),
        order=tuple(pairs),  # type: ignore[arg-type]
        max_calls=raw.get("max_calls"),
    )


def _expectations(raw: dict[str, Any], where: str) -> Expectations:
    allowed = {"refunds", "tickets", "escalated", "intent", "tools"}
    if unknown_keys := set(raw) - allowed:
        raise ScenarioError(f"{where}: unknown keys {sorted(unknown_keys)}")
    intent = raw.get("intent")
    if intent is not None and intent not in INTENTS:
        raise ScenarioError(f"{where}: unknown intent {intent!r}")
    refunds = raw.get("refunds")
    return Expectations(
        refunds=None if refunds is None else tuple(_cents(a, f"{where}.refunds") for a in refunds),
        tickets=None if "tickets" not in raw else _range(raw["tickets"], f"{where}.tickets"),
        escalated=raw.get("escalated"),
        escalation_checked="escalated" in raw and raw["escalated"] is not None,
        intent=intent,
        tools=_tools(raw.get("tools") or {}, f"{where}.tools"),
    )


def parse_scenario(raw: dict[str, Any], source: str = "") -> Scenario:
    where = f"{source}:{raw.get('id', '?')}"
    for key in ("id", "description", "customer", "turns", "expect"):
        if key not in raw:
            raise ScenarioError(f"{where}: missing {key!r}")
    turns = []
    for i, turn in enumerate(raw["turns"], start=1):
        if isinstance(turn, str):
            turns.append(Turn(turn))
        else:
            expect = turn.get("expect")
            turns.append(
                Turn(turn["say"], _expectations(expect, f"{where}.turn{i}") if expect else None)
            )
    faults = {}
    for target, modes in (raw.get("faults") or {}).items():
        if target not in KNOWN_FAULT_TARGETS or not set(modes) <= {"before", "after"}:
            raise ScenarioError(f"{where}: bad fault {target}: {modes}")
        faults[target] = tuple(modes)
    return Scenario(
        id=raw["id"],
        description=raw["description"],
        customer=raw["customer"],
        turns=tuple(turns),
        expect=_expectations(raw["expect"], f"{where}.expect"),
        faults=faults,
        source=source,
    )


def load_scenarios(directory: str | Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for path in sorted(Path(directory).glob("*.yaml")):
        for raw in yaml.safe_load(path.read_text()) or []:
            scenarios.append(parse_scenario(raw, path.name))
    ids = [s.id for s in scenarios]
    if duplicates := {i for i in ids if ids.count(i) > 1}:
        raise ScenarioError(f"duplicate scenario ids: {sorted(duplicates)}")
    return scenarios
