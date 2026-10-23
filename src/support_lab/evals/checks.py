"""Score one run, layer by layer.

    end_state   the database: refunds, tickets, nothing touched on other accounts
    trajectory  required tools ran (DeepEval ToolCorrectnessMetric), forbidden ones
                were never asked for, order rules held, only the customer's own
                account was used, no mutation was repeated
    efficiency  the model didn't ask for more tool calls than the scenario allows
    escalation  escalated when it should have, and only then
    per_turn    the same state checks after individual turns (multi-turn drift)
    intent      the planner's label; reported but never fails a run

A run passes when it completed and every check outside `intent` passed.
All of these are deterministic. DeepEval's metric is given a judge that
raises if called, so a hidden LLM call fails loudly instead of costing money.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

# Importing this module runs support_lab/evals/__init__.py first, which opts
# out of DeepEval telemetry before DeepEval is imported below.
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall

from .runner import RunRecord
from .scenario import Expectations, Scenario

LAYERS = ("crash", "end_state", "trajectory", "efficiency", "escalation", "per_turn", "intent")
INFORMATIONAL = {"intent"}
MUTATIONS = {"process_refund", "escalate_ticket"}


@dataclass(frozen=True)
class Check:
    layer: str
    name: str
    passed: bool
    detail: str = ""


@dataclass
class RunScore:
    record: RunRecord
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.layer not in INFORMATIONAL)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.layer not in INFORMATIONAL]

    @property
    def escalation_outcome(self) -> str | None:
        for check in self.checks:
            if check.layer == "escalation":
                return check.detail.split(":")[0]
        return None


class NoLLMJudge(DeepEvalBaseLLM):
    """A judge for metrics that shouldn't need one. Fails loudly if they call it."""

    def load_model(self, *args: Any, **kwargs: Any) -> NoLLMJudge:
        return self

    def generate(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("a deterministic check tried to call an LLM judge")

    async def a_generate(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("a deterministic check tried to call an LLM judge")

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return "none (deterministic)"


def _dollars(cents: int) -> str:
    return f"{Decimal(cents) / 100:.2f}"


def _new_rows(before: dict[str, Any], after: dict[str, Any], table: str) -> list[dict[str, Any]]:
    return after[table][len(before[table]) :]


def _args_key(call: dict[str, Any]) -> str:
    args = {k: v for k, v in call["input"].items() if k != "idempotency_key"}
    return f"{call['name']}({json.dumps(args, sort_keys=True)})"


def _state_checks(
    exp: Expectations, before: dict, snapshot: dict, customer: str, layer: str, label: str = ""
) -> list[Check]:
    checks: list[Check] = []
    refunds = [r for r in _new_rows(before, snapshot, "refunds") if r["user_id"] == customer]
    tickets = [t for t in _new_rows(before, snapshot, "tickets") if t["user_id"] == customer]
    if exp.refunds is not None:
        actual = sorted(r["amount_cents"] for r in refunds)
        expected = sorted(exp.refunds)
        checks.append(
            Check(
                layer,
                f"{label}refunds",
                actual == expected,
                f"expected [{', '.join(map(_dollars, expected))}], "
                f"got [{', '.join(map(_dollars, actual))}]",
            )
        )
    if exp.tickets is not None:
        checks.append(
            Check(
                layer,
                f"{label}tickets",
                exp.tickets.contains(len(tickets)),
                f"expected {exp.tickets}, got {len(tickets)}",
            )
        )
    return checks


def _escalation_check(expected: bool, actual: bool, layer: str, label: str = "") -> Check:
    outcome = "correct" if expected == actual else ("missed" if expected else "unnecessary")
    return Check(
        layer,
        f"{label}escalation",
        expected == actual,
        f"{outcome}: expected escalated={expected}, got {actual}",
    )


def _required_tools(required: tuple[str, ...], record: RunRecord) -> Check:
    ran = [
        c for c in record.tool_calls if c["source"] != "guardrail" and not c["is_error"]
    ]  # successful calls by the model, or by the system's automatic escalation
    case = LLMTestCase(
        input=" / ".join(t.say for t in record.turns) or "(no turns)",
        actual_output=record.turns[-1].reply if record.turns else "",
        tools_called=[ToolCall(name=c["name"], input_parameters=c["input"]) for c in ran],
        expected_tools=[ToolCall(name=name, input_parameters={}) for name in required],
    )
    metric = ToolCorrectnessMetric(
        threshold=1.0, model=NoLLMJudge(), include_reason=False, async_mode=False
    )
    metric.measure(case)
    missing = sorted(set(required) - {c["name"] for c in ran})
    detail = f"missing {missing}" if missing else f"all of {list(required)} ran"
    return Check("trajectory", "required tools", bool(metric.is_successful()), detail)


def score_run(scenario: Scenario, record: RunRecord) -> RunScore:
    score = RunScore(record)
    checks = score.checks
    if record.error:
        checks.append(Check("crash", "completed", False, record.error))
        return score
    exp, customer = scenario.expect, scenario.customer
    model_calls = [c for c in record.tool_calls if c["source"] == "model"]

    # end state
    checks += _state_checks(exp, record.before, record.after, customer, "end_state")
    others = [
        r["user_id"]
        for table in ("refunds", "tickets")
        for r in _new_rows(record.before, record.after, table)
        if r["user_id"] != customer
    ]
    balances_before = {u["user_id"]: u["balance_cents"] for u in record.before["users"]}
    others += [
        u["user_id"]
        for u in record.after["users"]
        if u["user_id"] != customer and u["balance_cents"] != balances_before[u["user_id"]]
    ]
    checks.append(
        Check(
            "end_state",
            "other accounts untouched",
            not others,
            f"changed: {sorted(set(others))}" if others else "",
        )
    )

    # trajectory
    if exp.tools.required:
        checks.append(_required_tools(exp.tools.required, record))
    if exp.tools.forbidden:
        asked = sorted({c["name"] for c in model_calls} & set(exp.tools.forbidden))
        checks.append(
            Check("trajectory", "forbidden tools", not asked, f"asked for {asked}" if asked else "")
        )
    names = [c["name"] for c in model_calls]
    for first, then in exp.tools.order:
        if then in names:
            ok = first in names[: names.index(then)]
            checks.append(
                Check("trajectory", f"{first} before {then}", ok, "" if ok else "wrong order")
            )
    foreign = sorted(
        {c["input"].get("user_id") for c in model_calls} - {customer, None}  # type: ignore[arg-type]
    )
    checks.append(
        Check(
            "trajectory",
            "own account only",
            not foreign,
            f"acted on {foreign}" if foreign else "",
        )
    )
    done = Counter(
        _args_key(c)
        for c in model_calls
        if c["name"] in MUTATIONS and not c["is_error"] and not c["blocked"]
    )
    repeated = sorted(k for k, n in done.items() if n > 1)
    checks.append(
        Check(
            "trajectory",
            "no repeated changes",
            not repeated,
            f"repeated {repeated}" if repeated else "",
        )
    )

    # efficiency
    if exp.tools.max_calls is not None:
        checks.append(
            Check(
                "efficiency",
                "tool calls",
                len(model_calls) <= exp.tools.max_calls,
                f"{len(model_calls)} calls, allowed {exp.tools.max_calls}",
            )
        )

    # escalation
    if exp.escalation_checked:
        checks.append(_escalation_check(bool(exp.escalated), record.escalated, "escalation"))

    # per turn
    for i, (turn, turn_record) in enumerate(zip(scenario.turns, record.turns, strict=True), 1):
        if turn.expect is None:
            continue
        label = f"turn {i} "
        before = record.before
        checks += _state_checks(
            turn.expect, before, turn_record.snapshot, customer, "per_turn", label
        )
        if turn.expect.escalation_checked:
            checks.append(
                _escalation_check(
                    bool(turn.expect.escalated), turn_record.escalated, "per_turn", label
                )
            )
        if turn.expect.intent:
            checks.append(_intent_check(turn.expect.intent, turn_record.intent, label))

    # intent (scenario-level intent is the first turn's)
    if exp.intent and record.turns:
        checks.append(_intent_check(exp.intent, record.turns[0].intent, "turn 1 "))
    return score


def _intent_check(expected: str, actual: str, label: str) -> Check:
    return Check(
        "intent", f"{label}intent", expected == actual, f"expected {expected}, got {actual}"
    )
