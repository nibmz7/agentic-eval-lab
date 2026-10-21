"""Summaries of scored runs: per scenario, per layer, and overall.

pass@1 is the share of all runs that passed. pass^k is the share of scenarios
that passed in *every* one of their k runs, which is what matters when a
customer gets one try: an agent that passes 80% of runs can still fail most
scenarios at least once.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .checks import INFORMATIONAL, LAYERS, RunScore


@dataclass
class ScenarioSummary:
    scenario_id: str
    runs: int
    passed: int
    failing_checks: dict[str, int]  # "layer: check" -> number of runs it failed in

    @property
    def passed_every_run(self) -> bool:
        return self.passed == self.runs


@dataclass
class Summary:
    scenarios: list[ScenarioSummary]
    runs: int
    pass_at_1: float
    pass_k: float
    k: int
    layer_pass_rate: dict[str, float]  # share of completed runs with no failure in that layer
    completed: int
    escalation: dict[str, int]  # correct / missed / unnecessary
    intent_accuracy: float | None
    crashes: int
    crash_reasons: dict[str, int]


def summarize(scores: list[RunScore], k: int) -> Summary:
    by_scenario: dict[str, list[RunScore]] = defaultdict(list)
    for score in scores:
        by_scenario[score.record.scenario_id].append(score)
    scenarios = []
    for scenario_id, runs in by_scenario.items():
        failing: Counter[str] = Counter()
        for run in runs:
            failing.update({f"{c.layer}: {c.name}" for c in run.failures})
        scenarios.append(
            ScenarioSummary(scenario_id, len(runs), sum(r.passed for r in runs), dict(failing))
        )
    # A crashed run never reaches the other checks, so counting it would make
    # those layers look better than they are. Layer rates cover completed runs.
    completed = [s for s in scores if not s.record.error]
    layer_rate = {}
    for layer in LAYERS:
        if layer in INFORMATIONAL or layer == "crash" or not completed:
            continue
        failed = sum(any(c.layer == layer and not c.passed for c in s.checks) for s in completed)
        layer_rate[layer] = 1 - failed / len(completed)
    intents = [c for s in scores for c in s.checks if c.layer == "intent"]
    escalation = Counter(o for s in scores if (o := s.escalation_outcome))
    return Summary(
        scenarios=scenarios,
        runs=len(scores),
        pass_at_1=sum(s.passed for s in scores) / len(scores) if scores else 0.0,
        pass_k=sum(s.passed_every_run for s in scenarios) / len(scenarios) if scenarios else 0.0,
        k=k,
        layer_pass_rate=layer_rate,
        escalation={o: escalation.get(o, 0) for o in ("correct", "missed", "unnecessary")},
        intent_accuracy=sum(c.passed for c in intents) / len(intents) if intents else None,
        completed=len(completed),
        crashes=len(scores) - len(completed),
        crash_reasons=dict(
            Counter(s.record.error.splitlines()[0][:160] for s in scores if s.record.error)
        ),
    )


def render(summary: Summary) -> str:
    width = max([len(s.scenario_id) for s in summary.scenarios] + [8])
    lines = [f"{'scenario':<{width}}  passed  every run  failing checks"]
    for s in summary.scenarios:
        failing = ", ".join(
            f"{name} ({n}/{s.runs})" for name, n in sorted(s.failing_checks.items())
        )
        mark = "yes" if s.passed_every_run else "no"
        lines.append(f"{s.scenario_id:<{width}}  {s.passed:>2}/{s.runs:<3}  {mark:<9}  {failing}")
    lines.append("")
    lines.append(
        f"runs: {summary.runs}   pass@1: {summary.pass_at_1:.0%}   "
        f"pass^{summary.k}: {summary.pass_k:.0%}   crashes: {summary.crashes}"
    )
    if summary.layer_pass_rate:
        rates = "   ".join(f"{layer} {r:.0%}" for layer, r in summary.layer_pass_rate.items())
        lines.append(f"by layer ({summary.completed} completed runs): {rates}")
    else:
        lines.append("by layer: no run completed")
    for reason, n in sorted(summary.crash_reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"crashed ({n}x): {reason}")
    e = summary.escalation
    lines.append(
        f"escalation: {e['correct']} correct, {e['missed']} missed, {e['unnecessary']} unnecessary"
    )
    if summary.intent_accuracy is not None:
        lines.append(f"intent accuracy (not scored): {summary.intent_accuracy:.0%}")
    return "\n".join(lines)


def write_json(path: Path, summary: Summary, scores: list[RunScore], meta: dict[str, Any]) -> None:
    """Everything needed to debug a failure later: replies, tool calls, every check."""
    path.parent.mkdir(parents=True, exist_ok=True)
    runs = [
        {
            "scenario_id": s.record.scenario_id,
            "run": s.record.run,
            "passed": s.passed,
            "seconds": round(s.record.seconds, 2),
            "error": s.record.error,
            "checks": [asdict(c) for c in s.checks],
            "turns": [
                {
                    "say": t.say,
                    "reply": t.reply,
                    "intent": t.intent,
                    "escalated": t.escalated,
                    "tool_calls": t.tool_calls,
                }
                for t in s.record.turns
            ],
        }
        for s in scores
    ]
    body = {"meta": meta, "summary": asdict(summary), "runs": runs}
    path.write_text(json.dumps(body, indent=2, default=str))
