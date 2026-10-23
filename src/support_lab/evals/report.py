"""Summaries of scored runs: per scenario, per layer, and overall.

Each scenario runs n times. pass^k is the chance the agent gets a scenario
right k times in a row, estimated the way τ-bench does: for a scenario with c
passes out of n runs, it's the chance that k runs picked at random from those
n are all passes, C(c, k) / C(n, k), averaged over scenarios.

pass^1 is plain accuracy. pass^n is the share of scenarios that passed every
run. The values in between show how fast reliability drops when the agent has
to be right repeatedly, all from the same n runs. Being right once is not
enough for an agent handling money: the same request comes in again and again.
"""

from __future__ import annotations

import json
import math
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


def pass_hat_k(passed: int, runs: int, k: int) -> float:
    """τ-bench's unbiased estimate of k-in-a-row success for one scenario."""
    if not 1 <= k <= runs:
        raise ValueError(f"k must be between 1 and {runs}, got {k}")
    return math.comb(passed, k) / math.comb(runs, k)


@dataclass
class Summary:
    scenarios: list[ScenarioSummary]
    runs: int
    runs_per_scenario: int  # n
    pass_hat: dict[int, float]  # k -> pass^k, for k = 1..n
    layer_pass_rate: dict[str, float]  # share of completed runs with no failure in that layer
    completed: int
    escalation: dict[str, int]  # correct / missed / unnecessary
    intent_accuracy: float | None
    crashes: int
    crash_reasons: dict[str, int]


def summarize(scores: list[RunScore]) -> Summary:
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
    n = min((s.runs for s in scenarios), default=0)
    pass_hat = {
        k: sum(pass_hat_k(s.passed, s.runs, k) for s in scenarios) / len(scenarios)
        for k in range(1, n + 1)
    }
    intents = [c for s in scores for c in s.checks if c.layer == "intent"]
    escalation = Counter(o for s in scores if (o := s.escalation_outcome))
    return Summary(
        scenarios=scenarios,
        runs=len(scores),
        runs_per_scenario=n,
        pass_hat=pass_hat,
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
        f"runs: {summary.runs} ({summary.runs_per_scenario} per scenario)   "
        f"crashes: {summary.crashes}"
    )
    curve = "   ".join(f"^{k} {v:.0%}" for k, v in summary.pass_hat.items())
    lines.append(f"pass^k (right k times in a row): {curve}")
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
