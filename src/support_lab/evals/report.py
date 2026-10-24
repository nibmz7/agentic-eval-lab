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

from .checks import INFORMATIONAL, LAYERS, Check, RunScore
from .pricing import PRICES_AS_OF, total_cost
from .runner import RunRecord, TurnRecord


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
    reply_pass_rate: dict[str, float]  # judge criterion -> share of judged runs it passed
    judge_errors: int
    tokens: dict[str, dict[str, int]]  # "agent" / "judge" -> input, output
    cost: dict[str, float | None]  # "agent" / "judge" -> estimated USD, None if unknown


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
    replies: dict[str, list[bool]] = defaultdict(list)
    judge_errors = 0
    for s in scores:
        for c in s.checks:
            if c.layer != "reply":
                continue
            if c.detail.startswith("judge error"):
                judge_errors += 1
            else:
                replies[c.name].append(c.passed)
    tokens, cost = _usage(scores)
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
        reply_pass_rate={name: sum(v) / len(v) for name, v in replies.items()},
        judge_errors=judge_errors,
        tokens=tokens,
        cost=cost,
        crash_reasons=dict(
            Counter(s.record.error.splitlines()[0][:160] for s in scores if s.record.error)
        ),
    )


def _usage(scores: list[RunScore]) -> tuple[dict[str, dict[str, int]], dict[str, float | None]]:
    tokens: dict[str, dict[str, int]] = {}
    cost: dict[str, float | None] = {}
    for role, model_of, usage_of in (
        ("agent", lambda r: r.model, lambda r: r.usage),
        ("judge", lambda r: r.judge_model, lambda r: r.judge_usage),
    ):
        calls = [(model_of(s.record), u) for s in scores for u in usage_of(s.record)]
        if not calls:
            continue
        tokens[role] = {
            "input": sum(u["input_tokens"] for _, u in calls),
            "output": sum(u["output_tokens"] for _, u in calls),
            "calls": len(calls),
        }
        costs = [total_cost(model, [u]) if model else None for model, u in calls]
        cost[role] = None if None in costs else sum(c for c in costs if c is not None)
    return tokens, cost


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
    if summary.reply_pass_rate or summary.judge_errors:
        rates = "   ".join(f"{n} {r:.0%}" for n, r in sorted(summary.reply_pass_rate.items()))
        errors = f"   judge errors: {summary.judge_errors}" if summary.judge_errors else ""
        lines.append(f"reply, judged (not scored until calibrated): {rates or '-'}{errors}")
    for role, t in summary.tokens.items():
        dollars = summary.cost.get(role)
        price = f"≈ ${dollars:.4f}" if dollars is not None else "cost unknown for this model"
        lines.append(
            f"{role}: {t['calls']} calls, {t['input']:,} input + {t['output']:,} output tokens, "
            f"{price}"
        )
    if summary.tokens:
        lines.append(f"(estimates from prices as of {PRICES_AS_OF})")
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
            "model": s.record.model,
            "usage": s.record.usage,
            "judge_model": s.record.judge_model,
            "judge_usage": s.record.judge_usage,
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


def load_scores(results: dict[str, Any], customers: dict[str, str]) -> list[RunScore]:
    """Rebuild scored runs from a results file (the inverse of `write_json`).

    Database snapshots aren't saved, so the rebuilt runs can be judged and
    summarised again but not re-scored by the deterministic checks.
    """
    scores = []
    for run in results["runs"]:
        record = RunRecord(
            scenario_id=run["scenario_id"],
            run=run["run"],
            customer=customers.get(run["scenario_id"], ""),
            before={},
            turns=[
                TurnRecord(
                    say=t["say"],
                    reply=t["reply"],
                    intent=t["intent"],
                    escalated=t["escalated"],
                    tool_calls=t["tool_calls"],
                    snapshot={},
                )
                for t in run["turns"]
            ],
            error=run["error"],
            seconds=run.get("seconds", 0.0),
            model=run.get("model"),
            usage=run.get("usage", []),
            judge_model=run.get("judge_model"),
            judge_usage=run.get("judge_usage", []),
        )
        scores.append(RunScore(record, [Check(**c) for c in run["checks"]]))
    return scores
