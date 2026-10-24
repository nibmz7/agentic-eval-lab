"""Eval results on Langfuse traces.

Each scored run's trace gets its results as scores, so in the Langfuse UI you
can filter traces to the failing ones (passed = 0), or to one failing layer,
and open the conversation, the model calls and the tool calls behind it.
"""

from __future__ import annotations

from typing import Any

from support_lab import tracing

from .checks import LAYERS, RunScore
from .runner import RunRecord


def trace_scores(score: RunScore) -> list[dict[str, Any]]:
    """Scores for a run's trace: 1 for pass, 0 for fail. Comments say what failed.

    passed           the run's verdict
    <layer>          each scored layer that ran (crash, end_state, trajectory, ...)
    reply: <name>    each judged criterion, which doesn't count toward `passed`
    """
    failing = "; ".join(f"{c.layer}: {c.name}" for c in score.failures)
    items: list[dict[str, Any]] = [
        {"name": "passed", "value": float(score.passed), "comment": failing or None}
    ]
    for layer in LAYERS:
        checks = [c for c in score.checks if c.layer == layer]
        if layer == "reply" or not checks:
            continue
        failed = [c for c in checks if not c.passed]
        details = "\n".join(f"{c.name}: {c.detail}" if c.detail else c.name for c in failed)
        items.append({"name": layer, "value": float(not failed), "comment": details or None})
    for c in score.checks:
        if c.layer == "reply" and not c.detail.startswith("judge error"):
            name = f"reply: {c.name}"
            items.append({"name": name, "value": float(c.passed), "comment": c.detail})
    return items


def send_scores(scores: list[RunScore]) -> None:
    for score in scores:
        trace_id = score.record.trace_id
        if trace_id is None:
            continue
        for item in trace_scores(score):
            tracing.score(
                trace_id, item["name"], item["value"], boolean=True, comment=item["comment"]
            )


def link_traces(records: list[RunRecord]) -> None:
    """Fill in each record's trace URL, for the report."""
    for record in records:
        if record.trace_id is None:
            continue
        record.trace_url = tracing.trace_url(record.trace_id)
        if record.trace_url is None:
            return  # Langfuse didn't answer; the report shows trace ids instead
