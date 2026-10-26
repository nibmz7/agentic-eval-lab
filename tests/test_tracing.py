"""Langfuse tracing, checked against an in-memory span exporter.

Nothing is sent over the network. The spans are the ones Langfuse would
receive, so these tests pin down the trace tree the docs describe: an eval
run, its turns, the graph's nodes, and the model and tool calls inside them.
"""

import json
from pathlib import Path

import pytest
from test_anthropic_model import fake_api
from test_judge import judge_with

from support_lab import tracing
from support_lab.agent import AgentSettings, Intent
from support_lab.agent.scripted import ScriptedModel, text, tool_use
from support_lab.evals.checks import Check, RunScore, score_run
from support_lab.evals.judge import judge_scores
from support_lab.evals.report import render, summarize
from support_lab.evals.runner import RunRecord, run_once, run_scenarios
from support_lab.evals.scenario import load_scenarios
from support_lab.evals.traces import send_scores, trace_scores

pytestmark = pytest.mark.anyio

SCENARIOS = {s.id: s for s in load_scenarios(Path(__file__).parent.parent / "evals/scenarios")}
FAST = AgentSettings(retry_delays=())
TYPE = "langfuse.observation.type"


def parents(spans) -> dict[str, str | None]:
    """Span name -> its parent's name. Names repeated in a trace keep the last one."""
    by_id = {s.context.span_id: s.name for s in spans}
    return {s.name: by_id.get(s.parent.span_id) if s.parent else None for s in spans}


def named(spans, name: str):
    return next(s for s in spans if s.name == name)


def over_limit_model():
    return ScriptedModel(
        [
            [tool_use("process_refund", user_id="u_bob", amount=900)],
            [tool_use("escalate_ticket", user_id="u_bob", reason="900 is over the limit")],
            [text("That's over our $500 limit, so a colleague will handle it.")],
        ],
        intent=Intent.REFUND_REQUEST,
    )


async def test_an_eval_run_is_one_trace_with_the_turn_nodes_and_calls_nested(traced):
    scenario = SCENARIOS["refund-over-limit"]
    record = await run_once(scenario, 1, over_limit_model(), FAST, session="eval-test")
    spans = traced()

    root = named(spans, "eval refund-over-limit #1")
    assert record.trace_id == f"{root.context.trace_id:032x}"
    assert {f"{s.context.trace_id:032x}" for s in spans} == {record.trace_id}

    tree = parents(spans)
    assert tree["eval refund-over-limit #1"] is None
    assert tree["support turn"] == "eval refund-over-limit #1"
    assert tree["LangGraph"] == "support turn"
    for node in ("planner", "agent", "guardrail", "tools"):
        assert tree[node] == "LangGraph"
    # The guardrail's verdict, and the balance lookup it made, sit under the guardrail node.
    assert tree["refund check"] == "guardrail"
    assert tree["check_balance"] == "refund check"
    assert tree["escalate_ticket"] == "tools"

    check = named(spans, "refund check")
    assert check.attributes[TYPE] == "guardrail"
    assert "requires_escalation" in check.attributes["langfuse.observation.output"]
    assert named(spans, "escalate_ticket").attributes[TYPE] == "tool"
    assert named(spans, "support turn").attributes[TYPE] == "agent"

    # Trace attributes reach every span, so Langfuse can filter and group on them.
    assert root.attributes["session.id"] == "eval-test"
    assert root.attributes["langfuse.trace.name"] == "eval refund-over-limit"
    assert set(root.attributes["langfuse.trace.tags"]) == {"eval", "refund-over-limit"}


async def test_failed_tool_calls_are_marked_as_errors(traced):
    scenario = SCENARIOS["refund-over-limit"]
    model = ScriptedModel(
        [[tool_use("process_refund", user_id="u_nobody", amount=5)], [text("Sorry.")]],
        intent=Intent.REFUND_REQUEST,
    )
    await run_once(scenario, 1, model, FAST)
    call = named(traced(), "process_refund")
    assert call.attributes["langfuse.observation.level"] == "ERROR"
    assert call.attributes["langfuse.observation.status_message"] == "unknown_user"


async def test_concurrent_runs_get_separate_traces(traced):
    scenario = SCENARIOS["refund-over-limit"]
    records = await run_scenarios(
        [scenario], lambda s, r: over_limit_model(), runs=3, settings=FAST, session="eval-test"
    )
    spans = traced()
    trace_ids = [r.trace_id for r in records]
    assert len(set(trace_ids)) == 3
    for trace_id in trace_ids:
        names = {s.name for s in spans if f"{s.context.trace_id:032x}" == trace_id}
        assert len([n for n in names if n.startswith("eval ")]) == 1


async def test_claude_calls_are_generations_with_model_and_tokens(traced):
    model, _ = fake_api([{"type": "text", "text": "Hi"}], "end_turn")
    await model.respond(system="be nice", messages=[{"role": "user", "content": "hi"}], tools=[])
    generation = named(traced(), "respond")
    assert generation.attributes[TYPE] == "generation"
    assert generation.attributes["langfuse.observation.model.name"] == model.model
    usage = json.loads(generation.attributes["langfuse.observation.usage_details"])
    assert usage["input"] == 10
    assert usage["output"] == 5


async def test_judge_calls_join_the_runs_trace(traced):
    scenario = SCENARIOS["refund-over-limit"]
    record = await run_once(scenario, 1, over_limit_model(), FAST)
    score = score_run(scenario, record)
    judge, _ = judge_with([9, 9])
    await judge_scores({scenario.id: scenario}, [score], lambda: judge)
    spans = traced()

    evaluators = [s for s in spans if s.attributes.get(TYPE) == "evaluator"]
    assert {s.name for s in evaluators} == {
        "judge honest-about-outcomes",
        "judge explains-escalation",
    }
    assert {f"{s.context.trace_id:032x}" for s in evaluators} == {record.trace_id}
    assert (parents(spans)["judge call"] or "").startswith("judge ")


async def test_with_tracing_off_nothing_is_recorded(_span_exporter):
    _span_exporter.clear()
    scenario = SCENARIOS["refund-over-limit"]
    record = await run_once(scenario, 1, over_limit_model(), FAST, session="eval-test")
    assert record.trace_id is None
    assert not tracing.enabled()
    assert _span_exporter.get_finished_spans() == ()


def test_tracing_follows_the_environment(monkeypatch):
    monkeypatch.setattr(tracing, "_public_key", tracing._UNSET)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert not tracing.enabled()  # one key isn't enough


# -- results on traces ----------------------------------------------------------------


def scored_run(checks: list[Check], trace_id: str | None = "t1") -> RunScore:
    record = RunRecord("refund-over-limit", 1, "u_bob", before={}, trace_id=trace_id)
    return RunScore(record, checks)


def test_scores_cover_the_verdict_each_layer_and_each_judged_criterion():
    score = scored_run(
        [
            Check("end_state", "refunds", True),
            Check("trajectory", "forbidden tools", False, "asked for process_refund"),
            Check("trajectory", "order", True),
            Check("intent", "intent", True),
            Check("reply", "explains-escalation", False, "0.2: never says why"),
            Check("reply", "honest-about-outcomes", False, "judge error: timeout"),
        ]
    )
    items = {i["name"]: (i["value"], i["comment"]) for i in trace_scores(score)}
    assert items == {
        "passed": (0.0, "trajectory: forbidden tools"),
        "end_state": (1.0, None),
        "trajectory": (0.0, "forbidden tools: asked for process_refund"),
        "intent": (1.0, None),
        "reply: explains-escalation": (0.0, "0.2: never says why"),
        # a judge error is not a verdict, so it isn't sent as one
    }


def test_scores_go_to_the_runs_trace(monkeypatch):
    sent = []
    monkeypatch.setattr(tracing, "score", lambda *args, **kwargs: sent.append((args, kwargs)))
    send_scores([scored_run([Check("end_state", "refunds", True)]), scored_run([], None)])
    assert [args for args, _ in sent] == [("t1", "passed", 1.0), ("t1", "end_state", 1.0)]
    assert all(kwargs["boolean"] for _, kwargs in sent)


def test_report_lists_failed_runs_with_their_trace():
    failing = scored_run([Check("escalation", "escalation", False)])
    failing.record.trace_url = "http://localhost:3000/project/p/traces/t1"
    untraced = scored_run([Check("end_state", "tickets", False)], trace_id=None)
    report = render(summarize([failing, untraced, scored_run([])]))
    assert "failed runs:\n  refund-over-limit #1: escalation: escalation\n" in report
    assert "    trace: http://localhost:3000/project/p/traces/t1\n" in report
    assert "  refund-over-limit #1: end_state: tickets" in report
