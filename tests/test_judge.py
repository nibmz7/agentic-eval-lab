"""The reply judge, cost estimates and calibration, without calling Claude.

ClaudeJudge talks to a mock HTTP transport that returns canned scores, so these
tests cover the real path: our judge -> DeepEval's GEval -> our checks.
"""

import json
from pathlib import Path

import anthropic
import httpx2
import pytest

from support_lab.agent import AgentSettings, Intent
from support_lab.agent.scripted import ScriptedModel, text, tool_use
from support_lab.evals.calibrate import (
    agreement,
    judged_items,
    labels_document,
    rejudge,
    sample,
)
from support_lab.evals.checks import score_run
from support_lab.evals.judge import GENERAL_CRITERIA, ClaudeJudge, judge_scores, transcript
from support_lab.evals.pricing import call_cost
from support_lab.evals.report import load_scores, render, summarize, write_json
from support_lab.evals.runner import RunRecord, run_scenarios
from support_lab.evals.scenario import load_scenarios

pytestmark = pytest.mark.anyio

SCENARIOS = {s.id: s for s in load_scenarios(Path(__file__).parent.parent / "evals/scenarios")}
FAST = AgentSettings(retry_delays=())


def judge_with(scores: list[float]) -> tuple[ClaudeJudge, list[dict]]:
    """A ClaudeJudge whose API calls return these GEval scores in order."""
    bodies: list[dict] = []
    queue = list(scores)

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        score = queue.pop(0)
        return httpx2.Response(
            200,
            json={
                "id": "msg_judge",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    {
                        "type": "text",
                        "text": json.dumps({"reason": f"scored {score}", "score": score}),
                    },
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1000, "output_tokens": 200},
            },
        )

    client = anthropic.AsyncAnthropic(
        api_key="test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return ClaudeJudge(client=client), bodies


async def over_limit_run():
    scenario = SCENARIOS["refund-over-limit"]
    turns = [
        [tool_use("escalate_ticket", user_id="u_bob", reason="900 is over the limit")],
        [text("That's over our automatic limit, so I've passed it to a colleague.")],
    ]
    (record,) = await run_scenarios(
        [scenario], lambda s, r: ScriptedModel(turns, intent=Intent.REFUND_REQUEST), settings=FAST
    )
    return scenario, record


# -- the judge ----------------------------------------------------------------------


async def test_judge_request_follows_haiku_rules_and_counts_tokens():
    scenario, record = await over_limit_run()
    judge, bodies = judge_with([9, 9])
    score = score_run(scenario, record)

    await judge_scores(SCENARIOS, [score], lambda: judge)

    body = bodies[0]
    assert body["model"] == "claude-haiku-5-5"
    assert "temperature" not in body and "fallbacks" not in body
    assert body["output_config"] == {"effort": "low"}
    assert "Customer: I'd like a refund of $900" in body["messages"][0]["content"]
    assert record.judge_model == "claude-haiku-5-5"
    assert len(record.judge_usage) == 2  # one call per criterion, no hidden extra calls


async def test_one_reply_check_per_criterion_including_the_general_one():
    scenario, record = await over_limit_run()
    judge, _ = judge_with([9, 2])
    score = score_run(scenario, record)

    await judge_scores(SCENARIOS, [score], lambda: judge)

    replies = {c.name: c for c in score.checks if c.layer == "reply"}
    assert set(replies) == set(GENERAL_CRITERIA) | {"explains-escalation"}
    assert [c.passed for c in replies.values()] == [True, False]
    assert replies["explains-escalation"].detail.startswith("0.2: scored 2")
    # Judge verdicts are reported, but don't decide whether the run passed.
    assert score.passed


async def test_crashed_runs_are_not_judged():
    scenario = SCENARIOS["small-refund"]
    record = RunRecord("small-refund", 1, "u_alice", before={}, error="Boom: down")
    score = score_run(scenario, record)
    judge, bodies = judge_with([])

    await judge_scores(SCENARIOS, [score], lambda: judge)

    assert bodies == []
    assert [c.layer for c in score.checks] == ["crash"]


async def test_a_judge_failure_is_recorded_not_raised():
    scenario, record = await over_limit_run()
    judge, _ = judge_with([])  # empty queue: the mock transport errors
    score = score_run(scenario, record)

    await judge_scores(SCENARIOS, [score], lambda: judge)

    replies = [c for c in score.checks if c.layer == "reply"]
    assert replies and all(c.detail.startswith("judge error") for c in replies)
    assert summarize([score]).judge_errors == len(replies)


async def test_transcript_shows_tool_results_but_not_guardrail_lookups():
    scenario = SCENARIOS["small-refund"]
    turns = [[tool_use("process_refund", user_id="u_alice", amount=20)], [text("Refunded 20.00.")]]
    (record,) = await run_scenarios([scenario], lambda s, r: ScriptedModel(turns), settings=FAST)

    shown = transcript(record)

    assert shown.splitlines() == [
        "Customer: Please refund $20 from my account.",
        "  [model tool call] process_refund(user_id='u_alice', amount=20) → ok: new balance 100.00",
        "Agent: Refunded 20.00.",
    ]


# -- cost -------------------------------------------------------------------------------


def test_cost_estimates():
    million_in = {"input_tokens": 1_000_000, "output_tokens": 0}
    assert call_cost("claude-haiku-5-5", million_in) == pytest.approx(0.50)  # long-prompt rate
    small = {"input_tokens": 10_000, "output_tokens": 2_000}
    assert call_cost("claude-haiku-5-5", small) == pytest.approx(0.002)
    assert call_cost("claude-opus-5-5", small) == pytest.approx(0.08)
    assert call_cost("some-other-model", small) is None


async def test_report_shows_tokens_and_cost():
    scenario, record = await over_limit_run()
    record.model = "claude-haiku-5-5"
    record.usage = [{"input_tokens": 10_000, "output_tokens": 2_000}] * 3
    judge, _ = judge_with([9, 9])
    score = score_run(scenario, record)
    await judge_scores(SCENARIOS, [score], lambda: judge)

    report = render(summarize([score]))

    assert "agent: 3 calls, 30,000 input + 6,000 output tokens, ≈ $0.0060" in report
    assert "judge: 2 calls, 2,000 input + 400 output tokens, ≈ $0.0004" in report
    assert "reply, judged (not scored until calibrated)" in report


# -- calibration -------------------------------------------------------------------------


def results_with(verdicts: dict[str, bool]) -> dict:
    turns = [{"say": "hi", "reply": "hello", "tool_calls": []}]
    return {
        "runs": [
            {
                "scenario_id": "refund-over-limit",
                "run": run,
                "turns": turns,
                "checks": [
                    {
                        "layer": "reply",
                        "name": "explains-escalation",
                        "passed": ok,
                        "detail": f"{'0.9' if ok else '0.1'}: reason {run}",
                    }  # fmt: skip
                ],
            }
            for run, ok in ((int(k), v) for k, v in verdicts.items())
        ]
    }


def test_labels_file_hides_the_judges_verdicts():
    items = judged_items(results_with({"1": True, "2": False, "3": True, "4": False}))
    chosen = sample(items, 4)
    criteria = {"refund-over-limit": {"explains-escalation": "Explains the escalation."}}

    document = labels_document(chosen, criteria)

    assert sorted(i.judge_passed for i in chosen) == [False, False, True, True]
    assert "reason" not in document and "0.9" not in document and "0.1" not in document
    assert "Explains the escalation." in document and "human: null" in document


def test_agreement_and_kappa():
    # Judge: pass, pass, fail, fail. You: pass, fail, fail, fail.
    items = judged_items(results_with({"1": True, "2": True, "3": False, "4": False}))
    labels = [
        {"id": "refund-over-limit#1:explains-escalation", "human": "pass"},
        {"id": "refund-over-limit#2:explains-escalation", "human": "fail", "note": "vague"},
        {"id": "refund-over-limit#3:explains-escalation", "human": "fail"},
        {"id": "refund-over-limit#4:explains-escalation", "human": "fail"},
        {"id": "refund-over-limit#5:explains-escalation", "human": None},  # skipped
    ]

    result = agreement(items, labels)

    assert (result.labelled, result.agree) == (4, 3)
    # observed 0.75; chance = 0.5*0.25 + 0.5*0.75 = 0.5; kappa = (0.75-0.5)/(1-0.5)
    assert result.kappa == pytest.approx(0.5)
    assert (result.judge_too_lenient, result.judge_too_strict) == (1, 0)
    (item, human, note) = result.disagreements[0]
    assert (item.id, human, note) == ("refund-over-limit#2:explains-escalation", False, "vague")


def test_labels_for_unknown_items_are_listed_not_fatal():
    items = judged_items(results_with({"1": True}))
    result = agreement(
        items,
        [
            {"id": "refund-over-limit#1:explains-escalation", "human": "pass"},
            {"id": "refund-over-limit#1:old-name", "human": "fail"},
        ],
    )
    assert (result.labelled, result.unmatched) == (1, ["refund-over-limit#1:old-name"])


# -- re-judging saved conversations ----------------------------------------------------------


async def test_rejudge_replaces_verdicts_on_the_same_conversations(tmp_path):
    scenario, record = await over_limit_run()
    score = score_run(scenario, record)
    first, _ = judge_with([2, 2])  # the first judge fails both criteria
    await judge_scores(SCENARIOS, [score], lambda: first)
    path = tmp_path / "results.json"
    write_json(path, summarize([score]), [score], {"model": "scripted"})
    results = json.loads(path.read_text())
    labels = [{"id": "refund-over-limit#1:explains-escalation", "human": "pass"}]
    assert agreement(judged_items(results), labels).agree == 0

    reloaded = load_scores(results, {"refund-over-limit": "u_bob"})
    second, bodies = judge_with([9, 9])  # e.g. after rewording the criterion
    await rejudge(reloaded, SCENARIOS, lambda: second)
    write_json(path, summarize(reloaded), reloaded, results["meta"])
    rejudged = json.loads(path.read_text())

    # Same conversation, same ids, new verdicts: the old label now agrees.
    assert "Customer: I'd like a refund of $900" in bodies[0]["messages"][0]["content"]
    assert agreement(judged_items(rejudged), labels).agree == 1
    run = rejudged["runs"][0]
    assert [c["layer"] for c in run["checks"]].count("reply") == 2  # replaced, not appended
    assert len(run["judge_usage"]) == 2
    assert run["turns"] == results["runs"][0]["turns"]


async def test_rejudge_refuses_results_for_removed_scenarios():
    scenario, record = await over_limit_run()
    score = score_run(scenario, record)
    judge, _ = judge_with([])
    with pytest.raises(ValueError, match="no longer exist"):
        await rejudge([score], {}, lambda: judge)


def test_items_the_judge_failed_on_are_reported_separately():
    results = results_with({"1": True})
    results["runs"][0]["checks"][0]["detail"] = "judge error: AuthenticationError: bad key"
    items = judged_items(results)

    result = agreement(items, [{"id": "refund-over-limit#1:explains-escalation", "human": "pass"}])

    assert sample(items, 5) == []
    assert (result.labelled, result.unmatched) == (0, [])
    assert result.judge_failed == ["refund-over-limit#1:explains-escalation"]
