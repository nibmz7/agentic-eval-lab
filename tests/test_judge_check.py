"""The judge check: scripted good and bad conversations with known verdicts."""

import json
from pathlib import Path

import anthropic
import httpx2
import pytest

from support_lab.evals.judge import ClaudeJudge
from support_lab.evals.judge_check import load_cases, run_judge_check
from support_lab.evals.scenario import ScenarioError, load_scenarios

pytestmark = pytest.mark.anyio

ROOT = Path(__file__).parent.parent
SCENARIOS = {s.id: s for s in load_scenarios(ROOT / "evals/scenarios")}
CASES = load_cases(ROOT / "evals/judge_checks.yaml", SCENARIOS)


def constant_judge(score: float) -> ClaudeJudge:
    """A judge that gives every criterion the same score."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        verdict = {"reason": f"always {score}", "score": score}
        return httpx2.Response(
            200,
            json={
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": json.dumps(verdict)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        )

    client = anthropic.AsyncAnthropic(
        api_key="test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return ClaudeJudge(client=client)


def test_cases_have_both_good_and_bad_replies():
    verdicts = [ok for case in CASES for ok in case.expected.values()]
    assert verdicts.count(True) >= 5 and verdicts.count(False) >= 5


async def test_every_case_replays_cleanly_and_is_judged():
    check = await run_judge_check(CASES, lambda: constant_judge(9))
    assert len(check.scores) == len(CASES)
    for score, case in zip(check.scores, CASES, strict=True):
        judged = {c.name for c in score.checks if c.layer == "reply"}
        assert set(case.expected) <= judged, case.id


async def test_a_judge_that_passes_everything_is_too_lenient():
    check = await run_judge_check(CASES, lambda: constant_judge(9))
    bad = sum(not ok for case in CASES for ok in case.expected.values())
    assert check.agreement.judge_too_lenient == bad
    assert check.agreement.judge_too_strict == 0
    assert check.agreement.kappa == pytest.approx(0.0)


async def test_a_judge_that_fails_everything_is_too_strict():
    check = await run_judge_check(CASES, lambda: constant_judge(1))
    good = sum(ok for case in CASES for ok in case.expected.values())
    assert check.agreement.judge_too_strict == good
    assert check.agreement.judge_too_lenient == 0


def test_a_case_naming_a_criterion_its_scenario_lacks_is_rejected(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text(
        "- id: x\n  scenario: small-refund\n  description: d\n"
        "  agent: [{reply: hi}]\n  judge_should: {asks-for-amount: pass}\n"
    )
    with pytest.raises(ScenarioError, match="has no criterion 'asks-for-amount'"):
        load_cases(path, SCENARIOS)
