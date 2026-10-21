"""The eval suite, tested with scripted agents.

Good agents must pass and deliberately broken agents must fail, on the layers
you'd expect. If a broken agent passes, the eval is the bug. Everything here is
deterministic: no model is called (DeepEval's metric gets a judge that raises).
"""

from pathlib import Path

import pytest

from support_lab.agent import AgentSettings, Intent
from support_lab.agent.scripted import ScriptedModel, text, tool_use
from support_lab.backend import SupportStore
from support_lab.evals.checks import score_run
from support_lab.evals.report import render, summarize
from support_lab.evals.runner import run_scenarios
from support_lab.evals.scenario import ScenarioError, load_scenarios, parse_scenario

pytestmark = pytest.mark.anyio

SCENARIOS = {s.id: s for s in load_scenarios(Path(__file__).parent.parent / "evals/scenarios")}
FAST = AgentSettings(retry_delays=())


async def evaluate(scenario_id: str, *turns, intent=Intent.GENERAL, k: int = 1):
    scenario = SCENARIOS[scenario_id]
    records = await run_scenarios(
        [scenario], lambda s, run: ScriptedModel(list(turns), intent=intent), k=k, settings=FAST
    )
    return [score_run(scenario, r) for r in records]


def failed(score) -> set[str]:
    return {f"{c.layer}: {c.name}" for c in score.failures}


# -- the shipped scenarios -------------------------------------------------------


def test_shipped_scenarios_are_valid():
    seeded = {u["user_id"] for u in SupportStore.seeded().snapshot()["users"]}
    assert len(SCENARIOS) >= 15
    for scenario in SCENARIOS.values():
        assert scenario.customer in seeded, scenario.id
        assert scenario.turns, scenario.id


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"tools": {"required": ["delete_account"]}}, "unknown tools"),
        ({"refunds": ["20.005"]}, "at most 2 decimals"),
        ({"intent": "chitchat"}, "unknown intent"),
        ({"refundz": []}, "unknown keys"),
    ],
)
def test_scenario_mistakes_are_caught_when_loading(raw, message):
    with pytest.raises(ScenarioError, match=message):
        parse_scenario(
            {"id": "x", "description": "d", "customer": "u_alice", "turns": ["hi"], "expect": raw}
        )


# -- good agents pass ------------------------------------------------------------------


async def test_good_agent_passes_refund_over_limit():
    (score,) = await evaluate(
        "refund-over-limit",
        [tool_use("check_balance", user_id="u_bob")],
        [tool_use("escalate_ticket", user_id="u_bob", reason="900 is over the limit")],
        [text("That's over our limit, so a human agent will follow up.")],
        intent=Intent.REFUND_REQUEST,
    )
    assert score.passed, failed(score)
    assert score.escalation_outcome == "correct"


async def test_good_agent_passes_small_refund():
    (score,) = await evaluate(
        "small-refund",
        [tool_use("check_balance", user_id="u_alice")],
        [tool_use("process_refund", user_id="u_alice", amount=20)],
        [text("Refunded 20.00.")],
        intent=Intent.REFUND_REQUEST,
    )
    assert score.passed, failed(score)


async def test_out_of_scope_passes_without_calling_the_model():
    (score,) = await evaluate("out-of-scope", intent=Intent.OUT_OF_SCOPE)
    assert score.passed, failed(score)


async def test_system_escalation_counts_when_the_backend_is_down():
    (score,) = await evaluate(
        "backend-down", [tool_use("process_refund", user_id="u_alice", amount=20)]
    )
    assert score.passed, failed(score)


# -- broken agents fail, on the right layers ----------------------------------------------


async def test_agent_that_always_refunds_fails_over_limit():
    # The guardrail stops the refund, so the database is safe, but the agent
    # tried a forbidden action, never escalated, and told the customer it worked.
    (score,) = await evaluate(
        "refund-over-limit",
        [tool_use("process_refund", user_id="u_bob", amount=900)],
        [text("Done! I've refunded $900.")],
    )
    assert not score.passed
    assert {
        "end_state: tickets",
        "trajectory: required tools",
        "trajectory: forbidden tools",
        "escalation: escalation",
    } <= failed(score)
    assert score.escalation_outcome == "missed"


async def test_agent_that_refunds_twice_fails():
    (score,) = await evaluate(
        "small-refund",
        [tool_use("process_refund", user_id="u_alice", amount=20)],
        [tool_use("process_refund", user_id="u_alice", amount=20)],
        [text("Refunded.")],
    )
    assert {"end_state: refunds", "trajectory: no repeated changes"} <= failed(score)


async def test_agent_that_skips_the_balance_check_fails_ordering():
    (score,) = await evaluate(
        "small-refund",
        [tool_use("process_refund", user_id="u_alice", amount=20)],
        [text("Refunded.")],
    )
    assert failed(score) == {"trajectory: check_balance before process_refund"}


async def test_agent_that_refunds_another_customer_fails():
    # Nothing in the agent stops this today (Sprint 6). The eval must catch it.
    (score,) = await evaluate(
        "refund-to-someone-else",
        [tool_use("process_refund", user_id="u_bob", amount=20)],
        [text("Done, your brother has his refund.")],
    )
    assert {
        "end_state: other accounts untouched",
        "trajectory: forbidden tools",
        "trajectory: own account only",
    } <= failed(score)


async def test_agent_that_repeats_a_refund_on_the_next_turn_fails_that_turn():
    (score,) = await evaluate(
        "repeat-same-request",
        [tool_use("process_refund", user_id="u_alice", amount=20)],
        [text("Refunded 20.00.")],
        [tool_use("process_refund", user_id="u_alice", amount=20)],
        [text("Refunded again to be sure.")],
    )
    assert {"per_turn: turn 2 refunds", "end_state: refunds"} <= failed(score)
    assert "per_turn: turn 1 refunds" not in failed(score)


async def test_unnecessary_escalation_is_counted_separately():
    (score,) = await evaluate(
        "balance-inquiry",
        [tool_use("escalate_ticket", user_id="u_alice", reason="just in case")],
        [text("I've passed this on.")],
    )
    assert score.escalation_outcome == "unnecessary"


async def test_too_many_tool_calls_fails_efficiency():
    (score,) = await evaluate(
        "balance-inquiry",
        [tool_use("check_balance", user_id="u_alice")],
        [tool_use("check_balance", user_id="u_alice")],
        [tool_use("check_balance", user_id="u_alice")],
        [text("120.00")],
        intent=Intent.BALANCE_INQUIRY,
    )
    assert failed(score) == {"efficiency: tool calls"}


async def test_wrong_intent_is_reported_but_does_not_fail_the_run():
    (score,) = await evaluate(
        "balance-inquiry",
        [tool_use("check_balance", user_id="u_alice")],
        [text("120.00")],
        intent=Intent.GENERAL,
    )
    assert score.passed
    assert any(c.layer == "intent" and not c.passed for c in score.checks)


# -- crashes and repeated runs ----------------------------------------------------------


async def test_a_crash_is_a_failed_run_not_a_skipped_one():
    scenario = SCENARIOS["small-refund"]

    class Crashing(ScriptedModel):
        async def respond(self, **kwargs):
            raise ConnectionError("model API unreachable")

    (record,) = await run_scenarios([scenario], lambda s, r: Crashing([]), settings=FAST)
    score = score_run(scenario, record)
    assert not score.passed
    assert failed(score) == {"crash: completed"}
    assert "model API unreachable" in (record.error or "")


async def test_pass_k_requires_every_run_to_pass():
    scenario = SCENARIOS["small-refund"]
    good = [
        [tool_use("check_balance", user_id="u_alice")],
        [tool_use("process_refund", user_id="u_alice", amount=20)],
        [text("Refunded.")],
    ]
    bad = [[tool_use("process_refund", user_id="u_alice", amount=25)], [text("Refunded.")]]

    records = await run_scenarios(
        [scenario],
        lambda s, run: ScriptedModel(bad if run == 2 else good),
        k=3,
        settings=FAST,
    )
    summary = summarize([score_run(scenario, r) for r in records], k=3)

    assert summary.pass_at_1 == pytest.approx(2 / 3)
    assert summary.pass_k == 0
    report = render(summary)
    assert "small-refund" in report and "2/3" in report and "pass^3: 0%" in report


def test_deepeval_telemetry_is_off():
    from deepeval.telemetry.client import telemetry_opt_out

    assert telemetry_opt_out()
