"""Test the reply judge on conversations with a known right answer.

evals/judge_checks.yaml holds scripted conversations, some deliberately bad
and some good, each saying what a correct judge concludes. They are replayed
(no agent model is called) and judged by the real judge. The result is the
judge's agreement and Cohen's kappa against those expected verdicts, with no
manual labelling: a quick answer to "can I trust the judge?".

Run it after changing the judge model or reply criteria.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from support_lab.agent import AgentSettings
from support_lab.agent.model import Block
from support_lab.agent.scripted import ScriptedModel, text, tool_use

from .calibrate import Agreement, agreement, judged_items
from .checks import RunScore
from .judge import ClaudeJudge, criteria_for, judge_scores
from .report import summarize, write_json
from .runner import run_once
from .scenario import KNOWN_TOOLS, Scenario, ScenarioError

DEFAULT_CASES = Path("evals/judge_checks.yaml")


@dataclass(frozen=True)
class JudgeCase:
    id: str
    scenario: Scenario
    description: str
    agent_turns: tuple[tuple[Block, ...], ...]  # one model turn each
    expected: dict[str, bool]  # criterion -> should the judge pass it?


def load_cases(path: Path, scenarios: dict[str, Scenario]) -> list[JudgeCase]:
    cases = []
    for raw in yaml.safe_load(path.read_text()) or []:
        where = f"{path.name}:{raw.get('id', '?')}"
        scenario = scenarios.get(raw.get("scenario", ""))
        if scenario is None:
            raise ScenarioError(f"{where}: unknown scenario {raw.get('scenario')!r}")
        turns: list[tuple[Block, ...]] = []
        for step in raw["agent"]:
            if "call" in step:
                if step["call"] not in KNOWN_TOOLS:
                    raise ScenarioError(f"{where}: unknown tool {step['call']!r}")
                turns.append((tool_use(step["call"], **step.get("args", {})),))
            elif "reply" in step:
                turns.append((text(step["reply"].strip()),))
            else:
                raise ScenarioError(f"{where}: each agent step needs `call` or `reply`")
        known = criteria_for(scenario)
        expected = {}
        for name, verdict in raw["judge_should"].items():
            if name not in known:
                raise ScenarioError(f"{where}: {scenario.id} has no criterion {name!r}")
            if verdict not in {"pass", "fail"}:
                raise ScenarioError(f"{where}: {name} should be pass or fail")
            expected[name] = verdict == "pass"
        ids = [c.id for c in cases]
        if raw["id"] in ids:
            raise ScenarioError(f"{where}: duplicate id")
        cases.append(JudgeCase(raw["id"], scenario, raw["description"], tuple(turns), expected))
    return cases


@dataclass
class JudgeCheck:
    scores: list[RunScore]
    agreement: Agreement
    labels: list[dict[str, Any]]


async def run_judge_check(
    cases: list[JudgeCase],
    make_judge: Callable[[], ClaudeJudge],
    *,
    concurrency: int = 4,
) -> JudgeCheck:
    settings = AgentSettings(retry_delays=())
    scores = []
    for case in cases:
        model = ScriptedModel([list(t) for t in case.agent_turns])
        record = await run_once(case.scenario, 1, model, settings)
        if record.error:
            raise RuntimeError(f"judge check {case.id} didn't replay cleanly: {record.error}")
        record.scenario_id = case.id  # one result per case, even when cases share a scenario
        scores.append(RunScore(record))
    await judge_scores(
        {c.id: c.scenario for c in cases}, scores, make_judge, concurrency=concurrency
    )
    labels = [
        {"id": f"{c.id}#1:{name}", "human": "pass" if ok else "fail"}
        for c in cases
        for name, ok in c.expected.items()
    ]
    return JudgeCheck(scores, agreement(judged_items(_as_results(scores)), labels), labels)


def _as_results(scores: list[RunScore]) -> dict[str, Any]:
    return {
        "runs": [
            {
                "scenario_id": s.record.scenario_id,
                "run": s.record.run,
                "turns": [
                    {"say": t.say, "reply": t.reply, "tool_calls": t.tool_calls}
                    for t in s.record.turns
                ],
                "checks": [asdict(c) for c in s.checks],
            }
            for s in scores
        ]
    }


def save(check: JudgeCheck, path: Path, meta: dict[str, Any]) -> None:
    write_json(path, summarize(check.scores), check.scores, {**meta, "judge_check": True})
