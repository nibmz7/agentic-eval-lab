"""Check the judge against your own labels (`support-eval label / agreement / rejudge`).

`label` picks judged replies, half the judge passed and half it failed where
possible, and writes them *without* the judge's verdicts, so its answer can't
sway yours. `agreement` compares your labels with the judge's.

Two numbers matter:
- agreement: the share of items where you and the judge agree.
- Cohen's kappa: agreement after removing what chance alone would give. If 90%
  of replies are fine, a judge that passes everything agrees 90% of the time
  but has a kappa of 0. Above about 0.8 is strong; below about 0.6, rework the
  criteria before trusting the judge.

Disagreements are listed with the judge's reason. Read them: often the
criterion is ambiguous, and rewording it fixes the judge. `rejudge` then runs
the judge again on the *same* saved conversations, so the labels you already
made still apply and you can compare wordings without labelling again.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import yaml

from .checks import RunScore
from .judge import ClaudeJudge, judge_scores, transcript_from_turns
from .scenario import Scenario


@dataclass(frozen=True)
class JudgedItem:
    id: str  # "<scenario>#<run>:<criterion>"
    scenario_id: str
    criterion_name: str
    judge_passed: bool
    judge_detail: str
    turns: list[dict[str, Any]]
    judge_error: bool = False  # the judge failed to score this item


def judged_items(results: dict[str, Any]) -> list[JudgedItem]:
    items = []
    for run in results["runs"]:
        for check in run["checks"]:
            if check["layer"] != "reply":
                continue
            items.append(
                JudgedItem(
                    id=f"{run['scenario_id']}#{run['run']}:{check['name']}",
                    scenario_id=run["scenario_id"],
                    criterion_name=check["name"],
                    judge_passed=check["passed"],
                    judge_detail=check["detail"],
                    turns=run["turns"],
                    judge_error=check["detail"].startswith("judge error"),
                )
            )
    return items


def sample(items: list[JudgedItem], n: int, seed: int = 0) -> list[JudgedItem]:
    """Up to n items, balanced between judge passes and failures where possible."""
    rng = random.Random(seed)
    usable = [i for i in items if not i.judge_error]
    passed = [i for i in usable if i.judge_passed]
    failed = [i for i in usable if not i.judge_passed]
    rng.shuffle(passed)
    rng.shuffle(failed)
    take_failed = min(len(failed), n // 2)
    take_passed = min(len(passed), n - take_failed)
    take_failed = min(len(failed), n - take_passed)
    chosen = failed[:take_failed] + passed[:take_passed]
    rng.shuffle(chosen)
    return chosen


class _BlockDumper(yaml.SafeDumper):
    pass


def _str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_BlockDumper.add_representer(str, _str)


def labels_document(chosen: list[JudgedItem], criteria: dict[str, dict[str, str]]) -> str:
    entries = [
        {
            "id": item.id,
            "criterion": criteria[item.scenario_id][item.criterion_name],
            "transcript": transcript_from_turns(item.turns) + "\n",
            "human": None,
            "note": "",
        }
        for item in chosen
    ]
    header = (
        "# For each item, read the transcript and set `human: pass` if the criterion is met,\n"
        "# `human: fail` if not. Leave `human` empty to skip an item. `note` is optional.\n"
    )
    return header + yaml.dump(entries, Dumper=_BlockDumper, sort_keys=False, allow_unicode=True)


def _verdict(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"pass", "yes", "true", "y", "1"}:
        return True
    if text in {"fail", "no", "false", "n", "0"}:
        return False
    raise ValueError(f"human label must be pass or fail, got {value!r}")


@dataclass
class Agreement:
    labelled: int
    agree: int
    kappa: float | None
    judge_too_lenient: int  # judge passed, you failed
    judge_too_strict: int  # judge failed, you passed
    disagreements: list[tuple[JudgedItem, bool, str]]  # item, your label, your note
    unmatched: list[str]  # labelled ids not in the results (criterion renamed or removed?)
    judge_failed: list[str]  # labelled ids the judge couldn't score in these results

    @property
    def rate(self) -> float:
        return self.agree / self.labelled if self.labelled else 0.0


def agreement(items: list[JudgedItem], labels: list[dict[str, Any]]) -> Agreement:
    by_id = {i.id: i for i in items}
    pairs: list[tuple[JudgedItem, bool, str]] = []
    unmatched: list[str] = []
    judge_failed: list[str] = []
    for label in labels:
        human = _verdict(label.get("human"))
        if human is None:
            continue
        if label["id"] not in by_id:
            unmatched.append(label["id"])
            continue
        if by_id[label["id"]].judge_error:
            judge_failed.append(label["id"])
            continue
        pairs.append((by_id[label["id"]], human, label.get("note") or ""))
    n = len(pairs)
    agree = sum(item.judge_passed == human for item, human, _ in pairs)
    lenient = sum(item.judge_passed and not human for item, human, _ in pairs)
    strict = sum(not item.judge_passed and human for item, human, _ in pairs)
    kappa = None
    if n:
        judge_yes = sum(item.judge_passed for item, _, _ in pairs) / n
        human_yes = sum(human for _, human, _ in pairs) / n
        chance = judge_yes * human_yes + (1 - judge_yes) * (1 - human_yes)
        kappa = None if chance == 1 else (agree / n - chance) / (1 - chance)
    return Agreement(
        labelled=n,
        agree=agree,
        kappa=kappa,
        judge_too_lenient=lenient,
        judge_too_strict=strict,
        disagreements=[(i, h, note) for i, h, note in pairs if i.judge_passed != h],
        unmatched=unmatched,
        judge_failed=judge_failed,
    )


def render(result: Agreement) -> str:
    unmatched = ""
    if result.judge_failed:
        unmatched += (
            f"\nThe judge couldn't score these (see the judge errors in the results): "
            f"{', '.join(result.judge_failed)}"
        )
    if result.unmatched:
        unmatched += (
            f"\nNot in this results file (criterion renamed or removed?): "
            f"{', '.join(result.unmatched)}"
        )
    if not result.labelled:
        return (
            "No labelled items yet: set `human: pass` or `human: fail` in the labels file."
            + unmatched
        )
    kappa = "n/a (everything got the same label)" if result.kappa is None else f"{result.kappa:.2f}"
    lines = [
        f"labelled: {result.labelled}   agreement: {result.rate:.0%}   Cohen's kappa: {kappa}",
        f"judge too lenient (passed what you failed): {result.judge_too_lenient}",
        f"judge too strict (failed what you passed): {result.judge_too_strict}",
    ]
    for item, human, note in result.disagreements:
        yours = "pass" if human else "fail"
        judge = "pass" if item.judge_passed else "fail"
        lines.append(f"\n{item.id}: you {yours}, judge {judge}")
        lines.append(f"  judge said: {item.judge_detail}")
        if note:
            lines.append(f"  your note: {note}")
    return "\n".join(lines) + unmatched


async def rejudge(
    scores: list[RunScore],
    scenarios: dict[str, Scenario],
    make_judge: Callable[[], ClaudeJudge],
    *,
    concurrency: int = 4,
) -> None:
    """Replace each run's judge verdicts with fresh ones from the current criteria."""
    missing = sorted({s.record.scenario_id for s in scores} - set(scenarios))
    if missing:
        raise ValueError(f"scenarios no longer exist: {missing}")
    for score in scores:
        score.checks = [c for c in score.checks if c.layer != "reply"]
        score.record.judge_model = None
        score.record.judge_usage = []
    await judge_scores(scenarios, scores, make_judge, concurrency=concurrency)
