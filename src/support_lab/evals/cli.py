"""The eval suite's command line.

    uv run support-eval run           # test the agent: run, score, judge, report  (the usual one)
    uv run support-eval judge-check   # test the judge on known-good and known-bad replies

Manual calibration on real conversations, when you want it:

    uv run support-eval label RESULTS              # write replies for you to mark pass/fail
    uv run support-eval agreement RESULTS LABELS   # compare your marks with the judge
    uv run support-eval rejudge RESULTS            # judge the same conversations again,
                                                   # e.g. after rewording a criterion

RESULTS is a file from evals/results/. `support-eval` with no action means `run`.
Everything that calls Claude needs ANTHROPIC_API_KEY or `ant auth login`.

With LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY set, every run is traced to
Langfuse with its results attached as scores, and the report links each
failed run to its trace. See docs/TRACING.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import anthropic
import anyio
import yaml

from support_lab import tracing
from support_lab.agent import AnthropicModel
from support_lab.agent.model import CHEAPEST_MODEL

from .calibrate import agreement, judged_items, labels_document, rejudge, sample
from .calibrate import render as render_agreement
from .checks import score_run
from .judge import ClaudeJudge, criteria_for, judge_scores
from .judge_check import DEFAULT_CASES, load_cases, run_judge_check, save
from .pricing import PRICES_AS_OF, total_cost
from .report import load_scores, render, summarize, write_json
from .runner import RunRecord, run_scenarios
from .scenario import load_scenarios
from .traces import link_traces, send_scores

DEFAULT_SCENARIOS = Path("evals/scenarios")
RESULTS_DIR = Path("evals/results")
LABELS_DIR = Path("evals/labels")
ACTIONS = ("run", "judge-check", "label", "agreement", "rejudge")


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _quiet_logs() -> None:
    # Each run builds an in-process MCPServer, which turns on INFO logging for the
    # process. Per-request HTTP and MCP logs would bury the progress lines.
    for name in ("httpx2", "httpx", "mcp"):
        logging.getLogger(name).setLevel(logging.WARNING)


def cmd_run(args: argparse.Namespace) -> None:
    scenarios = load_scenarios(args.scenarios)
    if args.only:
        unknown = set(args.only) - {s.id for s in scenarios}
        if unknown:
            sys.exit(f"Unknown scenario ids: {sorted(unknown)}")
        scenarios = [s for s in scenarios if s.id in args.only]
    stamp = _stamp()
    out = args.out or RESULTS_DIR / f"{stamp}.json"
    judge = "no judge" if args.no_judge else f"judged by {args.judge_model}"
    print(f"{len(scenarios)} scenarios × {args.runs} runs = {len(scenarios) * args.runs} runs "
          f"on {args.model} (effort {args.effort}), {judge}")  # fmt: skip
    session = f"eval-{stamp}" if tracing.enabled() else None
    print(f"Tracing to Langfuse, session {session}.\n" if session else "")

    client = anthropic.AsyncAnthropic()

    def make_model(scenario, run):
        return AnthropicModel(client, model=args.model, effort=args.effort)

    def progress(record: RunRecord) -> None:
        status = "crashed" if record.error else "done"
        print(f"  {record.scenario_id} #{record.run} {status} ({record.seconds:.1f}s)")

    records = anyio.run(
        lambda: run_scenarios(
            scenarios,
            make_model,
            runs=args.runs,
            concurrency=args.concurrency,
            on_done=progress,
            session=session,
        )
    )
    by_id = {s.id: s for s in scenarios}
    scores = [score_run(by_id[r.scenario_id], r) for r in records]
    if not args.no_judge:
        print("\nJudging replies…")
        anyio.run(
            lambda: judge_scores(
                by_id,
                scores,
                lambda: ClaudeJudge(args.judge_model, client=client),
                concurrency=args.concurrency,
            )
        )
    if session:
        send_scores(scores)
        link_traces(records)
    summary = summarize(scores)
    print("\n" + render(summary))
    meta = {
        "model": args.model,
        "effort": args.effort,
        "runs": args.runs,
        "judge_model": None if args.no_judge else args.judge_model,
        "started": stamp,
        "langfuse_session": session,
    }
    write_json(out, summary, scores, meta)
    print(f"\nFull results: {out}")


def cmd_judge_check(args: argparse.Namespace) -> None:
    scenarios = {s.id: s for s in load_scenarios(args.scenarios)}
    cases = load_cases(args.cases, scenarios)
    expected = sum(len(c.expected) for c in cases)
    print(f"Judge check: {len(cases)} scripted conversations, {expected} verdicts with a known "
          f"answer, judged by {args.judge_model}\n")  # fmt: skip
    client = anthropic.AsyncAnthropic()
    check = anyio.run(
        lambda: run_judge_check(
            cases,
            lambda: ClaudeJudge(args.judge_model, client=client),
            concurrency=args.concurrency,
        )
    )
    verdicts = {
        (s.record.scenario_id, c.name): c
        for s in check.scores
        for c in s.checks
        if c.layer == "reply"
    }
    width = max(len(c.id) for c in cases)
    for case in cases:
        for name, should_pass in case.expected.items():
            got = verdicts.get((case.id, name))
            if got is None or got.detail.startswith("judge error"):
                mark, judged = "?", "judge error" if got else "not judged"
            else:
                mark = "✓" if got.passed == should_pass else "✗"
                judged = f"judge {'passed' if got.passed else 'failed'} ({got.detail[:70]})"
            should = "should pass" if should_pass else "should fail"
            print(f"  {mark} {case.id:<{width}}  {name:<22} {should}, {judged}")
    a = check.agreement
    kappa = "n/a" if a.kappa is None else f"{a.kappa:.2f}"
    print(f"\nagreement: {a.rate:.0%} ({a.agree}/{a.labelled})   Cohen's kappa: {kappa}")
    print(f"too lenient (passed a bad reply): {a.judge_too_lenient}   "
          f"too strict (failed a good reply): {a.judge_too_strict}")  # fmt: skip
    errors = [
        c.detail for s in check.scores for c in s.checks if c.detail.startswith("judge error")
    ]
    if errors:
        print(f"judge errors: {len(errors)}, e.g. {errors[0][len('judge error: ') :][:160]}")
    usage = [u for s in check.scores for u in s.record.judge_usage]
    dollars = total_cost(args.judge_model, usage)
    price = f"≈ ${dollars:.4f}" if dollars is not None else "cost unknown"
    print(f"judge: {len(usage)} calls, {price} (prices as of {PRICES_AS_OF})")
    out = args.out or RESULTS_DIR / f"judge-check-{_stamp()}.json"
    save(check, out, {"judge_model": args.judge_model})
    print(f"\nFull results: {out}")


def cmd_label(args: argparse.Namespace) -> None:
    results = json.loads(args.results.read_text())
    if results.get("meta", {}).get("judge_check"):
        sys.exit("This is a judge-check result: its right answers are already known, so there's "
                 "nothing to label. Use `label` on results from `support-eval run`.")  # fmt: skip
    items = judged_items(results)
    if not any(not i.judge_error for i in items):
        sys.exit(
            "No judged replies in this results file (run with --no-judge, or the judge failed)"
        )
    criteria = {s.id: criteria_for(s) for s in load_scenarios(args.scenarios)}
    chosen = sample(items, args.n, args.seed)
    out = args.out or LABELS_DIR / f"labels-{_stamp()}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(labels_document(chosen, criteria))
    print(f"Wrote {len(chosen)} items to {out}.")
    print("Mark each `human: pass` or `human: fail`, then run `support-eval agreement`.")


def cmd_agreement(args: argparse.Namespace) -> None:
    items = judged_items(json.loads(args.results.read_text()))
    labels = yaml.safe_load(args.labels.read_text()) or []
    print(render_agreement(agreement(items, labels)))


def cmd_rejudge(args: argparse.Namespace) -> None:
    results = json.loads(args.results.read_text())
    scenarios = {s.id: s for s in load_scenarios(args.scenarios)}
    scores = load_scores(results, {i: s.customer for i, s in scenarios.items()})
    client = anthropic.AsyncAnthropic()
    print(f"Judging {len(scores)} saved conversations again with {args.judge_model}…")
    anyio.run(
        lambda: rejudge(
            scores,
            scenarios,
            lambda: ClaudeJudge(args.judge_model, client=client),
            concurrency=args.concurrency,
        )
    )
    out = args.out or args.results.with_name(f"{args.results.stem}.rejudged-{_stamp()}.json")
    meta = {**results.get("meta", {}), "judge_model": args.judge_model}
    meta["rejudged_from"] = str(args.results)
    summary = summarize(scores)
    write_json(out, summary, scores, meta)
    print(render(summary))
    print(f"\nWrote {out}. Compare with your labels using `support-eval agreement`.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="support-eval", description="Evaluate the support agent, and the judge that scores it."
    )
    sub = parser.add_subparsers(dest="action", required=True, metavar="ACTION")

    def common(p: argparse.ArgumentParser, *, judge: bool = True) -> None:
        p.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
        if judge:
            p.add_argument(
                "--judge-model", default=CHEAPEST_MODEL, help=f"default {CHEAPEST_MODEL}"
            )
            p.add_argument("--concurrency", type=int, default=4)

    run = sub.add_parser("run", help="test the agent: run the scenarios, score, judge, report")
    common(run)
    run.add_argument("--only", nargs="+", metavar="ID", help="run only these scenario ids")
    run.add_argument("--runs", type=int, default=5, help="runs per scenario (default 5)")
    run.add_argument(
        "--model", default=CHEAPEST_MODEL, help=f"agent model, default {CHEAPEST_MODEL}"
    )
    run.add_argument("--effort", default="medium")
    run.add_argument("--no-judge", action="store_true", help="skip judging replies")
    run.add_argument("--out", type=Path)
    run.set_defaults(func=cmd_run)

    check = sub.add_parser("judge-check", help="test the judge on known-good and known-bad replies")
    common(check)
    check.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    check.add_argument("--out", type=Path)
    check.set_defaults(func=cmd_judge_check)

    label = sub.add_parser("label", help="write judged replies for you to mark pass or fail")
    common(label, judge=False)
    label.add_argument("results", type=Path)
    label.add_argument("--n", type=int, default=20)
    label.add_argument("--seed", type=int, default=0)
    label.add_argument("--out", type=Path)
    label.set_defaults(func=cmd_label)

    agree = sub.add_parser("agreement", help="compare your marks with the judge's")
    agree.add_argument("results", type=Path)
    agree.add_argument("labels", type=Path)
    agree.set_defaults(func=cmd_agreement)

    again = sub.add_parser("rejudge", help="judge saved conversations again with current criteria")
    common(again)
    again.add_argument("results", type=Path)
    again.add_argument("--out", type=Path)
    again.set_defaults(func=cmd_rejudge)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in ACTIONS and argv[0] not in {"-h", "--help"}):
        argv.insert(0, "run")  # `support-eval` and `support-eval --runs 3` still mean `run`
    args = build_parser().parse_args(argv)
    _quiet_logs()
    if args.func in (cmd_run, cmd_judge_check, cmd_rejudge) and (problem := tracing.connect()):
        print(problem)
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit("Stopped.")
    finally:
        tracing.flush()
