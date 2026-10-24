"""Run the scenario suite against Claude.

    uv run support-eval                       # every scenario, 5 runs each, Claude Haiku 5.5
    uv run support-eval --only refund-over-limit --runs 8
    uv run support-eval --model claude-sonnet-5-5
    uv run support-eval --no-judge            # skip the Claude judge for replies

Needs Claude credentials (ANTHROPIC_API_KEY, or `ant auth login`). Every run
calls the model several times, so the run count is printed before starting.
Results go to evals/results/<timestamp>.json.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import anthropic
import anyio

from support_lab.agent import AnthropicModel
from support_lab.agent.model import CHEAPEST_MODEL

from .checks import score_run
from .judge import ClaudeJudge, judge_scores
from .report import render, summarize, write_json
from .runner import RunRecord, run_scenarios
from .scenario import load_scenarios

DEFAULT_SCENARIOS = Path("evals/scenarios")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the scenario suite against Claude.")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--only", nargs="+", metavar="ID", help="run only these scenario ids")
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="runs per scenario, n (default 5); pass^k is shown up to k=n",
    )
    parser.add_argument("--model", default=CHEAPEST_MODEL, help=f"default {CHEAPEST_MODEL}")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--judge-model", default=CHEAPEST_MODEL, help=f"judge for replies, default {CHEAPEST_MODEL}"
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="skip judging replies (deterministic layers only)"
    )
    parser.add_argument("--out", type=Path, help="JSON results file")
    args = parser.parse_args()
    # Each run builds an in-process MCPServer, which turns on INFO logging for the
    # process. Per-request HTTP and MCP logs would bury the progress lines.
    for name in ("httpx2", "httpx", "mcp"):
        logging.getLogger(name).setLevel(logging.WARNING)

    scenarios = load_scenarios(args.scenarios)
    if args.only:
        unknown = set(args.only) - {s.id for s in scenarios}
        if unknown:
            sys.exit(f"Unknown scenario ids: {sorted(unknown)}")
        scenarios = [s for s in scenarios if s.id in args.only]
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = args.out or Path("evals/results") / f"{stamp}.json"
    judge = "no judge" if args.no_judge else f"judged by {args.judge_model}"
    print(f"{len(scenarios)} scenarios × {args.runs} runs = {len(scenarios) * args.runs} runs "
          f"on {args.model} (effort {args.effort}), {judge}\n")  # fmt: skip

    client = anthropic.AsyncAnthropic()

    def make_model(scenario, run):
        return AnthropicModel(client, model=args.model, effort=args.effort)

    def progress(record: RunRecord) -> None:
        status = "crashed" if record.error else "done"
        print(f"  {record.scenario_id} #{record.run} {status} ({record.seconds:.1f}s)")

    try:
        records = anyio.run(
            lambda: run_scenarios(
                scenarios,
                make_model,
                runs=args.runs,
                concurrency=args.concurrency,
                on_done=progress,
            )
        )
    except KeyboardInterrupt:
        sys.exit("Stopped.")
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
    summary = summarize(scores)
    print("\n" + render(summary))
    meta = {
        "model": args.model,
        "effort": args.effort,
        "runs": args.runs,
        "judge_model": None if args.no_judge else args.judge_model,
        "started": stamp,
    }
    write_json(out, summary, scores, meta)
    print(f"\nFull results: {out}")
