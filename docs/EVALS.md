# Running the evals

Everything that calls Claude needs `ANTHROPIC_API_KEY` or `ant auth login`. Nothing else has to be running: each eval run builds its own database and MCP server in memory.

## The two commands you'll use

| When | Command | What it does | Claude calls |
|---|---|---|---|
| Testing the agent (after changing its prompt, model, graph or guardrail) | `uv run support-eval run` | Runs every scenario 5 times, scores each run, judges the replies, prints a report | agent + judge |
| Checking the judge (after changing the judge model or a reply criterion) | `uv run support-eval judge-check` | Shows the judge good and bad replies with known right answers, and reports how often it agrees | judge only |

`support-eval` on its own means `run`. Useful options for `run`: `--runs 3` for a quicker run, `--only refund-over-limit` for one scenario, `--model claude-sonnet-5-5` to test another model, `--no-judge` to skip the judge.

Results go to `evals/results/<time>.json`, with every conversation, tool call and check.

## Reading the `run` report

```
scenario                 passed  every run  failing checks
refund-over-limit         5/5    yes
injection-admin-override  3/5    no         trajectory: forbidden tools (2/5)

pass^k (right k times in a row): ^1 92%   ^2 85%   ^3 80%   ^4 76%   ^5 72%
by layer: end_state 100%   trajectory 96%   ...
reply, judged (not scored until calibrated): explains-escalation 90%   ...
agent: 512 calls, ... ≈ $0.07     judge: 220 calls, ... ≈ $0.03

failed runs:
  injection-admin-override #3: trajectory: forbidden tools
    trace: http://localhost:3000/project/support-agent/traces/4f2a…
```

- **passed / every run**: how many of the 5 runs passed, and whether all of them did.
- **failing checks**: which checks failed and in how many runs. Open the results file to see the conversation.
- **pass^k**: ^1 is plain accuracy; ^5 is the chance of being right 5 times in a row. The gap between them shows how inconsistent the agent is.
- **reply, judged**: the judge's verdicts. Reported only, until the judge is trusted.
- **failed runs**: each run that failed and why. With Langfuse running ([TRACING.md](TRACING.md)), each has a link to its trace, showing every step the agent took. Without it, look the run up in the results file.

What these numbers can and can't tell you: [MEASUREMENT.md](MEASUREMENT.md).

## Reading the `judge-check` report

```
  ✓ claims-refund-that-was-blocked  honest-about-outcomes  should fail, judge failed (...)
  ✗ escalates-without-saying-why    explains-escalation    should fail, judge passed (...)

agreement: 93% (25/27)   Cohen's kappa: 0.85
too lenient (passed a bad reply): 1   too strict (failed a good reply): 1
```

Each line is one conversation and one criterion: ✓ the judge got it right, ✗ it didn't. The cases are in `evals/judge_checks.yaml`, half deliberately bad and half good.

### What kappa means

Agreement alone can flatter a lazy judge. If most replies are good, a judge that passes everything without reading agrees most of the time, yet never catches a bad reply. Kappa removes the agreement you'd get by chance:

- **1.0**: the judge always matches the right answer.
- **0**: no better than chance. A judge that passes everything scores 0.
- Rough guide: above 0.8, trust the judge; 0.6–0.8, usable; below 0.6, fix the criteria first.

Too lenient (passing bad replies) is usually worse than too strict, because it hides real problems.

## When the judge gets something wrong

The fix is almost always to reword the criterion, in `evals/scenarios/*.yaml` under `reply:`, so it says exactly what a good reply must contain. Then run `judge-check` again. Only the judge is called, so it costs a cent or two.

Change the criteria to match what you think a good reply is, never to make the agent's score look better. If the judge correctly fails the agent, the agent needs work, not the criterion.

## Optional: checking the judge on real conversations

`judge-check` uses scripted conversations. To check the judge on what the real agent said:

```bash
uv run support-eval label evals/results/<time>.json       # writes evals/labels/labels-<time>.yaml
# mark each item human: pass or human: fail (the judge's verdicts are hidden from you)
uv run support-eval agreement evals/results/<time>.json evals/labels/labels-<time>.yaml
# after rewording a criterion, judge the SAME conversations again (judge calls only):
uv run support-eval rejudge evals/results/<time>.json
uv run support-eval agreement evals/results/<time>.rejudged-<time2>.json evals/labels/labels-<time>.yaml
```

## Two loops, in this order

1. **Make the judge trustworthy:** `judge-check` (and optionally labelling) until kappa is high. Only criteria and the judge's setup change.
2. **Improve the agent:** change its prompt, guardrail or model, run `run`, and compare pass^k and the judge's reply scores with the previous results. The scenarios and the judge stay fixed.

Doing them the other way round means tuning the agent against a measurement you can't trust yet.
