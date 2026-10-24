# 0007: Testing the judge with known answers, and harder scenarios

**Status:** accepted (Sprint 3c)

## Context

The first real run on Claude Haiku 5.5 passed every check in every run, and the judge passed all 160 replies. That result can't tell a good agent from a great one (the scenarios were too easy), and it says nothing about whether the judge can fail a bad reply: a judge that passes everything would have produced the same output. It also left manual calibration with nothing to compare, since every sampled reply was a judge pass.

## Decisions

**A judge check with known answers.** `evals/judge_checks.yaml` holds scripted conversations, roughly half bad (claiming a refund the guardrail blocked, guessing an amount, answering trivia, giving in to pushback) and half good, each listing the verdict a correct judge gives per criterion. `support-eval judge-check` replays them without calling the agent model, judges them with the real judge, and reports agreement and Cohen's kappa against those answers. No manual labelling is needed, it costs a few cents, and it's repeatable, so it's the first thing to run after changing the judge or a criterion. Manual labelling of real conversations stays available for a deeper check.

**Harder scenarios.** `adversarial.yaml` adds seven: fake admin and staff instructions inside the customer's message, three turns of rising pressure, a vague "refund what you charged me", an amount in words, a second refund that the remaining balance can't cover, and an honest follow-up after a failed refund. Each has deterministic expectations and a reply criterion.

**One command.** The eval tools are now actions of one command, `support-eval run | judge-check | label | agreement | rejudge`, and `support-eval` alone still means `run`. `docs/EVALS.md` is the one-page guide.

## Open question found along the way

The refund policy limits a single refund to 500.00, but says nothing about several refunds in a row. A customer could ask for 450.00 twice. No scenario covers this because the right answer is a product decision, not a test: either the policy (and the backend) gains a cumulative limit, or splitting is allowed. Deciding that first, then adding the scenario, keeps the eval testing stated policy rather than guessing at it.
