# 0005: How the eval suite scores a run

**Status:** accepted (Sprint 3a)

## Context

A support agent's replies vary from run to run, and several different tool sequences can be equally correct. Matching the reply text against a reference would fail good runs and pass bad ones. We need scores that are cheap, repeatable, and hard to fool with a convincing reply.

## Decisions

**Score what happened, not what was said.** The main signal is the database after the conversation: which refunds and tickets exist, and whether any other customer's account changed. A reply saying "refunded!" counts for nothing if no refund row exists. This only works because the backend owns the policy (ADR 0001).

**Layers, cheapest and most reliable first.** Each layer catches failures the one before can't see:

| Layer | Catches | Example |
|---|---|---|
| `end_state` | Wrong outcome | Refunded twice; refunded the wrong customer |
| `trajectory` | Right outcome, wrong path | Tried a forbidden refund that the guardrail stopped; skipped the balance check; acted on another account |
| `efficiency` | Wasted calls | Looked up the balance three times |
| `escalation` | Wrong hand-off decision | Missed escalations and unnecessary ones are counted separately, because they cost different things |
| `per_turn` | Drift during a conversation | Correct after turn 1, refunded again on turn 2 |
| `intent` | Planner mistakes | Reported only: a wrong label doesn't fail a run if the agent still did the right thing |

**`trajectory` uses DeepEval where it fits.** "Did the required tools run?" is DeepEval's `ToolCorrectnessMetric`. Order rules, forbidden tools, own-account and no-repeat checks are ours, because they depend on our tool log (sources, blocked calls, arguments).

**No LLM is called in 3a.** DeepEval's metrics create a judge model in their constructor even when they never use it, defaulting to OpenAI. We pass `NoLLMJudge`, which raises if called, so a metric that quietly started using an LLM would fail the tests instead of adding cost or randomness.

**Every run is isolated.** A fresh in-memory database and in-process MCP server per run, with faults injected per scenario. Runs can't leak state into each other and can run concurrently. The agent receives only the customer's messages, never the expectations.

**Crashes are failures.** An exception, a model API error or the step limit makes the run fail in the `crash` layer. Layer pass rates are computed over completed runs and the report says how many, so crashes can't make the other layers look better.

**pass^k, estimated the way τ-bench does.** Each scenario runs n times (`--runs`, default 5). For a scenario with c passes, pass^k is the chance that k runs picked at random from its n are all passes, `C(c, k) / C(n, k)`, averaged over scenarios. The report prints pass^1 to pass^n from the same runs:

- pass^1 is plain accuracy.
- pass^n is the share of scenarios that passed every run.
- The values in between show how fast reliability drops when the agent must be right repeatedly.

A customer gets one try, and the same request arrives many times, so the high end of that curve matters more than the average. Compared with counting "passed all k runs", the estimate uses every run and gives a fraction per scenario instead of yes/no, so one unlucky run moves the score a little instead of flipping it. It's unbiased, and it doesn't assume every scenario is equally reliable, which raising accuracy to the power k would.

**The suite is tested with broken agents.** Scripted agents that always refund, refund twice, skip the balance check, refund another customer, or repeat a refund on the next turn must each fail, on the expected layers. If a broken agent ever passes, the suite has a bug.

## Known limits

- Reply quality isn't scored yet: whether the agent asked for a missing amount or explained a refusal politely. That's the judge in 3b.
- Expectations are hand-written. A scenario with a wrong expectation scores good behaviour as bad. Sprint 5's ground-truth audit re-derives them from the policy.
- Some scenarios accept either outcome (`escalated: null`) where reasonable agents could differ, such as a suspended account. That's a judgement call worth revisiting with real results.
