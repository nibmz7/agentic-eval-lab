# What the evals measure, and what they don't

A green report is only as good as what it checks. This page lists what the suite measures, how far each number can be trusted, and the known gaps. Use it to read a report honestly, and as the to-do list for later sprints.

## What one run is

A scripted customer sends a fixed list of messages to the real agent: the real graph, guardrail and Claude calls, with a fresh SQLite database behind a real MCP server (in-process). Afterwards the run is scored against the scenario's expectations. The agent never sees those expectations. `support-eval run` does this for 26 scenarios, 5 times each.

## What is measured

| Layer | Question | How | Counts toward pass? |
|---|---|---|---|
| crash | Did the run finish? | A crash, step limit or API error fails the run. | yes |
| end_state | Is the database right? | Exact refunds and ticket count, other accounts untouched. | yes |
| trajectory | Did it use the tools correctly? | Required tools ran, forbidden ones were never requested, order rules held, only the customer's account was used, no mutation was repeated. | yes |
| efficiency | Did it avoid needless calls? | Tool calls requested ≤ the scenario's `max_calls`. | yes |
| escalation | Did it hand off to a person exactly when it should? | Correct / missed / unnecessary. | yes |
| per_turn | Did each turn of a longer conversation leave the right state? | The end_state checks after chosen turns. | yes |
| intent | Did the planner label the request correctly? | Exact match. | no, reported |
| reply | Did it tell the customer the right things, honestly? | A Claude judge, one call per criterion. | no, reported |

Everything except `reply` is deterministic code: the same run always gets the same score. Langfuse traces (Sprint 4) don't add measurements. They make each result inspectable.

## How far the numbers go

**Five runs is a small sample.** A scenario that passed 5 out of 5 times could still fail up to about 45% of the time (95% confidence: 1 − 0.05^(1/5)). Across the whole suite, 130 passing runs out of 130 puts the per-run failure rate below about 2.3%. That's a statement about these 26 situations, not about customers in general. Raise `--runs` when a decision depends on the number.

**pass^k assumes runs are independent.** They are close to it here: every run gets a fresh database and server. But they all use the same prompt and model, so a systematic weakness shows up as the same failure every time, not as noise.

**Randomness comes only from the model.** Haiku 5.5 doesn't accept a temperature setting, so variation between runs is whatever the model produces. Repeated runs are the only way to see it.

**Costs are estimates** from a dated price table (`evals/pricing.py`), not from a bill.

## What isn't measured

- **Speed.** Each run's time is recorded, but nothing fails for being slow.
- **Cost per task.** Reported, never scored.
- **Anything a scripted customer can't do.** The customer's messages are fixed in advance. They can't answer a question the agent asks, change their mind because of a reply, or react to a mistake. A simulated customer that responds (as τ²-bench does) would test far more.
- **Tone and helpfulness**, beyond the specific reply criteria.
- **The real transports.** Evals use the in-process MCP client. Stdio, HTTP and the OAuth sign-in are covered by unit tests, not by agent evals.
- **Other models.** Only the model you pass with `--model` (Sprint 7 compares providers).
- **Change over time.** Each report stands alone. Comparing two runs (before and after a prompt change) is manual, either from the results files or from Langfuse sessions.

## Known blind spots

**The suite has hit its ceiling.** Claude Haiku 5.5 passes every scenario in every run. That's good news about the agent, but it means the suite can't currently tell a better prompt from a worse one, since both score 100%. It needs harder scenarios before it can guide improvements.

**The judge is checked, but only on our own examples.** `judge-check` agrees with the right answer on 27 of 27 items (kappa 1.00). Those 15 conversations were written by us, and a criterion was reworded where the judge disagreed, so the criteria may fit those cases better than real replies. It hasn't yet been compared with hand labels on real replies. Until it has, `reply` stays informational.

**The judge grades its own model.** By default Haiku 5.5 judges Haiku 5.5. Models can favour answers that sound like their own. A judge from a different model or provider would show whether that matters here.

**Ground truth is hand-written.** Each scenario's expected refunds, tickets and tool rules were derived by hand from the policy. A mistake there makes a wrong agent pass, or a right one fail. Sprint 5 re-derives them.

**The policy has a gap.** It limits a single refund to 500.00 but says nothing about several refunds that add up to more ([ADR 0007](decisions/0007-testing-the-judge.md)). No scenario tests this until the rule is decided.

**The guardrail hides some mistakes from end_state.** If the agent tries a refund it shouldn't, the guardrail blocks it and the database stays correct. The trajectory layer still fails the run, because the forbidden call was requested. But the score measures the agent and guardrail together, not the agent alone.

**Hand-set thresholds.** `max_calls` limits were chosen by judgement. Too loose and waste goes unnoticed; too tight and a careful extra check fails.

**Faults are narrow.** Injected failures hit the backend's refund method (fail before or after the write). Slow or dropped MCP connections, and Claude API errors such as rate limits, aren't simulated. A real API error crashes the run, which counts as a failure.

**The evaluator isn't isolated by test.** The agent only receives the customer's messages, but nothing asserts that it couldn't read the scenario files or write to the database outside MCP. Sprint 5 adds those checks.

## Reading a green report

"Every scenario passed every run" means the agent handled these 26 situations correctly 5 times each, as judged by these checks. It doesn't mean the agent is safe in general. The gaps above are where to look next.
