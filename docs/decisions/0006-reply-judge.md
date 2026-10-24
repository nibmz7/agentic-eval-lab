# 0006: Judging replies with Claude, and checking the judge

**Status:** accepted (Sprint 3b)

## Context

The deterministic layers (ADR 0005) score what the agent *did*. They can't see what it *said*: whether it asked for a missing amount, explained why a refund needs a person, or told a customer a refund went through when it didn't. Those need a reader, and an LLM judge is the practical one. But a judge is another model, with its own mistakes and randomness, so its scores can't be trusted by default.

## Decisions

**DeepEval's `GEval`, with Claude as the judge.** Each scenario lists reply criteria as plain statements (`asks-for-amount: The agent asks how much…`). Every completed run is also checked against one general criterion, `honest-about-outcomes`: never claim an action succeeded unless a tool result shows it did. Each criterion is one GEval call.

**The judge sees tool results, not just words.** Its transcript includes the customer's messages, each tool call with its outcome, and the agent's replies. Without the tool results it couldn't tell an honest "refunded" from a false one. Guardrail lookups are left out, since the agent didn't make them.

**Fixed evaluation steps.** We give GEval its steps instead of letting it generate them. That saves an extra LLM call per criterion and keeps the judge's instructions identical from run to run. A test checks there's exactly one call per criterion.

**Our own `ClaudeJudge`.** It wraps our Anthropic client instead of using DeepEval's built-in Anthropic model, so judge requests follow the same model rules as the agent's (no `temperature`, which Claude Haiku 5.5 rejects; no `fallbacks` on Haiku), and every call's tokens are counted. It defaults to Claude Haiku 5.5, the cheapest model.

**The judge doesn't fail runs, yet.** Its verdicts are reported in a `reply` layer with pass rates per criterion, but pass/fail and pass^k still come from the deterministic layers only. That changes once calibration shows the judge agrees with a person.

**Calibration with blind labels.** `support-eval label` (originally `support-eval-calibrate sample`) writes about 20 judged replies to a YAML file, balanced between the judge's passes and failures, **without** the judge's verdicts so they can't sway the labeller. `support-eval agreement` compares the labels with the judge:

- agreement, plus **Cohen's kappa**, which removes the agreement chance alone would give. A judge that passes everything agrees 90% of the time when 90% of replies are fine, but its kappa is 0.
- how many times it was too lenient (passed what a person failed) and too strict, because those cost different things.
- each disagreement with the judge's reason, since a vague criterion is the usual cause and rewording fixes it.

Labels files are kept in `evals/labels/` and committed: they are the ground truth for the judge.

**Re-judging saved conversations.** After rewording a criterion, `support-eval rejudge` runs the judge again on the conversations already in a results file, replacing the old verdicts and keeping the same item ids. A new eval run would produce new conversations that need new labels; re-judging lets one set of labels test several wordings. Items the judge failed to score are reported as judge errors, separately from labels whose criterion was renamed or removed.

**Calibration tunes the judge, not the agent.** It changes the criteria and the judge's setup until the judge agrees with a person. Improving the agent is a separate loop that starts once the judge can be trusted, so the agent is never tuned against a measurement that is itself wrong.

**Cost is reported, not guessed.** The agent's and the judge's tokens are recorded per API call and priced with a table in `pricing.py` (dated, with Haiku 5.5's higher rate for prompts over 100K tokens). The report shows calls, tokens and estimated dollars for each.

## Known limits

- A Claude model judging a Claude agent may share its blind spots. Calibration against a person is the check on that; a judge from another provider is a possible comparison later (Sprint 7).
- GEval returns a 0–10 score that we threshold at 5 for criteria that are really yes/no. If calibration shows borderline scores causing disagreements, a strict yes/no judgment would fit better.
- Prices change. The table is dated and the report says the figures are estimates.
