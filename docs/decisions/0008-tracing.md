# 0008: Tracing with Langfuse

**Status:** accepted (Sprint 4)

## Context

The eval report says which checks failed, and the results file holds each conversation and its tool calls. That's enough to see *what* went wrong but not always *why*: what the model was sent, what it answered (thinking included), which node made which call, and what the judge reasoned. Langfuse shows all of that as a tree, and can filter and compare runs.

## Decisions

**Optional, and off by default.** Tracing turns on when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. Without them, `tracing.observe(...)` does nothing, so the agent, the tests and the evals behave identically. Tests force tracing off, even when keys are in the environment, unless a test turns it on against an in-memory exporter.

**One module talks to Langfuse.** `support_lab/tracing.py` wraps the few calls we use (observations, trace attributes, scores, links). The rest of the code doesn't import Langfuse.

**One trace per eval run, one session per `support-eval run`.** A multi-turn scenario stays in one trace, so the whole conversation reads top to bottom. The session groups every run of one invocation, which is the unit you compare before and after a change. The chat uses one trace per turn and one session per conversation, the usual shape for a live app.

**Results go onto the traces as scores.** `passed`, one score per layer, and one per judged criterion, so Langfuse can filter to failing runs or to one kind of failure. The report links each failed run to its trace. Scores are sent after scoring, against the trace id each run recorded.

**LangGraph nodes from Langfuse's handler, everything else by hand.** Langfuse's LangChain callback handler records each node. Claude calls, MCP tool calls and the guardrail's verdict are recorded where they happen, because the agent calls the Anthropic SDK directly. Two adjustments were needed:

- The handler runs inline. By default LangChain runs callback handlers in a thread pool, so a node's span never became current inside the node, and the calls made inside it attached to the turn instead.
- The Langfuse client gets its own OpenTelemetry tracer provider rather than the global one. The MCP SDK emits spans through the global provider, and they would otherwise appear as extra traces, and keep being exported after tracing was turned off.

**Check the connection once, then carry on.** The CLIs check that Langfuse accepts the keys before starting. If it doesn't, they say so and run untraced, instead of logging an export error for every batch.

**Local Langfuse from a trimmed compose file.** `infra/langfuse/docker-compose.yml` is Langfuse's own compose file cut down to what the lab needs. Only the web UI is published, on 127.0.0.1, and its first start creates the project and keys. The secrets are fixed values, which is fine on one machine and wrong anywhere else.

## Not verified here

The code was tested against an in-memory exporter and a stand-in HTTP server that accepts Langfuse's endpoints. It hasn't been run against a real Langfuse instance from this repo's CI, so the first real run is the check that the UI shows what the docs describe.
