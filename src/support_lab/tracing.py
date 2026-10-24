"""Optional tracing to Langfuse.

Tracing is off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set
(and LANGFUSE_BASE_URL, when Langfuse isn't Langfuse Cloud). When it's off,
every helper here does nothing, so the agent and the evals behave exactly the
same with or without it.

A trace is a tree of observations. Ours look like this:

    eval refund-over-limit #1            one eval run (evals/runner.py), or
      support turn                       one chat turn (agent/runner.py)
        LangGraph                        the graph, from Langfuse's LangGraph handler
          planner                        each node the turn went through
            classify                     a call to Claude (a "generation")
          agent
            respond
          guardrail
            refund check                 the guardrail's verdict
              check_balance              an MCP tool call
          tools
            process_refund

Langfuse is built on OpenTelemetry: an observation started "as current" becomes
the parent of anything started inside it, in the same task or in tasks it spawns.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from langfuse import Langfuse, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from opentelemetry.sdk.trace import TracerProvider

_UNSET = object()
_public_key: Any = _UNSET  # the Langfuse project's public key, or None when tracing is off


def configure(public_key: str | None) -> None:
    """Trace to the Langfuse client for `public_key`, or turn tracing off with None.

    Without a call to this, tracing follows the environment variables.
    """
    global _public_key
    _public_key = public_key


def new_client(**settings: Any) -> Langfuse:
    """A Langfuse client, configured from the environment unless `settings` say otherwise.

    It gets a tracer provider of its own instead of becoming OpenTelemetry's
    global one. Libraries that trace through the global provider (the MCP SDK
    does) then stay out of our traces, and nothing is exported unless we
    started it. get_client(public_key=...) finds the client afterwards.
    """
    return Langfuse(tracer_provider=TracerProvider(), **settings)


def client() -> Langfuse | None:
    global _public_key
    if _public_key is _UNSET:
        keys = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
        if all(keys):
            new_client()
            _public_key = keys[0]
        else:
            _public_key = None
    return None if _public_key is None else get_client(public_key=_public_key)


def enabled() -> bool:
    return client() is not None


def connect() -> str | None:
    """Check that Langfuse accepts our keys. If it doesn't, turn tracing off and
    say why, rather than logging an export error for every batch of spans."""
    langfuse = client()
    if langfuse is None:
        return None
    try:
        langfuse.auth_check()
    except Exception as exc:
        configure(None)
        reason = str(exc).splitlines()[0][:120] if str(exc) else type(exc).__name__
        return f"Langfuse keys are set, but Langfuse didn't accept them ({reason}). Not tracing."
    return None


class _Off:
    """Stands in for an observation when tracing is off."""

    def update(self, **fields: Any) -> _Off:
        return self


@contextmanager
def observe(name: str, *, as_type: str = "span", **fields: Any) -> Iterator[Any]:
    """Record the block as an observation, nested under the current one.

    `as_type` is a Langfuse observation type: span, agent, generation, tool,
    guardrail, evaluator... `fields` are its input, metadata, model, and so on.
    Yields the observation, so the block can `update(output=...)` it.
    """
    langfuse = client()
    if langfuse is None:
        yield _Off()
        return
    with langfuse.start_as_current_observation(
        name=name,
        as_type=as_type,  # type: ignore[arg-type]
        **fields,
    ) as observation:
        yield observation


def attributes(**attrs: Any) -> Any:
    """Set trace attributes (session_id, user_id, tags, trace_name, metadata)
    for everything started inside the block."""
    if not enabled():
        return nullcontext()
    return propagate_attributes(**attrs)


class _InlineHandler(CallbackHandler):
    # LangChain runs callback handlers in a thread pool by default, so the span the
    # handler starts for a node never becomes "current" inside that node, and our
    # own observations would attach to the turn instead of the node that made them.
    # Running inline makes each node's span the parent of what happens inside it.
    run_inline = True


def usage_details(tokens: dict[str, int]) -> dict[str, int]:
    """Anthropic token counts under the names Langfuse prices them by."""
    return {
        "input": tokens["input_tokens"],
        "output": tokens["output_tokens"],
        "cache_creation_input_tokens": tokens["cache_creation_input_tokens"],
        "cache_read_input_tokens": tokens["cache_read_input_tokens"],
    }


def langgraph_callbacks() -> list[CallbackHandler]:
    """Callbacks for a LangGraph run: one observation per node."""
    if not enabled():
        return []
    return [_InlineHandler(public_key=_public_key)]


def current_trace_id() -> str | None:
    langfuse = client()
    return langfuse.get_current_trace_id() if langfuse else None


def trace_url(trace_id: str | None) -> str | None:
    """A link to the trace in the Langfuse UI, or None if it can't be built."""
    langfuse = client()
    if langfuse is None or trace_id is None:
        return None
    try:
        return langfuse.get_trace_url(trace_id=trace_id)  # looks up the project id once
    except Exception:
        return None


def score(
    trace_id: str, name: str, value: float, *, boolean: bool = False, comment: str | None = None
) -> None:
    """Attach a score to a trace. Boolean scores are 1 (pass) or 0 (fail)."""
    langfuse = client()
    if langfuse is not None:
        langfuse.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            data_type="BOOLEAN" if boolean else "NUMERIC",
            comment=comment,
        )


def flush() -> None:
    """Send everything still buffered. Call before a short-lived process exits."""
    langfuse = client()
    if langfuse is not None:
        langfuse.flush()
