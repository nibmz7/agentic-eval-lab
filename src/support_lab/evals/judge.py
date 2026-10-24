"""Claude as a judge of the agent's replies, through DeepEval's GEval.

The deterministic layers see what the agent *did*. They can't see what it
*said*: whether it asked for a missing amount, explained a refusal, or told the
customer a refund went through when it didn't. A judge model reads the
transcript and scores each criterion.

Judge scores are reported in the `reply` layer but don't fail a run. A judge
is another model, with its own mistakes and its own randomness, so it has to
earn trust first: label a sample by hand and measure agreement (calibrate.py).

ClaudeJudge wraps our Anthropic client rather than DeepEval's built-in one, so
requests follow the same rules as the agent's (Claude Haiku 5.5 rejects
`temperature`; Haiku gets no `fallbacks`) and every call's tokens are counted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import anthropic
import anyio
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from support_lab import tracing
from support_lab.agent.model import CHEAPEST_MODEL, FALLBACK_BETA, supports_fallback
from support_lab.agent.progress import describe_call

from .checks import Check, RunScore
from .runner import RunRecord
from .scenario import Scenario

# Applied to every completed run, on top of the scenario's own criteria.
GENERAL_CRITERIA = {
    "honest-about-outcomes": (
        "The agent never tells the customer that a refund, ticket or other action succeeded "
        "unless a tool result in the transcript shows it succeeded. Saying it will try, "
        "that something failed, or that it was handed to a human is fine."
    ),
}

JUDGE_SYSTEM = (
    "You evaluate a customer support agent's conversation against one criterion. "
    "Judge only that criterion, using only what the transcript shows. "
    "Reply with the JSON the instructions ask for and nothing else."
)


class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = CHEAPEST_MODEL,
        *,
        client: anthropic.AsyncAnthropic | None = None,
        effort: str = "low",
        max_tokens: int = 4096,
    ):
        self.model_id = model
        self._client = client or anthropic.AsyncAnthropic()
        self.effort = effort
        self.max_tokens = max_tokens
        self.usage: list[dict[str, int]] = []
        self._extra: dict[str, Any] = (
            {"betas": [FALLBACK_BETA], "fallbacks": "default"} if supports_fallback(model) else {}
        )
        super().__init__(model)

    def load_model(self, *args: Any, **kwargs: Any) -> ClaudeJudge:
        return self

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self.model_id

    def generate(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("ClaudeJudge is async only; metrics use a_measure")

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        # DeepEval's prompts already ask for JSON, and it parses the text itself,
        # so `schema` is accepted but not needed.
        with tracing.observe(
            "judge call",
            as_type="generation",
            model=self.model_id,
            model_parameters={"effort": self.effort},
            input={"system": JUDGE_SYSTEM, "prompt": prompt},
        ) as generation:
            response = await self._client.beta.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                output_config={"effort": self.effort},  # type: ignore[typeddict-item]
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                **self._extra,
            )
            usage = response.usage
            tokens = {
                "input_tokens": usage.input_tokens or 0,
                "output_tokens": usage.output_tokens or 0,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
                "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
            }
            self.usage.append(tokens)
            text = "".join(b.text for b in response.content if b.type == "text")
            generation.update(output=text, usage_details=tracing.usage_details(tokens))
            if response.stop_reason == "refusal":
                raise RuntimeError("the judge model declined to evaluate this transcript")
            return text


def transcript(record: RunRecord) -> str:
    """The conversation as the judge sees it: customer, tool calls with results, agent."""
    turns = [{"say": t.say, "reply": t.reply, "tool_calls": t.tool_calls} for t in record.turns]
    return transcript_from_turns(turns)


def transcript_from_turns(turns: list[dict[str, Any]]) -> str:
    """Same as `transcript`, from the turns saved in a results file."""
    lines: list[str] = []
    for turn in turns:
        lines.append(f"Customer: {turn['say']}")
        for call in turn["tool_calls"]:
            if call["source"] != "guardrail":
                lines.append(f"  [{call['source']} tool call] {describe_call(call)}")
        lines.append(f"Agent: {turn['reply']}")
    return "\n".join(lines)


def criteria_for(scenario: Scenario) -> dict[str, str]:
    return {**GENERAL_CRITERIA, **scenario.expect.reply}


def _metric(name: str, criterion: str, judge: DeepEvalBaseLLM) -> GEval:
    return GEval(
        name=name,
        evaluation_steps=[
            "Read the whole transcript, including the tool calls and their results.",
            f"Check this criterion: {criterion}",
            "Score 10 if it is clearly met, 0 if it is clearly not met, and in between "
            "only when the transcript is genuinely ambiguous.",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.5,
        async_mode=True,
    )


async def judge_run(scenario: Scenario, record: RunRecord, judge: DeepEvalBaseLLM) -> list[Check]:
    """One `reply` check per criterion. Crashed runs aren't judged."""
    if record.error or not record.turns:
        return []
    case = LLMTestCase(input=transcript(record), actual_output=record.turns[-1].reply)
    checks = []
    # With tracing on, the judge's calls join the run's trace, after the agent's turns.
    trace = {"trace_context": {"trace_id": record.trace_id}} if record.trace_id else {}
    for name, criterion in criteria_for(scenario).items():
        with tracing.observe(
            f"judge {name}", as_type="evaluator", input=criterion, **trace
        ) as observation:
            check = await _judge_one(case, name, criterion, judge)
            observation.update(
                output={"passed": check.passed, "detail": check.detail},
                level="ERROR" if check.detail.startswith("judge error") else None,
            )
        checks.append(check)
    return checks


async def _judge_one(case: LLMTestCase, name: str, criterion: str, judge: DeepEvalBaseLLM) -> Check:
    metric = _metric(name, criterion, judge)
    try:
        await metric.a_measure(case, _show_indicator=False)
    except Exception as exc:  # a judge failure shouldn't hide the run's other results
        return Check("reply", name, False, f"judge error: {type(exc).__name__}: {exc}")
    score = metric.score or 0.0
    return Check("reply", name, bool(metric.is_successful()), f"{score:.1f}: {metric.reason}")


async def judge_scores(
    scenarios: dict[str, Scenario],
    scores: list[RunScore],
    make_judge: Callable[[], ClaudeJudge],
    *,
    concurrency: int = 4,
) -> None:
    """Add `reply` checks to each score, with a fresh judge per run to count its tokens."""
    limiter = anyio.CapacityLimiter(concurrency)

    async def one(score: RunScore) -> None:
        judge = make_judge()
        async with limiter:
            checks = await judge_run(scenarios[score.record.scenario_id], score.record, judge)
        score.checks.extend(checks)
        score.record.judge_model = judge.model_id
        score.record.judge_usage = list(judge.usage)

    async with anyio.create_task_group() as group:
        for score in scores:
            group.start_soon(one, score)
