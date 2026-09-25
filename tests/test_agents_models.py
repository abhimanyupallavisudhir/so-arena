import asyncio
import math

import pytest

import oversight_arena as oa
from oversight_arena.agents.base import Observation, ResponseSpec
from oversight_arena.core.types import TokenLogprob, ToolCall
from oversight_arena.models import CachedModel, FunctionModel, ModelOutput
from conftest import make_tasks


def obs(response, tools=None, **kw):
    t = make_tasks(1)[0]
    return Observation(role="judge", role_title="Judge", task=t.view(), task_text="Q?", prompt="Decide.",
                       response=response, tools=tools or [], brief="BRIEF", strategy="STRAT", **kw)


async def test_llm_agent_renders_and_parses():
    seen = {}

    def fn(messages, config, tools, sample):
        seen["sys"] = messages[0].content
        seen["user"] = messages[-1].content
        return 'Reasoning. {"A": 0.25, "B": 0.75}'

    a = oa.LLMAgent(FunctionModel(fn))
    act = await a.act(obs(ResponseSpec.distribution(["A", "B"])))
    assert "BRIEF" in seen["sys"] and "STRAT" in seen["sys"] and "Q?" in seen["user"]
    assert abs(act.parsed["probs"]["B"] - 0.75) < 1e-9 and act.parsed["choice"] == "B"


async def test_llm_agent_retries_unparseable():
    calls = []

    def fn(messages, config, tools, sample):
        calls.append(1)
        return "no idea" if len(calls) == 1 else "ANSWER: A"

    act = await oa.LLMAgent(FunctionModel(fn)).act(obs(ResponseSpec.choice(["A", "B"])))
    assert act.parsed["choice"] == "A" and len(calls) == 2


async def test_llm_agent_tool_loop():
    @oa.tool
    def add(x: int, y: int) -> str:
        """Add numbers.

        Args:
            x: first
            y: second
        """
        return str(x + y)

    def fn(messages, config, tools, sample):
        if messages[-1].role == "tool":
            return f"The sum is {messages[-1].content}."
        return {"text": "", "tool_calls": [{"id": "c1", "name": "add", "arguments": {"x": 2, "y": 3}}]}

    act = await oa.LLMAgent(FunctionModel(fn)).act(obs(ResponseSpec.text(), tools=[add]))
    assert act.text == "The sum is 5." and act.tool_trace[0].result == "5"


async def test_logprob_and_sample_elicitation():
    def fn(messages, config, tools, sample):
        if config.logprobs:
            return ModelOutput(text="B", logprobs=[TokenLogprob(token="B", logprob=math.log(0.8), top=[("B", math.log(0.8)), ("A", math.log(0.2))])])
        return "ANSWER: A" if sample % 3 else "ANSWER: B"

    a = oa.LLMAgent(FunctionModel(fn), elicitation="logprobs")
    act = await a.act(obs(ResponseSpec.distribution(["A", "B"], reasoning=False)))
    assert abs(act.parsed["probs"]["B"] - 0.8) < 1e-3
    b = oa.LLMAgent(FunctionModel(fn), elicitation="sample", n_samples=6)
    act = await b.act(obs(ResponseSpec.distribution(["A", "B"])))
    assert act.parsed["probs"]["A"] > act.parsed["probs"]["B"]


async def test_cached_model_distinct_samples(tmp_path):
    n = {"calls": 0}

    def fn(messages, config, tools, sample):
        n["calls"] += 1
        return f"sample {sample}"

    m = CachedModel(FunctionModel(fn, name="f"), path=tmp_path / "c.sqlite")
    msgs = [oa.core.types.ChatMessage.user("hi")]
    a = await m.generate(msgs, sample=0)
    b = await m.generate(msgs, sample=1)
    c = await m.generate(msgs, sample=0)
    assert a.text != b.text and c.text == a.text and c.cached and n["calls"] == 2


async def test_inspect_backend_mockllm():
    pytest.importorskip("inspect_ai")
    from oversight_arena.models.inspect_model import InspectModel

    m = InspectModel("mockllm/model")
    out = await m.generate([oa.core.types.ChatMessage.user("hello")])
    assert isinstance(out.text, str) and out.usage.calls == 1
    agent = oa.LLMAgent(m)
    act = await agent.act(obs(ResponseSpec.text()))
    assert isinstance(act.text, str)
