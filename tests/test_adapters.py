import asyncio
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.log import EvalSample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import Score

from oversight_arena import Observation, PublicTask, Role
from oversight_arena.adapters.inspect import InspectPolicy, ScoreSelector, from_eval_sample


def test_real_inspect_model_mock_backend():
    model = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", '{"answer": "yes"}')],
        memoize=False,
    )
    policy = InspectPolicy(model, json_output=True)
    response = asyncio.run(policy(Observation(PublicTask("t", "q"), "agent", "answer", (), {}, 3)))
    assert response.content == {"answer": "yes"}
    assert response.cost is None


def test_import_explicit_scores_and_exclude_target():
    sample = EvalSample(
        id="id",
        epoch=1,
        input="input",
        target="SECRET TARGET",
        scores={"reward": Score(value=0.4), "truth": Score(value="C")},
    )
    record = from_eval_sample(
        sample,
        task=PublicTask("t", "public"),
        roles=[Role("a")],
        reward_map={"a": ScoreSelector("reward")},
        truth_map={"a": {"quality": ScoreSelector("truth", labels={"C": 1})}},
    )
    assert record.rewards["a"].value == 0.4
    assert record.evaluations[0].per_agent["a"]["quality"] == 1
    assert "SECRET TARGET" not in str(record)
    with pytest.raises(ValueError, match="separate"):
        from_eval_sample(
            sample,
            task=PublicTask("t", "public"),
            roles=[Role("a")],
            reward_map={"a": ScoreSelector("reward")},
            truth_map={"a": {"quality": ScoreSelector("reward")}},
        )


def test_full_controlarena_evaluation_and_import(tmp_path):
    pytest.importorskip("control_arena")
    path = Path(__file__).parents[1] / "examples" / "controlarena_bridge.py"
    spec = importlib.util.spec_from_file_location("controlarena_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run(str(tmp_path))
    assert result.rewards["worker"].value == 0.75
    assert result.evaluations[0].per_agent["worker"]["quality"] == 1
    assert list(tmp_path.glob("*.eval"))
