import re

import pytest

pytest.importorskip("gepa")

from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge  # noqa: E402
from so_arena.integrations.gepa import MechanismAdapter, gepa_search, model_callable  # noqa: E402
from so_arena.mechanisms import Propaganda  # noqa: E402
from so_arena.models import FunctionModel  # noqa: E402
from so_arena.samplers.prompt_search import PromptSearch  # noqa: E402


def level_factory(strategy: str):
    m = re.search(r"level (\d+)", strategy or "")
    lv = int(m.group(1)) if m else 0
    return synthetic_arguer(honest_mean=1.0 + 0.3 * lv, dishonest_mean=0.3 * lv, sd=0.2, label=f"level{lv}")


def reflection_model():
    seen = []

    def fn(messages, options, i):
        text = messages[-1].content
        seen.append(text)
        levels = [int(x) for x in re.findall(r"level (\d+)", text)] or [0]
        return f"```\nArgue at level {max(levels) + 1}.\n```"

    return FunctionModel(fn, name="mock-reflector"), seen


def make_search():
    dom = SyntheticPersuasion(n_items=10, seed=4)
    items = dom.load()
    opt, seen = reflection_model()
    search = PromptSearch(Propaganda(affordances={"agents": ["answer_key"]}), items[:6], role="agent",
                          policy_factory=level_factory, others={"judge": synthetic_judge()}, optimizer=opt,
                          arms=["false"], directive="deceptive", eval_items=items[6:], ctx=dom.context())
    return search, seen


def test_adapter_scores_by_mechanism_reward_and_hides_ground_truth():
    search, _ = make_search()
    adapter = MechanismAdapter(search)
    out = adapter.evaluate(search.items[:3], {"strategy": "argue at level 2"}, capture_traces=True)
    assert len(out.scores) == 3 and all(isinstance(s, float) for s in out.scores)
    ds = adapter.make_reflective_dataset({"strategy": "argue at level 2"}, out, ["strategy"])
    rows = ds["strategy"]
    assert rows and all("reward" in r["Feedback"] for r in rows)
    text = str(rows).lower()
    assert "ground truth" not in text and "'value'" not in text
    rec = adapter.frame()
    assert set(rec.columns) >= {"strategy", "reward", "value"} and (rec["value"] == -1.0).all()


def test_gepa_search_climbs_the_mechanism_reward():
    search, seen = make_search()
    result, records = gepa_search(search, max_metric_calls=40, reflection_minibatch_size=2)
    best = result.best_candidate["strategy"]
    assert re.search(r"level (\d+)", best) and int(re.search(r"level (\d+)", best).group(1)) >= 1
    by = records.groupby("strategy")["reward"].mean()
    assert by[best] > by["Play your role well."]
    # the reflection prompt knew the mechanism and the directive, not the truth
    assert seen and "Propaganda" in seen[0] or "propaganda" in seen[0].lower()
    assert "deceptive" in seen[0].lower()


def test_model_callable_accepts_message_lists():
    m, _ = reflection_model()
    call = model_callable(m)
    assert "level" in call("level 3")
    assert "level" in call([{"role": "user", "content": "level 5"}])
