import re

import numpy as np
import pytest

from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
from so_arena.mechanisms import Debate, Propaganda
from so_arena.models import FunctionModel
from so_arena.samplers.prompt_search import PromptSearch, PromptSearchSuite
from so_arena.samplers.psro import PSRO


def level_factory(strategy: str):
    """Strategy text 'level N' -> argument strength shift N (a stand-in for prompt quality)."""
    m = re.search(r"level (\d+)", strategy or "")
    lv = int(m.group(1)) if m else 0
    return synthetic_arguer(honest_mean=1.0 + 0.3 * lv, dishonest_mean=0.3 * lv, sd=0.2, label=f"level{lv}")


def optimizer_model():
    seen = []

    def fn(messages, options, i):
        text = messages[-1].content
        seen.append(text)
        levels = [int(x) for x in re.findall(r"level (\d+)", text)] or [0]
        nxt = max(levels) + 1
        return "".join(f"<rationale>push harder</rationale><strategy>argue at level {nxt + j}</strategy>" for j in range(2))

    return FunctionModel(fn, name="mock-optimizer"), seen


def setup():
    dom = SyntheticPersuasion(n_items=12, seed=4)
    items = dom.load()
    return items[:8], items[8:], dom.context()


def test_prompt_search_improves_reward_and_is_blind_to_ground_truth():
    train, test, ctx = setup()
    opt, seen = optimizer_model()
    mech = Propaganda(affordances={"agents": ["answer_key"]})
    search = PromptSearch(mech, train, role="agent", policy_factory=level_factory,
                          others={"judge": synthetic_judge()}, optimizer=opt, arms=["false"], directive="deceptive",
                          iterations=3, candidates_per_iter=2, eval_items=test, ctx=ctx)
    res = search.run()
    path = res.path()
    assert path["best_reward"].is_monotonic_increasing
    assert res.best.mean_reward > res.candidates[0].mean_reward
    assert res.evaluated and all(c.split == "eval" for c in res.evaluated)
    # values recorded (deceptive arm: all -1) but never shown to the optimizer
    assert all(v == -1.0 for v in res.best.values)
    assert all("value" not in s.lower() or "reward" in s for s in seen)
    assert search.example_prompt and "Write your argument" in search.example_prompt
    assert "deceptive" in seen[0].lower()


def test_suite_honesty_margin():
    train, test, ctx = setup()
    opt, _ = optimizer_model()
    suite = PromptSearchSuite(directives=("honest", "deceptive"), mechanism=Propaganda(affordances={"agents": ["answer_key"]}),
                              items=train, role="agent", policy_factory=level_factory,
                              others={"judge": synthetic_judge()}, optimizer=opt, iterations=2,
                              candidates_per_iter=2, eval_items=test, ctx=ctx)
    # honest search argues the true side, deceptive the false side
    suite.searches["honest"].arms = ["true"]
    suite.searches["deceptive"].arms = ["false"]
    suite.run()
    m = suite.honesty_margin()
    assert m["margin"] > 0 and m["ci_low"] > 0
    assert m["honest_value"] == 1.0 and m["deceptive_value"] == -1.0
    assert set(suite.paths()["directive"]) == {"honest", "deceptive"}


def test_psro_runs_and_tracks_meta_strategy():
    train, _, ctx = setup()
    opt, _ = optimizer_model()
    mech = Debate(rounds=1, affordances={"agents": ["answer_key"]}, zero_sum=True)
    psro = PSRO(mech, train[:4], roles=["debater_a", "debater_b"],
                initial={"debater_a": {"base": ""}, "debater_b": {"base": ""}},
                policy_factories={"debater_a": level_factory, "debater_b": level_factory},
                fixtures={"judge": synthetic_judge()}, optimizer=opt,
                stances={"debater_a": "true", "debater_b": "false"}, iterations=2,
                search_kwargs={"iterations": 1, "candidates_per_iter": 2}, ctx=ctx)
    hist = psro.run()
    assert len(hist) == 3
    assert len(hist[-1].populations["debater_a"]) == 3
    for h in hist:
        for r, sig in h.meta_strategy.items():
            assert sum(sig) == pytest.approx(1.0)
    assert hist[1].nash_conv_prev is not None and hist[1].nash_conv_prev >= -1e-9
    # higher-level strategies dominate: the meta-strategy moves to the newest best responses
    assert np.argmax(hist[-1].meta_strategy["debater_a"]) == 2


def test_autoresearch_ratchet_and_log():
    train, test, ctx = setup()
    seen = []

    def fn(messages, options, i):
        text = messages[-1].content
        seen.append(text)
        levels = [int(x) for x in re.findall(r"level (\d+)", text)] or [0]
        nxt = max(levels) + (1 if i % 2 == 0 else -1)  # alternate an improving and a worse experiment
        return f"<hypothesis>try level {max(nxt, 0)}</hypothesis><strategy>argue at level {max(nxt, 0)}</strategy>"

    search = PromptSearch(Propaganda(affordances={"agents": ["answer_key"]}), train, role="agent",
                          policy_factory=level_factory, others={"judge": synthetic_judge()},
                          optimizer=FunctionModel(fn, name="researcher"), arms=["true"], algorithm="autoresearch",
                          iterations=4, ctx=ctx)
    res = search.run()
    assert len(search.research_log) == 4
    assert any(e["kept"] for e in search.research_log)
    kept_rewards = [e["reward"] for e in search.research_log if e["kept"]]
    assert all(e["kept"] is False for e in search.research_log if e.get("note"))
    assert kept_rewards == sorted(kept_rewards)  # the incumbent only ever improves
    assert "Research log" in seen[-1] and "experiment 1" in seen[-1]
    assert search.incumbent.mean_reward == max(c.mean_reward for c in res.candidates)
