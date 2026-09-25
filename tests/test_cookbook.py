"""The cookbook's API calls (docs/experiments.md), run with real model names in simulate mode (no API calls)."""

import pytest

import so_arena as soa
from so_arena.core.game import Player
from so_arena.core.runner import Profile
from so_arena.domains import get_domain
from so_arena.mechanisms import Consultancy, Debate, DirectJudge, Propaganda, ReviewedWork


@pytest.fixture(autouse=True)
def simulate():
    soa.configure(simulate=True)
    yield
    soa.configure(simulate=False)


@pytest.fixture
def setup():
    dom = get_domain("gsm8k")
    items = dom.load(limit=4)
    expert = soa.LLMPolicy("anthropic/claude-haiku-4-5", temperature=0.7)
    judge = soa.LLMPolicy("openai/gpt-4o-mini", elicitation="logprobs", temperature=0)
    return dom, items, expert, judge


def test_asd_recipe(setup, tmp_path):
    from so_arena.samplers.arms import ASDExperiment

    dom, items, expert, judge = setup
    exp = ASDExperiment([DirectJudge(), Propaganda(), Consultancy(rounds=2), Debate(rounds=2)], items,
                        agent=expert, fixtures={"judge": judge}, ctx=dom.context(), store=str(tmp_path / "asd"))
    eps = exp.run()
    assert eps and all(e.error is None for e in eps)
    assert sum(e.total_usage.cost_usd for e in eps) > 0  # simulated calls are priced
    s = exp.summary()
    assert set(s["mechanism"]) == {"direct", "propaganda", "consultancy", "debate"}


def test_optimization_recipe(setup):
    from so_arena.samplers.pools import OptimizationExperiment

    dom, items, expert, judge = setup
    mech = ReviewedWork(critique_rounds=1, rebuttal=True, transform="prob")
    profile = Profile(name="open", players={"worker": expert, "critic": expert, "reviewer": judge})
    # proposal pooled x3, critique x2, rebuttal x2 (phase-specific keys override role keys)
    exp = OptimizationExperiment(mech, items[:2], profile,
                                 pool_sizes={"worker": 3, "worker:rebuttal1": 2, "critic": 2})
    trees = exp.run()
    assert len(trees) == 2 and all(t.n_leaves == 12 for t in trees)
    surface = exp.grid({"worker:work": [1, 3], "critic": [1, 2], "worker:rebuttal1": [1, 2]})
    assert len(surface) == 8 and "reward_worker" in surface and "level_worker:rebuttal1" in surface


def test_prompt_search_recipe(setup):
    from so_arena.samplers.prompt_search import PromptSearchSuite

    dom, items, expert, judge = setup
    suite = PromptSearchSuite(directives=("honest", "deceptive"), mechanism=Consultancy(rounds=1), items=items[:2],
                              eval_items=items[2:], role="consultant",
                              policy_factory=lambda s: soa.LLMPolicy("anthropic/claude-haiku-4-5", strategy=s),
                              others={"judge": judge}, optimizer="anthropic/claude-sonnet-4-5",
                              algorithm="reflective", iterations=1, candidates_per_iter=2)
    suite.searches["honest"].arms = ["true"]
    suite.searches["deceptive"].arms = ["false"]
    suite.run()
    m = suite.honesty_margin()
    assert "margin" in m and not suite.paths().empty


def test_game_and_rl_recipes(setup):
    from so_arena.games import EmpiricalGameExperiment
    from so_arena.integrations.rl import MechanismEnv, reward_function, rollout_prompts

    dom, items, expert, judge = setup
    strategies = {r: {"plain": soa.LLMPolicy("openai/gpt-4o"), "aggressive": soa.LLMPolicy("openai/gpt-4o", strategy="Attack every claim.")}
                  for r in ("debater_a", "debater_b")}
    exp = EmpiricalGameExperiment(Debate(zero_sum=True, rounds=1), items[:2], strategies, fixtures={"judge": judge},
                                  stances={"debater_a": "true", "debater_b": "false"})
    exp.run()
    game = exp.game()
    assert game.shape == (2, 2)
    game.pure_nash()

    async def drive():
        env = MechanismEnv(Consultancy(rounds=2), {"judge": Player(policy=judge)}, trainees={"consultant": "false"})
        res = await env.reset(items[0])
        steps = 0
        while not res.done:
            steps += 1
            res = await env.step("An argument for my side.")
        return steps, res

    steps, res = soa.run_sync(drive())
    assert steps == 2 and res.done
    data = rollout_prompts(Propaganda(), "agent", items[:2], {"judge": Player(policy=judge)}, stance="false")
    fn = reward_function(Propaganda(), "agent", items, {"judge": Player(policy=judge)}, stances={"agent": "false"})
    assert len(fn(prompts=[None], completions=["text"], item_id=[items[0].id])) == 1 and len(data) == 2
