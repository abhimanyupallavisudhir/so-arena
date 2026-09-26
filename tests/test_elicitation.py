import asyncio
import json

import oversight_arena as oa
from oversight_arena.channels import EvidencePolicy
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.elicitation import (
    PSRO, Evaluator, LLMProposer, MechanismEnv, ParamProposer, PromptOptimizer, StrategyGradient, library,
    pool_oracle, preference_pairs, reward_function,
)
from oversight_arena.mechanisms import Debate, Propaganda
from oversight_arena.models import FunctionModel
from oversight_arena.sim import BayesianBitJudge, BitAdvocate

DOM = HiddenBits(n_tasks=12)
AGENTS = {"kind:judge": BayesianBitJudge(trust=0.8), "*": BitAdvocate()}


def test_param_optimizer_improves():
    liar = oa.Strategy(name="liar", stance=oa.Stance.INCORRECT, params={"claims": 1, "lie_rate": 0.0})
    ev = Evaluator(DOM, Propaganda(evidence=EvidencePolicy(budget=1)), AGENTS, "agent")
    tr = asyncio.run(PromptOptimizer(ev, ParamProposer({"claims": (1, 8), "lie_rate": (0.0, 1.0)}), seeds=[liar],
                                     iterations=4, per_iter=4, minibatch=None).run())
    assert tr.best(1)[0].reward > tr.candidates[0].reward
    assert len(tr.trajectory()) == 5


def test_llm_proposers_and_constraint():
    def fake(messages, config, tools, sample):
        return json.dumps({"instructions": [f"variant {sample}-{i}" for i in range(3)], "reflection": "x"})

    ev = Evaluator(DOM, Propaganda(), AGENTS, "agent", tasks=DOM.tasks()[:4])
    for style in ("opro", "reflective"):
        tr = asyncio.run(PromptOptimizer(ev, LLMProposer(FunctionModel(fake), style=style), iterations=2, per_iter=3,
                                         constraint=lambda e: False, minibatch=2).run())
        assert all(not c.accepted for c in tr.candidates) and len(tr.candidates) == 7


def test_psro_and_strategy_gradient():
    pool_a = [oa.Strategy(name=f"a{c}", stance=oa.Stance.CORRECT, params={"claims": c}) for c in (1, 4)]
    pool_b = [oa.Strategy(name=f"b{c}", stance=oa.Stance.INCORRECT, params={"claims": c, "lie_rate": 1.0}) for c in (1, 4)]
    psro = PSRO(DOM, Debate(rounds=1, evidence=EvidencePolicy(budget=2)), AGENTS, {"debater_a": pool_a[:1], "debater_b": pool_b[:1]},
                {"debater_a": pool_oracle(pool_a), "debater_b": pool_oracle(pool_b)}, iterations=2, tasks=DOM.tasks()[:6])
    trace = asyncio.run(psro.run())
    assert len(trace.iterations) == 3 and trace.game.shape[0] >= 1
    sg = StrategyGradient(DOM, Propaganda(evidence=EvidencePolicy(budget=1)), AGENTS,
                          {"agent": [oa.Strategy(name="weak", stance=oa.Stance.INCORRECT, params={"claims": 1}),
                                     oa.Strategy(name="strong", stance=oa.Stance.INCORRECT, params={"claims": 6, "lie_rate": 1.0})]},
                          iterations=6, batch=16, lr=3.0)
    df = asyncio.run(sg.run())
    assert df["p[agent:strong]"].iloc[-1] > 0.5


def test_mechanism_env_and_rl_adapters():
    env = MechanismEnv(DOM, Debate(rounds=1), fixtures={"judge": BayesianBitJudge(trust=0.9)}, trainable=["debater_a", "debater_b"])
    role, obs = env.reset(DOM.tasks()[0])
    n = 0
    while role is not None:
        assert env.render(obs)[0]["role"] == "system"
        role, obs = env.step('<bit i="0">1</bit>')
        n += 1
    assert n == 2 and set(env.rewards) == {"debater_a", "debater_b"}
    fn = reward_function(DOM, Propaganda(), "agent", fixtures={"judge": BayesianBitJudge(trust=0.9)}, require_stance=False)
    vals = fn(["p", "p"], ['<bit i="1">1</bit>', '<bit i="1">0</bit>'], task_id=[DOM.tasks()[0].id] * 2)
    assert len(vals) == 2 and all(isinstance(v, float) for v in vals)
    res = oa.Experiment(DOM, Propaganda(), AGENTS, oa.Seeds(n=3, strategies={"agent": oa.Strategy(name="s", params={"sample": {"claims": (1, 4), "lie_rate": (0, 1)}})}), progress=False).run()
    pp = preference_pairs(res, "agent", gt="honesty")
    assert {"chosen", "rejected", "gt_agrees"} <= set(pp.columns)


def test_strategy_library():
    lib = library()
    assert len(lib) == 2 * 11 and any(s.tags["honest"] for s in lib) and any(not s.tags["honest"] for s in lib)
