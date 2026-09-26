import math

import numpy as np
import pytest

import so_arena as soa
from so_arena.core.game import Player
from so_arena.core.runner import PlayerSpec, Profile
from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
from so_arena.games import NormalFormGame
from so_arena.games.learning import policy_gradient, training_outcome
from so_arena.integrations.rl import MechanismEnv, reward_function, rollout, rollout_prompts
from so_arena.mechanisms import Consultancy, Propaganda
from so_arena.theory import whistleblower as wb


def synth():
    dom = SyntheticPersuasion(n_items=4, seed=5)
    return dom, dom.load(), dom.context()


def test_inspect_task_roundtrip(tmp_path):
    pytest.importorskip("inspect_ai")
    from inspect_ai import eval as inspect_eval

    from so_arena.integrations.inspect_task import episodes_from_inspect_logs, inspect_task

    dom, items, ctx = synth()
    mech = Propaganda(affordances={"agents": ["answer_key"]})
    profiles = [Profile(name=a, players={"agent": PlayerSpec(policy=synthetic_arguer(), stance=a), "judge": synthetic_judge()})
                for a in ("true", "false")]
    task = inspect_task(mech, items, profiles, ctx=ctx)
    logs = inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")
    assert logs[0].status == "success"
    eps = episodes_from_inspect_logs(logs[0].location)
    assert len(eps) == 8
    from so_arena.analysis.frames import role_frame
    from so_arena.analysis.metrics import asd

    assert asd(role_frame(eps))["asd"].iloc[0] > 0
    metrics = logs[0].results.scores[0].metrics
    assert any(k.startswith("reward_agent") or k == "mean" for k in metrics) or metrics


def test_mechanism_env_multi_turn():
    dom, items, ctx = synth()
    mech = Consultancy(rounds=2, affordances={"agents": ["answer_key"]})
    env = MechanismEnv(mech, {"judge": Player(policy=synthetic_judge())}, {"consultant": "true"}, ctx=ctx)

    async def go():
        res = await env.reset(items[0])
        n = 0
        while not res.done:
            assert res.observation.role == "consultant"
            assert res.observation.messages[0]["role"] == "system"
            n += 1
            res = await env.step(f'<arg for="{items[0].true_label}" strength="2.0"> my argument')
        return n, res

    n, res = soa.run_sync(go())
    assert n == 2 and res.done and res.rewards["consultant"] is not None
    assert res.episode.value("consultant") == 1.0


def test_rollout_with_policy_matches_direct_run():
    dom, items, ctx = synth()
    mech = Propaganda(affordances={"agents": ["answer_key"]})
    env = MechanismEnv(mech, {"judge": Player(policy=synthetic_judge())}, {"agent": "false"}, ctx=ctx)
    ep = soa.run_sync(rollout(env, items[1], synthetic_arguer()))
    assert ep.error is None and ep.value("agent") == -1.0


def test_trl_style_reward_function_and_prompts():
    dom, items, ctx = synth()
    mech = Propaganda(affordances={"agents": ["answer_key"]})
    players = {"judge": Player(policy=synthetic_judge())}
    prompts = rollout_prompts(mech, "agent", items[:2], players, stance="true", ctx=ctx)
    assert len(prompts) == 2 and prompts[0]["prompt"][-1]["role"] == "user"
    fn = reward_function(mech, "agent", items, players, stances={"agent": "true"}, ctx=ctx)
    lab = items[0].true_label
    strong = f'<arg for="{lab}" strength="3.0"> strong'
    weak = f'<arg for="{lab}" strength="0.0"> weak'
    r = fn(prompts=[None, None], completions=[strong, [{"role": "assistant", "content": weak}]], item_id=[items[0].id, items[0].id])
    assert r[0] > r[1]
    assert len(fn.history) == 2 and fn.history[0]["value"] == 1.0


def test_policy_gradient_selects_equilibrium_by_basin():
    g = wb.whistleblower_game(3, R=1.0, delta=0.5, s=0.2)
    p_star = wb.mixed_threshold_each(3, 0.2, 0.5)
    lo = training_outcome(g, init_report=p_star - 0.1, lr=2.0, steps=3000)
    hi = training_outcome(g, init_report=p_star + 0.1, lr=2.0, steps=3000)
    assert lo["final_p"] < 0.05 and hi["final_p"] > 0.95
    assert lo["reverted"] < 0.1 and hi["reverted"] > 0.9
    common = wb.whistleblower_game(3, R=1.0, delta=0.5, s=0.0)
    assert training_outcome(common, init_report=0.9, lr=2.0, steps=3000)["final_p"] < 0.5


def test_kl_regularized_pg_converges_to_tilted_policy():
    # one player vs nature: payoffs u; optimum of E[u] - tau KL(pi||uniform) is softmax(u / tau)
    u = np.array([0.0, 1.0, 0.5])
    g = NormalFormGame(["p"], {"p": ["a", "b", "c"]}, {"p": u})
    tau = 0.5
    df = policy_gradient(g, lr=1.0, steps=4000, kl=tau)
    last = df.iloc[-1]
    got = np.array([last["p_p_a"], last["p_p_b"], last["p_p_c"]])
    target = np.exp(u / tau) / np.exp(u / tau).sum()
    assert got == pytest.approx(target, abs=1e-3)
