"""Expected one-step logit update of StrategyGradient(natural=True) vs the documented eta*(u_a - u_bar)."""
import asyncio, types
import numpy as np
import oversight_arena.elicitation.rl as rl
from oversight_arena.core.strategy import Strategy

U = {"rare_good": 1.0, "common_a": 0.0, "common_b": 0.0}   # deterministic payoff of each strategy
async def fake_run_episode(mech, task, prof, agents, domain, seed=0, **kw):
    s = prof.assignments["p"].strategy.name
    return types.SimpleNamespace(rewards={"p": U[s]}, gt={}, error=None)
rl.run_episode = fake_run_episode
pop = [Strategy(name=n) for n in U]
init = [0.02, 0.49, 0.49]
deltas = {True: [], False: []}
for natural in (True, False):
    for seed in range(400):
        sg = rl.StrategyGradient(domain=None, mechanism=None, agents={}, populations={"p": pop}, lr=0.5, batch=16, iterations=1,
                                 init={"p": init}, tasks=[types.SimpleNamespace(id="t")], seed=seed, natural=natural)
        th0 = sg.logits["p"].copy()
        asyncio.run(sg.run())
        d = sg.logits["p"] - th0
        deltas[natural].append(d[0] - d[1:].mean())   # relative logit gain of the rare good strategy
ubar = float(np.dot(init, list(U.values())))
print("documented natural update (relative):  eta*(u_rare - u_common) =", 0.5 * (1 - 0))
print("StrategyGradient natural=True, mean over 400 seeds:", round(float(np.mean(deltas[True])), 3))
print("StrategyGradient natural=False (REINFORCE), mean  :", round(float(np.mean(deltas[False])), 3),
      " (doc: eta*pi_a*(u_a-u_bar) =", round(0.5 * 0.02 * (1 - ubar), 4), "+ others' term)")
print("P(rare strategy sampled in a batch of 16) =", round(1 - 0.98 ** 16, 3))
