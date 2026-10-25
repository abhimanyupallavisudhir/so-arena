"""PeerPrediction(rule='multitask') draws peer/task pairings from g.episode_id.
In a sampled game tree the episode id encodes the path (the plan), so identical reporter
actions get different rewards depending on which (identical) candidate was followed;
best-of-N over a pool then selects pairing luck. Across profiles the id also differs."""
from so_arena.core.items import TaskItem
from so_arena.core.policy import FunctionPolicy
from so_arena.core.game import Player, RunContext
from so_arena.core.runner import run_sync, run_episodes, Profile
from so_arena.mechanisms.elicitation import PeerPrediction
from so_arena.samplers.pools import expand_tree

subs = [{"id": f"t{i}", "question": f"q{i}", "labels": ["A", "B"]} for i in range(4)]
item = TaskItem(id="bundle", question="bundle", context={"subitems": subs})
# fixed answer patterns per reporter (deterministic: every candidate in a pool is identical)
pat = {"reporter_1": "ABAB", "reporter_2": "AABB", "reporter_3": "ABBA"}
def pol(role):
    return FunctionPolicy(lambda req, ctx: f"Answer: {pat[ctx.role][int(req.phase.split(':t')[1])]}", label="fixed")
players = {r: Player(policy=pol(r)) for r in pat}
mech = PeerPrediction(n_reporters=3, rule="multitask")
tree, eps = run_sync(expand_tree(mech, item, players, pool_sizes={"reporter_1:answer:t0": 6}, ctx=RunContext(),
                                 ground_truth=[], keep_episodes=True))
print("reporter_1 answers per leaf:", {tuple(e.outcome.data["all_answers"]["reporter_1"]) for e in eps})
print("reporter_1 rewards across 6 identical candidates:", [round(e.rewards["reporter_1"], 3) for e in eps])
print("best-of-6 picks", max(e.rewards["reporter_1"] for e in eps), "vs mean",
      sum(e.rewards["reporter_1"] for e in eps) / len(eps))

# same actions, two profiles that differ only by name -> different rewards
profs = [Profile(name=n, players={r: pol(r) for r in pat}) for n in ("arm_x", "arm_y")]
out = run_sync(run_episodes(mech, [item], profs, ground_truth=[]))
print("profile rewards (identical play):", {e.profile: {r: round(v, 3) for r, v in e.rewards.items()} for e in out})
