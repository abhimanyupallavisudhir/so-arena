# Release pseudonymizes players' labels/policy ids, but turn.metadata["mixture_component"] keeps the arm.
import json, tempfile, pathlib
import so_arena as soa
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.mechanisms import Consultancy
from so_arena.release import release

items = [soa.binary_item(f"q{i}", "?", correct="x", incorrect="y", shuffle_seed=i) for i in range(4)]
mix = soa.MixturePolicy([soa.FixedPolicy("I argue my side.", id="honest_arm"),
                         soa.FixedPolicy("I argue my side.", id="deceptive_arm")])
judge = soa.ScriptedPolicy('{"A": 0.5, "B": 0.5}', label="judge")
prof = Profile(name="p", players={"consultant": PlayerSpec(policy=mix, stance="true"), "judge": judge})
eps = run_sync(run_episodes(Consultancy(rounds=1, judge_questions=False), items, [prof], repeats=2))
out = pathlib.Path(tempfile.mkdtemp()) / "rel"
release(eps, items, out)
for line in (out / "episodes.jsonl").read_text().splitlines():
    e = json.loads(line)
    comps = {t["metadata"].get("mixture_component") for t in e["turns"] if t["role"] == "consultant"}
    print(e["item_id"], "label:", e["players"]["consultant"]["label"], "| turn metadata:", comps)
