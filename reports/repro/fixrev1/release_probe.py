# Look for any per-arm signal in a names="auto" release of a Stances() experiment.
import json, tempfile, collections, oversight_arena as oa
from types import SimpleNamespace as NS
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.mechanisms import Debate, Consultancy
from oversight_arena.sim import BayesianBitJudge, BitAdvocate
from oversight_arena.release import create_release
from oversight_arena.release.release import _item
dom = HiddenBits(n_tasks=6); tasks = {t.id: t for t in dom.tasks()}
for mech in [Debate(rounds=1), Consultancy(rounds=1)]:
    res = oa.Experiment(dom, [mech], {"kind:judge": BayesianBitJudge(trust=0.8), "*": BitAdvocate(lie_rate=0.5)}, oa.Stances(), progress=False).run()
    d = tempfile.mkdtemp(); create_release(res, d)
    items = json.load(open(d + "/items.json"))
    print(type(mech).__name__, "items", len(items), "keys", sorted(items[0].keys()))
    txt = json.dumps(items).lower()
    for w in ["incorrect", "honest", "argue_", "stance", "dishonest", "lie"]:
        print(f"   '{w}' occurrences:", txt.count(w))
    # does any field (other than rewards/decision) differ systematically between correct/incorrect arms?
    by = collections.defaultdict(collections.Counter)
    for it in items:
        t = tasks[it["task"]]
        for role, pos in (it.get("positions") or {}).items():
            arm = pos in t.correct_ids()
            by[(role, "agents")][(json.dumps(it["agents"].get(role)), arm)] += 1
            by[(role, "n_entries")][(len(it.get("transcript") or []), arm)] += 1
    for k, c in by.items(): print("  ", k, dict(c))
    sys_msgs = {e["content"][:120] for it in items for e in (it.get("transcript") or []) if e["role"] is None}
    print("   system entries:", list(sys_msgs)[:4])
# adapted release_tags: tags present but no stance
from oversight_arena.core.strategy import Strategy
from oversight_arena.core.episode import BoundInfo
s1, s2 = Strategy(name="sandbag_forecaster", tags={"honest": False}), Strategy(name="honest_forecaster", tags={"honest": True})
b = {"forecaster_1": BoundInfo(agent="m", strategy_id=s1.id, strategy_name=s1.name, tags=s1.tags),
     "forecaster_2": BoundInfo(agent="m", strategy_id=s2.id, strategy_name=s2.name, tags=s2.tags)}
from oversight_arena.release.release import _names_safe
rec = NS(outcome={"forecasts": {"forecaster_1": 0.2, "forecaster_2": 0.7}}, bound=b, id="e1", task_id="q1",
         mechanism="Forecast", mechanism_hash="h", reward_rule="r", profile=NS(label="honest vs sandbagger", id="p-123"),
         rewards={}, meta={}, created_at="now", transcript=NS(entries=[]))
it = _item(rec, transcripts=False, show=_names_safe(rec))
print("release_tags adapted: names_safe", _names_safe(rec), "profile:", it["profile"], "strategies:", it["strategies"])
