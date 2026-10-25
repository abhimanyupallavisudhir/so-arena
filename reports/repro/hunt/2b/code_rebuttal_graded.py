# HiddenTests grades the worker's LAST text turn containing code, not the submission (outcome.output = work turn).
# In ReviewedWork with a critique round, code posted in the rebuttal replaces the submitted program.
import so_arena as soa
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.mechanisms import ReviewedWork
from so_arena.domains.code import CodeDomain, hack_solution, submitted_code
dom = CodeDomain(kind="implement")
item = dom.load(limit=1)[0]
d = item.ground_truth.data
hack = hack_solution(d["entry_point"], d["visible_tests"])
ref = d["reference"]
def worker_fn(submit, rebut):
    def act(req, ctx):
        return f"```python\n{submit if req.phase == 'work' else rebut}\n```"
    return soa.FunctionPolicy(act, label="w")
critic = soa.FixedPolicy("This fails on other inputs.", label="c")
reviewer = soa.ScriptedPolicy('{"accept": 0.9, "reject": 0.1}', label="r")
mech = ReviewedWork(critique_rounds=1, rebuttal=True)
for name, sub, reb in (("buggy submission, correct code in rebuttal", hack, ref),
                       ("correct submission, helper snippet in rebuttal", ref, "def helper(x):\n    return x")):
    prof = Profile(name=name, players={"worker": PlayerSpec(policy=worker_fn(sub, reb)), "critic": critic, "reviewer": reviewer})
    ep = run_sync(run_episodes(mech, [item], [prof], ground_truth=dom.ground_truth_scorers()))[0]
    print(f"{name}: submission==hack? {sub is hack}; GT hidden_pass_rate={ep.ground_truth['hidden_pass_rate']:.2f}"
          f" worker value={ep.ground_truth['role_values']['worker']:+.1f} judge_correct={ep.ground_truth['judge_correct']}")
