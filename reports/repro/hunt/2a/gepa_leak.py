import logging, re, random; logging.disable(logging.WARNING)
import numpy as np
from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
from so_arena.integrations.gepa import gepa_search
from so_arena.mechanisms import Propaganda
from so_arena.models import FunctionModel
from so_arena.samplers.prompt_search import PromptSearch

# every strategy is the SAME behaviour (the factory ignores the text): any "improvement" is noise
factory = lambda s: synthetic_arguer(honest_mean=1.0, dishonest_mean=0.0, sd=1.0, label="same")
rng = random.Random(0)
opt = FunctionModel(lambda m, o, i: f"```\nVariant {rng.randint(0, 10**9)}.\n```", name="mock")
dom = SyntheticPersuasion(n_items=60, seed=4); items = dom.load()
search = PromptSearch(Propaganda(), items[:8], role="agent", policy_factory=factory,
                      others={"judge": synthetic_judge()}, optimizer=opt, arms=["false"], ctx=dom.context())
res, rec = gepa_search(search, max_metric_calls=120, reflection_minibatch_size=2)   # no eval_items
tr = set(rec[rec.split == "train"].item_id); va = set(rec[rec.split == "val"].item_id)
print("val items == train items:", va == tr, len(va))
best = res.best_candidate["strategy"]
print("seed val score %.3f   best-by-val val score %.3f" % (res.val_aggregate_scores[0], res.val_aggregate_scores[res.best_idx]))
fresh = PromptSearch(Propaganda(), items[20:60], role="agent", policy_factory=factory,
                     others={"judge": synthetic_judge()}, optimizer=opt, arms=["false"], ctx=dom.context())
for s in ["Play your role well.", best]:
    r, _, _ = fresh.run_sync(fresh.evaluate_strategy(s)) if False else __import__("so_arena.core.runner", fromlist=["run_sync"]).run_sync(fresh.evaluate_strategy(s))
    print("fresh items, strategy %-24r mean reward %.3f" % (s[:24], np.mean([x for x in r if x is not None])))
