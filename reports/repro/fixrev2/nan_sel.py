import math, warnings; warnings.filterwarnings("ignore")
from so_arena.analysis.optimization import GameTree, TreeNode, TreeLeaf, evaluate_tree, BestOfN, Tilted
for rs in ([math.nan, 0.5, 0.2], [0.5, math.nan, 0.2], [math.inf, 0.5, 0.2], [-math.inf, 0.5, 0.2]):
    leaves = {f"l{i}": TreeLeaf(id=f"l{i}", rewards={"p": r}, values={"v": float(i)}) for i, r in enumerate(rs)}
    t = GameTree(item_id="x", mechanism="m", root="n", nodes={"n": TreeNode(id="n", key="k", role="p", children=list(leaves))}, leaves=leaves)
    for sel in (BestOfN(3), Tilted(float("inf")), Tilted(2.0)):
        try:
            tv = evaluate_tree(t, {"p": sel}); print(rs, sel, "reward", round(tv.rewards["p"], 3), "v", tv.values, "rcov", tv.reward_coverage)
        except Exception as e:
            print(rs, sel, "ERR", repr(e)[:100])
