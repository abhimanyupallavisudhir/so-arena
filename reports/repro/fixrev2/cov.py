import sys; sys.path.insert(0,"/home/user/karmax/so-arena/reports/repro")
from so_arena.analysis.optimization import GameTree, TreeNode, TreeLeaf, evaluate_tree, BestOfN
leaves = {f"l{i}": TreeLeaf(id=f"l{i}", rewards={"p": r}, values=({} if v is None else {"correct": v}))
          for i, (r, v) in enumerate([(1.0, None), (0.2, 1.0), (0.1, 1.0), (0.0, 0.0)])}
t = GameTree(item_id="x", mechanism="m", root="n", nodes={"n": TreeNode(id="n", key="k", role="p", children=list(leaves))}, leaves=leaves)
tv = evaluate_tree(t, {"p": BestOfN(64)})
print(tv)
