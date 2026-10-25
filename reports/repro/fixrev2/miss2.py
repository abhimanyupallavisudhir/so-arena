from so_arena.analysis.optimization import GameTree, TreeNode, TreeLeaf, evaluate_tree, optimization_grid, BestOfN
leaves = {f"l{i}": TreeLeaf(id=f"l{i}", rewards={"p": r}, values=({} if v is None else {"correct": v}))
          for i, (r, v) in enumerate([(1.0, None), (0.2, 1.0), (0.1, 1.0), (0.0, 0.0)])}
t = GameTree(item_id="x", mechanism="m", root="n", nodes={"n": TreeNode(id="n", key="k", role="p", children=list(leaves))}, leaves=leaves)
for n in (1, 2, 4, 8, 64):
    tv = evaluate_tree(t, {"p": BestOfN(n)})
    print(n, tv.values, getattr(tv,'coverage',None), [f for f in type(tv).model_fields] if hasattr(type(tv),'model_fields') else vars(tv).keys())
print(optimization_grid([t], {"p": [1, 2, 4, 8, 64]}).to_string())
