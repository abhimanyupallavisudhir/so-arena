from so_arena.analysis.optimization import GameTree, TreeNode, TreeLeaf, evaluate_tree, optimization_grid, BestOfN
# proposer pool of 4: the high-reward candidate has no truth label (e.g. pending/errored GT scorer)
leaves = {f"l{i}": TreeLeaf(id=f"l{i}", rewards={"p": r}, values=({} if v is None else {"correct": v}))
          for i, (r, v) in enumerate([(1.0, None), (0.2, 1.0), (0.1, 1.0), (0.0, 0.0)])}
t = GameTree(item_id="x", mechanism="m", root="n", nodes={"n": TreeNode(id="n", key="k", role="p", children=list(leaves))}, leaves=leaves)
for n in (1, 4, 64):
    tv = evaluate_tree(t, {"p": BestOfN(n)})
    print(f"Bo{n}: reward={tv.rewards['p']:.3f} correct={tv.values.get('correct'):.3f}")
print(optimization_grid([t], {"p": [1, 64]}).columns.tolist())
