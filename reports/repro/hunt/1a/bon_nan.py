import numpy as np
from oversight_arena.analysis.bon import bon_weights, tree_value, tree_mesh, Node
print("numpy", np.__version__)
# one sample in the pool has a missing (NaN) reward
s = [0.2, 0.9, np.nan]
for n in (1, 4, 64):
    print("n=",n,"max weights", bon_weights(s, n).round(3), " min weights", bon_weights(s, n, maximize=False).round(3))
# tree: proposer picks between a good proposal (payoff .9, gt 1) and a bad one whose one leaf lacks a payoff
good = Node(children=[Node(payoff=0.9, gt=1.0), Node(payoff=0.8, gt=1.0)])
bad = Node(children=[Node(payoff=0.1, gt=0.0), Node(payoff=None, gt=0.0)])
root = Node(children=[good, bad])
for ks in ([64, 64], [64, 1]):
    print("ks", ks, "tree_value ->", tree_value(root, ks, [True, False]))
print(tree_mesh([root, Node(children=[Node(payoff=.5, gt=1.0)])], [[64, 64]], [True, False])[["payoff","gt","n"]])
