"""General-sum conditional self-play; replace leaves with measured rollout payoffs."""

from oversight_arena.optimization import Decision, Leaf, nested_best_of_n

# Each proposal has its own conditional critique pool. Quality is independent of reward.
tree = Decision(
    "proposer",
    (
        Decision(
            "critic",
            (
                Leaf({"proposer": 0.8, "critic": 0.2}, {"proposer": 1.0}),
                Leaf({"proposer": 0.6, "critic": 0.7}, {"proposer": 1.0}),
            ),
        ),
        Decision(
            "critic",
            (
                Leaf({"proposer": 0.9, "critic": 0.1}, {"proposer": 0.0}),
                Leaf({"proposer": 0.1, "critic": 0.9}, {"proposer": 0.0}),
            ),
        ),
    ),
)
for n in (1, 2, 4, 8):
    for m in (1, 2, 4, 8):
        result = nested_best_of_n(tree, {"proposer": n, "critic": m})
        print(n, m, result.rewards["proposer"], result.qualities["proposer"])
