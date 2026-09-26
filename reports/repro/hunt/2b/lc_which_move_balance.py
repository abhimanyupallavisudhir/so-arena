"""which_move: A/B balance and surface tells (check/capture) of the correct move."""
import chess, collections
from so_arena.domains.chess import ChessDomain
items = ChessDomain("which_move").load()
print("n", len(items), "P(correct=A) =", sum(i.ground_truth.correct == "A" for i in items) / len(items))
c = collections.Counter()
for it in items:
    b = chess.Board(it.context["fen"]); cands = it.context["candidates"]
    forcing = {k: ("+" in v["san"] or "#" in v["san"]) for k, v in cands.items()}
    g = it.ground_truth.correct; o = "B" if g == "A" else "A"
    c[(forcing[g], forcing[o])] += 1
print("(best is check, alternative is check):", dict(c))
