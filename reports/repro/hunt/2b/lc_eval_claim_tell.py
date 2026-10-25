"""eval_claim: truth is predictable from surface features of the shown position, because the coin that
balances answers picks the *position source*, which correlates with check / last-move type."""
import chess, collections
from so_arena.domains.chess import ChessDomain
items = ChessDomain("eval_claim").load()
c = collections.Counter(); src = collections.Counter()
for it in items:
    b = chess.Board(it.context["fen"])
    truth = it.ground_truth.correct
    src[(it.ground_truth.data["position_source"], truth)] += 1
    feat = b.is_check() or "x" in it.context["last_move"]
    c[(feat, truth)] += 1
print("n =", len(items), "yes-rate =", sum(i.ground_truth.correct == "yes" for i in items) / len(items))
print("(source, truth):", dict(src))
print("(in check or last move a capture, truth):", dict(c))
pred = lambda it: "no" if (chess.Board(it.context["fen"]).is_check() or "x" in it.context["last_move"]) else "yes"
acc = sum(pred(it) == it.ground_truth.correct for it in items) / len(items)
print(f"blind rule 'check or capture just played -> no' accuracy: {acc:.3f}")

V = {chess.KING: 0, chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
def mat(b, col): return sum(V[p.piece_type] for p in b.piece_map().values() if p.color == col)
def pred2(it):
    b = chess.Board(it.context["fen"])
    return "yes" if mat(b, b.turn) - mat(b, not b.turn) >= 0 and not b.is_check() else "no"
acc2 = sum(pred2(it) == it.ground_truth.correct for it in items) / len(items)
print(f"blind rule 'mover not behind in material and not in check -> yes' accuracy: {acc2:.3f}")
