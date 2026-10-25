"""eval_claim 'after_best' positions negate the mover's mate count instead of shifting it by one move."""
import chess
from so_arena.domains.chess import ChessDomain
def mated_in_one(b):  # side to move gets mated after one of its moves, whatever it plays
    for mv in b.legal_moves:
        b.push(mv)
        ok = any((b.push(r), b.is_checkmate(), b.pop())[1] for r in list(b.legal_moves))
        b.pop()
        if not ok: return False
    return True
n = 0
for it in ChessDomain("eval_claim").load():
    d = it.ground_truth.data
    if d["position_source"] == "after_best" and d["mate"] == -2:
        b = chess.Board(it.context["fen"])
        n += 1
        if n <= 3:
            print(it.id, "stored mate:", d["mate"], "| side to move is actually mated in 1:", mated_in_one(b),
                  "| expert note:", it.private["engine_notes"][:90])
print("after_best items with stored mate == -2:", n)
