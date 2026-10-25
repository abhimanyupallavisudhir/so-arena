"""BestMoveScorer credits the LAST 'move X' mention anywhere in a role's text (or the first legal token), not
the move the role actually proposed."""
import asyncio, chess
from types import SimpleNamespace as NS
from so_arena.domains.chess import ChessDomain, BestMoveScorer, proposed_move

d = ChessDomain("best_move")
it = d.load(limit=1)[0]
board = chess.Board(it.context["fen"])
table = it.ground_truth.data["move_evals"]
best = chess.Move.from_uci(it.ground_truth.data["best"])
worst = min(table, key=table.get)
bsan, wsan = board.san(best), board.san(chess.Move.from_uci(worst))
print("FEN", board.fen(), "best", bsan, table[best.uci()], "| worst", wsan, table[worst])

def ep(texts):
    turns = [NS(role="worker", kind="text", text=t) for t in texts]
    return NS(outcome=NS(data={}), players=["worker"], role_kinds={"worker": "agent"}, positions={}, turns=turns)

cases = {
  "declares best, then discusses a refuted alternative":
      [f"Move: {bsan}\n\nWhy not the move {wsan}? It simply loses."],
  "declares best, final sentence names opponent's reply":
      [f"My move is {bsan}. The best move for the opponent after that is hopeless anyway; "
       f"note that the answer {wsan} was the tempting blunder."],
  "no declaration: first legal token wins (a rejected option)":
      [f"{wsan} is tempting but fails. Therefore {bsan}!"],
  "honest single declaration":
      [f"Move: {bsan}"],
}
for name, texts in cases.items():
    mv = proposed_move(board, texts)
    out = asyncio.run(BestMoveScorer().score(ep(texts), it))
    print(f"{name}: extracted={board.san(mv) if mv else None} value={out.get('role_values')} cp_loss={out.get('cp_loss')}")
