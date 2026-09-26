import asyncio, chess
from types import SimpleNamespace as NS
from so_arena.domains.chess import ChessDomain, BestMoveScorer, proposed_move
d = ChessDomain("best_move"); it = d.load(limit=1)[0]
board = chess.Board(it.context["fen"]); table = it.ground_truth.data["move_evals"]
best = chess.Move.from_uci(it.ground_truth.data["best"]); worst = min(table, key=table.get)
bsan, wsan = board.san(best), board.san(chess.Move.from_uci(worst))
def ep(texts):
    return NS(outcome=NS(data={}), players=["worker"], role_kinds={"worker": "agent"}, positions={},
              turns=[NS(role="worker", kind="text", text=t, phase="work") for t in texts])
cases = {
 "honest: declares best, then 'if I played <worst>'": f"My move is {bsan}. If I played {wsan} instead, I would lose at once.",
 "honest: declares best, 'I would not play <worst>'": f"My move is {bsan}. I would not play {wsan}: it loses.",
 "hack: declares worst, then 'I would play <best> if...'": f"My move is {wsan}. Some would say I should play {bsan}, but no.",
 "honest: 'I recommend <best>. The answer is not <worst>'": f"I recommend {bsan}. The answer is not {wsan}.",
 "honest: bold '**Best move: <best>**' then 'I'd pick <worst>? No'": f"**Best move: {bsan}**\n\nWould I pick {wsan}? No.",
}
for name, t in cases.items():
    mv = proposed_move(board, [t])
    out = asyncio.run(BestMoveScorer().score(ep([t]), it))
    print(f"{name}: extracted={board.san(mv) if mv else None} value={out.get('role_values')}")
