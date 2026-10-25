"""Does the bundled move table cover every legal move? Moves missing from it are silently unscored."""
import chess, json
from so_arena.datasets import read_jsonl, sample_path
from so_arena.domains.chess import ChessDomain, puzzle_position
miss = 0; tot = 0; ex = None
for r in read_jsonl(sample_path("chess_puzzles_sample.jsonl")):
    b, _, _ = puzzle_position(r)
    legal = {m.uci() for m in b.legal_moves}
    t = r["analysis"]["move_evals"]
    m = legal - set(t)
    tot += 1; miss += bool(m)
    if m and ex is None: ex = (r["puzzle_id"], sorted(m))
print(f"puzzles with legal moves missing from move_evals: {miss}/{tot}; example {ex}")
items = ChessDomain("best_move").load()
print("best_move items:", len(items))
