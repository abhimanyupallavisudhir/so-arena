"""Engine-backed chess agents: no LLM, real capability gap.

- :class:`EngineAdvocate` (expert): argues for its assigned move with verifiable ``<line>``
  claims. ``style="honest"`` shows principal variations (best play by both sides) and, from
  the second turn, the true refutation of the opposing move; ``style="cherry_pick"`` shows
  lines in which the *other* side plays a plausible-but-poor move, making its own move look
  good and the opposing move look bad.
- :class:`EngineJudge` (weak judge): builds the tree of *revealed, verified* lines and runs
  minimax over it, evaluating leaves (and, with ``self_check``, internal nodes) with its own
  shallow search (``depth``) or pure material (``depth=0``). With no lines it just searches
  shallowly from the root — the NaiveJudge baseline.

This is Irving et al.'s debate-as-revealed-minimax in a domain where the capability gap is a
measurable quantity (search depth / Elo).
"""

from __future__ import annotations

import math
from typing import Any

from ..agents.base import Action, Agent, Observation
from ..core.util import rng_for
from ..domains.chess import MATE, Engine, apply_line, judge_leaf_value


def _opt_san(obs: Observation, oid: str) -> str:
    o = next(o for o in obs.task.options if o.id == oid)
    return o.data.get("san") or o.text


class EngineAdvocate(Agent):
    def __init__(self, depth: int = 12, line_len: int = 6, top_k: int = 4, engine_path: str | None = None, id: str | None = None):
        self.depth = depth
        self.line_len = line_len
        self.top_k = top_k
        self.engine_path = engine_path
        self.id = id or f"engine_advocate(d={depth})"
        self._cache: dict[tuple, list[str]] = {}

    def describe(self) -> dict[str, Any]:
        return {"type": "EngineAdvocate", "depth": self.depth, "line_len": self.line_len, "top_k": self.top_k}

    def _eng(self) -> Engine:
        return Engine.get(self.engine_path, "expert")

    def _pv(self, board: Any, n: int, depth: int) -> list:
        if n <= 0 or board.is_game_over():
            return []
        info = self._eng().analyse(board, depth)[0]
        return list(info.get("pv", []))[:n]

    def _line(self, fen: str, first: str, depth: int, bend_ply: int | None, maximize_for: Any) -> list[str]:
        """Line starting with move ``first``. If ``bend_ply`` is set, at that ply choose — among
        the top-k moves — the one best for ``maximize_for`` (a cherry-picked deviation)."""
        key = (fen, first, depth, bend_ply, bool(maximize_for), self.line_len, self.top_k)
        if key not in self._cache:
            self._cache[key] = self._line_uncached(fen, first, depth, bend_ply, maximize_for)
        return list(self._cache[key])

    def _line_uncached(self, fen: str, first: str, depth: int, bend_ply: int | None, maximize_for: Any) -> list[str]:
        board, applied, err = apply_line(fen, first)
        if err:
            return [first]
        line = list(applied)
        while len(line) < self.line_len and not board.is_game_over():
            ply = len(line) + 1
            if bend_ply is not None and ply == bend_ply:
                infos = self._eng().analyse(board, max(2, depth // 2), multipv=self.top_k)
                cands = [(i["score"].pov(maximize_for).score(mate_score=MATE), i["pv"][0]) for i in infos if i.get("pv")]
                if not cands:
                    break
                mv = max(cands, key=lambda x: x[0])[1]
                line.append(board.san(mv))
                board.push(mv)
                continue
            pv = self._pv(board, 1, depth)
            if not pv:
                break
            line.append(board.san(pv[0]))
            board.push(pv[0])
        return line

    async def act(self, obs: Observation) -> Action:
        import asyncio

        import chess

        p = obs.params
        style = p.get("style", "honest")
        depth = int(p.get("depth", self.depth))
        fen = obs.task.data["fen"]
        mover = chess.Board(fen).turn
        target = obs.target or obs.task.option_ids[0]
        mine = _opt_san(obs, target)
        others = [_opt_san(obs, o) for o in obs.task.option_ids if o != target]
        n_prev = sum(1 for e in obs.entries if e.role == obs.role)
        opponent_side = not mover
        claims = []
        if n_prev == 0:
            # support line for my move: honest PV, or the opponent's reply cherry-picked to be poor
            bend = 2 if style == "cherry_pick" else None
            line = await asyncio.to_thread(self._line, fen, mine, depth, bend, mover)
            claims.append(("my move", line))
        else:
            for other in others:
                # attack line for the opposing move: honest refutation (PV), or the mover's
                # follow-up (ply 3) cherry-picked to be poor
                bend = 3 if style == "cherry_pick" else None
                line = await asyncio.to_thread(self._line, fen, other, depth, bend, opponent_side)
                claims.append((f"against {other}", line))
        body = " ".join(f"Consider {what}: <line>{' '.join(l)}</line>." for what, l in claims)
        return Action(text=f"The better move is {mine}. {body}", parsed={"choice": target} if obs.response.kind == "choice" else {})


class EngineJudge(Agent):
    def __init__(self, depth: int = 1, temperature_cp: float = 150.0, self_check: bool = False,
                 use_verifier_evals: bool = False, engine_path: str | None = None, id: str | None = None):
        self.depth = depth
        self.temperature_cp = temperature_cp
        self.self_check = self_check
        self.use_verifier_evals = use_verifier_evals
        self.engine_path = engine_path
        self.id = id or f"engine_judge(d={depth})"
        self._leaf_cache: dict[str, float] = {}

    def describe(self) -> dict[str, Any]:
        return {"type": "EngineJudge", "depth": self.depth, "T": self.temperature_cp, "self_check": self.self_check,
                "use_verifier_evals": self.use_verifier_evals}

    def _leaf(self, board: Any, mover: Any, cache: dict) -> float:
        key = f"{board.fen()}|{int(bool(mover))}"
        if key in self._leaf_cache and key not in cache:
            cache[key] = self._leaf_cache[key]
        if key not in cache:
            cache[key] = judge_leaf_value(board, mover, self.depth, self.engine_path)
            self._leaf_cache[key] = cache[key]
        return cache[key]

    def evaluate(self, obs: Observation) -> dict[str, float]:
        import chess

        fen = obs.task.data["fen"]
        root = chess.Board(fen)
        mover = root.turn
        trie: dict = {}
        verifier_eval: dict[str, float] = {}
        for e in obs.entries:
            for ev in e.evidence:
                if ev.kind == "line" and ev.verified and ev.data.get("legal"):
                    node = trie
                    for mv in ev.data.get("line", []):
                        node = node.setdefault(mv, {})
                    if "eval_cp" in ev.data and "fen" in ev.data:
                        verifier_eval[ev.data["fen"]] = ev.data["eval_cp"]
        cache: dict = {}
        if self.use_verifier_evals:
            cache.update({f"{k}|{int(bool(mover))}": v for k, v in verifier_eval.items()})

        def value(board: Any, node: dict) -> float:
            if not node:
                return self._leaf(board, mover, cache)
            vals = []
            for san, child in node.items():
                b2 = board.copy()
                try:
                    b2.push_san(san)
                except Exception:
                    continue
                vals.append(value(b2, child))
            if self.self_check:
                vals.append(self._leaf(board, mover, cache))
            if not vals:
                return self._leaf(board, mover, cache)
            return max(vals) if board.turn == mover else min(vals)

        out = {}
        for o in obs.task.options:
            san = o.data.get("san") or o.text
            b = root.copy()
            b.push_san(san)
            out[o.id] = value(b, trie.get(san, {}))
        return out

    async def act(self, obs: Observation) -> Action:
        import asyncio

        vals = await asyncio.to_thread(self.evaluate, obs)
        m = max(vals.values())
        w = {k: math.exp((v - m) / self.temperature_cp) for k, v in vals.items()}
        s = sum(w.values())
        probs = {k: max(min(v / s, 1 - 1e-4), 1e-4) for k, v in w.items()}
        z = sum(probs.values())
        probs = {k: v / z for k, v in probs.items()}
        text = "; ".join(f"{k}: {v / 100:+.2f}" for k, v in vals.items())
        if obs.response.kind == "distribution":
            return Action(text=text, parsed={"probs": probs, "choice": max(probs, key=probs.get)})
        if obs.response.kind == "choice":
            return Action(text=text, parsed={"choice": max(probs, key=probs.get)})
        return Action(text="Which line refutes the other move?")
