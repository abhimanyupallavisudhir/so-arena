import asyncio
import sqlite3
from dataclasses import replace

import pytest

from oversight_arena import Action, Claim, Outcome, Reward, Role, Task, run
from oversight_arena.demo import reporting_game
from oversight_arena.domains import SQLiteSnapshot, SQLScorer, chess_legality
from oversight_arena.games import EmpiricalGame
from oversight_arena.resources import Price, tree_calls
from oversight_arena.sampling import Strategy, sample_strategies
from oversight_arena.training import reinforce
from oversight_arena.verification import VerifierRegistry


def test_swarm_equilibria_depend_on_bonus_and_beliefs():
    game = reporting_game(0.3)
    assert set(game.pure_equilibria()) == {("silent", "silent"), ("report", "report")}
    assert reporting_game(1.2).pure_equilibria() == [("report", "report")]
    assert game.regrets([[1, 0], [1, 0]]) == {"alice": 0, "bob": 0}
    assert game.regrets([[0, 1], [1, 0]])["alice"] == 0.7
    cycle = game.best_response_dynamics(("silent", "report"), 10)
    assert cycle["cycle_start"] == 0 and not cycle["equilibrium"]
    assert game.best_response_dynamics(("silent", "report"), 10, "alternating")["equilibrium"]


def test_coalitions_and_correlated_distributions():
    game = EmpiricalGame(
        ("a", "b"),
        (("D", "C"),) * 2,
        {("D", "D"): (1, 1), ("C", "D"): (0, 3), ("D", "C"): (3, 0), ("C", "C"): (2, 2)},
    )
    assert game.pure_equilibria() == [("D", "D")]
    assert game.coalition_gain(("D", "D"), ("a", "b")) == 1
    assert game.coarse_correlated_regret({("D", "D"): 1}) == {"a": 0, "b": 0}
    assert game.coarse_correlated_regret({("C", "C"): 1}) == {"a": 1, "b": 1}


def test_actual_rl_increases_optimal_action_probability_and_freezes_fixture():
    game = EmpiricalGame(
        ("agent", "fixture"),
        (("bad", "good"), ("x", "y")),
        {(a, b): (float(a == "good"), 1.0) for a in ("bad", "good") for b in ("x", "y")},
    )
    history = reinforce(game, steps=300, trainable=["agent"], seed=4)
    assert history[0].policies[0][1] == 0.5
    assert history[-1].policies[0][1] > 0.97
    assert history[-1].policies[1] == (0.5, 0.5)
    assert history == reinforce(game, steps=300, trainable=["agent"], seed=4)


def sql_run(query):
    async def policy(obs):
        return Action(data={"query": query})

    async def mechanism(ctx):
        await ctx.act("a")
        return Outcome({"a": Reward(0.8)})

    return asyncio.run(
        run(Task("t", "Count rows"), (Role("a"),), {"a": policy}, mechanism, name="sql")
    )


def test_sql_private_execution_checks_bags_and_rejects_writes():
    db = SQLiteSnapshot("CREATE TABLE x(v); INSERT INTO x VALUES (1),(1),(2);")
    assert db.query("SELECT SUM(v) FROM x") == [[4]]
    for query in (
        "DELETE FROM x",
        "ATTACH DATABASE '/tmp/oops.db' AS evil",
        "PRAGMA table_info(x)",
    ):
        with pytest.raises(sqlite3.DatabaseError):
            db.query(query)
    scorer = SQLScorer({"t": (db,)}, {"t": "SELECT v FROM x"}, ("a",))
    assert scorer(sql_run("SELECT DISTINCT v FROM x")).scores["a"]["quality"].value == 0
    assert scorer(sql_run("SELECT v FROM x ORDER BY v DESC")).scores["a"]["quality"].value == 1
    assert scorer(sql_run("bad sql")).scores["a"]["quality"].value == 0
    bad_oracle = replace(scorer, reference_queries={"t": "bad sql"})
    assert bad_oracle(sql_run("SELECT v FROM x")).scores["a"]["quality"].status == "error"
    with pytest.raises(ValueError, match="row budget"):
        replace(db, max_rows=1).query("SELECT v FROM x")
    with pytest.raises(sqlite3.DataError):
        replace(db, max_bytes=1000).query("SELECT zeroblob(1000000)")
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        db.query(
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT sum(x) FROM c"
        )


def test_claim_is_bound_to_artifact_and_scope_comes_from_verifier():
    db = SQLiteSnapshot("CREATE TABLE x(v); INSERT INTO x VALUES(1);")
    registry = VerifierRegistry()
    registry.register("sql", "1", "Query output, not semantic correctness", db.check_claim)
    claim = Claim(
        "This answers the user's question",
        {"query": "SELECT count(*) FROM x", "rows": [[1]]},
        "sql",
        "all claims are true",
    )
    result = asyncio.run(registry.verify(claim))
    assert result.status == "verified" and result.scope != claim.scope
    changed = replace(claim, artifact={"query": "SELECT count(*) FROM x", "rows": [[100]]})
    assert changed.id != claim.id
    assert asyncio.run(registry.verify(changed)).status == "refuted"
    assert asyncio.run(registry.verify(replace(claim, verifier="unknown"))).status == "unknown"


def test_chess_legality_does_not_claim_strength():
    chess = pytest.importorskip("chess")
    claim = Claim("White wins", {"fen": chess.STARTING_FEN, "moves": ["e2e4", "e7e5"]}, "legal")
    valid, detail = asyncio.run(chess_legality(claim))
    assert valid and "does not establish" in detail
    assert not asyncio.run(
        chess_legality(replace(claim, artifact={"fen": chess.STARTING_FEN, "moves": ["e2e5"]}))
    )[0]


def test_cost_and_balanced_strata():
    p = Price(2, 10, 0.5, "user supplied", "2026-09-25")
    assert p.cost(input_tokens=1000, output_tokens=100, cached_tokens=400) == pytest.approx(0.0024)
    assert tree_calls([50, 50, 20]) == 52_550
    sampled = sample_strategies(
        lambda: None,
        [
            Strategy("a", "truth", "honest"),
            Strategy("b", "lie", "deceptive"),
            Strategy("c", "evade", "deceptive"),
        ],
        count=10,
        balanced=True,
    )
    assert sum(s.strategy.stratum == "honest" for s in sampled) == 5


def test_chess_engine_scores_initial_players_perspective(monkeypatch):
    pytest.importorskip("chess")
    import chess.engine

    from oversight_arena.domains import ChessScorer
    from oversight_arena.types import Event

    class Engine:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def analyse(self, board, limit, root_moves=None):
            assert limit.nodes == 1234
            return {
                "score": chess.engine.PovScore(
                    chess.engine.Cp(40 if root_moves else 80), board.turn
                )
            }

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **k: Engine())
    record = replace(
        sql_run("SELECT 1"),
        task=Task("t", "Choose a move", {"fen": chess.STARTING_FEN}),
        events=(Event(0, "a", "action", {"data": {"move": "e2e4"}}),),
    )
    score = ChessScorer("fixture-engine", "fixture-v1", ("a",), nodes=1234)(record)
    assert score.scores["a"]["quality"].value == -40
    assert score.scores["a"]["quality"].kind == "proxy"


def test_human_packet_blinds_reward_and_retains_provenance():
    from oversight_arena.human import import_rating, review_packet

    record = sql_run("SELECT 1")
    packet = review_packet(record, "a", "Does the query answer the request?")
    assert "reward" not in packet and "condition" not in packet
    rating = import_rating(packet, value=0.3, rater="rater-17", seconds=42)
    assert rating.scores["a"]["quality"].kind == "human"
    assert "review_seconds=42" in rating.scores["a"]["quality"].detail
