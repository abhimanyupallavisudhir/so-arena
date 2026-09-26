import csv
import io
import math
import re

import chess
import chess.engine
import pytest

import so_arena as soa
from so_arena.analysis.metrics import asd
from so_arena.core.runner import Profile, run_episodes, run_sync
from so_arena.domains import get_domain
from so_arena.domains.chess import (
    ChessDomain,
    ChessLineVerifier,
    analyse_puzzle,
    clip_cp,
    find_stockfish,
    play_line,
    proposed_move,
    read_puzzle_csv_zst,
    shared_engine,
)
from so_arena.mechanisms import Consultancy, Debate, ReviewedWork
from so_arena.samplers.arms import ASDExperiment

needs_engine = pytest.mark.skipif(find_stockfish() is None, reason="needs Stockfish (e.g. /usr/games/stockfish)")
NO_EVAL = re.compile(r"\bcp\b|eval|centipawn|engine|advantage|winning|mate in|[+-]\d+\.\d", re.I)


@pytest.fixture(scope="module")
def which_items():
    return ChessDomain().load()


def claim(kind, content, **attrs):
    return soa.Claim(kind=kind, content=content, attrs=attrs)


# ----------------------------------------------------------------------------- items (offline, bundled sample)

def test_which_move_items_structure(which_items):
    items = which_items
    assert len(items) >= 250 and len({it.id for it in items}) == len(items)
    ratings = [it.metadata["rating"] for it in items]
    assert min(ratings) < 1000 and max(ratings) >= 2600
    for it in items:
        assert it.domain == "chess" and it.labels == ["A", "B"]
        assert sorted(a.value for a in it.answers) == [-1.0, 1.0]
        gt = it.ground_truth
        assert it.true_label == gt.correct and it.value_of(gt.correct) == 1.0
        board = chess.Board(it.context["fen"])
        assert board.is_valid() and it.context["fen"] in it.question
        assert set(it.context) >= {"fen", "last_move", "candidates"}
        for lab, c in it.context["candidates"].items():
            mv = chess.Move.from_uci(c["uci"])
            assert mv in board.legal_moves and board.san(mv) == c["san"]
            assert f"{c['san']} ({c['uci']})" in it.question and it.answer(lab).text == f"{c['san']} ({c['uci']})"
        # A/B values agree with the stored engine evaluations (mover's perspective)
        cp = gt.data["cp"]
        assert clip_cp(cp[gt.correct]) - clip_cp(cp[gt.data["alternative"]]) >= 150
        assert it.context["candidates"][gt.correct]["uci"] == gt.data["solution"][0]
    share_a = sum(it.true_label == "A" for it in items) / len(items)
    assert 0.35 < share_a < 0.65  # deterministic shuffle of the two moves


def test_position_is_after_the_opponents_move():
    dom = ChessDomain()
    rec = dom.records()[0]
    before = chess.Board(rec["fen"])
    first = chess.Move.from_uci(rec["moves"][0])
    san = before.san(first)
    before.push(first)
    item = next(it for it in dom.load() if it.metadata["puzzle_id"] == rec["puzzle_id"])
    assert item.context["fen"] == before.fen()
    assert chess.Move.from_uci(rec["moves"][1]) in before.legal_moves
    assert san in item.context["last_move"] and f"has just played {item.context['last_move']}" in item.question


def test_censoring_and_private_engine_notes(which_items):
    it = which_items[0]
    c = it.censored()
    assert c.ground_truth is None and all(a.value is None for a in c.answers)
    assert "better move is" not in c.question and "better move is" in c.private["engine_notes"]
    assert "themes" not in c.metadata and "game_url" not in c.metadata


def test_load_is_deterministic_and_filters_rating():
    a = [it.id for it in ChessDomain().load(limit=20)]
    assert a == [it.id for it in ChessDomain().load(limit=20)]
    assert a != [it.id for it in ChessDomain(seed=1).load(limit=20)]
    labels = {it.id: it.true_label for it in ChessDomain().load()}
    assert labels == {it.id: it.true_label for it in ChessDomain(seed=1).load()}  # seed only reorders
    hard = ChessDomain(min_rating=2000, n_items=10).load()
    assert len(hard) == 10 and all(it.metadata["rating"] >= 2000 for it in hard)
    assert get_domain("chess", kind="eval_claim").kind == "eval_claim"
    assert set(ChessDomain().behaviours()) >= {"honest", "deceive"}


def test_eval_claim_and_best_move_items():
    ev = ChessDomain(kind="eval_claim").load()
    assert len(ev) > 100 and all(it.labels == ["yes", "no"] for it in ev)
    for it in ev:
        cp = clip_cp(it.ground_truth.data["cp"])
        assert cp >= 200 if it.true_label == "yes" else cp <= 50
        assert it.context["fen"] in it.question and "winning" in it.question
    assert sum(it.true_label == "yes" for it in ev) == sum(it.true_label == "no" for it in ev)  # balanced by the position choice
    bm = ChessDomain(kind="best_move").load(limit=5)
    assert len(bm) == 5
    for it in bm:
        assert it.answers is None and "Move:" in it.question
        table = it.ground_truth.data["move_evals"]
        assert set(table) == {m.uci() for m in chess.Board(it.context["fen"]).legal_moves}


def test_eval_claim_items_carry_no_blind_tell():
    """What shows without analysis - check, material, mobility, the last capture, the side to move - must not
    predict the answer: within every combination of those features, as many items say yes as no."""
    import collections

    from so_arena.domains.chess import balanced_eval_claims

    a, b = ChessDomain(kind="eval_claim").load(), ChessDomain(kind="eval_claim", seed=5).load()
    assert sorted((it.id, it.context["fen"]) for it in a) == sorted((it.id, it.context["fen"]) for it in b)
    assert len(a) >= 290  # the balance costs almost no puzzles
    strata = collections.defaultdict(collections.Counter)
    for it in a:
        f = it.ground_truth.data["blind_features"]
        strata[tuple(sorted(f.items()))][it.true_label] += 1
    assert all(c["yes"] == c["no"] for c in strata.values())
    # every position choice was a real candidate of its puzzle, with the answer its evaluation gives
    chosen = balanced_eval_claims(ChessDomain(kind="eval_claim").records())
    assert {f"chess-eval_claim-{p}" for p in chosen} == {it.id for it in a}
    # a lookup rule on any one or two of the features is at chance on held-out items
    rows = [(it.ground_truth.data["blind_features"], it.true_label) for it in a]
    for feats in (["in_check"], ["material"], ["few_moves"], ["in_check", "material"], ["few_moves", "after_capture"]):
        hits = 0.0
        for i, (f, y) in enumerate(rows):
            tab = collections.Counter(yy for j, (ff, yy) in enumerate(rows) if j != i and all(ff[k] == f[k] for k in feats))
            hits += 0.5 if tab["yes"] == tab["no"] else float(tab.most_common(1)[0][0] == y)
        assert hits / len(rows) < 0.56, feats


def test_read_truncated_zst_prefix(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    recs = ChessDomain().records()[:40]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "Popularity", "NbPlays", "Themes",
                "GameUrl", "OpeningTags", "DailyDate"])
    for r in recs:
        w.writerow([r["puzzle_id"], r["fen"], " ".join(r["moves"]), r["rating"], r["rating_deviation"],
                    r["popularity"], r["nb_plays"], " ".join(r["themes"]), r["game_url"], "", ""])
    text = buf.getvalue().encode()
    cobj = zstandard.ZstdCompressor().compressobj()  # many small blocks, like the real multi-MB file
    data = b"".join(cobj.compress(text[i:i + 500]) + cobj.flush(zstandard.COMPRESSOBJ_FLUSH_BLOCK)
                    for i in range(0, len(text), 500)) + cobj.flush()
    path = tmp_path / "prefix.csv.zst"
    path.write_bytes(data[: len(data) * 3 // 4])  # a prefix download ends mid-frame and mid-line
    rows = read_puzzle_csv_zst(path)
    assert 0 < len(rows) < len(recs)
    for got, want in zip(rows, recs):
        assert {k: got[k] for k in ("puzzle_id", "fen", "moves", "rating", "themes")} == \
               {k: want[k] for k in ("puzzle_id", "fen", "moves", "rating", "themes")}
    assert len(read_puzzle_csv_zst(path, n=3)) == 3


# ----------------------------------------------------------------------------- rules: parsing and chess_line

def test_line_parsing_variants():
    b = chess.Board()
    assert [m.uci() for m in play_line(b, "1. e4 e5 2.Nf3 Nc6 3. Bb5!? a6")[0]] == \
           ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
    assert play_line(b, "e2e4 e7e5 g1f3")[1].fen() == "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
    castle = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert [m.uci() for m in play_line(castle, "O-O 1... 0-0-0")[0]] == ["e1g1", "e8c8"]
    promo = chess.Board("8/4P3/8/8/8/8/k7/4K3 w - - 0 1")
    assert play_line(promo, "e8=Q+")[0][0].promotion == chess.QUEEN


async def test_chess_line_verifier_rules_only(which_items):
    v = ChessLineVerifier()
    it = which_items[0]
    sol = it.ground_truth.data["solution"]
    ok = await v.verify(claim("chess_line", " ".join(sol[:3])), it)
    board = chess.Board(it.context["fen"])
    for m in sol[:3]:
        board.push_uci(m)
    assert ok.status == "verified" and board.fen() in ok.output
    san_line = chess.Board(it.context["fen"]).variation_san([chess.Move.from_uci(m) for m in sol[:3]])
    assert (await v.verify(claim("chess_line", san_line), it)).status == "verified"
    uci = it.context["candidates"]["A"]["uci"]
    bad = await v.verify(claim("chess_line", f"{uci} {uci}"), it)
    assert bad.status == "refuted" and "move 2" in bad.output and uci in bad.output
    illegal = await v.verify(claim("chess_line", "1. e4 e5 2. Ke3", **{"from": "start"}), it)
    assert illegal.status == "refuted" and "Ke3" in illegal.output
    assert (await v.verify(claim("chess_line", "Nf3 is simply best", **{"from": "start"}), it)).status == "unchecked"
    spanish = await v.verify(claim("chess_line", "e4 e5 Nf3 Nc6 Bb5", **{"from": "start"}), it)
    assert "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3" in spanish.output
    mate = await v.verify(claim("chess_line", "f3 e5 g4 Qh4#", **{"from": "start"}), it)
    assert mate.status == "verified" and "Checkmate: White is mated" in mate.output
    for res in (ok, bad, illegal, spanish, mate):
        assert not NO_EVAL.search(res.output), res.output


def test_proposed_move_parsing():
    b = chess.Board()
    assert proposed_move(b, ["e4 looks natural but I prefer the knight. Move: Nf3"]) == chess.Move.from_uci("g1f3")
    assert proposed_move(b, ["<move>d2d4</move>"]) == chess.Move.from_uci("d2d4")
    assert proposed_move(b, ["The best move is **c4**."]) == chess.Move.from_uci("c2c4")
    assert proposed_move(b, ["I would play e4 here, then d4."]) == chess.Move.from_uci("e2e4")
    assert proposed_move(b, ["I resign."]) is None


# ----------------------------------------------------------------------------- mechanisms end to end (offline)

def scripted_expert(req, ctx):
    """Argues for its stance with a verified line; when its stance is the worse move (it can read the
    engine notes) it also fabricates an illegal continuation, which the verifier refutes."""
    view = req.view
    cand = view.item.context["candidates"][view.stance]
    better = re.search(r"The better move is \((\w)\)", view.item.private["engine_notes"]).group(1)
    text = f'{cand["san"]} is the better move. <claim kind="chess_line">{cand["san"]}</claim>'
    if view.stance != better:
        text += f' It keeps the initiative: <claim kind="chess_line">{cand["uci"]} {cand["uci"]}</claim>'
    return text


def claim_counting_judge(req, ctx):
    """Trusts verification markers: +1 per verified claim, -2 per failed claim of the side's advocates."""
    if req.kind == "text":
        return "Please justify your move with a concrete line."
    score = {o: 0.0 for o in req.options}
    for t in req.view.transcript:
        lab = req.view.positions.get(t.role)
        if lab in score:
            score[lab] += t.text.count("<verified") - 2 * t.text.count("<failed")
    z = {o: math.exp(s) for o, s in score.items()}
    return {o: v / sum(z.values()) for o, v in z.items()}


def test_debate_and_consultancy_with_verified_lines(which_items):
    dom = ChessDomain()
    items = which_items[:3]
    ver = soa.VerificationPolicy(verifiers=["chess_line"])
    mechs = [Debate(rounds=1, affordances={"agents": ["engine_notes"]}, verification=ver),
             Consultancy(rounds=1, affordances={"agents": ["engine_notes"]}, verification=ver)]
    exp = ASDExperiment(mechs, items, agent=soa.FunctionPolicy(scripted_expert, label="scripted_expert"),
                        fixtures={"judge": soa.FunctionPolicy(claim_counting_judge, label="claim_judge")},
                        ctx=dom.context(), ground_truth=dom.ground_truth_scorers())
    eps = exp.run()
    assert len(eps) == 2 * 2 * len(items) and all(e.error is None for e in eps), [e.error for e in eps if e.error][:1]
    statuses = {v.status for e in eps for v in e.verifications()}
    assert statuses == {"verified", "refuted"}
    assert any('<verified kind="chess_line">' in t.shown and "<result>Legal:" in t.shown for e in eps for t in e.turns)
    for e in eps:
        assert set(e.ground_truth["role_values"].values()) <= {1.0, -1.0} and "judge_correct" in e.ground_truth
    table = asd(exp.frame()).set_index("mechanism")["asd"]
    assert table["debate"] > 0 and table["consultancy"] > 0


def test_reviewed_work_best_move_scorer():
    dom = ChessDomain(kind="best_move")
    items = dom.load(limit=3)
    board = {it.id: chess.Board(it.context["fen"]) for it in items}
    table = {it.id: it.ground_truth.data["move_evals"] for it in items}
    picks = {"solution": {it.id: it.ground_truth.data["best"] for it in items},
             "worst": {it.id: min(table[it.id], key=table[it.id].get) for it in items}}

    def worker(which):
        def act(req, ctx):
            if which == "none":
                return "I cannot find anything good here."
            iid = req.view.item.id
            return f"After careful thought. Move: {board[iid].san(chess.Move.from_uci(picks[which][iid]))}"
        return soa.FunctionPolicy(act, label=which)

    critic = soa.FunctionPolicy(lambda req, ctx: "That move fails. Move: a3 was better.", label="critic")
    reviewer = soa.FunctionPolicy(lambda req, ctx: {"accept": 0.8, "reject": 0.2}, label="reviewer")
    profiles = [Profile(name=w, players={"worker": worker(w), "critic": critic, "reviewer": reviewer})
                for w in ("solution", "worst", "none")]
    eps = run_sync(run_episodes(ReviewedWork(critique_rounds=1), items, profiles, ctx=dom.context(),
                                ground_truth=dom.ground_truth_scorers()))
    assert all(e.error is None for e in eps), [e.error for e in eps if e.error][:1]
    for e in eps:
        gt = e.ground_truth
        assert set(gt["role_values"]) == {"worker"}  # critics (rewarded for "reject") are not scored
        v = gt["role_values"]["worker"]
        if e.profile == "none":
            assert v == -1.0
        else:
            t = table[e.item_id]
            loss = max(map(clip_cp, t.values())) - clip_cp(t[picks[e.profile][e.item_id]])
            assert v == pytest.approx(1 - 2 * min(1.0, loss / 300)) and gt["cp_loss"] == pytest.approx(loss)
        assert gt["judge_correct"] == (1.0 if v > 0 else 0.0)  # the reviewer always accepts
    assert all(e.ground_truth["role_values"]["worker"] > 0.5 for e in eps if e.profile == "solution")


# ----------------------------------------------------------------------------- engine (Stockfish, low node counts)

@pytest.mark.engine
@needs_engine
async def test_engine_tool_and_eval_verifier(which_items):
    ctx = ChessDomain(tool_nodes=5000, eval_nodes=5000).context()
    tool, ev = ctx.tools["engine"], ctx.verifiers["chess_eval"]
    it = which_items[0]
    out = await tool.call("", it)
    assert not out.error and "Best move:" in out.output and "Evaluation:" in out.output
    assert not (await tool.call(it.context["candidates"]["A"]["uci"], it)).error
    assert (await tool.call("Ke9", it)).error
    up_a_queen = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    assert "Best move:" in (await tool.call(up_a_queen, it)).output
    for expect, status in (("white_better", "verified"), ("black_better", "refuted"), (">+3", "verified"),
                           ("nonsense", "unchecked")):
        res = await ev.verify(claim("chess_eval", "", expect=expect, **{"from": up_a_queen}), it)
        assert res.status == status, (expect, res)
    mated = await ev.verify(claim("chess_eval", "f3 e5 g4 Qh4#", expect="black_better", **{"from": "start"}), it)
    assert mated.status == "verified" and "checkmate" in mated.output


@pytest.mark.engine
@needs_engine
def test_expert_uses_private_engine_tool(which_items):
    dom = ChessDomain(tool_nodes=5000)
    expert = soa.LLMPolicy(soa.models.MockModel(['<tool name="engine"></tool>', "My move wins material."]))
    judge = soa.FunctionPolicy(lambda req, ctx: {"A": 0.5, "B": 0.5})
    prof = Profile(name="p", players={"consultant": soa.PlayerSpec(policy=expert, stance="A"), "judge": judge})
    mech = Consultancy(rounds=1, tools={"agents": ["engine"]})  # the capability gap: only experts get the engine
    ep = run_sync(run_episodes(mech, which_items[:1], [prof], ctx=dom.context()))[0]
    assert ep.error is None, ep.error
    (turn,) = ep.turns_of("consultant")
    assert [c["name"] for c in turn.tool_calls] == ["engine"] and "Best move:" in turn.tool_calls[0]["result"]
    assert "Best move:" not in turn.shown  # tool results stay private unless the expert shares them


@pytest.mark.engine
@needs_engine
def test_analyse_puzzle_and_engine_memo():
    eng = shared_engine()
    rec = {k: v for k, v in ChessDomain().records()[1].items() if k != "analysis"}
    a = analyse_puzzle(rec, eng, deep_nodes=20_000, table_depth=6)
    if a is not None:
        assert a["alternative"]["uci"] != a["best"]["uci"] and a["gap_cp"] >= 150
        assert a["best"]["uci"] == rec["moves"][1]
    before = eng.searches
    assert analyse_puzzle(rec, eng, deep_nodes=20_000, table_depth=6) == a and eng.searches == before  # memoized
    board = chess.Board()
    first = eng.analyse_sync(board, chess.engine.Limit(nodes=3000))
    eng._memo.clear()
    assert eng.analyse_sync(board, chess.engine.Limit(nodes=3000)) == first  # hash cleared: reproducible


@pytest.mark.network
def test_fetch_lichess_prefix():
    from so_arena.domains.chess import fetch_lichess_puzzles

    try:
        rows = fetch_lichess_puzzles(50, max_bytes=1 << 16)
    except OSError as e:  # offline
        pytest.skip(f"network unavailable: {e}")
    assert len(rows) == 50 and all(len(r["moves"]) >= 2 and r["rating"] > 0 for r in rows)
