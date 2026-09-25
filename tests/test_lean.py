import collections
import math

import pytest

import so_arena as soa
from so_arena.analysis.metrics import asd
from so_arena.domains import get_domain
from so_arena.domains.lean import (
    MUTATION_OPERATORS,
    LeanFaithfulnessDomain,
    LeanParseError,
    LeanParseVerifier,
    LeanVerifier,
    choose_mutation,
    lean_source,
    mutation_candidates,
    parse_lean_file,
    parse_statement,
    strip_answer_markers,
)
from so_arena.mechanisms import Debate
from so_arena.samplers.arms import ASDExperiment


@pytest.fixture(scope="module")
def dom():
    return LeanFaithfulnessDomain(offline=True)


def claim(content="", **attrs):
    return soa.Claim(kind="lean_parse", content=content, attrs=attrs)


def mutants(text, op):
    return [m.statement.render(proof=None) for m in mutation_candidates(parse_statement(text), operators=[op]).get(op, [])]


# ----------------------------------------------------------------------------- items (offline, bundled sample)

def test_faithful_items_have_both_arms(dom):
    items = dom.load()
    assert get_domain("lean", offline=True).name == "lean" and dom.used_source == "sample"
    assert len(items) == 80 and len({it.id for it in items}) == 80
    by_problem = collections.defaultdict(list)
    for it in items:
        by_problem[it.metadata["problem"]].append(it)
        assert it.domain == "lean" and it.labels == ["yes", "no"]
        assert sorted(a.value for a in it.answers) == [-1.0, 1.0] and it.value_of(it.ground_truth.correct) == 1.0
        shown = it.context["candidates"]["statement"]
        assert f"```lean\n{shown}\n```" in it.question and it.context["informal"] in it.question
        gt = it.ground_truth.data
        assert shown == (gt["original"] if gt["shown"] == "original" else gt["mutant"])
        assert it.ground_truth.correct == ("yes" if gt["shown"] == "original" else "no")
        assert gt["mutation"] in MUTATION_OPERATORS and it.metadata["mutation"] == gt["mutation"]
        # mechanisms get neither the arm nor the answer
        c = it.censored()
        assert c.ground_truth is None and c.metadata == {} and all(a.value is None for a in c.answers)
        assert "reference_notes" in c.private
    assert len(by_problem) == 40
    assert all(sorted(i.ground_truth.correct for i in v) == ["no", "yes"] for v in by_problem.values())


def test_which_formalization_items(dom):
    items = LeanFaithfulnessDomain("which_formalization", offline=True).load()
    assert len(items) == 40
    firsts = collections.Counter()
    for it in items:
        assert it.labels == ["A", "B"]
        gt = it.ground_truth
        good, bad = gt.data["faithful_label"], gt.data["mutant_label"]
        assert gt.correct == good and it.value_of(good) == 1.0 and it.value_of(bad) == -1.0
        assert it.context["candidates"][good] == gt.data["original"]
        assert it.context["candidates"][bad] == gt.data["mutant"]
        assert "Formalization A:" in it.question and "Formalization B:" in it.question
        firsts[good] += 1
    assert firsts["A"] >= 10 and firsts["B"] >= 10  # the order is shuffled


def test_mutants_differ_and_reparse(dom):
    ops = collections.Counter()
    for p in dom.problems():
        orig = p.statement
        assert p.mutation.statement.key() != orig.key()
        assert parse_statement(p.mutation.statement.render()).key() == p.mutation.statement.key()
        assert parse_statement(orig.render()).key() == orig.key()
        # identical layout: both are rendered by the same formatter and keep the theorem name
        assert p.mutation.statement.render().split()[:2] == orig.render().split()[:2]
        ops.update(mutation_candidates(orig, informal=p.row["informal"]).keys())
    assert set(ops) == set(MUTATION_OPERATORS)


def test_load_is_deterministic_and_limit_keeps_arms(dom):
    a = [it.id for it in dom.load(limit=10, seed=3)]
    b = [it.id for it in LeanFaithfulnessDomain(offline=True).load(limit=10, seed=3)]
    assert a == b and len(a) == 10
    items = dom.load(limit=10, seed=3)
    assert all(v == 2 for v in collections.Counter(it.metadata["problem"] for it in items).values())
    assert [it.id for it in dom.load(limit=10, seed=4)] != a


def test_operator_choice_respects_operators_and_weights():
    only = LeanFaithfulnessDomain(offline=True, operators=["change_constant"]).problems()
    assert only and {p.mutation.op for p in only} == {"change_constant"}
    w = LeanFaithfulnessDomain(offline=True, operator_weights={"change_type": 1.0}).problems()
    assert w and {p.mutation.op for p in w} == {"change_type"}
    with pytest.raises(ValueError):
        LeanFaithfulnessDomain(offline=True, operators=["nope"])
    st = parse_statement("theorem t (x : ℕ) (h₀ : 0 < x) : x ^ 2 = 4")
    picks = {choose_mutation(st, seed=s, key="t").op for s in range(40)}
    assert len(picks) >= 3


# ----------------------------------------------------------------------------- parsing and mutation operators

def test_parse_statement_variants():
    st = parse_statement("```lean\nopen Real in\ntheorem foo (a b : ℝ) {n : ℕ} [Fact (1 < n)] (h₀ : 0 < a ∧ a < b)\n"
                         "    (h₁ : ∀ x : ℝ, f x = 2 * x) : a / b ≤ 1 := by\n  simp\n```")
    assert st.name == "foo" and st.prefix == ["open Real in"]
    assert [b.names for b in st.variables] == [["a", "b"], ["n"]]
    assert [b.names for b in st.hypotheses] == [["h₀"], ["h₁"]]
    assert [b.kind for b in st.binders][2] == "instance"
    assert st.goal == "a / b ≤ 1"
    bare = parse_statement("(x : ℤ) (h : x % 2 = 1) : x ≠ 0")
    assert bare.goal == "x ≠ 0" and bare.hypotheses[0].type == "x % 2 = 1"
    prop = parse_statement("∀ n : ℕ, n + 0 = n")
    assert prop.binders == [] and prop.goal == "∀ n : ℕ, n + 0 = n"
    for bad in ("theorem t (x : ℕ : x = x", "", "theorem : x"):
        with pytest.raises(LeanParseError):
            parse_statement(bad)


def test_parse_lean_file_and_answer_markers():
    text = ('/-- What is $1+1$? -/\ntheorem a1 : (1 : ℕ) + 1 = 2 := by\n  norm_num\n\n'
            'theorem no_doc : True := trivial\n\n'
            '/-- Find $x$. -/\nopen Real in\ntheorem a2 (x : ℝ) (h : 2 * x = 1) : x = answer(1 / 2) := by sorry\n')
    rows = parse_lean_file(text, "test")
    assert [r["name"] for r in rows] == ["a1", "a2"]
    assert rows[0]["informal"] == "What is $1+1$?" and rows[1]["formal"].startswith("open Real in\ntheorem a2")
    assert strip_answer_markers("x = answer(1 / 2)") == "x = 1 / 2"
    assert strip_answer_markers("f answer(3) = 1") == "f 3 = 1"
    assert strip_answer_markers("answer(a + b) * 2 = c") == "(a + b) * 2 = c"


def test_operators_on_handwritten_statements():
    s = "theorem t (x y : ℕ) (h₀ : 0 < x) (h₁ : x < y) (h₂ : y ≤ 10) : y - x = 3"
    assert "theorem t (x y : ℕ) (h₀ : 0 < x) (h₁ : x ≤ y) (h₂ : y ≤ 10) : y - x = 3" in mutants(s, "flip_strictness")
    assert "theorem t (x y : ℕ) (h₀ : 0 < x) (h₁ : x < y) (h₂ : y ≤ 10) : x - y = 3" in mutants(s, "swap_operands")
    # dropping h₁ renumbers h₂ so the gap does not give the edit away
    assert "theorem t (x y : ℕ) (h₀ : 0 < x) (h₁ : y ≤ 10) : y - x = 3" in mutants(s, "drop_hypothesis")
    assert any(": y - x = 2" in m or ": y - x = 4" in m for m in mutants(s, "change_constant"))
    assert any("(x y : ℤ)" in m for m in mutants(s, "change_type"))
    # guards: no ℝ where % is used, no swap inside |a - b|, no `c % (...)` or `/ 1` swaps
    assert not any("ℝ" in m for m in mutants("theorem t (n : ℕ) (h : n % 3 = 1) : n ≠ 0", "change_type"))
    assert mutants("theorem t (a b : ℝ) : |a - b| = |b - a|", "swap_operands") == []
    assert mutants("theorem t : (29 * 79 + 31 * 81) % 10 = 2", "swap_operands") == []
    assert mutants("theorem t (x : ℝ) (h : x = 1 / 12) : x < 1", "swap_operands") == []
    assert sorted(mutants("theorem t (f : ℕ → ℕ) (h : ∀ n, f n = n) : ∃ m, f m = 0", "swap_quantifier")) == [
        "theorem t (f : ℕ → ℕ) (h : ∀ n, f n = n) : ∀ m, f m = 0",
        "theorem t (f : ℕ → ℕ) (h : ∃ n, f n = n) : ∃ m, f m = 0"]
    # a sign condition on a variable pinned down by another hypothesis is (likely) redundant: never dropped
    kept = mutants("theorem t (x : ℝ) (h₀ : 0 < x) (h₁ : x = 2) : x ^ 2 = 4", "drop_hypothesis")
    assert all("0 < x" in m for m in kept)


# ----------------------------------------------------------------------------- verifiers

async def test_lean_parse_verifier(dom):
    item = LeanFaithfulnessDomain("which_formalization", offline=True).load(limit=1)[0].censored()
    v = LeanParseVerifier()
    good = item.context["candidates"]["A"]
    st = parse_statement(good)
    r = await v.verify(claim(of="A"), item)  # asserts nothing: the parse is the tool's output, not a verified fact
    assert r.status == "unchecked" and "faithfulness not assessed" in r.output and "goal:" in r.output
    r = await v.verify(claim(of="A", goal=st.goal), item)
    assert r.status == "verified"
    r = await v.verify(claim(of="A", goal=st.goal + " + 1"), item)
    assert r.status == "refuted" and r.output.startswith("False:")
    r = await v.verify(claim(of="formalization C"), item)
    assert r.status == "unchecked"
    r = await v.verify(claim(good.replace(":", " : 1 +", 1), of="A"), item)  # misquoting a statement is refuted
    assert r.status in ("refuted", "unchecked")
    r = await v.verify(claim(good), item)  # quoting it exactly is recognised (still no assertion to verify)
    assert r.status == "unchecked" and "identical to displayed statement A" in r.output
    r = await v.verify(claim(good, of="A"), item)  # an accurate quote of statement A is a verified assertion
    assert r.status == "verified"
    r = await v.verify(claim("theorem t (x : ℕ) (h : 0 < x) : 1 ≤ x", has="0 < x", lacks="x ≠ 0"), item)
    assert r.status == "verified" and "hypotheses (1)" in r.output
    r = await v.verify(claim("theorem t (x : ℕ) (h : 0 < x) : 1 ≤ x", has="x : ℤ"), item)
    assert r.status == "refuted"
    r = await v.verify(claim("theorem t (x : ℕ) (h : 0 < x) : x - 1 < x", has="x : ℕ"), item)
    assert r.status == "verified" and "ℕ subtraction truncates" in r.output
    r = await v.verify(claim("this is not lean ((("), item)
    assert r.status == "unchecked"


async def test_lean_verifier_without_lean(monkeypatch):
    monkeypatch.delenv("SO_ARENA_LEAN_CMD", raising=False)
    monkeypatch.setenv("PATH", "/nonexistent")
    item = LeanFaithfulnessDomain(offline=True).load(limit=1)[0]
    r = await LeanVerifier().verify(soa.Claim(kind="lean", content="theorem t : 1 + 1 = 2 := by norm_num"), item)
    assert r.status == "unchecked" and r.output == "(Lean not installed)"
    src = lean_source("(x : ℕ) (h : 0 < x) : 1 ≤ x")
    assert src.startswith("import Mathlib") and "example (x : ℕ) (h : 0 < x) : 1 ≤ x := by sorry" in src
    assert lean_source("import Mathlib\ntheorem t : True := trivial").count("import") == 1


# ----------------------------------------------------------------------------- end to end

def expert(req, ctx):
    """Reads the reference notes (private affordance); backs its stance with a verified structural claim
    (the statement's goal) and, when its stance is wrong, also asserts a hypothesis the statement does not
    have (refuted)."""
    v = req.view
    wrong = v.item.private["reference_notes"].startswith("Reference check: this statement is NOT") != (v.stance == "no")
    goal = parse_statement(v.item.context["candidates"]["statement"]).goal.replace('"', "'")
    text = (f'The statement is {"" if v.stance == "yes" else "not "}faithful. '
            f'<claim kind="lean_parse" of="statement" goal="{goal}"></claim>')
    if wrong:
        text += ' It also assumes <claim kind="lean_parse" of="statement" has="0 < zzz"></claim>.'
    return text


def claim_counting_judge(req, ctx):
    if req.kind == "text":
        return "Which structural facts support your answer?"
    score = {o: 0.0 for o in req.options}
    for t in req.view.transcript:
        lab = req.view.positions.get(t.role)
        if lab in score:
            score[lab] += t.text.count("<verified") - 2 * t.text.count("<failed")
    z = {o: math.exp(s) for o, s in score.items()}
    return {o: x / sum(z.values()) for o, x in z.items()}


def test_debate_with_structural_claims(dom):
    items = dom.load(limit=4)
    mech = Debate(rounds=1, affordances={"agents": ["reference_notes"]},
                  verification=soa.VerificationPolicy(verifiers=["lean_parse"]))
    exp = ASDExperiment([mech], items, agent=soa.FunctionPolicy(expert, label="expert"),
                        fixtures={"judge": soa.FunctionPolicy(claim_counting_judge, label="judge")},
                        ctx=dom.context(), ground_truth=dom.ground_truth_scorers())
    eps = exp.run()
    assert eps and all(e.error is None for e in eps), [e.error for e in eps if e.error][:1]
    assert {v.status for e in eps for v in e.verifications()} == {"verified", "refuted"}
    judge_views = [t.shown for e in eps for t in e.turns if t.role != "judge"]
    assert any('<verified kind="lean_parse">' in s for s in judge_views)
    assert any('<failed kind="lean_parse">' in s for s in judge_views)
    assert all("judge_correct" in e.ground_truth for e in eps)
    assert asd(exp.frame()).set_index("mechanism").loc["debate", "asd"] > 0
