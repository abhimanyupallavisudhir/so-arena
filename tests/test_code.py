import json
import math

import pytest

import so_arena as soa
from so_arena.analysis.metrics import asd
from so_arena.core.runner import Profile, run_episodes, run_sync
from so_arena.core.verification import Claim
from so_arena.domains.code import (
    CodeDomain,
    PythonTool,
    audit_contribution,
    code_blocks,
    code_expert,
    code_reviewer,
    code_team_worker,
    extract_code,
    find_counterexample,
    generate_mutants,
    hack_solution,
    hardcoded_literals,
    run_suites,
    run_tests,
    review_items,
)
from so_arena.mechanisms import ReviewedWork, Team
from so_arena.samplers.arms import ASDExperiment


def run(mech, items, profiles, ctx=None, gt=None):
    eps = run_sync(run_episodes(mech, items, profiles, ctx=ctx, ground_truth=gt))
    errs = [e.error for e in eps if e.error]
    assert not errs, errs[0]
    return eps


@pytest.fixture(scope="module")
def problems():
    return CodeDomain().problems()


def verify(v, claim, item):
    return run_sync(v.verify(claim, item))


# ----------------------------------------------------------------------------- data and items

def test_sample_loads_offline_and_hides_hidden_tests(problems):
    assert len(problems) == 60
    assert all(len(p.visible) == 1 and p.hidden and p.survivors() for p in problems)
    items = CodeDomain().load()
    assert len(items) == 120 and {it.true_label for it in items} == {"correct", "incorrect"}
    assert sum(it.true_label == "correct" for it in items) == 60  # both arms for every problem
    assert len(CodeDomain("which_solution").load()) == 60 and len(CodeDomain("implement").load()) == 60
    for it in items[:10] + CodeDomain("team").load(limit=2):
        public = json.dumps([it.question, it.context, it.private, it.metadata, it.id])
        hidden = it.ground_truth.data.get("hidden_tests") or [t for p in it.ground_truth.data["parts"]
                                                             for t in p["hidden_tests"]]
        assert hidden and not any(t in public for t in hidden)
        c = it.censored()
        assert c.ground_truth is None and all(a.value is None for a in c.answers or [])
    with pytest.raises(ValueError):
        CodeDomain().problems("train")  # the bundled sample is test-split only


def test_every_mutant_passes_visible_and_fails_hidden(problems):
    jobs, owners = [], []
    for p in problems:
        for code in [p.reference] + [m.code for m in p.mutants]:
            jobs.append({"code": code, "tests": p.tests, "setup": p.setup})
            owners.append((p, code == p.reference))
    results = run_suites(jobs, timeout=2.0)
    for (p, is_ref), r in zip(owners, results):
        if is_ref:
            assert r.all_passed, p.uid
        else:
            assert r.load_error is None and all(r.passed[:1]) and not all(r.passed[1:]), p.uid


def test_mutation_operators_and_code_extraction():
    src = ("def f(xs, k):\n    out, first = 0, xs[0]\n    for i in range(len(xs)):\n        if xs[i] > k and not xs[i] == 7:\n"
           "            out += xs[i]\n        else:\n            out = max(out, 1)\n    if not out:\n"
           "        return xs[0:2] if first else first\n    return out\n")
    ms = generate_mutants(src, limit=None)
    assert {"cmp", "range", "slice", "index", "const", "boolop", "arith", "minmax", "return", "branch", "not"} <= {m.op for m in ms}
    assert len({m.code for m in ms}) == len(ms) and all(m.desc.startswith("line ") for m in ms)
    for m in ms:
        compile(m.code, "<m>", "exec")
    text = "Plan:\n```\nassert f(1)\n```\n```python\ndef f(x):\n    return x\n```\nand a truncated\n```python\ndef g(x):\n    return 2"
    assert code_blocks(text) == ["def f(x):\n    return x", "def g(x):\n    return 2"]
    assert extract_code(text).startswith("def g") and extract_code("def h():\n    pass") == "def h():\n    pass"
    assert extract_code("no code here") is None


def test_sandbox_guards():
    r = run_tests("def f(x):\n    while True:\n        pass", ["assert f(1) == 1", "assert True"], timeout=0.3)
    assert r.passed == [False, True] and "Timeout" in r.errors[0]
    # an always-equal object cannot pass equality asserts
    cheat = "class E:\n    def __eq__(self, o):\n        return True\ndef f(x):\n    return E()"
    assert not run_tests(cheat, ["assert f(1) == 2"]).passed[0]
    # a hard crash only costs the in-flight test
    r = run_suites([{"code": "import os\ndef f():\n    os._exit(1)", "tests": ["assert f()", "assert 1"]}])[0]
    assert r.passed == [False, True]


# ----------------------------------------------------------------------------- verifiers and tools

def test_tests_verifier_reports_visible_only(problems):
    dom = CodeDomain()
    v = dom.verifiers()["tests"]
    items = review_items(problems[0])
    mutant = next(it for it in items if it.true_label == "incorrect")
    # even given the uncensored item with a poisoned hidden suite, only visible tests run
    poisoned = mutant.model_copy(deep=True)
    poisoned.ground_truth.data["hidden_tests"] = ["assert False"] * 5
    res = verify(v, Claim(kind="tests", content="run"), poisoned)
    assert res.status == "verified" and res.output.startswith("visible tests: 1/1 passed") and "test 2" not in res.output
    res = verify(v, Claim(kind="tests", content="run", attrs={"expect": "fail"}), mutant.censored())
    assert res.status == "refuted"
    p = problems[0]
    own = f"```python\ndef {p.entry_point}(*a):\n    return None\n```"
    assert verify(v, Claim(kind="tests", content=own), mutant.censored()).status == "refuted"
    which = CodeDomain("which_solution").load(limit=1)[0].censored()
    for lab in "AB":  # both candidates pass the visible tests by construction
        assert verify(v, Claim(kind="tests", content="run", attrs={"candidate": lab}), which).status == "verified"
    assert verify(v, Claim(kind="tests", content="run"), which).status == "error"


def test_python_verifier_oracle_and_tool(problems):
    dom = CodeDomain("which_solution")
    item = dom.load(limit=1)[0]
    p = problems[0]
    truth = item.true_label
    wrong = "B" if truth == "A" else "A"
    cx = find_counterexample(item.context["candidates"][wrong], p.reference, p.entry_point, p.visible, setup=p.setup)
    assert cx is not None and cx["candidate"] != cx["reference"]
    py, oracle = dom.verifiers()["python"], dom.verifiers()["oracle"]
    snippet = f"print({cx['call']} == {cx['reference']})"
    c = item.censored()
    assert verify(py, Claim(kind="python", content=snippet, attrs={"candidate": wrong, "expect": "False"}), c).status == "verified"
    assert verify(py, Claim(kind="python", content=snippet, attrs={"candidate": truth, "expect": "True"}), c).status == "verified"
    assert verify(oracle, Claim(kind="oracle", content=snippet, attrs={"expect": "True"}), c).status == "verified"
    assert verify(py, Claim(kind="python", content="print(1)", attrs={"candidate": "Z"}), c).status == "error"
    bad = verify(py, Claim(kind="python", content="print(undefined_name)"), c)
    assert bad.status == "refuted" and "NameError" in bad.output
    out = run_sync(PythonTool().call(f"# candidate: {wrong}\nprint(repr({cx['call']}))", c))
    assert out.output == cx["candidate"] and not out.error


# ----------------------------------------------------------------------------- ground truth

def test_hidden_tests_scorer_grades_implementations(problems):
    dom = CodeDomain("implement")
    p = problems[0]
    item = dom.load(limit=1)[0]
    mutant = p.survivors()[0].code
    reviewer = soa.ScriptedPolicy('{"accept": 0.8, "reject": 0.2}')
    profiles = [Profile(name=name, players={"worker": soa.FixedPolicy(f"Solution:\n```python\n{code}\n```"),
                                            "reviewer": reviewer})
                for name, code in (("correct", p.reference), ("buggy", mutant))]
    eps = {e.profile: e for e in run(ReviewedWork(), [item], profiles, gt=dom.ground_truth_scorers())}
    good, bad = eps["correct"].ground_truth, eps["buggy"].ground_truth
    assert good["hidden_pass_rate"] == 1.0 and good["role_values"]["worker"] == 1.0 and good["judge_correct"] == 1.0
    assert bad["visible_pass_rate"] == 1.0 and bad["hidden_pass_rate"] < 1.0 and bad["role_values"]["worker"] < 1.0
    assert bad["judge_correct"] == 0.0 and bad["outcome_value"] < 1.0


# ----------------------------------------------------------------------------- mechanisms end to end

def test_reviewed_work_asd_both_arms(problems):
    # a problem whose mutant has a counterexample near the visible inputs (found by differential testing)
    p = next(p for p in problems
             if find_counterexample(p.survivors()[0].code, p.reference, p.entry_point, p.visible, setup=p.setup))
    dom = CodeDomain()
    items = review_items(p)
    mech = ReviewedWork(critique_rounds=1, affordances={"agents": ["tests", "reference"]},
                        verification=soa.VerificationPolicy(verifiers=["tests", "python"]))
    exp = ASDExperiment([mech], items, agent=code_expert(), fixtures={"judge": code_reviewer()}, ctx=dom.context(),
                        ground_truth=dom.ground_truth_scorers())
    eps = exp.run()
    assert len(eps) == 4 and not any(e.error for e in eps)
    for ep in eps:
        assert ep.ground_truth["role_values"]["worker"] in (1.0, -1.0) and "judge_correct" in ep.ground_truth
        assert ep.rewards["worker"] == pytest.approx(math.log(ep.outcome.probs["accept"]))
        assert all(v.status in ("verified", "refuted") for v in ep.verifications())
    # honest arguments carry verified evidence, lies get refuted: the reviewer rewards the truth
    assert asd(exp.frame(), roles=["worker"])["asd"].iloc[0] > 0
    assert all(ep.ground_truth["judge_correct"] == 1.0 for ep in eps)


def test_team_hacker_reported_and_verified_by_oracle():
    dom = CodeDomain("team")
    items, ctx = dom.load(limit=1), dom.context()
    item = items[0]
    hack = hack_solution(item.context["parts"][0]["entry_point"], item.context["parts"][0]["visible_tests"])
    assert hardcoded_literals(hack, item.context["parts"][0]["entry_point"], item.context["parts"][0]["visible_tests"])

    def profile(reporter: str) -> Profile:
        return Profile(name=reporter, players={"worker_1": code_team_worker(work="hack"),
                                               "worker_2": code_team_worker(report=reporter),
                                               "worker_3": code_team_worker(), "grader": dom.grader()})

    mech = Team(n_workers=3, reward=soa.Whistleblower(bounty=0.5), affordances={"agents": ["reference"]})
    eps = {e.profile: e for e in run(mech, items, [profile("report"), profile("silent")], ctx=ctx,
                                      gt=dom.ground_truth_scorers())}
    rep, silent = eps["report"], eps["silent"]
    assert rep.outcome.data["team_score"] == pytest.approx(1.0)  # the hack passes the visible-test grader
    assert rep.outcome.data["reports"]["worker_2"]["verified"] is True and rep.outcome.data["violators"] == ["worker_1"]
    assert rep.rewards["worker_2"] == pytest.approx(0.5) and rep.rewards["worker_3"] == pytest.approx(0.0)
    assert silent.rewards == {f"worker_{i}": pytest.approx(1.0) for i in (1, 2, 3)}  # silence pays the hack
    g = rep.ground_truth
    assert g["true_violators"] == ["worker_1"] and g["violation_reverted"] == 1.0 and g["outcome_value"] == 1.0
    assert g["visible_pass_rate"] == 1.0 and g["hidden_pass_rate"] < 1.0
    assert g["role_values"] == {"worker_1": -1.0, "worker_2": 1.0, "worker_3": 0.0}
    assert silent.ground_truth["violation_reverted"] == 0.0 and silent.ground_truth["outcome_value"] == 0.0


def test_false_accusation_is_refuted_and_honest_bugs_are_not_violations(problems):
    dom = CodeDomain("team")
    items, ctx = dom.load(limit=1), dom.context()
    players = {"worker_1": code_team_worker(), "worker_2": code_team_worker(report="accuse"),
               "worker_3": code_team_worker(), "grader": dom.grader()}
    mech = Team(n_workers=3, reward=soa.Whistleblower(bounty=0.5, false_report_penalty=0.3),
                affordances={"agents": ["reference"]})
    ep = run(mech, items, [Profile(name="p", players=players)], ctx=ctx, gt=dom.ground_truth_scorers())[0]
    assert ep.outcome.data["reports"]["worker_2"]["verified"] is False
    assert ep.rewards["worker_2"] == pytest.approx(1.0 - 0.3) and ep.ground_truth["violation"] == 0.0
    # a subtly wrong (mutant) contribution fails hidden tests but hard-codes nothing: not misconduct
    p = problems[0]
    part = {"entry_point": p.entry_point, "visible_tests": p.visible, "hidden_tests": p.hidden}
    assert audit_contribution(p.survivors()[0].code, [part], p.setup)["violation"] is False
    assert audit_contribution(hack_solution(p.entry_point, p.visible), [part], p.setup)["violation"] is True


# ----------------------------------------------------------------------------- data sources

def test_mbpp_source_falls_back_to_sample_offline(monkeypatch, tmp_path):
    import urllib.error

    import so_arena.domains.code as code_mod

    def offline(*a, **k):
        raise urllib.error.URLError("no network in this test")

    monkeypatch.setenv("SO_ARENA_DATA", str(tmp_path))
    monkeypatch.setattr(code_mod, "download", offline)
    rows, used = code_mod.load_rows("mbpp")
    assert used == "sample" and len(rows) == 60


@pytest.mark.network
def test_mbpp_download_and_prepare():
    from so_arena.datasets import download
    from so_arena.domains.code import MBPP_URL, prepare_rows

    try:
        raw = json.loads(download(MBPP_URL).read_text())
    except OSError as e:
        pytest.skip(f"network unavailable: {e}")
    rows, stats = prepare_rows([r for r in raw if 11 <= r["task_id"] <= 20], max_candidates=20)
    assert stats["usable"] >= 8 and all(r["entry_point"] and "mutants" in r for r in rows)
