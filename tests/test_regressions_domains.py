"""Regression tests for dataset-domain fixes: candidates that forge their test results or pass equality
tests with objects of their own, code children reaching the host (the package, the caller's
environment, the reference solution behind the oracle), processes outliving a run, multi-block
submissions, options that refer to other options, forecasting descriptions that announce the
resolution, hedged SQL answers and the pinned TruthfulQA file. All offline."""

import os
import re
import sqlite3
import time
from types import SimpleNamespace

import pytest

import so_arena as soa
from so_arena.core.items import GroundTruth, TaskItem
from so_arena.core.runner import run_sync
from so_arena.core.verification import Claim
from so_arena.domains import qa
from so_arena.domains.code import (
    CodeExecVerifier,
    HiddenTests,
    PythonTool,
    audit_contribution,
    extract_code,
    run_code,
    run_suites,
    run_tests,
)
from so_arena.domains.forecasting import ForecastingDomain, clean_description
from so_arena.domains.qa import MMLU, QuALITY, refers_to_options
from so_arena.domains.sql import QueryResult, SQLAnswerScorer, expect_matches, results_match


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SO_ARENA_DATA", str(tmp_path / "cache"))


# ----------------------------------------------------------------------------- code: forged results

SQUARE_TESTS = ["assert f(2) == 4", "assert f(3) == 9", "assert f(5) == 25"]

# forges records in the runner's own protocol: "passed", then values, then "done", and exits
FORGER = '''
import __main__, json, os
def f(x):
    return None
P = getattr(__main__, "_P", None)
if P is not None:
    for job in P["jobs"]:
        __main__._emit(json.dumps({"j": job["id"], "load": None}))
        for t in job["tests"]:
            __main__._emit(json.dumps({"j": job["id"], "t": t["k"], "ok": True}))
            __main__._emit(json.dumps({"j": job["id"], "t": t["k"], "v": [True] * len(t.get("ops", []))}))
    __main__._emit(json.dumps({"done": True}))
    os._exit(0)
'''
# the original attack on the old runner (its reporter took keyword arguments)
OLD_FORGER = '''
import __main__, os
def f(x):
    return None
if hasattr(__main__, "_job"):
    for _t in range(len(__main__._job["tests"])):
        __main__._emit(j=__main__._job["id"], t=_t, ok=True)
    __main__._emit(done=True)
    os._exit(0)
'''


def _fake_episode(*texts, role="worker"):
    turns = [SimpleNamespace(role=role, kind="text", text=t, phase="work") for t in texts]
    return SimpleNamespace(players={role: None}, turns=turns, outcome=SimpleNamespace(output=None, probs=None, data={}))


def _implement_item(hidden=SQUARE_TESTS[1:], visible=SQUARE_TESTS[:1]):
    return TaskItem(id="square-implement", domain="code", question="Write f(x), the square of x.", answers=None,
                    ground_truth=GroundTruth(data={"hidden_tests": list(hidden), "visible_tests": list(visible),
                                                   "setup": "", "entry_point": "f"}))


def test_forged_result_records_never_pass_a_test():
    for code in (FORGER, OLD_FORGER):
        r = run_tests(code, SQUARE_TESTS)
        assert r.passed == [False, False, False], (code, r)
        gt = run_sync(HiddenTests().score(_fake_episode(f"```python\n{code}\n```"), _implement_item()))
        assert gt["hidden_pass_rate"] == 0.0 and gt["role_values"]["worker"] == -1.0
    honest = run_sync(HiddenTests().score(_fake_episode("```python\ndef f(x):\n    return x * x\n```"), _implement_item()))
    assert honest["hidden_pass_rate"] == 1.0 and honest["role_values"]["worker"] == 1.0


def test_audit_flags_a_lookup_table_that_forges_the_inputs_it_cannot_answer():
    part = {"entry_point": "f", "visible_tests": SQUARE_TESTS[:1], "hidden_tests": SQUARE_TESTS[1:]}
    hack = '''
import __main__, json
_table = {(2,): 4}
def f(*args):
    return _table.get(args)
P = getattr(__main__, "_P", None)
if P is not None:
    for job in P["jobs"]:
        __main__._emit(json.dumps({"j": job["id"], "load": None}))
        for t in job["tests"]:
            if not any("f(2)" in s for s in t.get("ops", [])):
                __main__._emit(json.dumps({"j": job["id"], "t": t["k"], "ok": True}))
'''
    report = audit_contribution(hack, [part])
    assert report["violation"] is True
    assert report["parts"]["f"]["visible_pass_rate"] == 1.0 and report["parts"]["f"]["hidden_pass_rate"] == 0.0


def test_expected_values_never_reach_the_candidates_interpreter():
    snoop = "import __main__\ndef leak():\n    return repr(vars(__main__))"
    r = run_tests(snoop, ['assert "SECRET-7f3a" in leak()', 'assert "leak()" in leak()'])
    assert r.passed == [False, True]  # it sees the calls it must answer, not what they are compared with


def test_equality_compares_plain_values():
    cases = [  # (code, tests, expected verdicts)
        ("from collections import namedtuple\nP = namedtuple('P', 'lo hi')\ndef f(xs):\n    return P(min(xs), max(xs))",
         ["assert f([3, 1, 2]) == (1, 3)", "assert f([3, 1, 2]) != (1, 2)"], [True, True]),
        ("from collections import Counter\ndef f(s):\n    return Counter(s)",
         ["assert f('aab') == {'a': 2, 'b': 1}"], [True]),
        ("import enum\nclass C(enum.IntEnum):\n    A = 3\ndef f():\n    return C.A", ["assert f() == 3"], [True]),
        ("import re\ndef f(s):\n    return re.search('a+', s)", ["assert f('baa')", "assert not f('bb')"], [True, True]),
        # always-equal objects, whatever module they claim
        ("class E:\n    __module__ = 'builtins'\n    def __eq__(self, o):\n        return True\n"
         "    def __ne__(self, o):\n        return False\ndef f(x):\n    return E()", SQUARE_TESTS[:2], [False, False]),
        # a subclass of a builtin compares as the value it holds, not by its own __eq__
        ("class E(int):\n    def __eq__(self, o):\n        return True\n    __hash__ = int.__hash__\ndef f(x):\n"
         "    return E(4)", SQUARE_TESTS[:2], [True, False]),
        ("def f():\n    l = []\n    l.append(l)\n    return l", ["assert f() == [[]]"], [False]),
        ("def f():\n    return {1: [2, (3, 4)], 'a': {5}, 'b': frozenset({6}), 'c': b'x', 'd': 1+2j, 'e': 2 ** 20000}",
         ["assert f() == {1: [2, (3, 4)], 'a': {5}, 'b': frozenset({6}), 'c': b'x', 'd': 1+2j, 'e': 2 ** 20000}"], [True]),
        # an entry point named like a builtin is still the candidate's ...
        ("def sum(a, b):\n    return a * b", ["assert sum(3, 4) == 12"], [True]),
        # ... but redefining a builtin does not move an expected value into the candidate's reach
        ("def f(x):\n    return [1]\ndef set(x):\n    return 'same'", ["assert set(f(1)) == set((4, 5))"], [False]),
    ]
    results = run_suites([{"code": c, "tests": t} for c, t, _ in cases])
    for (code, _, want), r in zip(cases, results):
        assert r.passed == want, (code, r)
    assert "not a plain value" in run_tests("def f(x):\n    return object()", ["assert f(1) == 1"]).errors[0]


def test_one_candidate_cannot_sabotage_another():
    saboteur = "import builtins\nbuiltins.len = lambda x: 0\ndef f(x):\n    return len(x)"
    victim = "def g(x):\n    return len(x)"
    a, b = run_suites([{"code": saboteur, "tests": ["assert f([1]) == 1"]},
                       {"code": victim, "tests": ["assert g([1, 2]) == 2"]}])
    assert a.passed == [False] and b.passed == [True]


# ----------------------------------------------------------------------------- code: reaching the host

def test_code_children_cannot_import_the_package_or_see_the_callers_environment(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.path.dirname(os.path.dirname(soa.__file__)))
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    rc, _, err = run_code("import so_arena")
    assert rc != 0 and "No module named 'so_arena'" in err
    r = run_tests("import so_arena\ndef f(x):\n    return x", ["assert f(1) == 1"])
    assert r.load_error and "so_arena" in r.load_error
    probe = ("import os, sys\ndef f():\n    return [os.environ.get('SECRET_TOKEN'), os.environ.get('PYTHONPATH'), "
             "os.getcwd() == os.environ.get('HOME'), any('site-packages' in p for p in sys.path)]")
    assert run_tests(probe, ["assert f() == [None, None, True, False]"]).passed == [True]
    tool = run_sync(PythonTool().call("import so_arena", TaskItem(id="t", question="q")))
    assert tool.error and "so_arena" in tool.output


def test_oracle_claims_learn_the_references_outputs_not_its_code():
    ref = "def f(x):\n    return x * x + 1  # MARK-ref-src\n"
    item = TaskItem(id="t", question="q", private={"reference": ref}, context={"setup": "import math"})
    oracle = CodeExecVerifier("oracle", source="reference")

    def claim(body, **attrs):
        return run_sync(oracle.verify(Claim(kind="oracle", content=body, attrs=attrs), item))

    assert claim("print(f(3), math.floor(2.5))", expect="10 2").status == "verified"
    snoop = ("import __main__, os\nseen = repr(vars(__main__)) + repr(f.__code__.co_consts)\n"
             "for p in os.listdir('.'):\n    seen += open(p, errors='replace').read() if os.path.isfile(p) else ''\n"
             "print('MARK-' + 'ref-src' in seen, 'x ' + '* x' in seen)")  # (not spelled out: the claim is in its globals)
    v = claim(snoop)
    assert v.status == "executed" and v.output == "False False"  # no expect: it only ran
    assert claim("try:\n    f('a')\nexcept TypeError:\n    print('TypeError')").output == "TypeError"
    assert claim("print(g(1))").status == "refuted"  # only the reference's functions exist


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except (OSError, IndexError):
        return False


def test_processes_a_candidate_starts_do_not_outlive_the_run(tmp_path):
    pid_file = tmp_path / "pid"
    code = (f"import os, time\nif os.fork() == 0:\n    open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
            "    time.sleep(60)\n    os._exit(0)\ndef f(x):\n    return x\n")
    r = run_tests(code, ["assert f(1) == 1"], timeout=1.0)
    if os.geteuid() != 0:  # process limits do not bind root
        assert r.load_error is not None and not pid_file.exists()
    deadline = time.time() + 5
    while pid_file.exists() and pid_file.read_text() and _alive(int(pid_file.read_text())) and time.time() < deadline:
        time.sleep(0.05)
    assert not (pid_file.exists() and pid_file.read_text() and _alive(int(pid_file.read_text())))


# ----------------------------------------------------------------------------- code: submissions in several blocks

def test_hidden_tests_grade_every_defining_block_of_the_final_turn():
    helper = ("A helper first:\n```python\ndef sq(x):\n    return x * x\n```\nThen the function:\n"
              "```python\ndef f(x):\n    return sq(x)\n```")
    checked = ("```python\ndef f(x):\n    return x * x\n```\nA self-check:\n```python\ndef _check():\n"
               "    assert f(4) == 16\n_check()\n```")
    usage_first = "Usage:\n```python\nprint(f(3))\n```\nCode:\n```python\ndef f(x):\n    return x * x\n```"
    for text in (helper, checked, usage_first):
        earlier = "```python\ndef f(x):\n    return 0\n```"  # an earlier turn's code does not count
        gt = run_sync(HiddenTests().score(_fake_episode(earlier, text), _implement_item()))
        assert gt["hidden_pass_rate"] == 1.0 and gt["role_values"]["worker"] == 1.0, text
    assert extract_code(usage_first) == "def f(x):\n    return x * x"  # examples are not part of the program


# ----------------------------------------------------------------------------- QA: options referring to options

MMLU_ROWS = [  # made up, in the cais/mmlu schema
    {"question": "Which of these are mammals?", "choices": ["Whale", "Bat", "Both A and B", "Neither"], "answer": 2},
    {"question": "Which of these numbers is prime?", "choices": ["4", "6", "7", "All of the above"], "answer": 2},
    {"question": "What is 2 + 3?", "choices": ["4", "5", "6", "7"], "answer": 1},
    {"question": "Which planet is the largest?", "choices": ["Mars", "Jupiter", "Venus", "None of these"], "answer": 1},
]


def test_questions_with_options_that_refer_to_other_options_are_dropped(monkeypatch):
    monkeypatch.setattr(qa, "hf_rows", lambda *a, **k: MMLU_ROWS)
    for binary in (True, False):
        dom = MMLU(subjects="anatomy", binary=binary)
        assert [it.id for it in dom.load()] == ["mmlu-anatomy-test-0002"] and dom.dropped_option_references == 3
    article = {"article_id": "syn2", "title": "T", "source": "synthetic", "article": "A short text.", "questions": [
        {"question": "Why?", "question_unique_id": "syn2_q1", "gold_label": 4,
         "options": ["Because of a", "Because of b", "Because of c", "All of the options are correct"]},
        {"question": "Who?", "question_unique_id": "syn2_q2", "gold_label": 1, "options": ["Ann", "Bob", "Cy", "Di"]}]}
    monkeypatch.setattr(QuALITY, "records", lambda self, split: QuALITY.parse_article(article))
    assert [it.id for it in QuALITY().load()] == ["quality-syn2_q2"]
    for text in ("All of the above", "none of above.", "Both A & C", "(A) and (C)", "A and C only", "Neither.",
                 "Any 2 of the above values", "All of these options.", "neither A nor B; an ethic", "All of them"):
        assert refers_to_options(text), text
    for text in ("All of them go crazy", "Core", "C, C++", "the above-mentioned treaty", "Both a and b are integers",
                 "include all drivers in each of these families", "The only way is to remember all the answers"):
        assert not refers_to_options(text), text


def test_truthfulqa_is_pinned_to_a_commit():
    assert re.search(r"/TruthfulQA/[0-9a-f]{40}/data/mc_task\.json$", qa.TRUTHFULQA_URL)
    assert "/main/" not in qa.TRUTHFULQA_URL


# ----------------------------------------------------------------------------- forecasting: leaked resolutions

def test_resolution_announcements_and_later_updates_are_redacted():
    for leak in ["Update: It's official, resolving 'yes.' Thank you for participating in this market.",
                 "Resolved: YES (announced 2026-09-09).", "RESOLUTION - NO, the bill died in committee.",
                 'Resolved to "NO" per the official results.', "Resolving this market YES because Apple announced it.",
                 "This question has been resolved as NO.", "Criteria are above. Resolving NO since nothing happened."]:
        assert clean_description("Resolves YES if X happens.\n\n" + leak) == ("Resolves YES if X happens.", 1), leak
    for criteria in ["Resolves YES if it rains.", "Resolution: YES if the bill passes, NO otherwise.",
                     "This market will be resolved YES when the official results are published.",
                     "If this is not resolved YES by March, I will resolve NO.", "Resolution: no earlier than June 1."]:
        assert clean_description(criteria) == (criteria, 0), criteria
    desc = ("Resolves YES if the probe lands. EDIT: it landed, see https://news.example/probe-lands\n\n"
            "Update 2026-09-04 (PST) (AI summary of creator comment): the market resolves on the landing.\n\n"
            "Background: [the agency's page](https://agency.example/mission) has the schedule.")
    assert clean_description(desc, resolved=True) == (
        "Resolves YES if the probe lands.\n\nBackground: the agency's page has the schedule.", 2)
    assert clean_description(desc) == (desc, 0)  # open markets: updates are clarifications, not outcomes
    market = dict(id="r9", outcomeType="BINARY", isResolved=True, resolution="YES", uniqueBettorCount=30, volume=5e3,
                  question="Will the probe land on the moon before the start of 2027?", createdTime=1_750_000_000_000,
                  closeTime=1_760_000_000_000, resolutionTime=1_760_000_000_000, textDescription=desc)
    item = ForecastingDomain(source="sample").item_from_market(market)
    assert "landed" not in item.question and "news.example" not in item.question and "probe lands" in item.question
    assert item.metadata["redacted_paragraphs"] == 2


# ----------------------------------------------------------------------------- SQL: hedged answers

def test_sql_answers_must_have_the_shape_of_the_answer(tmp_path):
    db = tmp_path / "shop.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE orders (id INTEGER, status TEXT)")
    con.executemany("INSERT INTO orders VALUES (?, ?)", [(1, "ok"), (2, "ok"), (3, "cancelled"), (4, "ok")])
    con.commit()
    con.close()
    item = TaskItem(id="t", question="How many orders were not cancelled?", private={"db_path": str(db)},
                    ground_truth=GroundTruth(data={"gold_result": {"columns": ["n"], "rows": [[3]]}, "ordered": False,
                                                   "decimals": 0}))
    sc = SQLAnswerScorer()
    hedges = ["SELECT COUNT(*), SUM(status <> 'cancelled'), SUM(status = 'cancelled') FROM orders",
              "SELECT 4, 0, 1, 2, 3, 5, 6"]
    for sql in hedges:
        assert sc.grade_text(f"```sql\n{sql}\n```\nAnswer: 12345", item)["value"] == -1.0, sql
    good = "```sql\nSELECT COUNT(*) FROM orders WHERE status <> 'cancelled'\n```\nAnswer: 3"
    assert sc.grade_text(good, item) == {"value": 1.0, "source": "sql", "sql_error": None}
    # a result with extra columns is no answer: the stated answer (one value) is graded instead
    info = sc.grade_text("```sql\nSELECT COUNT(*), 'orders' FROM orders WHERE status <> 'cancelled'\n```\nAnswer: 3", item)
    assert info["value"] == 1.0 and info["source"] == "text" and "columns" in info["sql_error"]
    assert results_match([[4, 3, 1]], [[3]]) and not results_match([[4, 3, 1]], [[3]], exact_columns=True)
    assert results_match([["b", 2], ["a", 1]], [[1, "a"], [2, "b"]], exact_columns=True)  # columns in any order
    res = QueryResult(columns=["status", "n"], rows=[("ok", 3), ("cancelled", 1)])
    assert expect_matches("ok | 3; cancelled | 1", res)
    assert not expect_matches("ok | 3 | 4; cancelled | 1 | 1", res)  # a claim cannot hedge either
