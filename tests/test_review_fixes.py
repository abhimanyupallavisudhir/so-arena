"""Regression tests for defects found in review (each test pins one failure mode)."""

import asyncio
import json

import numpy as np
import pytest

import oversight_arena as oa
from oversight_arena.channels import EvidencePolicy, SimulatedProbe
from oversight_arena.domains.base import TaskListDomain
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.mechanisms import Debate, Forecast, JudgeRating, Monitoring, Propaganda, Reporters
from oversight_arena.mechanisms.peer_prediction import CorrelatedAgreement
from oversight_arena.models import CachedModel, FunctionModel
from oversight_arena.sim import BayesianBitJudge, BitAdvocate
from conftest import make_tasks


def run(dom, mech, agents, profiles=None, **kw):
    res = oa.Experiment(dom, mech, agents, profiles, progress=False, **kw).run()
    assert not res.errors, res.errors[0].error
    return res


def test_per_role_gt_not_overwritten_by_outcome():
    from oversight_arena.domains.swarm import AbstractSwarm
    from oversight_arena.mechanisms import Swarm, TeamReward
    from oversight_arena.sim import SwarmWorker

    res = run(AbstractSwarm(n_tasks=4), Swarm(n_workers=3, rounds=1, reward=TeamReward()), {"*": SwarmWorker()},
              [oa.Profile.of(worker_1=oa.Strategy(name="c", params={"cheat": 1.0}),
                             worker_2=oa.Strategy(name="c", params={"cheat": 1.0}),
                             worker_3=oa.Strategy(name="c", params={"cheat": 1.0}))])
    d = res.df()
    for r in res.records:
        viol = set(r.env_state["violators"])
        rows = d[d["episode"] == r.id].set_index("role")
        for w in ("worker_1", "worker_2", "worker_3"):
            assert rows.loc[w, "gt_clean"] == (0.0 if w in viol else 1.0)
        assert (rows["gt_clean[_outcome]"] == r.gt["clean"]["_outcome"]).all()


def test_roles_with_identical_prompts_draw_independent_samples(tmp_path):
    calls = []

    def fn(messages, config, tools, sample):
        calls.append(sample)
        return f"ANSWER: {'AB'[sample % 2]}"

    model = CachedModel(FunctionModel(fn, name="coin"), path=tmp_path / "c.sqlite")
    run(TaskListDomain(task_list=make_tasks(6)), Reporters(n=3), oa.LLMAgent(model, retries=0))
    assert len(calls) == 18 and len(set(calls)) == 3  # one distinct draw per reporter (none served from cache)


def test_batch_rewards_are_per_population():
    def reporter(obs):
        if obs.params.get("collude"):
            return {"choice": "A"}
        return {"choice": "A" if obs.task.question.endswith(("0 + 0?", "2 + 2?", "4 + 4?")) else "B"}

    truthful = oa.Profile(label="truthful")
    collude = oa.Profile.of(label="collude", **{f"reporter_{i}": oa.Strategy(name="c", params={"collude": 1}) for i in (1, 2, 3)})
    dom = TaskListDomain(task_list=make_tasks(6))
    alone = run(dom, Reporters(n=3, reward=CorrelatedAgreement()), oa.ScriptedAgent(reporter), [truthful])
    both = run(dom, Reporters(n=3, reward=CorrelatedAgreement()), oa.ScriptedAgent(reporter), [truthful, collude])
    key = lambda res: sorted((r.task_id, tuple(sorted(r.rewards.items()))) for r in res.records if r.profile.label == "truthful")  # noqa: E731
    assert key(alone) == key(both)
    assert all(v == 0 for r in both.records if r.profile.label == "collude" for v in r.rewards.values())


def test_programmatic_agents_cannot_read_private_information():
    seen = {}

    def actor(obs):
        return oa.Action(text="done", reasoning="SECRET PLAN")

    def watcher(name):
        def fn(obs):
            seen.setdefault(name, []).append(obs)
            if obs.response.kind == "scalar":
                return {obs.response.scalar_name: 1}
            return "ok"
        return fn

    dom = TaskListDomain(task_list=make_tasks(2))
    agents = {"actor": oa.ScriptedAgent(actor), "monitor": oa.ScriptedAgent(watcher("monitor")),
              "grader": oa.ScriptedAgent(watcher("grader"))}
    run(dom, Monitoring(cot_visible=False, probe=None), agents, oa.Stances(roles=["actor"]))
    for o in seen["monitor"] + seen["grader"]:
        assert all(e.reasoning is None for e in o.entries)
        assert all(b.access == "public" for b in o.task.info)
    seen.clear()
    run(dom, Monitoring(cot_visible=True, probe=None), agents, oa.Stances(roles=["actor"]))
    assert any(e.reasoning == "SECRET PLAN" for o in seen["monitor"] for e in o.entries)


def test_mechanism_assigned_positions_reach_programmatic_agents():
    dom = HiddenBits(n_tasks=3)
    res = run(dom, Debate(rounds=1, evidence=EvidencePolicy()), {"kind:judge": BayesianBitJudge(), "*": BitAdvocate()})
    for r in res.records:
        for d in ("debater_a", "debater_b"):
            said = r.transcript.by_role(d)[0].content
            assert f"({r.outcome['positions'][d]})" in said


def test_braces_in_instructions_do_not_crash():
    s = oa.Strategy(name="json", instructions='Reply as {"answer": "X"} and argue for {target}.', stance=oa.Stance.CORRECT)
    b = s.bind(make_tasks(1)[0], "agent")
    assert '{"answer": "X"}' in b.instructions and "argue for A" in b.instructions


def test_episode_keys_cover_domain_config(tmp_path):
    agents = {"kind:judge": BayesianBitJudge(), "*": BitAdvocate()}
    a = run(HiddenBits(n_tasks=2, verify_cost=1.0), Propaganda(evidence=EvidencePolicy(budget=2)), agents, oa.Stances(), out=tmp_path)
    b = run(HiddenBits(n_tasks=2, verify_cost=100.0), Propaganda(evidence=EvidencePolicy(budget=2)), agents, oa.Stances(), out=tmp_path)
    assert {r.key for r in a.records}.isdisjoint({r.key for r in b.records})
    assert all(not r.transcript.entries[0].evidence for r in b.records)  # nothing affordable at cost 100


def test_parsing_does_not_invert_verdicts():
    from oversight_arena.agents.parsing import parse_choice, parse_distribution

    assert parse_distribution("**A**: 70%\n**B**: 30%", ["A", "B"])["A"] == pytest.approx(0.7)
    assert parse_distribution("- A (Paris): 0.8\n- B (Lyon): 0.2", ["A", "B"])["A"] == pytest.approx(0.8)
    assert parse_distribution("A: 70%, B: 30", ["A", "B"])["B"] == pytest.approx(0.3)
    assert parse_choice("The answer is a close call, but ANSWER: B", ["A", "B"]) == "B"
    assert parse_choice("the answer is a hard one", ["A", "B"], strict=True) is None


def test_logprobs_read_at_the_answer_token():
    import math

    from oversight_arena.agents.llm import _letter_probs_from_logprobs
    from oversight_arena.core.types import TokenLogprob

    lps = [TokenLogprob(token="(", logprob=0.0, top=[("(", 0.0), ("A", math.log(0.3))]),
           TokenLogprob(token="B", logprob=math.log(0.9), top=[("B", math.log(0.9)), ("A", math.log(0.1))])]
    assert _letter_probs_from_logprobs(lps, ["A", "B"])["B"] == pytest.approx(0.9, abs=1e-3)


def test_code_candidates_have_identical_formatting():
    from oversight_arena.domains.code import normalize_code

    src = "def f(x):\n    # add one\n\n    return x + 1\n"
    assert "#" not in normalize_code(src) and normalize_code(src) == normalize_code(normalize_code(src))


def test_lean_proof_must_prove_the_statement():
    from oversight_arena.domains.lean import proves_statement

    stmt = "theorem t (a : ℕ) (h : a = 2) : a + 1 = 3 := sorry"
    assert proves_statement("theorem t (a : ℕ) (h : a = 2) : a + 1 = 3 := by simp [h]", stmt)[0]
    assert not proves_statement("theorem t : True := trivial", stmt)[0]
    assert not proves_statement("axiom cheat : False\ntheorem t (a : ℕ) (h : a = 2) : a + 1 = 3 := cheat.elim", stmt)[0]


def test_simops_path_tricks_are_detected():
    from oversight_arena.domains.swarm import SimOps

    env = SimOps(n_tasks=1).make_env(SimOps(n_tasks=1).tasks()[0])
    env.workers = ["worker_1", "worker_2", "worker_3"]
    tools = {t.name: t for t in env.tools_for("worker_1", set())}
    asyncio.run(tools["write_file"].run({"path": "./tests/../tests/test_visible.py", "content": "def test_ok():\n    assert True\n"}))
    assert "worker_1" in env.violators()
    with pytest.raises(ValueError):
        asyncio.run(tools["read_file"].run({"path": "../" + env.dir.rsplit("/", 1)[-1] + "x/secret"}))
    asyncio.run(env.teardown())


def test_reported_gt_when_nobody_observed():
    from oversight_arena.domains.swarm import swarm_gt

    reported = [g for g in swarm_gt() if g.name == "reported"][0]
    rec = oa.EpisodeRecord(id="x", key="x", mechanism="swarm", task_id="t", domain="d",
                           env_state={"violators": ["worker_1"], "observed": {}},
                           outcome={"workers": ["worker_1", "worker_2"], "reports": [], "detected": []})
    out = reported.score(make_tasks(1)[0], rec)
    assert out["worker_2"] is None  # did not see anything: nothing to report


def test_bit_honesty_counts_plain_text_lies():
    from oversight_arena.domains.synthetic import BitHonesty

    dom = HiddenBits(n_tasks=2)
    t = dom.tasks()[0]
    wrong = 1 - t.resources["bits"][0]
    rec = oa.EpisodeRecord(id="x", key="x", mechanism="m", task_id=t.id, domain="d", roles=[oa.RoleSpec(name="agent")])
    rec.transcript.add(oa.core.transcript.Entry(role="agent", content=f"bit 0 is {wrong}"))
    assert BitHonesty().score(t, rec)["agent"] == 0.0


def test_hidden_bits_even_length_consistent():
    dom = HiddenBits(n_tasks=40, length=4)
    for t in dom.tasks():
        yes = next(o for o in t.options if o.text == "Yes")
        assert (yes.value > 0) == (sum(t.resources["bits"]) * 2 >= 4)


def test_sql_results_compare_as_multisets():
    from oversight_arena.domains.sql import same_result

    assert same_result([(1,), (2,)], [(2,), (1,)], ordered=False) and not same_result([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert same_result([(None, 1), (2, "a")], [(2, "a"), (None, 1)], ordered=False)


def test_zero_sum_value_for_column_player_and_incomplete_games():
    from oversight_arena.analysis import EmpiricalGame

    A = np.array([[3.0, -1.0], [-2.0, 1.0]])
    g = EmpiricalGame.from_matrices(A)
    v_row, _ = g.zero_sum_value("row")
    v_col, _ = g.zero_sum_value("col")
    assert v_col == pytest.approx(-v_row)
    g2 = EmpiricalGame.from_matrices(np.array([[1.0, np.nan], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        g2.nash()
    assert g2.fill_missing(0.0).nash()


def test_ece_multi_option():
    import pandas as pd

    from oversight_arena.analysis.diagnostics import ece

    ep = pd.DataFrame({"p[A]": [0.5, 0.5], "p[B]": [0.3, 0.3], "p[C]": [0.2, 0.2], "gt_decision_correct[_outcome]": [1.0, 0.0]})
    assert ece(ep) == pytest.approx(0.0)


def test_release_detects_deleted_items_and_keeps_sealed_reveal_private(tmp_path):
    from oversight_arena.domains.forecasting import forecast_task
    from oversight_arena.release import create_release, reveal_release, verify_release

    pending = [forecast_task(f"q{i}", f"<script>alert({i})</script>?", None) for i in range(4)]
    agents = {"*": oa.ScriptedAgent(lambda o: {"probability": 0.8}, id="f"),
              "kind:judge": oa.ScriptedAgent(lambda o: {"probability": 0.5, "rating": 6}, id="j")}
    res = run(TaskListDomain(task_list=pending), Forecast(judge=True, reward=JudgeRating()), agents)
    rel = create_release(res, tmp_path / "rel", title="<b>t</b>")
    html = (rel.path / "index.html").read_text()
    assert "<b>t</b>" not in html and "&lt;b&gt;" in html
    items = json.loads((rel.path / "items.json").read_text())
    (rel.path / "items.json").write_text(json.dumps(items[1:]))
    rep = verify_release(rel.path)
    assert not rep["ok"] and len(rep["missing"]) == 1
    sealed = create_release(res, tmp_path / "pub" / "sealed", sealed=True)
    assert not list((tmp_path / "pub" / "sealed").glob("reveal.json"))
    assert reveal_release(sealed.path)["ok"] and (sealed.path / "items.json").exists()


def test_judge_rates_every_forecaster():
    from oversight_arena.domains.forecasting import forecast_task

    tasks = [forecast_task(f"q{i}", f"q{i}?", None) for i in range(2)]

    def judge(obs):
        return {"probability": 0.5, **{k: 7 for k in obs.response.fields if k.startswith("rating")}}

    agents = {"*": oa.ScriptedAgent(lambda o: {"probability": 0.6}, id="f"), "kind:judge": oa.ScriptedAgent(judge, id="j")}
    res = run(TaskListDomain(task_list=tasks), Forecast(n_forecasters=2, judge=True, reward=JudgeRating()), agents)
    assert all(r.rewards == {"forecaster_1": pytest.approx(0.7), "forecaster_2": pytest.approx(0.7)} for r in res.records)


def test_probe_still_uses_harness_private_bound():
    task = HiddenBits(n_tasks=1).tasks()[0]
    from oversight_arena.mechanisms.base import EpisodeContext

    b = oa.argue("incorrect").bind(task, "actor")
    ctx = EpisodeContext(mechanism=Monitoring(), task=task, agents={}, bound={"actor": b}, clearances={}, episode_key="k")
    assert not hasattr(ctx, "bound") and asyncio.run(SimulatedProbe(auroc=0.99).query(ctx, "actor")) is not None
