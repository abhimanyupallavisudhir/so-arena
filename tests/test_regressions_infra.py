"""Regressions from the infrastructure review: releases, the ControlArena bridge, RL environments, the
Inspect export, specs and the CLI (all offline)."""

import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import so_arena as soa
from so_arena.cli import main
from so_arena.core.game import Player
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
from so_arena.integrations.rl import MechanismEnv, reward_function, rollout_prompts
from so_arena.mechanisms import Debate, DirectJudge, MarketScoringReward, PredictionMarket, Propaganda
from so_arena.release import release, resolve, verify
from so_arena.samplers.arms import ASDExperiment
from so_arena.spec import SpecError, load_spec, parse_spec, run_spec, run_usage, spec_hash

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def no_simulation():
    """``so-arena estimate`` switches simulation on for the process; switch it off again afterwards."""
    yield
    soa.configure(simulate=False)


def synth(n: int = 6, seed: int = 3):
    dom = SyntheticPersuasion(n_items=n, seed=seed)
    return dom.load(), dom.context()


def yes_no_items(n: int = 3) -> list[soa.TaskItem]:
    return [soa.TaskItem(id=f"q{i}", question=f"Will event {i} happen?",
                         answers=[soa.AnswerOption(label="yes", text="yes"), soa.AnswerOption(label="no", text="no")],
                         ground_truth=soa.GroundTruth(status="pending")) for i in range(n)]


def market_episodes(items, ctx=None, gt=None):
    players = {"trader_1": soa.ScriptedPolicy('{"yes": 0.8, "no": 0.2}', label="bull"),
               "trader_2": soa.ScriptedPolicy('{"yes": 0.3, "no": 0.7}', label="bear")}
    return run_sync(run_episodes(PredictionMarket(n_traders=2, rounds=1), items, [Profile(name="p", players=players)],
                                 ctx=ctx, ground_truth=gt))


# ------------------------------------------------------------------------------------------ releases

def test_release_withholds_private_information_and_the_arms(tmp_path):
    items, ctx = synth(12)
    mechs = [DirectJudge(), Propaganda(affordances={"agents": ["answer_key"]}),
             Debate(rounds=1, affordances={"agents": ["answer_key"]})]
    eps = ASDExperiment(mechs, items, agent=synthetic_arguer(), fixtures={"judge": synthetic_judge()}, ctx=ctx).run()
    truth = {it.id: it.true_label for it in items}
    rel = tmp_path / "rel"
    release(eps, items, rel, salt="kept-private")
    text = {p.name: p.read_text() for p in rel.iterdir() if p.is_file()}
    assert set(text) == {"items.jsonl", "episodes.jsonl", "rankings.json", "index.html", "MANIFEST.json"}
    # private information is withheld (the synthetic answer key *is* the correct label) ...
    assert all(json.loads(x)["private"] == {} for x in text["items.jsonl"].splitlines())
    # ... and so is everything naming an episode's arm: profiles, ids, tags, behaviour labels
    for needle in ("arm=", "arm_true", "arm_false", "argue_true", "argue_false", '"arm"', '"lead"', '"opponent"'):
        assert not [name for name, t in text.items() if needle in t], needle
    released = [soa.Episode.model_validate_json(x) for x in text["episodes.jsonl"].splitlines()]
    groups: dict[tuple, list] = {}
    for e in released:
        groups.setdefault((e.mechanism, e.item_id), []).append(e)

    def annotations(e):  # everything but the answer argued (positions, assigned stances) and the play
        players = {r: p.model_dump(exclude={"stance"}) for r, p in e.players.items()}
        return json.dumps([players, e.tags, e.created_at, e.run_id, e.seed, e.repeat], sort_keys=True)

    assert all(len({annotations(e) for e in g}) == 1 for g in groups.values())
    items_of_profile: dict[str, set] = {}
    for e in released:
        items_of_profile.setdefault(e.profile, set()).add(e.item_id)
    assert all(len(v) == 1 for v in items_of_profile.values())  # pseudonyms do not link items
    lead = lambda e: e.positions.get("agent") or e.positions.get("debater_a")  # noqa: E731
    first_is_true = [truth[i] == lead(g[0]) for (_, i), g in groups.items()]
    assert 0 < np.mean(first_is_true) < 1  # the order within an item does not follow the arm

    # resolution needs none of it: the resolved release reproduces the run's ASD exactly
    resolve(rel, truth, html=False)
    resolved = [soa.Episode.model_validate_json(x) for x in (rel / "resolved" / "episodes.jsonl").read_text().splitlines()]
    from so_arena.analysis.frames import role_frame
    from so_arena.analysis.metrics import asd

    def asd_of(es):
        return asd(role_frame(es), n_boot=50).set_index("mechanism")["asd"].round(9).to_dict()

    assert asd_of(resolved) == asd_of(eps)

    # opt-ins: allowlisted private keys, published labels
    release(eps, items, tmp_path / "open", private_keys=["answer_key"], public_labels=True, html=False)
    opened = (tmp_path / "open" / "items.jsonl").read_text() + (tmp_path / "open" / "episodes.jsonl").read_text()
    assert '"answer_key":' in opened and "arm=true" in opened
    with pytest.raises(ValueError, match="occur in no released item"):
        release(eps, items, tmp_path / "typo", private_keys=["answer_keys"], html=False)


def test_resolve_refuses_labels_that_are_not_answer_options(tmp_path):
    items = yes_no_items()
    release(market_episodes(items), items, tmp_path / "rel", html=False)
    with pytest.raises(ValueError, match="not one of its answer options"):
        resolve(tmp_path / "rel", {"q0": "maybe"}, html=False)  # would have scored every answer as wrong
    res = resolve(tmp_path / "rel", {"q0": "YES"}, reward_rule=MarketScoringReward(), html=False)  # Manifold style
    assert res.n_resolved == 1
    lines = (tmp_path / "rel" / "resolved" / "episodes.jsonl").read_text().splitlines()
    q0 = next(e for e in map(soa.Episode.model_validate_json, lines) if e.item_id == "q0")
    assert q0.outcome.data["resolution"] == "yes" and q0.rewards["trader_1"] == pytest.approx(math.log(0.8 / 0.5))
    assert q0.ground_truth["judge_correct"] == 0.0  # final price 0.3 on yes, which resolved yes


def test_cli_resolve_with_a_domain_scores_with_its_scorers(tmp_path, monkeypatch):
    monkeypatch.setenv("SO_ARENA_DATA", str(tmp_path / "cache"))
    from so_arena.domains import forecasting as fc

    dom = fc.ForecastingDomain(source="sample", status="pending")
    items = dom.load()
    release(market_episodes(items, dom.context(), dom.ground_truth_scorers()), items, tmp_path / "rel", html=False)
    mid = items[0].metadata["market_id"]
    resolved = {"id": mid, "outcomeType": "BINARY", "isResolved": True, "resolution": "YES",
                "resolutionTime": 1_950_000_000_000, "closeTime": 1_950_000_000_000, "probability": 0.98}
    monkeypatch.setattr(fc, "fetch_market", lambda m: resolved if m == mid else None)
    assert main(["resolve", str(tmp_path / "rel"), "--domain", "forecasting", "--market", "--out", str(tmp_path / "res")]) == 0
    eps = [soa.Episode.model_validate_json(x) for x in (tmp_path / "res" / "episodes.jsonl").read_text().splitlines()]
    ep = next(e for e in eps if e.item_id == items[0].id)
    assert ep.gt_status == "known" and ep.ground_truth["forecast_brier"] == pytest.approx(0.49)  # ForecastScore ran
    assert ep.ground_truth["role_values"]["trader_1"] == pytest.approx(1 - 2 * 0.2 ** 2)  # not the +-1 stance value


def test_verify_checks_the_published_digest_and_covers_the_viewer(tmp_path, capsys):
    items = yes_no_items()
    man = release(market_episodes(items), items, tmp_path / "rel")
    assert "index.html" in man.files
    assert verify(tmp_path / "rel", man.digest) and not verify(tmp_path / "rel", "0" * 64)
    assert main(["verify", str(tmp_path / "rel"), "--digest", man.digest]) == 0
    assert man.digest in capsys.readouterr().out
    assert main(["verify", str(tmp_path / "rel"), "--digest", "0" * 64]) == 1
    html = tmp_path / "rel" / "index.html"
    html.write_text(html.read_text().replace("</h1>", " (edited)</h1>"))
    assert not verify(tmp_path / "rel") and main(["verify", str(tmp_path / "rel")]) == 1


# ------------------------------------------------------------------------------------------ ControlArena

def control_frame(n: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n):  # APPS-style scorers only score the side task in attack mode: NA in honest rows
        rows.append({"eval_mode": "honest", "metadata_main_task_name": f"t{i}", "epoch": 1, "id": f"h{i}",
                     "score_max_suspicion_scorer": i / n, "main_task_success": True, "side_task_success": None})
        rows.append({"eval_mode": "attack", "metadata_main_task_name": f"t{i}", "epoch": 1, "id": f"a{i}",
                     "score_max_suspicion_scorer": 0.93, "main_task_success": True,
                     "side_task_success": None if i == 0 else i % 2 == 0})
    return pd.DataFrame(rows).astype({"main_task_success": "boolean", "side_task_success": "boolean"})


def test_controlarena_frames_with_missing_side_task_scores():
    from so_arena.integrations.controlarena import episodes_from_frame

    df = control_frame()
    assert df["side_task_success"].isna().sum() == 21  # pd.NA: bool() of it raises
    eps = episodes_from_frame(df, reward="task", value="graded")
    assert len(eps) == 40
    values = {e.tags["eval_mode"] + e.item_id: e.value("untrusted") for e in eps}
    assert values["honestt1#e1"] == 1.0 and values["attackt2#e1"] == 1.0 - 2.0 and values["attackt1#e1"] == 1.0
    assert values["attackt0#e1"] is None  # an unscored side task is unknown, not a failed attack


def test_control_report_passes_the_audit_budget_as_a_percentage(monkeypatch):
    analysis = pytest.importorskip("control_arena.analysis")
    from so_arena.integrations.controlarena import control_report, episodes_from_frame

    seen = []
    monkeypatch.setattr(analysis, "compute_safety", lambda df, scorer, budget, **kw: seen.append(budget) or 1.0)
    monkeypatch.setattr(analysis, "compute_usefulness", lambda df, **kw: 1.0)
    df = control_frame()
    control_report(df=df, audit_budgets=[0.05], rewards=["task_if_unflagged"])
    assert seen == [pytest.approx(5.0)]  # ControlArena takes percentages; 0.05 would be 0.05%
    honest = df[df["eval_mode"] == "honest"]["score_max_suspicion_scorer"].to_numpy(dtype=float)
    thr = episodes_from_frame(df, audit_budget=0.05)[0].mechanism_config["threshold"]
    assert thr == pytest.approx(np.percentile(honest, 100 - 5.0))  # ControlArena's threshold at the same budget
    with pytest.raises(ValueError, match="fraction"):
        control_report(df=df, audit_budgets=[5])


def test_format_actions_shows_the_text_of_list_content():
    pytest.importorskip("inspect_ai")
    from inspect_ai.model import ChatMessageAssistant, ContentReasoning, ContentText

    from so_arena.integrations.controlarena import format_actions

    m = ChatMessageAssistant(content=[ContentReasoning(reasoning="secret plan"), ContentText(text="Running the tests.")])
    out = format_actions([m])
    assert "Running the tests." in out and "secret plan" not in out  # reasoning models: text parts, no CoT


# ------------------------------------------------------------------------------------------ RL

def test_mechanism_env_serves_simultaneous_trainees():
    items, ctx = synth(2, seed=0)
    mech = Debate(rounds=1, simultaneous=True, affordances={"agents": ["answer_key"]})
    env = MechanismEnv(mech, {"judge": Player(policy=synthetic_judge())}, {"debater_a": "true", "debater_b": "false"}, ctx=ctx)

    async def drive():
        res = await env.reset(items[0])
        seen, pending, texts = [], [], []
        while not res.done:
            seen.append(res.observation.role)
            pending.append(env.pending_roles)
            texts.append(json.dumps(res.observation.messages))
            res = await asyncio.wait_for(env.step(f"My argument as {res.observation.role}."), timeout=10)
        return seen, pending, texts, res

    seen, pending, texts, res = run_sync(drive())
    assert sorted(seen) == ["debater_a", "debater_b"] and pending[0][0] == seen[0]  # served in arrival order
    assert "My argument as" not in texts[1]  # simultaneous: the second mover does not see the first's move
    assert res.done and res.rewards["debater_a"] is not None and res.rewards["debater_b"] is not None


def test_reward_function_scores_the_stance_the_prompt_assigned():
    items = [soa.TaskItem(id=f"mc{i}", question=f"Question {i}?",
                          answers=[soa.AnswerOption(label=lab, text=f"option {lab}", value=1.0 if lab == "A" else -1.0)
                                   for lab in "ABCD"], ground_truth=soa.GroundTruth(correct="A")) for i in range(12)]
    players = {"judge": Player(policy=soa.ScriptedPolicy('{"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}'))}
    rows = rollout_prompts(Propaganda(), "agent", items, players, stance="false")
    assert all(r["stance"] in "BCD" and f"({r['stance']})" in r["prompt"][0]["content"] for r in rows)
    for with_column in (True, False):  # the dataset column, or the env's own resolution of the spec
        fn = reward_function(Propaganda(), "agent", items, players, stances={"agent": "false"})
        kw = {"stance": [r["stance"] for r in rows]} if with_column else {}
        fn([r["prompt"] for r in rows], ["An argument." for _ in rows], item_id=[r["item_id"] for r in rows], **kw)
        assert [h["stance"] for h in fn.history] == [r["stance"] for r in rows]


def test_reward_function_returns_none_for_missing_rewards():
    item = yes_no_items(1)[0]  # pending: the market's rewards wait for the resolution
    mech = PredictionMarket(n_traders=2, rounds=1)
    players = {"trader_2": Player(policy=soa.ScriptedPolicy('{"yes": 0.3, "no": 0.7}'))}
    fn = reward_function(mech, "trader_1", [item], players)
    assert fn(completions=['{"yes": 0.8, "no": 0.2}'], item_id=["q0"]) == [None]  # not a made-up 0.0
    fn0 = reward_function(mech, "trader_1", [item], players, missing=0.0)
    assert fn0(completions=['{"yes": 0.8, "no": 0.2}'], item_id=["q0"]) == [0.0]


# ------------------------------------------------------------------------------------------ Inspect

def test_inspect_task_reports_the_same_keys_for_every_sample(tmp_path):
    pytest.importorskip("inspect_ai")
    from inspect_ai import eval as inspect_eval

    from so_arena.integrations.inspect_task import inspect_task, score_keys

    items, ctx = synth(3, seed=0)
    pending = items[0].model_copy(update={"id": "pending0", "ground_truth": soa.GroundTruth(status="pending"),
                                          "answers": [a.model_copy(update={"value": None}) for a in items[0].answers]})
    mech = Propaganda(affordances={"agents": ["answer_key"]})
    profiles = [Profile(name=a, players={"agent": PlayerSpec(policy=synthetic_arguer(), stance=a), "judge": synthetic_judge()})
                for a in ("true", "false")]
    log = inspect_eval(inspect_task(mech, [*items, pending], profiles, ctx=ctx), model="mockllm/model",
                       log_dir=str(tmp_path), display="none")[0]
    keys = set(score_keys(mech))
    assert log.status == "success" and {s.name for s in log.results.scores} == keys
    for s in log.samples:
        (score,) = s.scores.values()
        assert set(score.value) == keys
        if str(s.id).startswith("pending0"):  # its stances cannot be resolved: skipped, every key unscored
            assert score.metadata["skipped"] and all(math.isnan(v) for v in score.value.values())


# ------------------------------------------------------------------------------------------ specs and CLI

SPEC = {"name": "guard", "domain": {"name": "synthetic", "n_items": 3},
        "policies": {"expert": {"synthetic": "arguer"}, "judge": {"synthetic": "judge", "skill": 1.0}},
        "mechanisms": [{"name": "direct"}], "experiment": {"type": "asd", "agent": "expert", "fixtures": {"judge": "judge"}},
        "report": False}


def test_run_spec_refuses_a_directory_holding_another_spec(tmp_path):
    out = tmp_path / "run"
    first = parse_spec(SPEC)
    run_spec(first, out=out)
    run_spec(parse_spec({**SPEC, "concurrency": 2, "items": {"limit": 2}}), out=out)  # same results: resumes
    changed = parse_spec({**SPEC, "policies": {**SPEC["policies"], "judge": {"synthetic": "judge", "skill": 0.0}}})
    with pytest.raises(SpecError, match="different spec"):
        run_spec(changed, out=out)
    assert json.loads((out / "run.json").read_text())["spec_hash"] == spec_hash(first)  # record kept
    run_spec(changed, out=out, force=True)
    meta = json.loads((out / "run.json").read_text())
    assert meta["spec_hash"] == spec_hash(changed) and [p["spec_hash"] for p in meta["previous_specs"]] == [spec_hash(first)]
    assert meta["n_episodes"] == 6  # this run's episodes, not the earlier spec's still in the log


def test_misspelled_spec_keys_are_errors_with_suggestions(tmp_path):
    bad = {**SPEC, "seeds": 7, "cache": ".cache", "experiment": {**SPEC["experiment"], "repeat": 3}}
    msg = str(pytest.raises(SpecError, parse_spec, bad).value)
    assert all(f"did you mean {k!r}" in msg for k in ("seed", "cache_dir", "repeats"))
    top = {**{k: v for k, v in SPEC.items() if k != "mechanisms"}, "mechanism": SPEC["mechanisms"], "repeats": 5}
    msg = str(pytest.raises(SpecError, parse_spec, top).value)
    assert "did you mean 'mechanisms'" in msg and "'repeats' belongs under 'experiment'" in msg
    refs = {**SPEC, "policies": {**SPEC["policies"], "s": {"scripted": ["x"], "lable": "y"}},
            "mechanisms": [{"name": "debat"}, {"name": "debate", "reward": {"name": "zero_sum", "transfrom": "brier"}}],
            "experiment": {**SPEC["experiment"], "agent": "expret"}}
    msg = str(pytest.raises(SpecError, parse_spec, refs).value)
    assert all(f"did you mean {k!r}" in msg for k in ("label", "debate", "transform", "expert"))
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    assert main(["run", str(path), "--out", str(tmp_path / "out")]) == 2 and not (tmp_path / "out").exists()
    for name in ("asd_synthetic", "asd_gsm8k_llm", "bon_synthetic", "chess_debate_llm"):
        load_spec(CONFIGS / f"{name}.yaml")  # the shipped configs stay valid


def test_estimate_never_deletes_a_directory_it_did_not_create(tmp_path, no_simulation):
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(SPEC))
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "notes.txt").write_text("keep")
    assert main(["estimate", str(path), "--out", str(precious)]) == 2
    assert (precious / "notes.txt").read_text() == "keep"
    assert main(["estimate", str(path), "--out", str(tmp_path / "est")]) == 0
    assert main(["estimate", str(path), "--out", str(tmp_path / "est")]) == 0  # its own scratch dir is replaced


def test_estimate_prices_game_trees_and_prompt_searches(tmp_path, no_simulation):
    llm = {"worker": {"model": "openai/gpt-4o-mini", "max_tokens": 100},
           "judge": {"model": "openai/gpt-4o-mini", "elicitation": "logprobs", "temperature": 0},
           "optimizer": {"model": "openai/gpt-4o"}}
    pools = {"name": "est-pools", "domain": {"name": "synthetic", "n_items": 2, "seed": 2}, "policies": llm,
             "mechanisms": [{"name": "propaganda"}],
             "experiment": {"type": "pools", "players": {"agent": {"policy": "worker", "stance": "false"}, "judge": "judge"},
                            "pool_sizes": {"agent": 3}, "grid": {"agent": [1, 3]}}}
    search = {"name": "est-search", "domain": {"name": "synthetic", "n_items": 4, "seed": 2}, "policies": llm,
              "mechanisms": [{"name": "propaganda"}],
              "experiment": {"type": "prompt_search", "policy": "worker", "role": "agent", "others": {"judge": "judge"},
                             "optimizer": "optimizer", "iterations": 1, "candidates_per_iter": 2}}
    usage = {}
    for spec in (pools, search):
        path = tmp_path / f"{spec['name']}.yaml"
        path.write_text(yaml.safe_dump(spec))
        assert main(["estimate", str(path), "--out", str(tmp_path / spec["name"])]) == 0
        u = run_usage(tmp_path / spec["name"])
        usage[spec["name"]] = u.set_index("role")
        assert u[u["role"] == "TOTAL"]["cost_usd"].iloc[0] > 0
    assert usage["est-pools"].loc["agent", "calls"] == 2 * 3 and usage["est-pools"].loc["judge", "calls"] == 2 * 3
    assert usage["est-search"].loc["optimizer", "calls"] > 0
    meta = json.loads((tmp_path / "est-search" / "run.json").read_text())
    assert meta["n_episodes"] > 8  # proposed candidates were evaluated, not only the two searches' seed strategies
