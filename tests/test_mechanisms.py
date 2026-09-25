import pytest

import oversight_arena as oa
from oversight_arena.channels import Auditor, EvidencePolicy, Label, SimulatedProbe
from oversight_arena.core.rewards import Combined, JudgeLabelScore, JudgeProbability
from oversight_arena.mechanisms import (
    BTS, DMI, AcceptReward, Consultancy, CorrelatedAgreement, CrossExamination, Debate, Forecast, JudgeRating,
    MarketScoring, Monitoring, MonitoredReward, NaiveJudge, OpenConsultancy, OutputAgreement, PredictionMarket,
    Propaganda, ProperScoring, ProposerCritic, Reporters,
)


def run(domain, mech, agents, profiles=None):
    res = oa.Experiment(domain, mech, agents, profiles, progress=False).run()
    assert not res.errors, res.errors[0].error
    return res


@pytest.mark.parametrize("mech", [
    Debate(), Debate(simultaneous=False, judge_questions=True, rounds=3), Debate(n_debaters=3, rounds=1),
    CrossExamination(), Consultancy(), Consultancy(consultant_first=False), Propaganda(), NaiveJudge(),
])
def test_judge_mechanisms_run_and_reward(toy_domain, scripted_agents, mech):
    res = run(toy_domain, mech, scripted_agents, oa.Stances())
    for r in res.records:
        assert "probs" in r.outcome and "positions" in r.outcome
        assert set(r.rewards) == {x.name for x in r.roles if x.trainable}
        assert r.gt["correct"]


def test_asd_worlds_counterbalance(toy_domain, scripted_agents):
    res = run(toy_domain, Debate(), scripted_agents, oa.Stances())
    labels = {r.profile.label for r in res.records}
    assert len(labels) == 2
    df = res.df(trainable_only=True)
    # each debater argues correct in exactly one world per task
    assert (df.groupby(["task", "role"])["gt_correct"].sum() == 1).all()


def test_open_consultancy_and_proposer_critic(toy_domain, scripted_agents):
    res = run(toy_domain, OpenConsultancy(), scripted_agents, [oa.Profile.of(consultant=oa.HONEST)])
    assert all(r.outcome["answers"]["consultant"] == r.bound["consultant"].target for r in res.records)
    res = run(toy_domain, ProposerCritic(), scripted_agents, [oa.Profile.of(proposer=oa.DISHONEST)])
    r = res.records[0]
    assert "accept_prob" in r.outcome and set(r.rewards) == {"proposer", "critic"}
    assert r.rewards["proposer"] == pytest.approx(__import__("math").log(0.7))


def test_monitoring_with_probe_audit_and_trainable_monitor(toy_domain, scripted_agents):
    mech = Monitoring(monitor="both", cot_visible=True, audit=Auditor(prob=1.0), monitor_trainable=True,
                      probe=SimulatedProbe(auroc=0.9), reward=MonitoredReward())
    res = run(toy_domain, mech, scripted_agents, oa.Stances(roles=["actor"]))
    r = res.records[0]
    assert {"suspicion", "alarm", "task_score", "audit"} <= set(r.outcome)
    assert "monitor" in r.rewards and "actor" in r.rewards
    assert any(c.channel == "audit" for c in r.channels)


def test_simulated_probe_auroc():
    import random

    from oversight_arena.analysis.stats import bootstrap_ci  # noqa: F401

    # empirical AUROC of the probe over many draws should be near the target
    import asyncio
    from oversight_arena.domains.synthetic import HiddenBits
    from oversight_arena.mechanisms.base import EpisodeContext
    from oversight_arena.core.strategy import argue

    task = HiddenBits(n_tasks=1).tasks()[0]
    probe = SimulatedProbe(auroc=0.8)
    good, bad = [], []
    for k in range(400):
        for stance, bucket in [("correct", good), ("incorrect", bad)]:
            b = argue(stance).bind(task, "actor", seed=k)
            ctx = EpisodeContext(mechanism=Monitoring(), task=task, agents={}, bound={"actor": b}, clearances={}, episode_key=f"k{k}{stance}")
            bucket.append(asyncio.run(probe.query(ctx, "actor")))
    auc = sum((g < b_) + 0.5 * (g == b_) for g in good for b_ in bad) / (len(good) * len(bad))
    assert 0.74 < auc < 0.86


def test_peer_prediction_rules(toy_domain, scripted_agents):
    for rule in [OutputAgreement(), BTS()]:
        res = run(toy_domain, Reporters(n=3, predict_peers=isinstance(rule, BTS), reward=rule), scripted_agents)
        assert all(len(r.rewards) == 3 for r in res.records)
    for rule in [CorrelatedAgreement(), DMI()]:
        res = run(toy_domain, Reporters(n=3, reward=rule), scripted_agents)
        assert all(len(r.rewards) == 3 for r in res.records)


def test_forecast_and_market_delayed_rewards():
    from oversight_arena.domains.base import TaskListDomain
    from oversight_arena.domains.forecasting import default_forecast_gt, forecast_task

    pending = [forecast_task(f"q{i}", f"q{i}?", None) for i in range(4)]
    resolved = [forecast_task(f"q{i}", f"q{i}?", float(i % 2)) for i in range(4)]
    dom = TaskListDomain(task_list=pending)
    agents = {"*": oa.ScriptedAgent(lambda o: {"probability": 0.7}, id="f"),
              "kind:judge": oa.ScriptedAgent(lambda o: {"probability": 0.5, "rating": 8}, id="j")}
    res = oa.Experiment(dom, Forecast(), agents, progress=False).run()
    assert all(r.meta.get("rewards_pending") and not r.rewards for r in res.records)
    later = res.resolve(resolved, scorers=default_forecast_gt(), rule=ProperScoring())
    assert all(r.rewards and r.gt_status == "complete" for r in later.records)
    res2 = oa.Experiment(dom, Forecast(judge=True, reward=JudgeRating()), agents, progress=False).run()
    assert all(r.rewards["forecaster"] == pytest.approx(0.8) for r in res2.records)
    res3 = oa.Experiment(TaskListDomain(task_list=resolved), PredictionMarket(), agents, progress=False).run()
    assert all(len(r.outcome["trades"]) == 6 and r.rewards for r in res3.records)


def test_trainable_judge_with_labels(toy_domain, scripted_agents):
    mech = Debate(judge_trainable=True, judge_labels=Label(prob=1.0),
                  reward=Combined(rules=[JudgeProbability(), JudgeLabelScore()]))
    res = run(toy_domain, mech, scripted_agents, oa.Stances())
    assert all("judge" in r.rewards for r in res.records)
    assert all(any(c.channel == "label" for c in r.channels) for r in res.records)


def test_mechanism_decorator(toy_domain, scripted_agents):
    @oa.mechanism(roles=[oa.RoleSpec(name="expert", kind="expert"), oa.RoleSpec(name="judge", kind="judge", trainable=False)],
                  reward=JudgeProbability(transform="linear"))
    async def one_shot(ctx, words: int = 50):
        pos = ctx.positions(["expert"])
        await ctx.ask("expert", f"Argue in {words} words.")
        v = await ctx.ask("judge", "Which?", response=oa.ResponseSpec.distribution(ctx.task.option_ids))
        ctx.set_outcome(probs=v.data["probs"], positions=pos, decision=v.data["choice"])

    m = one_shot(words=20)
    assert m.words == 20 and m.name == "one_shot"
    res = run(toy_domain, m, scripted_agents, oa.Stances())
    assert all(r.rewards["expert"] in (0.7, 0.3) for r in res.records)
