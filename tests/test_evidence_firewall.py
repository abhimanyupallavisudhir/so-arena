import asyncio

import oversight_arena as oa
from oversight_arena.channels import EvidencePolicy, extract_claims
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.mechanisms import Debate, Propaganda
from oversight_arena.sim import BayesianBitJudge, BitAdvocate


def test_claim_extraction_and_annotation():
    cs = extract_claims('x <bit i="3">1</bit> y <quote>abc</quote>', ["bit", "quote"])
    assert [(c.kind, c.content, c.args) for c in cs] == [("bit", "1", {"i": "3"}), ("quote", "abc", {})]


def test_budget_and_statuses():
    dom = HiddenBits(n_tasks=3)
    agents = {"kind:judge": BayesianBitJudge(), "*": BitAdvocate(claims=3)}
    res = oa.Experiment(dom, Propaganda(evidence=EvidencePolicy(budget=2)), agents, oa.Stances(), progress=False).run()
    for r in res.records:
        claims = [c for e in r.transcript.entries for c in e.data.get("claims", [])]
        assert [c["status"] for c in claims][:2] == ["VERIFIED", "VERIFIED"]
        assert claims[2]["status"] == "UNCHECKED"


def test_verification_noise_flips():
    dom = HiddenBits(n_tasks=10)
    agents = {"kind:judge": BayesianBitJudge(), "*": BitAdvocate(claims=3)}
    res = oa.Experiment(dom, Propaganda(evidence=EvidencePolicy(noise=1.0)), agents, oa.Stances(), progress=False).run()
    evs = [ev for r in res.records for e in r.transcript.entries for ev in e.evidence]
    assert evs and all(ev.data.get("flipped") for ev in evs)


def test_liars_get_refuted_and_honesty_gt():
    dom = HiddenBits(n_tasks=10)
    agents = {"kind:judge": BayesianBitJudge(), "*": BitAdvocate(claims=6, lie_rate=1.0, lie_first=True)}
    res = oa.Experiment(dom, Propaganda(evidence=EvidencePolicy()), agents, [oa.Profile.of(agent=oa.DISHONEST)], progress=False).run()
    assert any(c["status"] == "REFUTED" for r in res.records for e in r.transcript.entries for c in e.data.get("claims", []))
    assert all(r.gt["honesty"]["agent"] < 1 for r in res.records)


def test_ground_truth_firewall_in_observations(toy_domain):
    seen = []

    def spy(obs):
        seen.append(obs)
        if obs.response.kind == "distribution":
            return {"probs": {o: 0.5 for o in obs.response.options}}
        return "ok"

    agents = {"*": oa.ScriptedAgent(spy)}
    oa.Experiment(toy_domain, Debate(rounds=1), agents, oa.Stances(), progress=False).run()
    for o in seen:
        dumped = o.model_dump_json()
        assert "hidden" not in dumped  # task.gt never reaches agents
        assert '"value"' not in o.task.model_dump_json()
        if o.role == "judge":
            assert "answer is" not in o.task_text and "answer" not in o.private  # no clearance
        else:
            assert "answer is" in o.task_text and o.private.get("answer") is not None


def test_judge_can_request_verification():
    dom = HiddenBits(n_tasks=2)
    calls = {}

    def judge(obs):
        calls["tools"] = [t.name for t in obs.tools]
        if obs.response.kind == "distribution":
            res = asyncio.get_event_loop  # noqa: F841
            return {"probs": {o: 0.5 for o in obs.response.options}}
        return "q"

    agents = {"kind:judge": oa.ScriptedAgent(judge), "*": BitAdvocate()}
    oa.Experiment(dom, Debate(rounds=1, evidence=EvidencePolicy(requests=["judge"])), agents, oa.Stances(), progress=False).run()
    assert "verify_bit" in calls["tools"]
