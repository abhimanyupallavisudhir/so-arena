"""Every agent is told which answer it has been assigned (guards against the 'Propaganda agent never
told its side' class of bug), and judges never see the ground truth."""

import pytest

import so_arena as soa
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.mechanisms import Consultancy, Debate, Propaganda, ReviewedWork


def recording_policy(log, role):
    def act(req, ctx):
        item = req.view.item
        assert item.ground_truth is None and all(a.value is None for a in item.answers or [])  # censored
        assigned = ctx.game.players[role].stance if ctx.game is not None else None
        log.append((role, req.kind, "\n".join(m.content for m in req.prompt), assigned))
        if req.kind == "probabilities":
            return {o: 1 / len(req.options) for o in req.options}
        return "An argument."

    return soa.FunctionPolicy(act, label=role)


@pytest.mark.parametrize("mech", [Propaganda(), Consultancy(rounds=2), Debate(rounds=2, simultaneous=False),
                                  Debate(rounds=1), ReviewedWork(critique_rounds=1)])
@pytest.mark.parametrize("arm", ["true", "false"])
def test_agents_see_their_assigned_answer_and_judges_do_not_see_truth(mech, arm):
    item = soa.binary_item("q", "Which city is the capital of Australia?", correct="Canberra", incorrect="Sydney",
                           shuffle_seed=3)
    log = []
    roles = mech.role_specs()
    players = {}
    agents = [r for r, s in roles.items() if s.kind == "agent" and s.required]
    for r, s in roles.items():
        if not s.required and r not in agents:
            continue
        stance = None
        if r == agents[0]:
            stance = arm
        elif r in agents:
            stance = f"opposite:{agents[0]}"
        players[r] = PlayerSpec(policy=recording_policy(log, r), stance=stance) if stance else recording_policy(log, r)
    if isinstance(mech, ReviewedWork):
        players["critic"] = recording_policy(log, "critic")
    eps = run_sync(run_episodes(mech, [item], [Profile(name="p", players=players)]))
    assert eps[0].error is None, eps[0].error
    for role, kind, text, stance in log:
        if stance is not None and role in agents and kind == "text":
            answer_text = item.answer(stance).text
            assert answer_text in text, f"{mech.name}/{role} was not told its side ({stance}={answer_text})"
    assert any(roles[r].kind == "judge" for r, *_ in log if r in roles)
