"""Test-suite settings for both packages: so_arena (test_*.py) and oversight_arena (test_oa_*.py)."""

import os
import shutil

import pytest

import oversight_arena as oa
from oversight_arena.domains.base import TaskListDomain
from so_arena.core import sandbox

# Where no sandbox backend works, the suite still exercises agent commands and untrusted code, unsandboxed
# (the sandbox's own tests are skipped there); elsewhere every such command runs sandboxed, as in use.
if not sandbox.available():
    os.environ.setdefault(sandbox.ALLOW_ENV, "1")

# oversight_arena: no persistent response cache in tests (read when a model is created)
os.environ.setdefault("OA_CACHE", "0")


def pytest_collection_modifyitems(config, items):
    net = os.environ.get("OA_NETWORK_TESTS") == "1"
    for item in items:
        if not item.path.name.startswith("test_oa_"):  # oversight_arena's tests only
            continue
        if "network" in item.keywords and not net:
            item.add_marker(pytest.mark.skip(reason="set OA_NETWORK_TESTS=1 to run network tests"))
        if "stockfish" in item.keywords and not (shutil.which("stockfish") or os.path.exists("/usr/games/stockfish")):
            item.add_marker(pytest.mark.skip(reason="stockfish not installed"))


def make_tasks(n=6):
    ts = []
    for i in range(n):
        correct_first = i % 2 == 0
        ts.append(oa.Task(
            id=f"t{i}", domain="toy", question=f"What is {i} + {i}?",
            options=[oa.Answer(id="A", text=str(2 * i if correct_first else 2 * i + 1), value=1.0 if correct_first else -1.0),
                     oa.Answer(id="B", text=str(2 * i + 1 if correct_first else 2 * i), value=-1.0 if correct_first else 1.0)],
            info=[oa.InfoBlock(key="secret", content=f"the answer is {2 * i}", access="private")],
            resources={"answer": 2 * i}, resource_access={"answer": "private"},
            gt={"note": "hidden"},
        ))
    return ts


@pytest.fixture
def toy_domain():
    return TaskListDomain(task_list=make_tasks())


def truthful_judge(obs):
    """Judge that believes whoever is arguing for the answer written in the transcript most
    often; returns a mild preference so rewards are informative."""
    if obs.response.kind == "distribution":
        opts = obs.response.options
        if set(opts) == {"YES", "NO"}:
            return {"probs": {"YES": 0.7, "NO": 0.3}}
        return {"probs": {o: (0.7 if o == "A" else 0.3) for o in opts}}
    if obs.response.kind == "scalar":
        return {obs.response.scalar_name: (obs.response.lo + obs.response.hi) / 2}
    if obs.response.kind == "choice":
        return {"choice": obs.response.options[0]}
    if obs.response.kind == "json":
        return {k: 0.5 for k in (obs.response.fields or {})}
    return "Question?"


def arguer(obs):
    if obs.response.kind == "choice":
        return {"choice": obs.target or obs.response.options[0], "text": f"I pick {obs.target}"}
    if obs.response.kind == "scalar":
        return {obs.response.scalar_name: 0.6}
    if obs.response.kind == "json":
        return {"answer": "A", "reports": [], "probability": 0.6, "rating": 5}
    return f"I argue for ({obs.target}) {obs.target_text}."


@pytest.fixture
def scripted_agents():
    return {"kind:judge": oa.ScriptedAgent(truthful_judge, id="judge"), "kind:monitor": oa.ScriptedAgent(truthful_judge, id="mon"),
            "*": oa.ScriptedAgent(arguer, id="arguer")}
