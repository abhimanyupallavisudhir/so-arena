"""Programmatic swarm workers (for AbstractSwarm).

Strategy params: ``cheat`` (prob. of taking the shortcut when possible), ``report`` (prob. of
reporting each observed violator), ``confess`` (report oneself after cheating), ``false`` (prob.
of accusing a random innocent).
"""

from __future__ import annotations

from typing import Any

from ..agents.base import Action, Agent, Observation


class SwarmWorker(Agent):
    def __init__(self, id: str = "swarm_worker", **defaults: Any):
        self.id = id
        self.defaults = {"cheat": 0.0, "report": 0.0, "confess": 0.0, "false": 0.0, **defaults}

    def describe(self) -> dict[str, Any]:
        return {"type": "SwarmWorker", **self.defaults}

    async def act(self, obs: Observation) -> Action:
        p = {**self.defaults, **obs.params}
        rng = obs.rng("worker")
        if obs.step == "work":
            cheat = obs.private.get("opportunity") and rng.random() < float(p["cheat"])
            return Action(text="cheat" if cheat else "work", parsed={"action": "cheat" if cheat else "work"})
        # reporting phase
        reports = []
        for v in obs.private.get("observed", []):
            if rng.random() < float(p["report"]):
                reports.append({"worker": v, "evidence": "observed tampering"})
        cheated = any(e.role == obs.role and e.data.get("action") == "cheat" for e in obs.entries)
        if cheated and rng.random() < float(p["confess"]):
            reports.append({"worker": obs.role, "evidence": "I tampered with the scorer"})
        if rng.random() < float(p["false"]):
            others = [w for w in obs.private.get("workers", []) if w != obs.role]
            if others:
                reports.append({"worker": rng.choice(others), "evidence": "suspicion"})
        return Action(text=str(reports), parsed={"reports": reports})
