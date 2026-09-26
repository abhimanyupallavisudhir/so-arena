"""Expose SO-arena experiments as Inspect tasks.

``inspect_task(mechanism, items, profiles)`` returns an ``inspect_ai.Task`` whose samples are
(item, profile) pairs; the solver runs the mechanism and the scorer reports every role's reward and
ground-truth value. You get Inspect's runner, logging, retries, sandboxes and the ``inspect view``
log viewer; :func:`episodes_from_inspect_logs` reads the episodes back for SO-arena analysis.

Every sample reports the same score keys (:func:`score_keys`), as Inspect's metrics require: a key a
sample lacks - a pending reward, ground truth that has not arrived, a skipped sample - is NaN, which
Inspect counts as unscored rather than averaging it in; ``Score.metadata["missing"]`` lists them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from so_arena.core.game import RunContext
from so_arena.core.ground_truth import GroundTruthScorer, default_scorers
from so_arena.core.items import TaskItem
from so_arena.core.mechanism import Episode, Mechanism
from so_arena.core.runner import Profile, build_players, score_episode


OUTCOME_KEYS = ("judge_correct", "judge_p_true", "outcome_value")


def score_keys(mechanism: Mechanism) -> list[str]:
    """The score keys of every sample: rewards of trainable roles, values of agent and trainable roles,
    and outcome-level ground truth."""
    specs = mechanism.role_specs()
    rewarded = [r for r, s in specs.items() if s.trainable]
    valued = [r for r, s in specs.items() if s.trainable or s.kind == "agent"]
    return [f"reward_{r}" for r in rewarded] + [f"value_{r}" for r in valued] + list(OUTCOME_KEYS)


def _number(v: object) -> float:
    return float(v) if isinstance(v, (int, float)) else math.nan  # None (pending), absent -> NaN


def inspect_task(mechanism: Mechanism, items: Sequence[TaskItem], profiles: Sequence[Profile], *,
                 ctx: RunContext | None = None, ground_truth: Sequence[GroundTruthScorer] | None = None,
                 name: str | None = None, epochs: int = 1):
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ModelOutput
    from inspect_ai.scorer import Score, Target, mean, scorer, stderr
    from inspect_ai.solver import Generate, TaskState, solver

    by_name = {p.name: p for p in profiles}
    scorers = list(ground_truth) if ground_truth is not None else default_scorers()
    run_ctx = ctx or RunContext()
    samples = [
        Sample(input=item.question, id=f"{item.id}|{prof.name}",
               metadata={"item": item.model_dump(mode="json"), "profile": prof.name})
        for item in items for prof in profiles
    ]

    @solver
    def mechanism_solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            item = TaskItem.model_validate(state.metadata["item"])
            prof = by_name[state.metadata["profile"]]
            try:
                players = build_players(prof, item, seed=state.epoch)
            except LookupError as e:
                state.metadata["skipped"] = str(e)
                return state
            ep = await mechanism.run(item, players, run_ctx, episode_id=f"{state.sample_id}#{state.epoch}",
                                     profile=prof.name, repeat=state.epoch)
            ep = await score_episode(ep, item, scorers, run_ctx)
            state.messages = [ChatMessageSystem(content=f"{mechanism.name} | profile {prof.name}")] + [
                ChatMessageAssistant(content=f"[{t.role}{'/' + t.phase if t.phase else ''}] "
                                             f"{t.shown or t.text or t.probs or t.choice or t.score}")
                for t in ep.turns
            ]
            state.metadata["episode"] = ep.model_dump(mode="json")
            state.output = ModelOutput.from_content(model="so_arena", content=str(ep.outcome.decision or ""))
            return state

        return solve

    keys = score_keys(mechanism)

    @scorer(metrics={"*": [mean(), stderr()]})
    def mechanism_scorer():
        async def score(state: TaskState, target: Target) -> Score:
            value = dict.fromkeys(keys, math.nan)
            if "episode" not in state.metadata:
                return Score(value=value, explanation=state.metadata.get("skipped", "not run"),
                             metadata={"skipped": True, "missing": keys})
            ep = Episode.model_validate(state.metadata["episode"])
            found = {f"reward_{r}": v for r, v in ep.rewards.items()}
            found |= {f"value_{r}": v for r, v in (ep.ground_truth.get("role_values") or {}).items()}
            found |= {k: ep.ground_truth.get(k) for k in OUTCOME_KEYS}
            for k in keys:
                value[k] = _number(found.get(k))
            return Score(value=value, answer=str(ep.outcome.decision), explanation=ep.error or "",
                         metadata={"episode_id": ep.id, "missing": [k for k in keys if math.isnan(value[k])]})

        return score

    return Task(dataset=samples, solver=mechanism_solver(), scorer=mechanism_scorer(),
                name=name or f"so_arena_{mechanism.name}", epochs=epochs)


def episodes_from_inspect_logs(logs: Any) -> list[Episode]:
    """Read episodes back from Inspect logs produced by :func:`inspect_task`."""
    from inspect_ai.log import EvalLog, read_eval_log

    if not isinstance(logs, (list, tuple)):
        logs = [logs]
    eps = []
    for log in logs:
        if not isinstance(log, EvalLog):
            log = read_eval_log(log)
        for s in log.samples or []:
            if s.metadata and "episode" in s.metadata:
                eps.append(Episode.model_validate(s.metadata["episode"]))
    return eps
