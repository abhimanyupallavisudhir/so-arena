"""Run OversightArena experiments as Inspect AI tasks, and read the logs back.

Why: Inspect is the evaluation framework ControlArena is built on. Exporting a (domain,
mechanism, profiles) grid as an Inspect ``Task`` gives you ``inspect eval`` / ``eval_set``,
model roles, rate limiting, retries, Inspect View for browsing transcripts, and log files that
other Inspect tooling understands — while every sample still carries the full
:class:`~oversight_arena.core.episode.EpisodeRecord`, so all OversightArena analyses (ASD,
best-of-N, equilibria, releases...) run on the logs::

    # my_eval.py
    from inspect_ai import task
    from oversight_arena import Stances
    from oversight_arena.domains.math import GSM8K
    from oversight_arena.mechanisms import Debate
    from oversight_arena.integrations.inspect import oversight_task

    @task
    def gsm8k_debate():
        return oversight_task(GSM8K(limit=50), Debate(rounds=2), profiles=Stances())

    $ inspect eval my_eval.py --model openai/gpt-4.1 --model-role trusted=openai/gpt-4.1-nano

    >>> results = results_from_logs("logs/")          # -> oversight_arena Results
    >>> asd(results.df(trainable_only=True))

Models: by default roles of judge-like kinds (judge, monitor, overseer, client) use the Inspect
model role ``trusted`` and everyone else ``untrusted`` — ControlArena's convention — each falling
back to the eval's ``--model``. Pass ``agents=`` for anything else (including programmatic
agents; then no model calls happen at all).

Scores: one dict-valued Inspect score per sample with ``reward/<role>`` for every trainable
role and ``gt/<scorer>/<role>`` for every ground-truth value, averaged by Inspect's metrics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..agents.base import Agent
from ..core.episode import EpisodeRecord
from ..core.strategy import Profile
from ..core.task import Task as OATask
from ..domains.base import JUDGE_KINDS, Domain
from ..experiment.profiles import ProfileSource, as_profile_source
from ..experiment.results import Results
from ..experiment.runner import AgentTable, run_episode
from ..ground_truth.base import GTScorer
from ..mechanisms.base import Mechanism

RECORD_KEY = "oa_record"


def default_inspect_agents(trusted: str = "trusted", untrusted: str = "untrusted", **llm_kw: Any) -> dict[str, Agent]:
    """LLM agents bound to Inspect model roles: judge-like roles → ``trusted``, others → ``untrusted``."""
    from ..agents.llm import LLMAgent, LLMJudge
    from ..models.inspect_model import InspectModel

    table: dict[str, Agent] = {f"kind:{k}": LLMJudge(InspectModel(role=trusted), id=f"role:{trusted}") for k in JUDGE_KINDS}
    table["*"] = LLMAgent(InspectModel(role=untrusted), id=f"role:{untrusted}", **llm_kw)
    return table


def _sample_id(task: OATask, profile: Profile) -> str:
    return f"{task.id}::{profile.id}"


def record_messages(rec: EpisodeRecord, task: OATask | None = None) -> list[Any]:
    """An episode transcript as Inspect chat messages (for Inspect View): the public task, then one
    message per transcript entry, labelled with the speaker's role; structured data in metadata."""
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

    from ..domains.base import default_render

    titles = {r.name: r.display for r in rec.roles}
    msgs: list[Any] = []
    if task is not None:
        msgs.append(ChatMessageUser(content=default_render(task.view(), set()), metadata={"oa": "task"}))
    for e in rec.transcript.entries:
        body = e.content
        if e.evidence and e.kind != "evidence":
            body += "\n\n" + "\n".join(ev.render() for ev in e.evidence)
        meta = {"oa_role": e.role, "kind": e.kind, "step": e.step, "turn": e.turn,
                "visible_to": e.visible_to, "data": {k: v for k, v in e.data.items() if not k.startswith("_")}}
        if e.role is None:
            msgs.append(ChatMessageUser(content=f"[Moderator] {body}", metadata=meta))
        else:
            msgs.append(ChatMessageAssistant(content=f"**{titles.get(e.role, e.role)}**: {body}", metadata=meta,
                                             model=rec.bound[e.role].agent if e.role in rec.bound else None))
    return msgs


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def score_values(rec: EpisodeRecord, gt_keys: Sequence[str] | None = None) -> dict[str, float | None]:
    """Flat dict of rewards and GT for an Inspect score (``reward/<role>``, ``gt/<scorer>/<who>``)."""
    out: dict[str, float | None] = {f"reward/{r}": _num(rec.rewards.get(r)) for r in rec.trainable_roles}
    for s, vals in rec.gt.items():
        for who, v in vals.items():
            if gt_keys is None or f"{s}/{who}" in gt_keys or s in gt_keys:
                out[f"gt/{s}/{who}"] = _num(v)
    return out


def oversight_task(
    domain: Domain,
    mechanism: Mechanism,
    agents: "AgentTable | dict[str, Agent] | Agent | None" = None,
    profiles: "ProfileSource | Sequence[Profile] | Profile | None" = None,
    *,
    gt: Sequence[GTScorer] | None = None,
    clearances: dict[str, Sequence[str]] | None = None,
    name: str | None = None,
    tasks: Sequence[OATask] | None = None,
    epochs: int = 1,
    trusted: str = "trusted",
    untrusted: str = "untrusted",
    metrics: bool = True,
) -> Any:
    """Build an Inspect ``Task``: one sample per (task, profile); each epoch is an independent
    episode (seed = epoch - 1). See the module docstring for models and scores."""
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import ModelOutput
    from inspect_ai.scorer import Score, mean, scorer, stderr
    from inspect_ai.solver import solver

    task_list = list(tasks) if tasks is not None else domain.tasks()
    by_id = {t.id: t for t in task_list}
    src = as_profile_source(profiles)
    roles = mechanism.roles()
    table = agents if isinstance(agents, AgentTable) else AgentTable(agents or default_inspect_agents(trusted, untrusted))
    samples = []
    for t in task_list:
        for prof in src.profiles(t, roles):
            samples.append(Sample(
                id=_sample_id(t, prof), input=t.question,
                metadata={"oa_task": t.id, "oa_profile": prof.model_dump(mode="json"), "oa_profile_label": prof.label,
                          "oa_domain": domain.name},
            ))
    gt_list = list(gt) if gt is not None else None

    @solver
    def oversight_solver() -> Any:
        async def solve(state: Any, generate: Any) -> Any:
            t = by_id[state.metadata["oa_task"]]
            prof = Profile.model_validate(state.metadata["oa_profile"])
            rec = await run_episode(mechanism, t, prof, table, domain, gt=gt_list, clearances=clearances,
                                    seed=max(0, int(state.epoch) - 1), experiment=name)
            try:  # which models the roles were actually bound to in this eval
                from inspect_ai.model import get_model

                rec.meta["inspect_models"] = {r: str(get_model(role=r)) for r in (trusted, untrusted)}
            except Exception:
                pass
            state.store.set(RECORD_KEY, rec.model_dump(mode="json"))
            state.messages = record_messages(rec, t)
            decision = rec.outcome.get("decision")
            summary = f"decision: {decision}; rewards: " + ", ".join(f"{k}={v:.3f}" for k, v in rec.rewards.items())
            state.output = ModelOutput.from_content(model="oversight_arena", content=summary)
            if rec.error:
                raise RuntimeError(rec.error)
            return state

        return solve

    @scorer(metrics={"*": [mean(), stderr()]} if metrics else [])
    def oversight_scorer() -> Any:
        async def score(state: Any, target: Any) -> Any:
            rec = EpisodeRecord.model_validate(state.store.get(RECORD_KEY))
            vals = {k: v for k, v in score_values(rec).items() if v is not None}
            return Score(value=vals, answer=str(rec.outcome.get("decision")),
                         metadata={"gt_status": rec.gt_status, "profile": rec.profile.label})

        return score

    return Task(
        dataset=MemoryDataset(samples), solver=oversight_solver(), scorer=oversight_scorer(), epochs=epochs,
        name=name or f"oa_{domain.name}_{mechanism.name}",
        metadata={"oversight_arena": {"domain": domain.describe(), "mechanism": mechanism.describe()}},
    )


def results_from_logs(logs: Any, tasks: Sequence[OATask] | None = None) -> Results:
    """Load OversightArena records from Inspect logs (EvalLog objects, log files or directories)."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    items = logs if isinstance(logs, (list, tuple)) else [logs]
    evals: list[Any] = []
    for it in items:
        if isinstance(it, (str, Path)) and Path(it).is_dir():
            evals += [read_eval_log(info) for info in list_eval_logs(str(it))]
        elif isinstance(it, (str, Path)):
            evals.append(read_eval_log(str(it)))
        else:
            evals.append(it)
    recs = []
    for log in evals:
        for s in log.samples or []:
            data = (s.store or {}).get(RECORD_KEY)
            if data:
                recs.append(EpisodeRecord.model_validate(data))
    return Results(recs, {t.id: t for t in tasks or []})


__all__ = ["oversight_task", "results_from_logs", "default_inspect_agents", "record_messages", "score_values", "RECORD_KEY"]
