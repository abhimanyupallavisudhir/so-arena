"""Candidate sampling, paired interventions and generic protocol/domain matrices."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from .core import JSON, Episode, Policy, Role, Scorer, Task
from .policies import PromptedPolicy
from .runtime import Mechanism, run_episode


@dataclass(frozen=True)
class Intervention:
    name: str
    policies: Mapping[str, Policy]
    metadata: dict[str, JSON] = field(default_factory=dict)


async def sample_profiles(
    tasks: Sequence[Task],
    mechanism: Mechanism,
    roles: Sequence[Role],
    base: Mapping[str, Policy],
    interventions: Sequence[Intervention],
    *,
    seeds: Sequence[int] = (0,),
    scorer: Scorer | None = None,
    **runtime_options,
) -> dict[str, list[Episode]]:
    if len({i.name for i in interventions}) != len(interventions):
        raise ValueError("Duplicate intervention names")
    if len({t.public.id for t in tasks}) != len(tasks) or len(set(seeds)) != len(seeds):
        raise ValueError("Tasks and seeds must be unique")
    results = {}
    for intervention in interventions:
        if set(intervention.policies) - set(base):
            raise ValueError("Intervention targets an unknown role")
        records = []
        for task in tasks:
            for seed in seeds:
                records.append(
                    await run_episode(
                        task,
                        mechanism,
                        roles,
                        {**base, **intervention.policies},
                        seed=seed,
                        scorer=scorer,
                        provenance={
                            "intervention": intervention.name,
                            "intervention_metadata": intervention.metadata,
                            "changed_roles": sorted(intervention.policies),
                        },
                        **runtime_options,
                    )
                )
        results[intervention.name] = records
    return results


async def paired_interventions(
    tasks: Sequence[Task],
    mechanism: Mechanism,
    roles: Sequence[Role],
    policies: Mapping[str, Policy],
    role: str,
    prompts: Mapping[str, Callable[[Task], str]],
    *,
    seeds: Sequence[int] = (0,),
    scorer: Scorer | None = None,
    **runtime_options,
) -> dict[str, list[Episode]]:
    """Prompt factories may use oracle data to assign a side; peers never see that data.

    This is a privileged experimental intervention, not a naturally sampled policy.
    """
    if len({t.public.id for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task ids")
    results = {name: [] for name in prompts}
    for task in tasks:
        interventions = [
            Intervention(
                name,
                {role: PromptedPolicy(policies[role], factory(task))},
                {"sampler": "privileged_prompt_intervention"},
            )
            for name, factory in prompts.items()
        ]
        sampled = await sample_profiles(
            [task],
            mechanism,
            roles,
            policies,
            interventions,
            seeds=seeds,
            scorer=scorer,
            **runtime_options,
        )
        for name, records in sampled.items():
            results[name].extend(records)
    return results


@dataclass(frozen=True)
class FailedRun:
    task_id: str
    mechanism: str
    seed: int
    error_type: str
    message: str


@dataclass(frozen=True)
class Campaign:
    records: tuple[Episode, ...]
    failures: tuple[FailedRun, ...]

    @property
    def completion_rate(self) -> float:
        total = len(self.records) + len(self.failures)
        return len(self.records) / total if total else 0.0


async def run_campaign(
    tasks: Sequence[Task],
    mechanisms: Sequence[Mechanism],
    roles: Sequence[Role],
    policies: Mapping[str, Policy],
    *,
    seeds: Sequence[int] = (0,),
    scorer: Scorer | None = None,
    **runtime_options,
) -> Campaign:
    """Cross domain/mechanism/seed matrix with explicit failures, never zero-filled scores."""
    records, failures = [], []
    for mechanism in mechanisms:
        for task in tasks:
            for seed in seeds:
                try:
                    records.append(
                        await run_episode(
                            task,
                            mechanism,
                            roles,
                            policies,
                            seed=seed,
                            scorer=scorer,
                            **runtime_options,
                        )
                    )
                except Exception as error:
                    failures.append(
                        FailedRun(
                            task.public.id, mechanism.name, seed, type(error).__name__, str(error)
                        )
                    )
    return Campaign(tuple(records), tuple(failures))
