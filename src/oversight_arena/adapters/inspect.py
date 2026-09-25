"""Inspect model policies and real ControlArena/Inspect evaluation-log imports."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from ..core import (
    Episode,
    Evaluation,
    Event,
    Observation,
    PublicTask,
    Response,
    Reward,
    Role,
    canonical,
    digest,
    finite,
)


@dataclass(frozen=True)
class InspectPolicy:
    model: Any  # Inspect Model or a provider/model identifier
    system: str = "Follow the protocol instructions."
    json_output: bool = False
    generation: dict[str, Any] = field(default_factory=dict)
    cost: Callable[[Any], float | None] | None = None

    async def __call__(self, observation: Observation) -> Response:
        from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model

        model = get_model(self.model) if isinstance(self.model, str) else self.model
        config = GenerateConfig(**{"seed": observation.seed, **self.generation})
        instruction = self.system + (" Return only valid JSON." if self.json_output else "")
        result = await model.generate(
            [
                ChatMessageSystem(content=instruction),
                ChatMessageUser(content=canonical(asdict(observation))),
            ],
            config=config,
        )
        content = json.loads(result.completion) if self.json_output else result.completion
        tokens = result.usage.total_tokens if result.usage else None
        return Response(content, cost=self.cost(result) if self.cost else None, tokens=tokens)


@dataclass(frozen=True)
class InspectPromptProposer:
    policy: InspectPolicy

    async def __call__(self, request) -> str:
        observation = Observation(
            PublicTask(
                "prompt_search",
                "Propose an improved agent instruction using reward feedback.",
                asdict(request),
            ),
            request.role,
            f"Search cohort: {request.cohort}. Return just the new prompt as plain text.",
            (),
            {},
            request.seed,
        )
        response = await self.policy(observation)
        if not isinstance(response.content, str):
            raise ValueError("Prompt proposer must return a string")
        return response.content


@dataclass(frozen=True)
class ScoreSelector:
    name: str
    key: str | None = None
    labels: Mapping[str, float] | None = None

    def read(self, scores: Mapping[str, Any]) -> float:
        value = scores[self.name].value
        if self.key is not None:
            value = value[self.key]
        if isinstance(value, str) and self.labels is not None:
            value = self.labels[value]
        return finite(value)


def from_eval_sample(
    sample: Any,
    *,
    task: PublicTask,
    roles: Sequence[Role],
    reward_map: Mapping[str, ScoreSelector],
    truth_map: Mapping[str, Mapping[str, ScoreSelector]] | None = None,
    events: Sequence[Event] = (),
    mechanism: str = "controlarena",
    config: dict | None = None,
    seed: int = 0,
    scorer_version: str = "user-specified-v1",
) -> Episode:
    """Import an actual Inspect EvalSample with explicit reward and truth mappings.

    Suspicion/main-task/side-task scores are never automatically treated as agent
    training rewards. Supply a sanitized PublicTask and optional projected events;
    sample input, metadata and target are not copied (they can contain secrets).
    """
    if sample.error is not None:
        raise ValueError("Cannot import failed sample as a successful episode")
    scores = sample.scores or {}
    reward_sources = {(s.name, s.key) for s in reward_map.values()}
    truth_sources = {
        (s.name, s.key) for metrics in (truth_map or {}).values() for s in metrics.values()
    }
    if reward_sources & truth_sources:
        raise ValueError("Reward and truth mappings must use separate score sources")
    rewards = {r: Reward(selector.read(scores)) for r, selector in reward_map.items()}
    evaluations = (
        (
            Evaluation(
                "inspect_import",
                scorer_version,
                {
                    r: {m: s.read(scores) for m, s in metrics.items()}
                    for r, metrics in truth_map.items()
                },
            ),
        )
        if truth_map
        else ()
    )
    record = Episode(
        "",
        task,
        mechanism,
        config or {},
        tuple(roles),
        seed,
        tuple(events),
        rewards,
        sample.output.completion if sample.output else None,
        {"known_cost": None, "calls": None},
        {
            "adapter": "inspect_eval_sample",
            "sample_id": sample.id,
            "epoch": sample.epoch,
            "reward_map": {k: asdict(v) for k, v in reward_map.items()},
            "truth_map": {
                r: {k: asdict(v) for k, v in metrics.items()}
                for r, metrics in (truth_map or {}).items()
            },
        },
        evaluations,
    )
    record = replace(record, id=digest(record.mechanism_payload()))
    record.validate()
    return record


def read_controlarena_log(
    path: str,
    *,
    task: Callable[[Any], PublicTask],
    event_projection: Callable[[Any], Sequence[Event]] | None = None,
    **mapping,
) -> list[Episode]:
    from inspect_ai.log import read_eval_log

    log = read_eval_log(path)
    if log.status != "success":
        raise ValueError("Import requires a completed successful Inspect evaluation")
    return [
        from_eval_sample(
            sample,
            task=task(sample),
            events=event_projection(sample) if event_projection else (),
            **mapping,
        )
        for sample in (log.samples or [])
    ]


def mechanism_reward_scorer(role: str, reward: Callable[[Any], Awaitable[float]]):
    """Create an Inspect scorer to attach explicitly defined training utility to a task.

    The callback receives TaskState, not Target. Researcher code must not smuggle
    hidden labels through TaskState.metadata. No reward is inferred from a task scorer.
    """
    from inspect_ai.scorer import Score, mean, scorer

    @scorer(metrics=[mean()], name=f"oversight_reward_{role}")
    def factory():
        async def score(state, target):
            return Score(
                value=finite(await reward(state)),
                metadata={"oversight_kind": "mechanism_reward", "role": role},
            )

        return score

    return factory()
