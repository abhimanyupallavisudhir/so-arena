"""Prompt search with explicit peer snapshots and held-out evaluation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Literal

from .core import Episode, Policy, Role, Scorer, Task
from .policies import PromptedPolicy
from .runtime import Mechanism, run_episode


@dataclass(frozen=True)
class SearchFeedback:
    prompt: str
    mean_reward: float
    episode_ids: tuple[str, ...]


@dataclass(frozen=True)
class SearchRequest:
    role: str
    cohort: str
    mechanism: str
    mechanism_description: str
    peer_prompts: dict[str, str]
    history: tuple[SearchFeedback, ...]
    seed: int


PromptProposer = Callable[[SearchRequest], Awaitable[str]]


@dataclass(frozen=True)
class SearchTrial:
    round: int
    role: str
    cohort: str
    peer_prompts: dict[str, str]
    feedback: SearchFeedback
    records: tuple[Episode, ...]


@dataclass(frozen=True)
class SearchResult:
    prompts: dict[str, str]
    trials: tuple[SearchTrial, ...]
    heldout: tuple[Episode, ...]
    schedule: str
    cohort: str


def _check_splits(train: Sequence[Task], heldout: Sequence[Task]) -> None:
    if not train or any(t.split != "train" for t in train):
        raise ValueError("Search requires nonempty tasks explicitly marked train")
    if any(t.split not in {"validation", "test"} for t in heldout):
        raise ValueError("Held-out tasks must be marked validation or test")
    ids = [t.public.id for t in [*train, *heldout]]
    if len(ids) != len(set(ids)):
        raise ValueError("Training and held-out task ids must be unique and disjoint")


async def prompt_search(
    *,
    train: Sequence[Task],
    heldout: Sequence[Task],
    mechanism: Mechanism,
    roles: Sequence[Role],
    policies: Mapping[str, Policy],
    proposers: Mapping[str, PromptProposer],
    initial: Mapping[str, str] | None = None,
    trials: int = 4,
    rounds: int = 1,
    schedule: Literal["simultaneous", "alternating"] = "simultaneous",
    cohort: str = "unrestricted",
    mechanism_description: str = "",
    seeds: Sequence[int] = (0,),
    scorer: Scorer | None = None,
    prompt_filter: Callable[[str, str], bool] | None = None,
    runtime_options: dict | None = None,
) -> SearchResult:
    """Optimize each focal prompt against frozen peer prompts for each search.

    Simultaneous updates commit after a complete round; alternating updates expose
    earlier accepted changes. There is no equilibrium or honesty guarantee. The
    proposer receives only mechanism reward feedback, never evaluator scores.
    The incumbent is evaluated first and retained on ties. Trials includes it.
    """
    _check_splits(train, heldout)
    if trials < 1 or rounds < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid search budget or seeds")
    if schedule not in {"simultaneous", "alternating"}:
        raise ValueError("Unknown update schedule")
    trainable = {r.id for r in roles if r.trainable}
    if not proposers or set(proposers) - trainable:
        raise ValueError("Search targets must be trainable roles")
    if set(initial or {}) - set(policies):
        raise ValueError("Unknown initial prompt role")
    options = runtime_options or {}
    if set(options) - {"budget", "tools", "channel_access"}:
        raise ValueError("Search runtime_options supports budget, tools and channel_access only")
    prompts = {role: (initial or {}).get(role, "") for role in policies}
    all_trials = []
    for round_index in range(rounds):
        snapshot = dict(prompts)
        updates = {}
        for role, proposer in proposers.items():
            peers = dict(snapshot if schedule == "simultaneous" else prompts)
            feedback = []
            for index in range(trials):
                prompt = (
                    peers[role]
                    if index == 0
                    else await proposer(
                        SearchRequest(
                            role,
                            cohort,
                            mechanism.name,
                            mechanism_description,
                            {k: v for k, v in peers.items() if k != role},
                            tuple(feedback),
                            seeds[0] + round_index * trials + index,
                        )
                    )
                )
                if not isinstance(prompt, str) or (
                    prompt_filter and not prompt_filter(cohort, prompt)
                ):
                    raise ValueError("Proposed prompt violates configured syntactic constraint")
                profile = {**peers, role: prompt}
                variants = {r: PromptedPolicy(p, profile[r]) for r, p in policies.items()}
                records = []
                for task in train:
                    for seed in seeds:
                        records.append(
                            await run_episode(
                                task,
                                mechanism,
                                roles,
                                variants,
                                seed=seed,
                                provenance={
                                    "sampler": "prompt_search",
                                    "cohort": cohort,
                                    "round": round_index,
                                    "focal_role": role,
                                    "prompts": profile,
                                    "schedule": schedule,
                                },
                                **options,
                            )
                        )
                item = SearchFeedback(
                    prompt,
                    mean(r.rewards[role].value for r in records),
                    tuple(r.id for r in records),
                )
                feedback.append(item)
                all_trials.append(
                    SearchTrial(round_index, role, cohort, peers.copy(), item, tuple(records))
                )
            best = max(feedback, key=lambda f: f.mean_reward)
            updates[role] = best.prompt
            if schedule == "alternating":
                prompts[role] = best.prompt
        prompts.update(updates)
    heldout_records = []
    for task in heldout:
        for seed in seeds:
            heldout_records.append(
                await run_episode(
                    task,
                    mechanism,
                    roles,
                    {r: PromptedPolicy(p, prompts[r]) for r, p in policies.items()},
                    seed=seed,
                    scorer=scorer,
                    provenance={
                        "sampler": "prompt_search_heldout",
                        "cohort": cohort,
                        "prompts": prompts,
                        "schedule": schedule,
                    },
                    **options,
                )
            )
    return SearchResult(prompts, tuple(all_trials), tuple(heldout_records), schedule, cohort)
