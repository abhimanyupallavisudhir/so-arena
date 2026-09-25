"""Training against mechanisms: step-wise environments and reward functions for RL libraries.

A mechanism is a reward function over multi-agent interactions. Two ways to train with one:

* :class:`MechanismEnv` - a step-wise, multi-agent environment. The trainee roles are driven from
  outside (by your RL loop) while fixtures (judges, graders) and non-trainee opponents act through
  their policies. ``reset`` returns the first observation (chat messages) for a trainee;
  ``step(text)`` supplies its action; when the protocol ends, ``done`` is True and ``rewards`` holds
  every role's reward. This maps directly onto multi-turn RL interfaces (e.g. Tinker's ``Env``,
  ``verifiers`` multi-turn environments, or a custom PPO/GRPO loop): the observation messages are
  rendered with your tokenizer's chat template, the sampled completion is passed to ``step``.
* :func:`reward_function` - for trainee roles with a single decision (propaganda, the proposer in a
  proposal->critique->judgment protocol, a consultant with one speech), a TRL-style batch reward
  function ``fn(prompts, completions, item_id=..., **kw) -> list[float]`` that completes each
  episode with the other roles' policies and returns the trainee's reward. :func:`rollout_prompts`
  builds the matching prompt dataset.

Ground-truth values are recorded alongside rewards (``env.last_episode``, ``fn.history``) so that
training curves can plot mechanism reward against ground truth - the point of the exercise.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from so_arena.core.actions import ActionRequest
from so_arena.core.game import Player, RunContext
from so_arena.core.ground_truth import GroundTruthScorer, default_scorers
from so_arena.core.items import TaskItem
from so_arena.core.mechanism import Episode, Mechanism
from so_arena.core.policy import ExternalPolicy, FixedPolicy, Policy, format_instructions, stable_hash
from so_arena.core.runner import resolve_stance, run_sync, score_episode
from so_arena.core.types import Message


def request_messages(request: ActionRequest, extra_system: str | None = None) -> list[dict[str, str]]:
    """The chat messages a trainee should see for a request (with output-format instructions)."""
    msgs = [m.model_dump(include={"role", "content"}) for m in request.prompt]
    if extra_system:
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] += "\n\n" + extra_system
        else:
            msgs.insert(0, {"role": "system", "content": extra_system})
    fmt = format_instructions(request)
    if fmt:
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"] += "\n\n" + fmt
        else:
            msgs.append({"role": "user", "content": fmt})
    return msgs


class Observation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: str
    messages: list[dict[str, str]]
    request: ActionRequest


class StepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    observation: Observation | None = None
    done: bool = False
    rewards: dict[str, float | None] = Field(default_factory=dict)
    episode: Episode | None = None


class MechanismEnv:
    """Step-wise view of a mechanism for RL.

    Args:
        players: policies (and stances) for non-trainee roles.
        trainees: trainee roles, optionally with stances ``{role: stance_label_or_None}``.
        ground_truth: scorers applied to finished episodes (values are logged, never rewards).
    """

    def __init__(self, mechanism: Mechanism, players: dict[str, Player], trainees: dict[str, str | None], *,
                 ctx: RunContext | None = None, ground_truth: Sequence[GroundTruthScorer] | None = None,
                 system_note: str | None = None):
        self.mechanism, self.players, self.trainees = mechanism, dict(players), dict(trainees)
        self.ctx = ctx or RunContext()
        self.scorers = list(ground_truth) if ground_truth is not None else default_scorers()
        self.system_note = system_note
        self._task: asyncio.Task | None = None
        self._externals: dict[str, ExternalPolicy] = {}
        self._pending: tuple[str, Any] | None = None
        self._item: TaskItem | None = None
        self.last_episode: Episode | None = None

    async def reset(self, item: TaskItem, *, episode_id: str | None = None, seed: int = 0) -> StepResult:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._item = item
        self._externals = {r: ExternalPolicy(label="trainee") for r in self.trainees}
        players = dict(self.players)
        rng = random.Random(stable_hash(seed, item.id))
        for r, stance in self.trainees.items():  # stance specs ("true", "false", labels) resolved per item
            players[r] = Player(policy=self._externals[r], stance=resolve_stance(stance, item, rng), label="trainee")
        self._task = asyncio.create_task(self.mechanism.run(item, players, self.ctx,
                                                            episode_id=episode_id or f"{item.id}:env", seed=seed))
        return await self._advance()

    async def _advance(self) -> StepResult:
        assert self._task is not None
        getters = {asyncio.ensure_future(q.requests.get()): r for r, q in self._externals.items()}
        try:
            done, _ = await asyncio.wait([self._task, *getters], return_when=asyncio.FIRST_COMPLETED)
        finally:
            for g in getters:
                if not g.done():
                    g.cancel()
        for g, role in getters.items():
            if g in done and not g.cancelled():
                request, actx, fut = g.result()
                self._pending = (role, fut)
                obs = Observation(role=role, messages=request_messages(request, self.system_note), request=request)
                return StepResult(observation=obs)
        ep = self._task.result()
        assert self._item is not None
        ep = await score_episode(ep, self._item, self.scorers, self.ctx)
        self.last_episode = ep
        self._pending = None
        return StepResult(done=True, rewards=dict(ep.rewards), episode=ep)

    async def step(self, action: Any) -> StepResult:
        """Supply the pending trainee's action (text, or a dict/Action for structured requests)."""
        if self._pending is None:
            raise RuntimeError("no pending trainee decision; call reset() first")
        _, fut = self._pending
        fut.set_result(action)
        return await self._advance()


async def rollout(env: MechanismEnv, item: TaskItem, policy: Policy) -> Episode:
    """Drive an env with an ordinary policy (useful for testing an RL setup end to end)."""
    from so_arena.core.policy import ActContext

    res = await env.reset(item)
    while not res.done:
        obs = res.observation
        assert obs is not None
        action = await policy.act(obs.request, ActContext(role=obs.role))
        res = await env.step(action)
    assert res.episode is not None
    return res.episode


class RewardFunction:
    """TRL-style batch reward function for a single-decision trainee role (see module docstring)."""

    def __init__(self, mechanism: Mechanism, role: str, items: Sequence[TaskItem], players: dict[str, Player], *,
                 stances: dict[str, str | None] | None = None, ctx: RunContext | None = None,
                 ground_truth: Sequence[GroundTruthScorer] | None = None):
        self.mechanism, self.role = mechanism, role
        self.items = {it.id: it for it in items}
        self.players, self.stances = dict(players), dict(stances or {})
        self.ctx = ctx or RunContext()
        self.scorers = list(ground_truth) if ground_truth is not None else default_scorers()
        self.history: list[dict[str, Any]] = []
        self.__name__ = f"so_arena_{mechanism.name}_{role}"

    @staticmethod
    def _text(completion: Any) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
        return str(completion)

    async def ascore(self, completions: Sequence[Any], item_ids: Sequence[str]) -> list[float]:
        async def one(c: Any, iid: str) -> float:
            item = self.items[iid]
            players = dict(self.players)
            spec = self.stances.get(iid, self.stances.get(self.role))
            players[self.role] = Player(policy=FixedPolicy(self._text(c), label="completion"),
                                        stance=resolve_stance(spec, item, random.Random(stable_hash(iid))))
            ep = await self.mechanism.run(item, players, self.ctx, episode_id=f"{iid}:rl:{len(self.history)}")
            ep = await score_episode(ep, item, self.scorers, self.ctx)
            r = ep.rewards.get(self.role)
            self.history.append({"item_id": iid, "reward": r, "value": ep.value(self.role),
                                 "judge_correct": ep.ground_truth.get("judge_correct")})
            return float(r) if r is not None else 0.0

        return list(await asyncio.gather(*[one(c, i) for c, i in zip(completions, item_ids)]))

    def __call__(self, prompts: Sequence[Any] | None = None, completions: Sequence[Any] = (), *,
                 item_id: Sequence[str] | None = None, **kwargs: Any) -> list[float]:
        if item_id is None:
            raise ValueError("reward function needs the dataset column 'item_id'")
        return run_sync(self.ascore(list(completions), list(item_id)))


def reward_function(mechanism: Mechanism, role: str, items: Sequence[TaskItem], players: dict[str, Player],
                    **kw: Any) -> RewardFunction:
    return RewardFunction(mechanism, role, items, players, **kw)


async def arollout_prompts(mechanism: Mechanism, role: str, items: Sequence[TaskItem], players: dict[str, Player], *,
                           stance: str | None = None, ctx: RunContext | None = None) -> list[dict[str, Any]]:
    """The trainee's first-decision prompt for each item: rows ``{"prompt": messages, "item_id": ...}``."""
    rows = []
    for item in items:
        env = MechanismEnv(mechanism, players, {role: stance}, ctx=ctx)
        res = await env.reset(item)
        if res.observation is not None:
            rows.append({"prompt": res.observation.messages, "item_id": item.id, "role": role})
        if env._task is not None:
            env._task.cancel()
            try:
                await env._task
            except (asyncio.CancelledError, Exception):
                pass
    return rows


def rollout_prompts(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return run_sync(arollout_prompts(*args, **kwargs))


def _as_messages(msgs: list[dict[str, str]]) -> list[Message]:
    return [Message(role=m["role"], content=m["content"]) for m in msgs]  # type: ignore[arg-type]
