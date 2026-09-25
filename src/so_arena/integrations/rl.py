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
  function ``fn(prompts, completions, item_id=..., stance=..., **kw) -> list[float | None]`` that
  completes each episode with the other roles' policies and returns the trainee's reward.
  :func:`rollout_prompts` builds the matching prompt dataset; its ``stance`` column carries the
  answer each prompt assigned, so the reward scores the stance the trainee was told to argue.

Rewards that do not exist yet or at all (a pending market, an unscored trainable monitor) are
``None``, never a made-up 0: TRL's ``GRPOTrainer`` reads ``None`` as "not applicable" (NaN, dropped by
its ``nansum`` over reward functions - so with a single reward function the sample contributes 0 and
TRL warns); mask such samples in your trainer, or pass ``missing=`` to substitute a value knowingly.

Ground-truth values are recorded alongside rewards (``env.last_episode``, ``fn.history``) so that
training curves can plot mechanism reward against ground truth - the point of the exercise.
"""

from __future__ import annotations

import asyncio
import collections
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


def trainee_rng(seed: int, item_id: str) -> random.Random:
    """RNG resolving a trainee's stance spec (e.g. which false answer ``"false"`` picks) on an item -
    shared by :class:`MechanismEnv` and :class:`RewardFunction`, so both assign the same answer."""
    return random.Random(stable_hash(seed, item_id))


class MechanismEnv:
    """Step-wise view of a mechanism for RL.

    Several trainees may be asked at once (simultaneous moves, e.g. self-play debate): every request is
    buffered and served one per observation, in arrival order; none of them sees the others' actions.
    ``pending_roles`` lists the trainees currently waiting for an action.

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
        self._queue: collections.deque[tuple[str, ActionRequest, asyncio.Future]] = collections.deque()
        self._item: TaskItem | None = None
        self.stances: dict[str, str | None] = {}  # the resolved stance of each trainee on the current item
        self.last_episode: Episode | None = None

    @property
    def pending_roles(self) -> list[str]:
        return [r for r, _, _ in self._queue]

    async def reset(self, item: TaskItem, *, episode_id: str | None = None, seed: int = 0) -> StepResult:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._queue.clear()
        self._item = item
        self._externals = {r: ExternalPolicy(label="trainee") for r in self.trainees}
        players = dict(self.players)
        rng = trainee_rng(seed, item.id)
        self.stances = {r: resolve_stance(stance, item, rng) for r, stance in self.trainees.items()}  # per item
        for r in self.trainees:
            players[r] = Player(policy=self._externals[r], stance=self.stances[r], label="trainee")
        self._task = asyncio.create_task(self.mechanism.run(item, players, self.ctx,
                                                            episode_id=episode_id or f"{item.id}:env", seed=seed))
        return await self._advance()

    def _drain(self) -> None:
        for role, ext in self._externals.items():
            while not ext.requests.empty():
                request, _actx, fut = ext.requests.get_nowait()
                self._queue.append((role, request, fut))

    async def _advance(self) -> StepResult:
        """The next buffered trainee request, else wait for one (or for the episode to end)."""
        assert self._task is not None
        while not self._task.done():
            self._drain()
            if self._queue:
                role, request, _ = self._queue[0]
                obs = Observation(role=role, messages=request_messages(request, self.system_note), request=request)
                return StepResult(observation=obs)
            getters = [asyncio.ensure_future(ext.requests.get()) for ext in self._externals.values()]
            try:
                await asyncio.wait([self._task, *getters], return_when=asyncio.FIRST_COMPLETED)
            finally:
                for g, role in zip(getters, self._externals):
                    if g.done() and not g.cancelled():
                        request, _actx, fut = g.result()  # already taken off its queue: keep it
                        self._queue.append((role, request, fut))
                    else:
                        g.cancel()  # a cancelled get() leaves its item on the queue for _drain
        self._queue.clear()  # the protocol has ended: nothing still waits for these
        ep = self._task.result()
        assert self._item is not None
        ep = await score_episode(ep, self._item, self.scorers, self.ctx)
        self.last_episode = ep
        return StepResult(done=True, rewards=dict(ep.rewards), episode=ep)

    async def step(self, action: Any) -> StepResult:
        """Supply the action of the trainee in the last observation (text, or a dict/Action for
        structured requests)."""
        if not self._queue:
            raise RuntimeError("no pending trainee decision; call reset() first")
        _, _, fut = self._queue.popleft()
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
    """TRL-style batch reward function for a single-decision trainee role (see module docstring).

    Each sample's stance is the dataset's ``stance`` column when given (as :func:`rollout_prompts`
    emits it: the answer that sample's prompt assigned), else the spec in ``stances`` (by item id, else
    by role) resolved exactly as :class:`MechanismEnv` resolves it for the same ``seed``. Rewards that
    are missing (pending, unscored) are returned as ``missing`` - ``None`` by default, never a silent 0.
    """

    def __init__(self, mechanism: Mechanism, role: str, items: Sequence[TaskItem], players: dict[str, Player], *,
                 stances: dict[str, str | None] | None = None, ctx: RunContext | None = None,
                 ground_truth: Sequence[GroundTruthScorer] | None = None, seed: int = 0,
                 missing: float | None = None):
        self.mechanism, self.role = mechanism, role
        self.items = {it.id: it for it in items}
        self.players, self.stances = dict(players), dict(stances or {})
        self.ctx = ctx or RunContext()
        self.scorers = list(ground_truth) if ground_truth is not None else default_scorers()
        self.seed, self.missing = seed, missing
        self.history: list[dict[str, Any]] = []
        self.__name__ = f"so_arena_{mechanism.name}_{role}"

    @staticmethod
    def _text(completion: Any) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
        return str(completion)

    def _stance(self, item: TaskItem, given: Any = ...) -> str | None:
        if given is not ...:  # from the dataset: a concrete label (or a spec, if it names no option)
            if given is None or given in item.labels:
                return given
            return resolve_stance(given, item, trainee_rng(self.seed, item.id))
        spec = self.stances.get(item.id, self.stances.get(self.role))
        return resolve_stance(spec, item, trainee_rng(self.seed, item.id))

    async def ascore(self, completions: Sequence[Any], item_ids: Sequence[str],
                     stances: Sequence[str | None] | None = None) -> list[float | None]:
        async def one(c: Any, iid: str, given: Any) -> float | None:
            item = self.items[iid]
            players = dict(self.players)
            stance = self._stance(item, given)
            players[self.role] = Player(policy=FixedPolicy(self._text(c), label="completion"), stance=stance)
            ep = await self.mechanism.run(item, players, self.ctx, episode_id=f"{iid}:rl:{len(self.history)}", seed=self.seed)
            ep = await score_episode(ep, item, self.scorers, self.ctx)
            r = ep.rewards.get(self.role)
            self.history.append({"item_id": iid, "stance": stance, "reward": r, "value": ep.value(self.role),
                                 "judge_correct": ep.ground_truth.get("judge_correct")})
            return float(r) if r is not None else self.missing

        given = list(stances) if stances is not None else [...] * len(item_ids)
        return list(await asyncio.gather(*[one(c, i, g) for c, i, g in zip(completions, item_ids, given)]))

    def __call__(self, prompts: Sequence[Any] | None = None, completions: Sequence[Any] = (), *,
                 item_id: Sequence[str] | None = None, stance: Sequence[str | None] | None = None,
                 **kwargs: Any) -> list[float | None]:
        if item_id is None:
            raise ValueError("reward function needs the dataset column 'item_id'")
        return run_sync(self.ascore(list(completions), list(item_id), None if stance is None else list(stance)))


def reward_function(mechanism: Mechanism, role: str, items: Sequence[TaskItem], players: dict[str, Player],
                    **kw: Any) -> RewardFunction:
    return RewardFunction(mechanism, role, items, players, **kw)


async def arollout_prompts(mechanism: Mechanism, role: str, items: Sequence[TaskItem], players: dict[str, Player], *,
                           stance: str | None = None, ctx: RunContext | None = None, seed: int = 0) -> list[dict[str, Any]]:
    """The trainee's first-decision prompt for each item: rows ``{"prompt", "item_id", "role", "stance"}``.

    ``stance`` is the answer the prompt assigned (None in open protocols): keep the column, and the
    reward function scores the completion under that same stance. Items whose stance cannot be
    resolved (e.g. ``"true"`` on an unresolved question) are skipped, as in the runner.
    """
    rows = []
    for item in items:
        env = MechanismEnv(mechanism, players, {role: stance}, ctx=ctx)
        try:
            res = await env.reset(item, seed=seed)
        except LookupError:
            continue
        if res.observation is not None:
            rows.append({"prompt": res.observation.messages, "item_id": item.id, "role": role,
                         "stance": env.stances.get(role)})
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
