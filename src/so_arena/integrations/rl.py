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
* :func:`preference_pairs` - offline preference data (DPO, reward models) from episodes or sampled game
  trees: same-context pairs ranked by the mechanism's reward, with whether ground truth agrees.

Rewards that do not exist yet or at all (a pending market, an unscored trainable monitor) are
``None``, never a made-up 0: TRL's ``GRPOTrainer`` reads ``None`` as "not applicable" (NaN, dropped by
its ``nansum`` over reward functions - so with a single reward function the sample contributes 0 and
TRL warns); mask such samples in your trainer, or pass ``missing=`` to substitute a value knowingly.

Ground-truth values are recorded alongside rewards (``env.last_episode``, ``fn.history``) so that
training curves can plot mechanism reward against ground truth - the point of the exercise.

**Audits are redrawn every training step.** Nature's moves (reward-time audits, in-protocol audits,
tie-breaks) are drawn from the item, repeat and seed only (:func:`so_arena.core.rewards.audit_draw`,
:meth:`so_arena.core.game.Game.chance`), so that all completions of one prompt face the same audit (a
group-relative advantage then compares behaviours, not audit luck). With one fixed seed the *same* items
would be audited at every step - over epochs the trainee could learn which items are never checked. The
reward function therefore runs step $t$ under its own seed (the trainer's ``global_step`` when TRL passes
``trainer_state``, an explicit ``step=``, else a count of calls): one shared draw within a step, fresh draws
across steps. :class:`MechanismEnv` does the same per episode: ``reset(item, train_step=t)`` runs under the
seed of step $t$ (pass it to every rollout of a group so they share the draw), and without it each reset
counts as a new step; only an explicit ``reset(seed=...)`` fixes the draw (reproducible evaluation). Stances
still resolve from the base seed, as in :func:`rollout_prompts`.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import math
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
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


def training_step_seed(seed: int, step: int) -> int:
    """The episode seed of training step ``step`` under base seed ``seed``: ``seed`` itself at step 0 (so a
    first step matches an evaluation run with ``seed``), a hash of (seed, step) after it - unrelated across
    base seeds. Shared by :class:`RewardFunction` and :class:`MechanismEnv`."""
    return seed if step == 0 else stable_hash("rl-step", seed, step) % 2**31


def trainee_rng(seed: int, item_id: str) -> random.Random:
    """RNG resolving a trainee's stance spec (e.g. which false answer ``"false"`` picks) on an item -
    shared by :class:`MechanismEnv` and :class:`RewardFunction`, so both assign the same answer."""
    return random.Random(stable_hash(seed, item_id))


class MechanismEnv:
    """Step-wise view of a mechanism for RL.

    Several trainees may be asked at once (simultaneous moves, e.g. self-play debate): every request is
    buffered and served one per observation, in arrival order; none of them sees the others' actions.
    ``pending_roles`` lists the trainees currently waiting for an action.

    Audits and other chance moves are redrawn per training step, as in :class:`RewardFunction` (one fixed
    seed would audit the same items in every epoch): an episode runs under :meth:`step_seed` of
    ``reset(train_step=t)``, or, without it, of the number of earlier resets. ``reset(seed=s)`` fixes the
    seed instead.

    Args:
        players: policies (and stances) for non-trainee roles.
        trainees: trainee roles, optionally with stances ``{role: stance_label_or_None}``.
        ground_truth: scorers applied to finished episodes (values are logged, never rewards).
        seed: the base seed (of step 0, and of the trainees' stances).
        redraw_per_step: False runs every episode under ``seed`` (the audits of an evaluation run, frozen).
    """

    def __init__(self, mechanism: Mechanism, players: dict[str, Player], trainees: dict[str, str | None], *,
                 ctx: RunContext | None = None, ground_truth: Sequence[GroundTruthScorer] | None = None,
                 system_note: str | None = None, seed: int = 0, redraw_per_step: bool = True):
        self.mechanism, self.players, self.trainees = mechanism, dict(players), dict(trainees)
        self.ctx = ctx or RunContext()
        self.scorers = list(ground_truth) if ground_truth is not None else default_scorers()
        self.system_note = system_note
        self.seed, self.redraw_per_step = seed, redraw_per_step
        self.resets = 0  # the default training step of the next reset
        self._task: asyncio.Task | None = None
        self._externals: dict[str, ExternalPolicy] = {}
        self._queue: collections.deque[tuple[str, ActionRequest, asyncio.Future]] = collections.deque()
        self._item: TaskItem | None = None
        self.stances: dict[str, str | None] = {}  # the resolved stance of each trainee on the current item
        self.last_episode: Episode | None = None

    @property
    def pending_roles(self) -> list[str]:
        return [r for r, _, _ in self._queue]

    def step_seed(self, step: int) -> int:
        """The episode seed of training step ``step`` (the base seed at step 0; see :func:`training_step_seed`)."""
        return training_step_seed(self.seed, step) if self.redraw_per_step else self.seed

    async def reset(self, item: TaskItem, *, episode_id: str | None = None, seed: int | None = None,
                    train_step: int | None = None) -> StepResult:
        """Start an episode on ``item``. It runs under ``seed`` if given (reproducible: the same seed, the
        same audits and stances), else under the seed of training step ``train_step`` - by default the number
        of earlier resets, so that audits are redrawn every episode."""
        step = self.resets if train_step is None else int(train_step)
        self.resets += 1
        stance_seed = self.seed if seed is None else seed
        if seed is None:
            seed = self.step_seed(step)
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._queue.clear()
        self._item = item
        self._externals = {r: ExternalPolicy(label="trainee") for r in self.trainees}
        players = dict(self.players)
        rng = trainee_rng(stance_seed, item.id)
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


async def rollout(env: MechanismEnv, item: TaskItem, policy: Policy, *, seed: int | None = None,
                  train_step: int | None = None) -> Episode:
    """Drive an env with an ordinary policy (useful for testing an RL setup end to end); ``seed`` and
    ``train_step`` are passed to :meth:`MechanismEnv.reset`."""
    from so_arena.core.policy import ActContext

    res = await env.reset(item, seed=seed, train_step=train_step)
    while not res.done:
        obs = res.observation
        assert obs is not None
        action = await policy.act(obs.request, ActContext(role=obs.role))
        res = await env.step(action)
    assert res.episode is not None
    return res.episode


class MultipleDecisionsError(RuntimeError):
    """A reward function's trainee role was asked for a second decision (see :class:`RewardFunction`)."""


class _Completion(FixedPolicy):
    """A trainee's completion, for its one decision; a second decision raises instead of reusing it."""

    def __init__(self, output: Any, *, label: str | None = None):
        super().__init__(output, label=label)
        self.phases: list[str] = []

    async def act(self, request, ctx):
        self.phases.append(request.phase)
        if len(self.phases) > 1:
            raise MultipleDecisionsError(f"decision {len(self.phases)} ({request.phase or request.kind!r}) of a "
                                         "single-completion trainee")
        return await super().act(request, ctx)


class RewardFunction:
    """TRL-style batch reward function for a single-decision trainee role (see module docstring).

    Each sample's stance is the dataset's ``stance`` column when given (as :func:`rollout_prompts`
    emits it: the answer that sample's prompt assigned), else the spec in ``stances`` (by item id, else
    by role) resolved exactly as :class:`MechanismEnv` resolves it for the same ``seed``. Rewards that
    are missing (pending, unscored) are returned as ``missing`` - ``None`` by default, never a silent 0.

    Audits and other chance moves are redrawn at every training step (``redraw_per_step=True``, see the
    module docstring): step $t$ runs its episodes under :meth:`step_seed`, where $t$ is ``step=`` if given,
    else ``trainer_state.global_step`` (TRL passes it), else the number of earlier calls - so pass all
    completions of a prompt in one call (TRL does). ``fn.history`` rows record the step and seed.

    A completion is one decision: if the role is asked again (a consultant with several speeches, a
    debater with several rounds) the call raises :class:`MultipleDecisionsError` rather than reusing the
    completion for decisions whose prompts the trainee never saw - train such roles with
    :class:`MechanismEnv`.
    """

    def __init__(self, mechanism: Mechanism, role: str, items: Sequence[TaskItem], players: dict[str, Player], *,
                 stances: dict[str, str | None] | None = None, ctx: RunContext | None = None,
                 ground_truth: Sequence[GroundTruthScorer] | None = None, seed: int = 0,
                 missing: float | None = None, redraw_per_step: bool = True):
        self.mechanism, self.role = mechanism, role
        self.items = {it.id: it for it in items}
        self.players, self.stances = dict(players), dict(stances or {})
        self.ctx = ctx or RunContext()
        self.scorers = list(ground_truth) if ground_truth is not None else default_scorers()
        self.seed, self.missing, self.redraw_per_step = seed, missing, redraw_per_step
        self.history: list[dict[str, Any]] = []
        self.calls = 0
        self.__name__ = f"so_arena_{mechanism.name}_{role}"

    def step_seed(self, step: int) -> int:
        """The episode seed of training step ``step``: the base seed at step 0 (so a first call matches
        an evaluation run with ``seed``), a hash of (seed, step) after it - unrelated across base seeds."""
        return training_step_seed(self.seed, step) if self.redraw_per_step else self.seed

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
                     stances: Sequence[str | None] | None = None, *, step: int | None = None) -> list[float | None]:
        """Rewards for one batch; the batch is training step ``step`` (default: the number of earlier calls)."""
        step = self.calls if step is None else int(step)
        self.calls += 1
        seed = self.step_seed(step)

        async def one(c: Any, iid: str, given: Any) -> float | None:
            item = self.items[iid]
            players = dict(self.players)
            stance = self._stance(item, given)
            policy = _Completion(self._text(c), label="completion")
            players[self.role] = Player(policy=policy, stance=stance)
            ep = await self.mechanism.run(item, players, self.ctx, episode_id=f"{iid}:rl:{len(self.history)}", seed=seed)
            if len(policy.phases) > 1:
                raise MultipleDecisionsError(
                    f"{self.mechanism.name}: role {self.role!r} was asked for more than one decision on item {iid!r} "
                    f"(phases {policy.phases}). A reward function scores one completion per episode; reusing it for "
                    "later decisions would score behaviour the trainee never produced. Train multi-decision roles "
                    "with MechanismEnv (one observation and completion per decision).")
            ep = await score_episode(ep, item, self.scorers, self.ctx)
            r = ep.rewards.get(self.role)
            self.history.append({"item_id": iid, "stance": stance, "reward": r, "value": ep.value(self.role),
                                 "judge_correct": ep.ground_truth.get("judge_correct"), "step": step, "seed": seed})
            return float(r) if r is not None else self.missing

        given = list(stances) if stances is not None else [...] * len(item_ids)
        return list(await asyncio.gather(*[one(c, i, g) for c, i, g in zip(completions, item_ids, given)]))

    def __call__(self, prompts: Sequence[Any] | None = None, completions: Sequence[Any] = (), *,
                 item_id: Sequence[str] | None = None, stance: Sequence[str | None] | None = None,
                 step: int | None = None, trainer_state: Any = None, **kwargs: Any) -> list[float | None]:
        if item_id is None:
            raise ValueError("reward function needs the dataset column 'item_id'")
        if step is None and getattr(trainer_state, "global_step", None) is not None:
            step = int(trainer_state.global_step)
        return run_sync(self.ascore(list(completions), list(item_id), None if stance is None else list(stance),
                                    step=step))


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


# ------------------------------------------------------------------------------------------------
# Preference data (DPO / reward-model training)
# ------------------------------------------------------------------------------------------------

def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _agrees(v_chosen: Any, v_rejected: Any) -> bool | None:
    """Whether ground truth prefers the behaviour the reward chose: None if either value is unknown (None or
    NaN) or they are equal - no ground-truth preference to agree with."""
    if not (_finite(v_chosen) and _finite(v_rejected)) or v_chosen == v_rejected:
        return None
    return bool(v_chosen > v_rejected)


def _render_item(item: TaskItem | None) -> str:
    if item is None:
        return ""
    opts = "\n".join(f"({a.label}) {a.text}" for a in item.answers or [])
    return item.question + ("\n\n" + opts if opts else "")


def _seen_before(ep: Episode, role: str, turn: Any) -> str:
    """The turns ``role`` saw before ``turn`` (its own included), as it saw them: visibility, verdict display
    (``show_to``) and simultaneous moves - a partner's move in the same group is not seen - as in the game."""
    lines = []
    for t in sorted(ep.turns, key=lambda t: t.slot):
        if t.slot >= turn.slot or (turn.group is not None and t.group == turn.group):
            continue
        if not (t.visible_to is None or role in t.visible_to or t.role == role):
            continue
        if t.role == role:
            text = t.text
        elif t.verdicts_to is None or t.unmarked is None:
            text = t.shown
        else:
            text = t.shown if role in t.verdicts_to or ep.role_kinds.get(role) in t.verdicts_to else t.unmarked
        lines.append(f"[{t.role}{'/' + t.phase if t.phase else ''}] {text}")
    return "\n\n".join(lines)


def _prompt(ep: Episode, role: str, turn: Any, item: TaskItem | None) -> str:
    stance = ep.players[role].stance if role in ep.players else None
    parts = [_render_item(item), f"You are {role}" + (f", arguing for ({stance})." if stance else "."),
             _seen_before(ep, role, turn)]
    return "\n\n".join(p for p in parts if p)


def _episode_pairs(episodes: Sequence[Episode], role: str, items: dict[str, TaskItem], min_gap: float) -> list[dict[str, Any]]:
    from so_arena.analysis.frames import config_key, mechanism_labels

    labels = mechanism_labels(episodes)
    groups: dict[str, list[tuple[Episode, str]]] = {}
    for ep in episodes:
        r = ep.rewards.get(role)
        turns = ep.turns_of(role)
        if ep.error is not None or ep.reward_status != "final" or not _finite(r) or not turns:
            continue
        first = min(turns, key=lambda t: t.slot)
        prompt = _prompt(ep, role, first, items.get(ep.item_id))
        me = ep.players.get(role)
        # the context: the same mechanism configuration, item, opponents and fixtures, the same assigned stance,
        # and the same transcript before the role's first move - so a pair differs only in how the role behaved
        others = sorted((q, p.policy_id, p.stance, p.label) for q, p in ep.players.items() if q != role)
        ctx = hashlib.sha256(json.dumps([ep.domain, ep.mechanism, config_key(ep), ep.item_id, me.stance if me else None,
                                         others, _seen_before(ep, role, first)], default=str).encode()).hexdigest()[:16]
        groups.setdefault(ctx, []).append((ep, prompt))
    rows = []
    for ctx, members in groups.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                (a, prompt), (b, _) = members[i], members[j]
                ra, rb = float(a.rewards[role]), float(b.rewards[role])  # type: ignore[arg-type]
                if abs(ra - rb) <= min_gap:
                    continue
                ch, rj = (a, b) if ra > rb else (b, a)
                vc, vr = ch.value(role), rj.value(role)
                rows.append({
                    "context": ctx, "mechanism": labels[(ch.mechanism, config_key(ch))], "mechanism_config": config_key(ch),
                    "item_id": ch.item_id, "role": role, "stance": ch.players[role].stance if role in ch.players else None,
                    "prompt": prompt,
                    "chosen": "\n\n".join(t.text for t in ch.turns_of(role)),
                    "rejected": "\n\n".join(t.text for t in rj.turns_of(role)),
                    "reward_chosen": ch.rewards[role], "reward_rejected": rj.rewards[role],
                    "reward_gap": abs(ra - rb), "value_chosen": vc, "value_rejected": vr, "gt_agrees": _agrees(vc, vr),
                    "chosen_label": ch.label(role), "rejected_label": rj.label(role),
                    "chosen_episode": ch.id, "rejected_episode": rj.id,
                })
    return rows


def _tree_pairs(trees: Sequence[Any], role: str, items: dict[str, TaskItem], min_gap: float, value_key: str | None,
                episodes: Sequence[Episode]) -> list[dict[str, Any]]:
    from so_arena.analysis.optimization import _Tree

    vkey = value_key or f"value_{role}"
    by_node: dict[tuple[str, str], tuple[Episode, Any]] = {}  # (episode's item, information set) -> a turn deciding it
    texts: dict[tuple[str, str, int], str] = {}
    for ep in episodes:
        for t in ep.turns:
            if t.role == role and t.node is not None:
                by_node.setdefault((ep.item_id, t.node), (ep, t))
                texts.setdefault((ep.item_id, t.node, t.candidate or 0), t.text)
    rows = []
    for tree in trees:
        t_ = _Tree(tree)
        reach = t_.reach()  # every pool sampled uniformly: the base policy
        comp = len(t_.rkeys) + t_.vkeys.index(vkey) if vkey in t_.vkeys else None
        for key, ids in t_.sets.items():
            node = tree.nodes[ids[0]]
            if node.role != role or role not in t_.ridx:
                continue
            u = t_.payoffs(key, reach)
            v = t_.payoffs(key, reach, comp) if comp is not None else np.full(len(u), math.nan)
            found = by_node.get((tree.item_id, key))
            prompt = _prompt(found[0], role, found[1], items.get(tree.item_id)) if found else None
            for i in range(len(u)):
                for j in range(i + 1, len(u)):
                    if not (math.isfinite(u[i]) and math.isfinite(u[j])) or abs(u[i] - u[j]) <= min_gap:
                        continue
                    c, r = (i, j) if u[i] > u[j] else (j, i)
                    vc, vr = float(v[c]), float(v[r])
                    rows.append({
                        "context": f"{tree.config}:{tree.item_id}:{key}", "mechanism": tree.mechanism,
                        "mechanism_config": tree.config, "item_id": tree.item_id, "role": role, "stance": None,
                        "prompt": prompt,
                        "chosen": texts.get((tree.item_id, key, c), node.candidates[c] if c < len(node.candidates) else ""),
                        "rejected": texts.get((tree.item_id, key, r), node.candidates[r] if r < len(node.candidates) else ""),
                        "reward_chosen": float(u[c]), "reward_rejected": float(u[r]), "reward_gap": float(abs(u[c] - u[r])),
                        "value_chosen": vc, "value_rejected": vr, "gt_agrees": _agrees(vc, vr),
                        "chosen_candidate": c, "rejected_candidate": r,
                    })
    return rows


def preference_pairs(source: Sequence[Any], role: str, *, items: Sequence[TaskItem] | dict[str, TaskItem] | None = None,
                     min_gap: float = 0.0, value_key: str | None = None,
                     episodes: Sequence[Episode] = ()) -> pd.DataFrame:
    """Preference data a mechanism would feed to DPO or a reward model: (prompt, chosen, rejected) pairs of
    ``role``'s behaviour ranked by the mechanism's reward, with ``gt_agrees`` - whether ground truth ranks
    them the same way (None when either value is unknown or they are equal) - so the share of pairs whose
    preference is wrong measures the data's quality before anyone trains on it.

    Pairs never mix contexts: two behaviours are paired only if everything but the role's own behaviour is
    the same - mechanism configuration, item, the other roles' policies and stances, the role's assigned
    stance and the transcript it saw before its first move (``context``). Pairs whose rewards differ by at
    most ``min_gap`` are left out; so are episodes that errored or whose reward is missing or pending.

    ``source`` is either episodes (a multi-turn role's texts are all its turns) or sampled game trees
    (:class:`~so_arena.analysis.optimization.GameTree`): then the candidates of each of the role's
    information sets are paired, each scored by its expected reward and value (``value_key``, default
    ``value_<role>``) over the rest of the tree under the base policy (every pool sampled uniformly) -
    what one step of RL from the base policy compares. Trees keep a short summary of each candidate; pass the
    trees' leaf episodes (``expand_tree(..., keep_episodes=True)``) as ``episodes`` for the full texts and
    the prompts. ``items`` (uncensored or not; only the question and options are shown) put the question in
    the prompt.
    """
    from so_arena.analysis.optimization import GameTree

    src = list(source)
    by_id = dict(items) if isinstance(items, dict) else {it.id: it for it in items or []}
    if src and all(isinstance(x, GameTree) for x in src):
        rows = _tree_pairs(src, role, by_id, min_gap, value_key, episodes)
    else:
        rows = _episode_pairs(src, role, by_id, min_gap)
    cols = ["context", "mechanism", "mechanism_config", "item_id", "role", "stance", "prompt", "chosen", "rejected",
            "reward_chosen", "reward_rejected", "reward_gap", "value_chosen", "value_rejected", "gt_agrees"]
    return pd.DataFrame(rows, columns=list(dict.fromkeys([*cols, *(k for r in rows for k in r)])))
