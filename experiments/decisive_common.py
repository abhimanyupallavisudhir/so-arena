"""Shared configuration of the decisive oversight study (experiments/decisive_*.py).

Models, policies, candidate items and spend accounting used by every stage:

* ``decisive_select.py``   - gap selection: items the debaters get right and the weak judges do not;
* ``decisive_protocols.py`` - every protocol, (judge, debater) pair, arm and sample on the selected items;
* ``decisive_optimize.py``  - best-of-n liars (and honest debaters) selected against each judge's reward;
* ``decisive_audit.py``     - audit of prompted and selected liar arguments;
* ``decisive_report.py``    - tables, bootstrap CIs and figures.

Every stage writes its own response cache (``<run>/cache/<stage>``), so a stage never replays another's
samples: the judge-alone and debater-alone accuracies measured in the protocol stage are fresh draws, not the
draws the items were selected on (which would overstate the gap by regression to the mean).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import so_arena as soa

JUDGES = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "llama-3.1-8b": "openrouter/meta-llama/llama-3.1-8b-instruct",
    "gemma-3-12b": "openrouter/google/gemma-3-12b-it",
}
# stronger judges for the judge-capability comparison (not used for item selection; reported separately)
STRONG_JUDGES = {
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "gpt-4.1": "openai/gpt-4.1",
}


def judge_set(names: list[str] | None) -> dict[str, str]:
    """The judges named (weak or strong), in order; all weak judges by default."""
    if not names:
        return dict(JUDGES)
    pool = {**JUDGES, **STRONG_JUDGES}
    return {n: pool[n] for n in names}


DEBATERS = {
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4-5",
    "gpt-5-mini": "openai/gpt-5-mini",
}
# Models newer than the bundled registry (Anthropic first-party prices, $ per million tokens)
for _name, _in, _out, _cached in (("anthropic/claude-sonnet-5", 2.0, 10.0, 0.2),
                                  ("anthropic/claude-opus-5-5", 4.0, 20.0, 0.2),
                                  ("anthropic/claude-opus-5", 5.0, 25.0, 0.5),
                                  ("anthropic/claude-fable-5-1", 10.0, 50.0, 0.25)):
    from so_arena.models.registry import register_model as _register

    _register(name=_name, family="claude-5", organization="Anthropic", price_input_per_mtok=_in,
              price_output_per_mtok=_out, price_cached_input_per_mtok=_cached)
# thinking models: room for hidden reasoning, effort instead of temperature (Claude 5 models think by default)
REASONING = ("gpt-5", "/o3", "/o4", "claude-sonnet-5", "claude-opus-5", "claude-fable-5")
KEY_ALIASES = {"OPENAI_API_KEY": "OPENAI_API_KEY_ASTRA_2", "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY_ASTRA_2",
               "OPENROUTER_API_KEY": "OPENROUTER_API_KEY_ASTRA"}


def map_keys() -> None:
    """Provider keys live under non-standard names in this environment; expose them under the standard ones.
    Also caps provider retries: a request a provider keeps failing (OpenRouter's Gemma returns server errors
    for some prompts indefinitely) raises after 6 retries of at most 5 minutes each instead of stalling a
    stage; its episode errors, is left out, and re-runs on resumption."""
    for std, alias in KEY_ALIASES.items():
        if not os.environ.get(std) and os.environ.get(alias):
            os.environ[std] = os.environ[alias]
    # Inspect allows 10 concurrent requests per model by default, which throttles these runs
    soa.configure(max_retries=6, attempt_timeout=300, max_connections=64)


def is_reasoning(model: str) -> bool:
    return any(t in model for t in REASONING)


# Output tokens per call for dry runs (``simulate=True``), measured on a few real calls: simulated models
# otherwise return 200 tokens, far below what reasoning (visible or hidden) costs.
SIM_OUTPUT = {"judge": 600, "solo": 1300, "solo_reasoning": 3500, "speech": 350, "speech_reasoning": 2500}


def _model(model: str, kind: str):
    from so_arena.config import settings
    from so_arena.models.base import get_model

    return get_model(f"sim/{model}", output_tokens=SIM_OUTPUT[kind]) if settings.simulate else model


def judge_policy(model: str) -> soa.LLMPolicy:
    """A weak judge: reasons in private before its verbalized probabilities; sampled (3 samples must differ)."""
    return soa.LLMPolicy(_model(model, "judge"), cot=True, temperature=0.7, max_tokens=2000)


def debater_policy(model: str, strategy: str | None = None) -> soa.LLMPolicy:
    """A strong agent arguing an assigned side. Reasoning models think internally at medium effort and need
    room for it (max_tokens is shared with hidden reasoning); others write their speech directly. Anthropic's
    API no longer takes a temperature: Claude samples at its default."""
    extra = {"strategy": strategy} if strategy else {}
    if is_reasoning(model):
        return soa.LLMPolicy(_model(model, "speech_reasoning"), temperature=None, max_tokens=12000,
                             reasoning_effort="medium", **extra)
    return soa.LLMPolicy(_model(model, "speech"), temperature=None, max_tokens=1500, **extra)


def solo_policy(model: str) -> soa.LLMPolicy:
    """A strong model answering alone (the 'debater alone' reference): reasoning first."""
    if is_reasoning(model):
        return soa.LLMPolicy(_model(model, "solo_reasoning"), temperature=None, max_tokens=12000,
                             reasoning_effort="medium")
    return soa.LLMPolicy(_model(model, "solo"), cot=True, temperature=None, max_tokens=3000)


class FreshSamples(soa.Policy):
    """``base`` drawing independent samples: every call's sample index is shifted by ``offset``, so the same
    prompt gets a new completion (a different cache key) instead of the one already drawn. Used to judge the
    same transcripts again (held-out evaluation of best-of-n) and for preview judges that must not share
    samples with the final verdict."""

    def __init__(self, base, offset: int = 50_000, *, label: str | None = None):
        super().__init__(label=label or f"fresh{offset}({base.label or base.id})")
        self.base, self.offset = base, offset

    def describe(self) -> dict:
        return {**super().describe(), "base": self.base.describe(), "offset": self.offset}

    async def act(self, request, ctx):
        from so_arena.core.policy import ActContext

        sub = ActContext(role=ctx.role, sample_index=ctx.sample_index + self.offset, game=ctx.game,
                         tools=ctx.tools, seed=ctx.seed, workspace=ctx._slot, draw=ctx.draw)
        action = await self.base.act(request, sub)
        ctx.usage = ctx.usage + sub.usage
        return action


def families(items) -> dict[str, str]:
    """Item id -> testbed family (ids do not encode it reliably)."""
    return {it.id: it.metadata.get("family", it.domain) for it in items}


def candidate_items(per_generated: int = 150, seed: int = 0):
    """Every labelled testbed question: all labelled questions of the experiment and dataset families, and
    ``per_generated`` questions of each generated family (decision theory, social choice, number theory)."""
    import cvtestbed

    from so_arena.domains.testbed import to_item

    out = []
    for name, fam in cvtestbed.families().items():
        n = per_generated if fam.cost == "free" else None
        qs = cvtestbed.load([name], n=n, seed=seed, status="labelled")
        out += [to_item(q) for q in qs]
    seen, items = set(), []
    for it in out:
        if it.id not in seen and it.true_label is not None:
            seen.add(it.id)
            items.append(it)
    return items


BUNDLE = Path(__file__).resolve().parents[1] / "docs" / "results" / "decisive_data"


def selected_items(run: str | Path, limit: int | None = None):
    """The gap-selected items in one fixed random order: every stage takes a prefix, so a stage stopped by the
    budget covers a random subset, and the optimization items are a subset of the protocol items."""
    import random

    from so_arena.core.store import RunStore

    items = RunStore(str(Path(run) / "select")).items()
    if not items:  # a fresh checkout: the committed data bundle has the same items
        import gzip

        from so_arena.core.items import TaskItem

        with gzip.open(BUNDLE / "select" / "items.jsonl.gz", "rt") as f:
            items = [TaskItem.model_validate_json(line) for line in f if line.strip()]
    return random.Random(0).sample(items, len(items))[:limit]


def cache_spend(run: str | Path, stage: str) -> float:
    """What a stage has paid: the sum of the recorded cost of every completion in its response cache. Every paid
    call is cached once and replays cost nothing, so this is exact across interrupted and resumed jobs."""
    import sqlite3

    path = Path(run) / "cache" / stage / "responses.sqlite"
    if not path.exists():
        return 0.0
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)
    try:
        got = con.execute("SELECT SUM(json_extract(value, '$.usage.cost_usd')) FROM cache").fetchone()[0]
    finally:
        con.close()
    return float(got or 0.0)


def episode_cost(ep) -> float:
    return float(sum((u.cost_usd or 0.0) for u in ep.usage.values()))


class Ledger:
    """Spend per stage, persisted to ``<run>/spend.json`` so separate jobs add up; ``check`` stops a stage
    before it would cross the study's budget."""

    def __init__(self, run: str | Path, budget: float = 1500.0):
        self.path = Path(run) / "spend.json"
        self.budget = budget
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _reload(self) -> None:
        if self.path.exists():
            self.data = json.loads(self.path.read_text())

    @property
    def total(self) -> float:
        self._reload()  # stages running in parallel jobs write the same file
        return float(sum(self.data.values()))

    def set(self, stage: str, usd: float) -> None:
        """Record a stage's total spend (recomputed from its stores, so resumed runs are not counted twice)."""
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._reload()
            self.data[stage] = round(float(usd), 6)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
            tmp.replace(self.path)

    def remaining(self) -> float:
        return self.budget - self.total

    def check(self, next_cost: float = 0.0) -> bool:
        return self.total + next_cost <= self.budget


def chunks(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
