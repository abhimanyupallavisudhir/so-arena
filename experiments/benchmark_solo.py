"""General model benchmark on the conceptual-validation testbed: every panel model alone on every labelled item.

Separate from the oversight experiments: no judge, no debaters. Each of the 20 model configurations of the panel
(``configs/panel/models.json``: Llama-3.1-8B to GPT-6 Astra, Artificial Analysis Intelligence Index v4.3 6.6 to 46,
with GPT-5.6 Sol at none / low / medium / high effort) answers each of the 3,437 labelled binary testbed questions
(``decisive_common.candidate_items``) once, reasoning before it answers, exactly as the panel's ``solo`` condition:
``DirectJudge`` with ``agent_policy(spec, solo=True)`` and verbalized probabilities.

Items are taken in the panel's fixed order (``random.Random(11)`` over the candidate pool), so the first 300 are the
panel preview's items. The preview drew two samples per (model, item) (its two arms); its ``arm == True`` sample is
reused as this benchmark's sample on those items for every model whose policy is unchanged since the preview, and
its ``arm == False`` sample is kept as an independent second draw (test-retest). Models whose policy was changed
(``CHANGED``) are run on all items.

Stages share one response cache (``<run>/cache/bench``), so a stage replays what an earlier stage paid for (the pilot
items are part of the full run). Spend per stage is the growth of that cache while the stage ran
(``<run>/spend.json``); every stage stops before the study's total would pass ``--budget``.

    python experiments/benchmark_solo.py --stage pilot --start 300 --n 20
    python experiments/benchmark_solo.py --stage full --budget 1000

Rows (one per model and item): ``<run>/rows/<model>.csv``. Analysis: ``benchmark_report.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import random
from pathlib import Path

import pandas as pd

import so_arena as soa
from so_arena.core.runner import run_sync
from so_arena.mechanisms import DirectJudge
from so_arena.samplers.arms import ASDExperiment

from decisive_common import cache_spend, map_keys
from decisive_panel import agent_policy, panel_config, register_prices
from decisive_protocols import GT, episode_rows

ROOT = Path(__file__).resolve().parents[1]
CACHE = "bench"
PREVIEW = ROOT / "docs" / "results" / "decisive_data" / "panel" / "rows.csv.gz"
SEED = 11  # decisive_panel.panel_items: the preview is the first 300 items of this order
log = logging.getLogger("benchmark")

# Changes to the panel's solo policy. The preview parsed under 95% of the answers of Claude Haiku 4.5, Claude Sonnet
# 4.5 and DeepSeek-V3.2 (87-92%): every failure was a visible chain of thought cut off at the panel's 3,000-token cap
# for non-reasoning models, before the answer. Reasoning configurations get 16,000 tokens (hidden reasoning plus
# answer), so every non-reasoning configuration gets the same budget here. A model listed in CHANGED is not
# comparable with its preview samples, so it runs on every item.
SOLO_MAX_TOKENS = 16000
CHANGED: dict[str, dict] = {m["name"]: {"max_tokens": SOLO_MAX_TOKENS} for m in panel_config()["agents"]
                            if m.get("reasoning_effort") in (None, "none")}


def models() -> list[dict]:
    return panel_config()["agents"]


def solo(spec: dict) -> soa.LLMPolicy:
    """The panel's solo policy, with this benchmark's fixes (``CHANGED``)."""
    pol = agent_policy(spec, solo=True)
    for k, v in CHANGED.get(spec["name"], {}).items():
        setattr(pol, k, v)
    return pol


def bench_items(run: Path, n: int | None = None):
    """Every labelled item in the panel's fixed random order (saved, so reruns and later analyses agree)."""
    from so_arena.core.items import TaskItem

    path = run / "items.jsonl.gz"
    if path.exists():
        with gzip.open(path, "rt") as f:
            items = [TaskItem.model_validate_json(x) for x in f if x.strip()]
    else:
        from decisive_common import candidate_items

        pool = candidate_items()
        items = random.Random(SEED).sample(pool, len(pool))
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as f:
            f.write("\n".join(it.model_dump_json() for it in items) + "\n")
    return items[:n]


ITEM_SET = ROOT / "docs" / "results" / "benchmark" / "item_set.csv"


def item_set(n: int | None = None, strata: list[str] | None = None, run: Path | str = "runs/benchmark",
             path: Path = ITEM_SET):
    """The benchmark-derived stratified item list (``benchmark_report.py``), as TaskItems in its fixed order: the
    first ``4k`` items hold ``k`` of each stratum (gap, easy, contested, hard) while every stratum lasts. ``strata``
    keeps only those strata (in the same order). Every model's solo correctness is in ``item_set.csv``."""
    df = pd.read_csv(path, usecols=["rank", "item", "stratum"]).sort_values("rank")
    if strata:
        df = df[df.stratum.isin(strata)]
    ids = df.item.tolist()[:n]
    by_id = {it.id: it for it in bench_items(Path(run))}
    return [by_id[i] for i in ids]


def preview_rows() -> pd.DataFrame:
    """The panel preview's solo rows (two samples per model and item)."""
    df = pd.read_csv(PREVIEW)
    return df[df.mechanism == "solo"].rename(columns={"debater": "model"}).drop(columns=["judge", "role", "value",
                                                                                         "reward"])


def rows_path(run: Path, name: str) -> Path:
    return run / "rows" / f"{name.replace('/', '_')}.csv"


def done(run: Path, name: str) -> set[str]:
    p = rows_path(run, name)
    return set(pd.read_csv(p, usecols=["item"]).item) if p.exists() else set()


class Spend:
    """Spend per stage in ``<run>/spend.json``: a stage's entry is what it had before this invocation plus the
    growth of the shared response cache since (paid calls are cached once; replays cost nothing)."""

    def __init__(self, run: Path, stage: str):
        self.run, self.stage, self.path = run, stage, run / "spend.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.base = float(self.data.get(stage, 0.0))
        self.start = cache_spend(run, CACHE)

    def update(self, max_age: float = 0.0) -> float:
        """Recompute this stage's spend from the cache (at most every ``max_age`` seconds) and return the total."""
        import time

        if max_age and time.monotonic() - getattr(self, "_at", -1e9) < max_age:
            return self.total
        self._at = time.monotonic()
        mine = round(self.base + cache_spend(self.run, CACHE) - self.start, 4)
        # other stages (e.g. the label audit) write the same file: keep their entries
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.data[self.stage] = mine
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        tmp.replace(self.path)
        return self.total

    @property
    def total(self) -> float:
        return float(sum(self.data.values()))


async def run_model(run: Path, spec: dict, items, *, chunk: int, concurrency: int, spend: Spend, budget: float,
                    stage: str) -> None:
    """Every item not yet in the model's rows, ``concurrency`` at a time. Each item is its own episode run, so a
    call a host leaves hanging holds one slot, not a batch; rows are appended every ``chunk`` items."""
    from so_arena.core.store import RunStore

    name = spec["name"]
    have = done(run, name)
    todo = [it for it in items if it.id not in have]
    store, policy = RunStore(str(run / "stores" / name)), solo(spec)
    sem = asyncio.Semaphore(concurrency)
    buf: list[dict] = []
    state = {"done": len(have), "stop": False}
    path = rows_path(run, name)

    def flush():
        if not buf:
            return
        df = pd.DataFrame(buf).rename(columns={"debater": "model"}).drop(columns=["judge", "role", "value",
                                                                                   "reward"], errors="ignore")
        df["stage"] = stage
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, mode="a", header=not path.exists(), index=False)
        log.info("%s: %d/%d items, parse %.0f%%, $%.4f/item; total $%.2f", name, state["done"], len(items),
                 100 * df.parse_ok.mean(), df.cost.sum() / len(df), spend.update())
        buf.clear()

    async def one(it):
        async with sem:
            if state["stop"] or spend.update(max_age=30) > budget:
                if not state["stop"]:
                    log.warning("%s: budget reached ($%.2f of $%.2f); stopping", name, spend.total, budget)
                state["stop"] = True
                return
            exp = ASDExperiment([DirectJudge(name="solo")], [it], agent=None, fixtures={"judge": policy},
                                ground_truth=GT, arms=("true",), repeats=1, concurrency=1)
            eps = await exp.arun()
            for ep in eps:  # appended here: an experiment given the store would reread it for every item
                store.append(ep)
            buf.extend(episode_rows(eps, judge="-", debater=name))
            state["done"] += 1
            if len(buf) >= chunk:
                flush()

    await asyncio.gather(*[one(it) for it in todo])
    flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/benchmark")
    ap.add_argument("--stage", default="full")
    ap.add_argument("--start", type=int, default=None,
                    help="first item index (default: 300 for preview-compatible models, 0 for CHANGED ones)")
    ap.add_argument("--n", type=int, default=None, help="number of items from --start (default: all)")
    ap.add_argument("--models", nargs="*", default=None, help="model names (default: all)")
    ap.add_argument("--chunk", type=int, default=100, help="rows appended every this many items")
    ap.add_argument("--concurrency", type=int, default=64, help="concurrent episodes per model")
    ap.add_argument("--budget", type=float, default=1000.0, help="stop before the study's total spend passes it")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic", "inspect_ai", "decisive"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    map_keys()
    run = Path(a.run)
    soa.configure(cache_dir=str(run / "cache" / CACHE), cache_reasoning=False)
    register_prices()
    specs = [m for m in models() if a.models is None or m["name"] in a.models]
    items = bench_items(run)
    spend = Spend(run, a.stage)

    def span(spec):
        start = a.start if a.start is not None else (0 if spec["name"] in CHANGED else 300)
        return items[start:None if a.n is None else start + a.n]

    async def go():
        await asyncio.gather(*[run_model(run, s, span(s), chunk=a.chunk, concurrency=a.concurrency, spend=spend,
                                         budget=a.budget, stage=a.stage) for s in specs])

    run_sync(go())
    print(json.dumps({"stage": a.stage, "spend": spend.data, "total": round(spend.update(), 2)}, indent=1))


if __name__ == "__main__":
    main()
