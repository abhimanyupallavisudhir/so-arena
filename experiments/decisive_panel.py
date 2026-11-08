"""Model panel: how does empirical incentive compatibility (EIC) change with the agent's and the judge's intelligence?

Agents: 20 configurations from Llama-3.1-8B to GPT-6 Astra (Artificial Analysis Intelligence Index v4.3, 6.6 to
46), including a reasoning-effort ladder (GPT-5.6 Sol at none / low / medium / high). Judges: 9 models from Gemma-3-12B
to DeepSeek-V3.2 (index 3.8 to 16), each reasoning before answering, as in the rest of the study. The panel follows
the legibench3 panel and its AA mapping conventions (``configs/panel/models.json``).

Items: a uniform random sample of the labelled testbed questions, **not** selected on any judge (the gap-selected
items were chosen because the weak judges fail, which makes their blind baselines negative by construction and
exaggerates any trend with judge capability).

Protocols, both instructed arms, ``--repeats`` samples per item and arm:

* ``debate_sim``: self-play, 3 rounds of simultaneous 200-word speeches, the same agent on both sides;
* ``consultancy_1r``: one 200-word argument for the assigned answer;
* ``direct``: the judge alone (blind baseline), and ``solo``: the agent alone (does it know the answer?).

Neither protocol lets an agent see the judge, so each transcript is generated once (with the first judge) and the
other judges' runs replay the agents' calls from the response cache: only judge calls are new.

Rows: ``<run>/panel/rows.csv``; stores of the first judge's episodes (for the liar audit): ``<run>/panel/stores``.
Analysis: ``decisive_panel_report.py``.

    python experiments/decisive_panel.py --run runs/decisive --items 300 --estimate
    python experiments/decisive_panel.py --run runs/decisive --items 300 --budget 2500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path

import pandas as pd

import so_arena as soa
from so_arena.core.runner import run_sync
from so_arena.mechanisms import Consultancy, Debate, DirectJudge
from so_arena.samplers.arms import ASDExperiment

from decisive_common import Ledger, cache_spend, chunks, map_keys
from decisive_protocols import GT, episode_rows

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "panel" / "models.json"
STAGE = "panel"
WORDS = 200


def panel_config() -> dict:
    return json.loads(CONFIG.read_text())


def register_prices() -> None:
    """Register $/Mtok for panel models the bundled registry lacks, from OpenRouter's live price list (OpenAI and
    Anthropic models served directly are priced from the same list, under their OpenRouter names)."""
    import urllib.request

    from so_arena.models.registry import get_spec, register_model

    cfg = panel_config()
    models = {m["model"] for m in cfg["agents"] + cfg["judges"]}
    missing = [m for m in models if get_spec(m) is None or get_spec(m).price_input_per_mtok is None]
    if not missing:
        return
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60) as r:
        prices = {m["id"]: m["pricing"] for m in json.load(r)["data"]}
    log = logging.getLogger("decisive")
    for m in missing:
        key = m.removeprefix("openrouter/")
        key = {"anthropic/claude-haiku-4-5": "anthropic/claude-haiku-4.5",
               "anthropic/claude-sonnet-4-5": "anthropic/claude-sonnet-4.5"}.get(key, key)
        p = prices.get(key)
        if p is None:
            log.warning("no price for %s: its cost will not be tracked", m)
            continue
        from so_arena.models.registry import ModelSpec

        register_model(ModelSpec(name=m, family=key.split("/")[-1], organization=key.split("/")[0],
                                 price_input_per_mtok=float(p["prompt"]) * 1e6,
                                 price_output_per_mtok=float(p["completion"]) * 1e6))


def _routed(spec: dict, model):
    """Pin OpenRouter hosts when the config names them (``openrouter_provider``): some DeepSeek-V3.2 hosts ignore
    "reasoning off" and think for 15-25k hidden tokens, which makes the model a different judge from call to call.
    The response cache key does not include the route, so cached calls still replay."""
    route = spec.get("openrouter_provider")
    if not route or not isinstance(model, str) or not model.startswith("openrouter/"):
        return model
    from so_arena.models.base import get_model

    return get_model(model, provider=route)


def agent_policy(spec: dict, *, solo: bool = False, strategy: str | None = None) -> soa.LLMPolicy:
    """An agent at its configured reasoning effort. Reasoning configurations get room for hidden reasoning (it
    shares max_tokens); the others write directly (solo answers reason in the open)."""
    from decisive_common import _model

    effort = spec.get("reasoning_effort")
    thinks = effort not in (None, "none")
    model = _routed(spec, _model(spec["model"], "speech_reasoning" if thinks else ("solo" if solo else "speech")))
    extra = {"strategy": strategy} if strategy else {}  # a searched strategy (decisive_panel_po.py)
    if thinks:
        return soa.LLMPolicy(model, temperature=None, max_tokens=16000, reasoning_effort=effort, **extra)
    return soa.LLMPolicy(model, temperature=None, max_tokens=3000 if solo else 1500, reasoning_effort=effort,
                         cot=solo, **extra)


def judge_policy(spec: dict) -> soa.LLMPolicy:
    """A judge reasoning in private before its verbalized probabilities (as the study's weak judges). Judges that
    work long problems out in the open (DeepSeek-V3.2) get a larger ``max_tokens`` in the config."""
    from decisive_common import _model

    return soa.LLMPolicy(_routed(spec, _model(spec["model"], "judge")), cot=True, temperature=0.7,
                         max_tokens=spec.get("max_tokens", 4000),
                         reasoning_effort=spec.get("reasoning_effort"))


def mechanisms():
    return [Debate(rounds=3, simultaneous=True, word_limit=WORDS, name="debate_sim"),
            Consultancy(rounds=1, word_limit=WORDS, judge_questions=False, name="consultancy_1r")]


def panel_items(run: Path, n: int, seed: int = 11):
    """``n`` labelled testbed questions drawn uniformly (fixed seed), saved so every chunk and rerun uses the same
    set; the first ``n`` of one fixed order, so a larger panel extends a smaller one."""
    from so_arena.core.items import TaskItem

    path = run / STAGE / "items.jsonl"
    if path.exists():
        items = [TaskItem.model_validate_json(x) for x in path.read_text().splitlines() if x.strip()]
        if len(items) >= n:
            return items[:n]
    from decisive_common import candidate_items

    pool = candidate_items()
    items = random.Random(seed).sample(pool, len(pool))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(it.model_dump_json() for it in items) + "\n")
    return items[:n]


async def run_chunk(run: Path, items, *, agents: list[dict], judges: list[dict], repeats: int, concurrency: int,
                    estimate: bool) -> list[dict]:
    rows: list[dict] = []

    async def go(mech, j, a, judge, agent, store=None, reps=repeats):
        exp = ASDExperiment([mech], items, agent=agent, fixtures={"judge": judge}, ground_truth=GT, repeats=reps,
                            store=None if estimate else store, concurrency=concurrency)
        got = episode_rows(await exp.arun(), judge=j, debater=a)
        rows.extend(got)

    first, *others = judges
    store = lambda *p: str(run / STAGE / "stores" / Path(*p))  # noqa: E731
    # phase A: the first judge pays for every transcript; references (blind judge, agent alone) alongside
    phase_a = [go(mech, first["name"], a["name"], judge_policy(first), agent_policy(a), store(a["name"], mech.name))
               for a in agents for mech in mechanisms()]
    phase_a += [go(DirectJudge(), j["name"], "-", judge_policy(j), None) for j in judges]
    phase_a += [go(DirectJudge(name="solo"), "-", a["name"], agent_policy(a, solo=True), None, reps=1)
                for a in agents]
    await asyncio.gather(*phase_a)
    # phase B: every other judge on the same transcripts (agents replay from the cache)
    await asyncio.gather(*[go(mech, j["name"], a["name"], judge_policy(j), agent_policy(a))
                           for j in others for a in agents for mech in mechanisms()])
    return rows


def done_items(path: Path, agents: list[str], judges: list[str]) -> set:
    """Items on which every (agent, protocol, judge) of this run already has rows: only those are skipped, so a
    rerun for agents that failed fills them in."""
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["item", "debater", "mechanism", "judge"])
    want = {(a, m.name, j) for a in agents for m in mechanisms() for j in judges}
    have = df[df.debater.isin(agents)].groupby("item").apply(
        lambda g: set(zip(g.debater, g.mechanism, g.judge)), include_groups=False)
    return {it for it, s in have.items() if want <= s}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--items", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--agents", nargs="*", default=None, help="agent names (default: all)")
    ap.add_argument("--judges", nargs="*", default=None, help="judge names (default: all; the first pays)")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--budget", type=float, default=2500.0)
    ap.add_argument("--estimate", action="store_true", help="dry run with simulated models; prints the cost")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic", "inspect_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    map_keys()
    run = Path(a.run)
    soa.configure(cache_dir=str(run / "cache" / STAGE), simulate=a.estimate or None, cache_reasoning=False)
    register_prices()
    cfg = panel_config()
    agents = [x for x in cfg["agents"] if a.agents is None or x["name"] in a.agents]
    judges = [x for x in cfg["judges"] if a.judges is None or x["name"] in a.judges]
    if a.judges:  # keep the requested order: the first judge generates the transcripts
        judges.sort(key=lambda x: a.judges.index(x["name"]))
    items = panel_items(run, a.items)
    log = logging.getLogger("decisive")
    path = run / STAGE / "rows.csv"
    ledger = Ledger(run)
    if a.estimate:
        rows = run_sync(run_chunk(run, items, agents=agents, judges=judges, repeats=a.repeats,
                                  concurrency=a.concurrency, estimate=True))
        df = pd.DataFrame(rows)
        print(df.groupby(["debater", "mechanism"]).cost.sum().round(2).to_string())
        print(f"estimated ${df.cost.sum():.2f} for {len(items)} items")
        return
    done = done_items(path, [x["name"] for x in agents], [x["name"] for x in judges])
    for batch in chunks([it for it in items if it.id not in done], a.chunk):
        spent = cache_spend(run, STAGE)
        if spent > a.budget:
            log.warning("budget reached ($%.2f of $%.2f); stopping", spent, a.budget)
            break
        rows = run_sync(run_chunk(run, batch, agents=agents, judges=judges, repeats=a.repeats,
                                  concurrency=a.concurrency, estimate=False))
        new = pd.DataFrame(rows)
        prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([prev, new]).drop_duplicates(["episode", "role", "judge"], keep="last").to_csv(path, index=False)
        ledger.set(STAGE, cache_spend(run, STAGE))
        log.info("chunk of %d items done; stage $%.2f", len(batch), ledger.data[STAGE])
    ledger.set(STAGE, cache_spend(run, STAGE))
    print(json.dumps({STAGE: ledger.data.get(STAGE)}, indent=1))


if __name__ == "__main__":
    main()
