import asyncio
import json
import subprocess
import sys

import pytest

from oversight_arena.optimization import Leaf, SamplingTurn, nested_best_of_n, sample_game_tree


def test_demo_cli_and_resolution(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "oversight_arena.cli", "demo", "--output", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["paired_asd"]["mean"] < 0
    assert (
        summary["curve"][-1]["rewards"]["consultant"] > summary["curve"][0]["rewards"]["consultant"]
    )
    assert summary["curve"][-1]["truth"]["consultant"] < summary["curve"][0]["truth"]["consultant"]
    assert summary["forecast"]["same_episode_id"]
    assert (tmp_path / "report.html").is_file()
    records = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    forecast = next(r for r in records if r["task"]["id"] == "forecast-1")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "oversight_arena.cli",
            "resolve-forecast",
            str(tmp_path / "runs.sqlite"),
            forecast["id"],
            "--role",
            "forecaster",
            "--outcome",
            "0",
            "--source",
            "corrected-fixture",
            "--resolved-at",
            "2026-09-25T00:00:00Z",
        ],
        check=True,
        capture_output=True,
    )
    from oversight_arena.storage import RunStore

    with RunStore(tmp_path / "runs.sqlite") as store:
        corrected = store.get(forecast["id"])
    assert len(corrected.evaluations) == 3
    assert corrected.evaluations[-1].per_agent["forecaster"]["quality"] < 0


def test_sampling_tree_preserves_prefix_and_conditions_continuations():
    seen = []

    async def expand(state, player, index, seed):
        seen.append((state.copy(), player, seed))
        state[player] = index
        return state

    async def score(state):
        payoff = float(state["a"] == state["b"])
        return Leaf({"a": payoff, "b": 1 - payoff}, {"a": float(state["a"] == 0)})

    initial = {}
    tree = asyncio.run(
        sample_game_tree(initial, [SamplingTurn("a", 2), SamplingTurn("b", 2)], expand, score)
    )
    result = nested_best_of_n(tree, {"a": 4, "b": 4})
    assert initial == {}
    assert sum(result.leaf_probabilities.values()) == pytest.approx(1)
    assert seen[1][0] == {"a": 0}
    assert seen[4][0] == {"a": 1}
    assert len({entry[2] for entry in seen}) == len(seen)


def test_plot_is_standalone_and_unresolved_is_rejected(tmp_path, episode_factory):
    pytest.importorskip("matplotlib")
    from dataclasses import replace

    from oversight_arena.optimization import best_of_n
    from oversight_arena.reporting import plot_optimization

    pool = [episode_factory(0.2, 1), episode_factory(0.9, 0)]
    path = plot_optimization(
        {"demo": [best_of_n(pool, "agent", n) for n in (1, 2, 4)]}, "agent", tmp_path / "curve.svg"
    )
    assert "<svg" in path.read_text()
    unknown = [replace(r, evaluations=()) for r in pool]
    with pytest.raises(ValueError, match="resolved"):
        plot_optimization(
            {"demo": [best_of_n(unknown, "agent", 1)]}, "agent", tmp_path / "missing.svg"
        )
