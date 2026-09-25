"""Tidy data frames from episodes: one row per (episode, role)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import pandas as pd

from so_arena.core.mechanism import Episode


def config_key(ep: Episode) -> str:
    """The episode's mechanism configuration, hashed (as :meth:`~so_arena.core.mechanism.Mechanism.config_hash`)."""
    return hashlib.sha256(json.dumps(ep.mechanism_config, sort_keys=True).encode()).hexdigest()[:10]


def _flat(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out: dict[str, Any] = {}
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: d}


def _short(v: Any) -> str:
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return s if len(s) <= 24 else s[:21] + "..."


def mechanism_labels(episodes: Sequence[Episode]) -> dict[tuple[str, str], str]:
    """``(mechanism name, config key) -> label`` for analysis.

    Mechanisms are named by class (``debate``), so two configurations of one mechanism - another reward
    transform, verification policy or number of rounds - would otherwise be pooled into one row of every
    analysis. A name used by a single configuration keeps it; otherwise each configuration is labelled by
    the settings that tell them apart (``debate(transform="prob")``), or by its config hash.
    """
    configs: dict[str, dict[str, dict[str, Any]]] = {}
    for ep in episodes:
        configs.setdefault(ep.mechanism, {}).setdefault(config_key(ep), ep.mechanism_config)
    labels: dict[tuple[str, str], str] = {}
    for name, by_key in configs.items():
        if len(by_key) == 1:
            labels[(name, next(iter(by_key)))] = name
            continue
        flat = {k: _flat({"config": c.get("config"), "reward_rule": c.get("reward_rule"),
                          "verification_policy": c.get("verification_policy")}) for k, c in by_key.items()}
        fields = sorted({f for fl in flat.values() for f in fl})
        differ = [f for f in fields if len({json.dumps(fl.get(f), sort_keys=True, default=str) for fl in flat.values()}) > 1]
        # the constructor arguments the experimenter set, when they differ (reward-rule fields follow from them)
        differ = ([f for f in differ if f.startswith("config.")] or differ)[:3]
        made: dict[str, str] = {}
        for k, fl in flat.items():
            parts = [f"{f.split('.', 1)[1] if f.startswith('config.') else f}={_short(fl.get(f))}" for f in differ]
            made[k] = f"{name}({', '.join(parts)})" if parts else f"{name}#{k[:6]}"
        if len(set(made.values())) < len(made):
            made = {k: f"{v}#{k[:6]}" for k, v in made.items()}
        labels.update({(name, k): v for k, v in made.items()})
    return labels


def role_frame(episodes: Sequence[Episode], roles: Sequence[str] | None = None, *,
               include_errors: bool = False, include_fixtures: bool = False) -> pd.DataFrame:
    """One row per (episode, role).

    Columns include ``mechanism`` (its name, or a label per configuration where several configurations
    share one: :func:`mechanism_labels`) and ``mechanism_config`` (the config hash), ``reward`` (mechanism
    reward), ``value`` (ground-truth value of the role's behaviour), ``label`` (behaviour label),
    ``stance`` (assigned) and ``position`` (final),
    ``judge_p_true``/``judge_correct`` (control-style measures of the outcome), claim statistics and
    cost. Tags are added as ``tag_<name>`` columns.
    """
    rows: list[dict[str, Any]] = []
    labels = mechanism_labels(episodes)
    for ep in episodes:
        if ep.error is not None and not include_errors:
            continue
        gt = ep.ground_truth or {}
        rv = gt.get("role_values") or {}
        manip = gt.get("manipulation_ok") or {}
        total = ep.total_usage
        role_list = roles if roles is not None else [
            r for r in ep.players
            if include_fixtures or r in ep.trainable_roles or r in rv
        ]
        for role in role_list:
            if role not in ep.players and role not in ep.rewards:
                continue
            p = ep.players.get(role)
            pos = ep.positions.get(role)
            verifs = ep.verifications(role)
            probs = ep.outcome.probs or {}
            row = {
                "episode_id": ep.id,
                "run_id": ep.run_id,
                "item_id": ep.item_id,
                "domain": ep.domain,
                "mechanism": labels[(ep.mechanism, config_key(ep))],
                "mechanism_config": config_key(ep),
                "profile": ep.profile,
                "repeat": ep.repeat,
                "role": role,
                "kind": ep.role_kinds.get(role, "agent"),
                "trainable": role in ep.trainable_roles,
                "label": p.label if p else None,
                "policy_id": p.policy_id if p else None,
                "model": p.model if p else None,
                "stance": p.stance if p else None,
                "position": pos,
                "reward": ep.rewards.get(role),
                "reward_status": ep.reward_status,
                "value": rv.get(role),
                "gt_status": ep.gt_status,
                "p_position": probs.get(pos) if pos is not None else None,
                "decision": ep.outcome.decision,
                "judge_p_true": gt.get("judge_p_true"),
                "judge_correct": gt.get("judge_correct"),
                "outcome_value": gt.get("outcome_value"),
                "manipulation_ok": manip.get(role),
                "n_turns": sum(1 for t in ep.turns if t.role == role),
                "n_claims": len(verifs),
                "n_verified": sum(v.status == "verified" for v in verifs),
                "n_failed": sum(v.status == "refuted" for v in verifs),
                "parse_ok": all(t.parse_ok for t in ep.turns if t.role == role),
                "cost_usd": total.cost_usd,
                "tokens": total.total_tokens,
                "oversight_tokens": sum(
                    u.total_tokens for r, u in ep.usage.items() if ep.role_kinds.get(r) in ("judge", "monitor", "auditor", "grader")
                ),
                "error": ep.error is not None,
            }
            for k, v in ep.tags.items():
                row[f"tag_{k}"] = v
            for k, v in gt.items():
                if k not in ("role_values", "manipulation_ok") and isinstance(v, (int, float, str, bool)) and k not in row:
                    row[f"gt_{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def episode_frame(episodes: Sequence[Episode]) -> pd.DataFrame:
    """One row per episode (outcome-level)."""
    rows = []
    labels = mechanism_labels(episodes)
    for ep in episodes:
        gt = ep.ground_truth or {}
        row = {
            "episode_id": ep.id, "item_id": ep.item_id, "mechanism": labels[(ep.mechanism, config_key(ep))],
            "mechanism_config": config_key(ep), "profile": ep.profile,
            "repeat": ep.repeat, "decision": ep.outcome.decision, "error": ep.error is not None,
            "cost_usd": ep.total_usage.cost_usd, "tokens": ep.total_usage.total_tokens,
            **{f"reward_{r}": v for r, v in ep.rewards.items()},
            **{f"label_{r}": p.label for r, p in ep.players.items()},
            **{f"position_{r}": v for r, v in ep.positions.items()},
            **{k: v for k, v in gt.items() if isinstance(v, (int, float, str, bool))},
            **{f"tag_{k}": v for k, v in ep.tags.items()},
        }
        if ep.outcome.probs:
            row.update({f"p_{k}": v for k, v in ep.outcome.probs.items()})
        rows.append(row)
    return pd.DataFrame(rows)
