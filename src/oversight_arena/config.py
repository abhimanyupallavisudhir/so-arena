"""YAML/JSON experiment configs.

Example (``configs/oversight_arena/gsm8k_asd.yaml``)::

    name: gsm8k-asd
    out: runs/gsm8k-asd
    domain: {type: gsm8k, limit: 50}
    mechanisms:
      - {type: debate, rounds: 2}
      - {type: consultancy, rounds: 2}
      - {type: propaganda}
      - {type: naive_judge}
    agents:
      "*": {type: llm, model: "${OA_EXPERT_MODEL:-mockllm/model}"}
      "kind:judge": {type: llm_judge, model: "${OA_JUDGE_MODEL:-mockllm/model}"}
    profiles: {type: stances}      # ASD worlds; roles inferred per mechanism
    clearances: null               # override per role / "kind:<kind>"
    concurrency: 8
    analysis: {gt: correct}
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .experiment.runner import Experiment
from .registry import build


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(text: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` (e.g. model names) before parsing."""
    return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else ""), text)


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = expand_env(p.read_text())
    return json.loads(text) if p.suffix == ".json" else yaml.safe_load(text)


def build_agents(spec: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, s in spec.items():
        if isinstance(s, str):
            s = {"type": "llm", "model": s}
        s = dict(s)
        typ = s.pop("type", "llm")
        out[key] = build("agent", {"type": typ, **s})
    return out


def experiment_from_config(cfg: dict[str, Any], **overrides: Any) -> Experiment:
    cfg = {**cfg, **{k: v for k, v in overrides.items() if v is not None}}
    domain = build("domain", cfg["domain"])
    if cfg.get("limit") is not None:
        domain.limit = int(cfg["limit"])
    mechs = [build("mechanism", m) for m in cfg.get("mechanisms", [cfg.get("mechanism")])]
    agents = build_agents(cfg["agents"])
    prof = cfg.get("profiles")
    profiles: Any = None
    if isinstance(prof, dict) and all(isinstance(v, dict) and "type" in v for v in prof.values()) and "type" not in prof:
        profiles = {k: build("profiles", v) for k, v in prof.items()}  # per-mechanism sources
    elif prof is not None:
        profiles = build("profiles", prof)
    gt = [build("gt", g) for g in cfg["gt"]] if cfg.get("gt") else None
    return Experiment(
        domain, mechs, agents, profiles, gt=gt, clearances=cfg.get("clearances"),
        name=cfg.get("name"), out=cfg.get("out"), concurrency=int(cfg.get("concurrency", 8)),
        repeats=int(cfg.get("repeats", 1)), seed=int(cfg.get("seed", 0)),
    )
