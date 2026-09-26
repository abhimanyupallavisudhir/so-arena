"""Releasing mechanism results before ground truth exists, and resolving them later.

Some ground truth arrives late (forecasting questions resolve months later) or never (open
research questions judged in public). A *release* publishes what the mechanisms concluded - each
item's outcome and every behaviour's rewards, i.e. what scored highly and what scored lowly - with
no claim about which was right. A SHA-256 manifest commits to the exact contents: publish the
digest (or timestamp it, e.g. with OpenTimestamps) and anyone can later verify the results were
not edited after the truth came out.

What a release contains is what the mechanisms saw and did: censored items (no ground truth, no
private information unless allowlisted), transcripts, positions, outcomes, rewards and model names.
What it withholds by default is the experimenter's account of *how* behaviour was produced - profile
names, tags, behaviour labels, policy ids and configurations, and the policies' annotations of their turns
(the component a mixture drew, a best-of-N scorer's values) - because in instructed-arm designs that
account is the ground truth: in profile ``arm=true`` (label ``argue_true``) the agent's position is
the correct answer. These are replaced by keyed pseudonyms that do not link episodes across items;
episodes are ordered by pseudonym within each item and keep only the date they were created, so
neither order nor timestamps reveal the arm either. Token counts and costs are withheld per turn and per
episode (a longer directive in one arm's system prompt shows in its input tokens); the manifest records
the whole release's total.

``resolve`` takes the release plus ground truth when it arrives, verifies the manifest, fills in
deferred rewards (e.g. market scoring rules), scores every episode, and writes metrics and a report.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
import logging
import secrets
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from so_arena.analysis.frames import config_key, mechanism_labels
from so_arena.core.game import Turn
from so_arena.core.ground_truth import GroundTruthScorer, default_scorers
from so_arena.core.items import GroundTruth, TaskItem
from so_arena.core.mechanism import Episode
from so_arena.core.rewards import RewardRule
from so_arena.core.runner import run_sync, score_episode
from so_arena.core.store import RunStore
from so_arena.core.types import Usage

log = logging.getLogger("so_arena")

FILES = ("items.jsonl", "episodes.jsonl", "rankings.json")  # data files every manifest covers
HTML = "index.html"  # the viewer; covered by the manifest too when written


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pseudonym(salt: str, *parts: Any) -> str:
    return hmac.new(salt.encode(), "|".join(map(str, parts)).encode(), hashlib.sha256).hexdigest()[:16]


# reward details safe to publish: whether an episode was audited is drawn from the item, repeat and seed alone
# (:func:`so_arena.core.rewards.audit_draw`); what an audit found - audited values, the label a judge audit
# returned - is ground truth (often simulated from it), and so is any detail a future rule adds: allowlist
PUBLIC_REWARD_DETAILS = ("audited",)
# turn fields safe to publish: what the agents did and what the mechanism saw; a field added later is reset to
# its default until it is listed here. Not ``usage``: input tokens count the prompt, whose directive (the arm's
# instructions) is not shown - arms whose directives differ in length would be told apart by it; likewise the
# usage of a turn's verifications and of the episode (the manifest keeps the release's total)
PUBLIC_TURN_FIELDS = ("index", "slot", "role", "phase", "kind", "text", "shown", "verdicts_to", "unmarked", "reasoning",
                      "visible_to", "choice",
                      "probs", "score", "data", "verifications", "tool_calls", "parse_ok", "node",
                      "candidate", "group", "state")
# Turn metadata holds the policies' annotations, among them the experimenter's account of how behaviour was
# produced: ``mixture_component`` names the component a MixturePolicy drew (in arm designs, the arm) and
# ``bon_scores`` what a best-of-N scorer (possibly ground truth) gave each candidate. Published: how a decision
# was elicited or observed, and - only with ``public_labels``, like policy ids - the drawn component.
PUBLIC_TURN_METADATA = ("elicitation", "votes", "probe_scores", "reported_tool_calls")
LABEL_TURN_METADATA = ("mixture_component",)
# mechanism configuration safe to publish: what was run - constructor arguments, the reward rule's description
# and the verifiers' names. The full reward-rule and verification-policy structures that episode ids hash
# describe functions by the values they close over or default to, and an audit oracle may close over the answer.
PUBLIC_MECHANISM_CONFIG = ("name", "class", "config", "reward", "verification")


def _without_closures(x: Any) -> Any:
    """A configuration description (:func:`so_arena.core.mechanism.describe_config`) with every function
    reduced to its name: the values it closes over, is bound to or partially applied to may be ground truth."""
    if isinstance(x, dict):
        if "function" in x and set(x) <= {"function", "params"}:
            return x["function"]
        if "partial" in x and set(x) <= {"partial", "args", "keywords"}:
            return {"partial": _without_closures(x["partial"])}
        if "method" in x and set(x) <= {"method", "of"}:
            return {"method": x["method"]}
        return {k: _without_closures(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_without_closures(v) for v in x]
    return x


def _public_config(ep: Episode, salt: str) -> dict[str, Any]:
    """The published summary of ``ep``'s mechanism configuration, with a keyed pseudonym of the full one: two
    configurations that differ only in withheld settings stay apart in every analysis of the release (whose
    labels, :func:`so_arena.analysis.frames.mechanism_labels`, can then quote published settings only)."""
    out = {k: _without_closures(v) for k, v in ep.mechanism_config.items() if k in PUBLIC_MECHANISM_CONFIG}
    return {**out, "configuration": _pseudonym(salt, "configuration", config_key(ep))}


def _public_turn(t: Turn, *, public_labels: bool) -> Turn:
    keep = PUBLIC_TURN_METADATA + (LABEL_TURN_METADATA if public_labels else ())
    fields = {f: getattr(t, f) for f in PUBLIC_TURN_FIELDS}
    # with verification noise, what a correct check would have said is the experimenter's: withheld
    fields["verifications"] = [v.model_copy(update={"usage": Usage(), "true_status": None, "true_output": None})
                               for v in t.verifications]
    return Turn(**fields, metadata={k: v for k, v in t.metadata.items() if k in keep})


def _strip(ep: Episode, *, salt: str | None = None, public_labels: bool = False) -> Episode:
    """The released form of an episode: no ground truth (nor audit findings in ``reward_details``), turns
    reduced to their public fields and metadata, the mechanism configuration to its public summary and,
    unless ``public_labels``, none of the experimenter's annotations that can encode it (see the module
    docstring). Positions and assigned stances are concrete answer labels - what the mechanism saw - and
    stay; so does ``item_id``."""
    e = ep.model_copy(deep=True)
    e.ground_truth = {}
    e.gt_status = "pending" if ep.gt_status in ("pending", "unscored") else "unknown"
    e.reward_details = {k: v for k, v in ep.reward_details.items() if k in PUBLIC_REWARD_DETAILS}
    e.turns = [_public_turn(t, public_labels=public_labels) for t in e.turns]
    e.usage = {}  # token counts reveal prompt lengths, i.e. directives (see PUBLIC_TURN_FIELDS)
    salt = salt if salt is not None else secrets.token_hex(16)
    e.mechanism_config = _public_config(ep, salt)
    if public_labels:
        return e
    # original ids embed the profile name (and hash it), so they are replaced too
    e.id = f"{ep.mechanism}:{ep.item_id}:{_pseudonym(salt, 'episode', ep.id)}"
    e.profile = f"profile-{_pseudonym(salt, 'profile', ep.item_id, ep.profile)[:10]}"  # unlinkable across items
    e.tags = {}
    e.created_at = ep.created_at[:10]  # the date: finer timestamps order an item's episodes by arm
    for r, p in e.players.items():
        name = p.model or r  # the behaviour is identified by its model (or role) only
        e.players[r] = p.model_copy(update={"label": name, "policy_id": name, "config": {}})
    return e


def rankings(episodes: Sequence[Episode]) -> dict[str, Any]:
    """Mechanism-only summaries: per item outcomes, and behaviours ranked by mean reward - per configuration
    of a mechanism (labelled as in :func:`so_arena.analysis.frames.mechanism_labels`), never pooled by name."""
    labels = mechanism_labels(episodes)
    per_item: dict[str, dict[str, Any]] = {}
    for ep in episodes:
        if ep.error:
            continue
        d = per_item.setdefault(ep.item_id, {})
        d.setdefault(labels[(ep.mechanism, config_key(ep))], []).append({
            "profile": ep.profile, "decision": ep.outcome.decision, "probs": ep.outcome.probs,
            "rewards": ep.rewards, "positions": ep.positions, "labels": {r: p.label for r, p in ep.players.items()},
        })
    rows = []
    for ep in episodes:
        if ep.error:
            continue
        for r, v in ep.rewards.items():
            if v is None:
                continue
            p = ep.players.get(r)
            rows.append({"mechanism": labels[(ep.mechanism, config_key(ep))], "role": r,
                         "behaviour": (p.label if p else None) or r, "reward": v})
    board = []
    if rows:
        df = pd.DataFrame(rows)
        g = df.groupby(["mechanism", "role", "behaviour"])["reward"].agg(["mean", "count", "std"]).reset_index()
        g = g.sort_values(["mechanism", "role", "mean"], ascending=[True, True, False])
        board = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()} for r in g.to_dict("records")]
    return {"per_item": per_item, "leaderboard": board}


class Manifest(BaseModel):
    title: str
    created_at: str
    files: dict[str, str]  # file name -> SHA-256: the data files and, when written, the viewer
    digest: str
    n_items: int
    n_episodes: int
    notes: str = ""
    mechanisms: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)  # model usage of all released episodes together (never per episode)


def _digest(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _released_item(item: TaskItem, private_keys: Sequence[str]) -> TaskItem:
    """Censored item without private information (answer keys, reference solutions, hidden passages)
    except the allowlisted keys; metadata stays because resolution needs it (e.g. market ids)."""
    it = item.censored(keep_metadata=True)
    return it.model_copy(update={"private": {k: v for k, v in it.private.items() if k in private_keys}})


def release(episodes: Sequence[Episode], items: Sequence[TaskItem], out_dir: str | Path, *, title: str = "Release",
            notes: str = "", html: bool = True, exclude_restricted: bool = False, private_keys: Sequence[str] = (),
            public_labels: bool = False, salt: str | None = None) -> Manifest:
    """Write a ground-truth-free release bundle and return its manifest (``manifest.digest`` is the commitment).

    Items whose licence forbids publication (``metadata["do_not_publish"]``, e.g. GPQA) are refused
    unless ``exclude_restricted=True``, which drops them and their episodes from the bundle.

    * ``TaskItem.private`` is removed except for ``private_keys`` (an explicit allowlist, e.g. a
      passage that is safe to publish): private information is what makes an overseer weak, and it
      often *is* the answer (``answer_key``, a reference ``solution``).
    * Item metadata is kept (resolution needs it), so do not release items whose metadata or ids
      reveal ground truth (e.g. review items of paired work designs, which name the arm).
    * ``public_labels=True`` publishes profile names, tags and behaviour labels as they are - only
      for designs where they are not defined relative to the truth (e.g. named forecasters in a
      market). Otherwise they are pseudonymized with ``salt`` (random if not given and never stored:
      pass and keep your own to reproduce a bundle or map pseudonyms back to the run's episodes).
    * Designs where a role always takes the same side of the truth (e.g. game stances
      ``{debater_a: true}``) reveal it through positions; run both side assignments before releasing.
    """
    restricted = {it.id for it in items if it.metadata.get("do_not_publish")}
    if restricted and not exclude_restricted:
        raise ValueError(f"{len(restricted)} items are marked do_not_publish (e.g. {sorted(restricted)[:3]}); "
                         "pass exclude_restricted=True to leave them out of the release")
    items = [it for it in items if it.id not in restricted]
    episodes = [e for e in episodes if e.item_id not in restricted]
    audited = sum(1 for e in episodes if (e.reward_details or {}).get("audited"))
    if audited:  # the findings are withheld, but an audited reward is computed from them
        log.warning("%d released episode(s) were audited: their rewards encode what the audit found (with "
                    "truth_oracle, the ground truth itself) - release them only if the audit was a real check",
                    audited)
    have = {k for it in items for k in it.private}
    if missing := [k for k in private_keys if k not in have]:
        raise ValueError(f"private_keys {missing} occur in no released item (private keys: {sorted(have)})")
    salt = salt if salt is not None else secrets.token_hex(16)
    total = sum((e.total_usage for e in episodes), Usage())
    released = [_strip(e, salt=salt, public_labels=public_labels) for e in episodes]
    # configurations of a mechanism are kept (and listed) apart, by labels made from what is published
    labels = mechanism_labels(released)
    group: dict[tuple[str, str], int] = {}
    for e in released:
        group.setdefault((labels[(e.mechanism, config_key(e))], e.item_id), len(group))
    # grouped as in the run, ordered by pseudonym within each (mechanism configuration, item): the run's order
    # follows the arms
    eps = sorted(released, key=lambda e: (group[(labels[(e.mechanism, config_key(e))], e.item_id)], e.id))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "items.jsonl", "w") as f:
        for it in items:
            f.write(_released_item(it, private_keys).model_dump_json() + "\n")
    with open(out / "episodes.jsonl", "w") as f:
        for e in eps:
            f.write(e.model_dump_json() + "\n")
    (out / "rankings.json").write_text(json.dumps(rankings(eps), indent=1, default=str))
    names = list(FILES)
    if html:
        from so_arena.analysis.report import build_report

        # written before hashing so the commitment covers it (so it cannot show its own digest)
        build_report(eps, out / HTML, title=title, subtitle="mechanism results · ground truth not included",
                     hide_ground_truth=True)
        names.append(HTML)
    files = {name: _sha(out / name) for name in names}
    man = Manifest(title=title, created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(), files=files,
                   digest=_digest(files), n_items=len(items), n_episodes=len(eps), notes=notes,
                   mechanisms=sorted(set(labels.values())), usage=total)
    (out / "MANIFEST.json").write_text(man.model_dump_json(indent=2))
    return man


def release_run(run_dir: str | Path, out_dir: str | Path, **kwargs: Any) -> Manifest:
    store = RunStore(run_dir)
    return release(store.episodes(), store.items(), out_dir, **kwargs)


def release_digest(release_dir: str | Path) -> str:
    """Digest of the manifest's files as they are now (equal to ``MANIFEST.json``'s if nothing changed)."""
    d = Path(release_dir)
    man = Manifest.model_validate_json((d / "MANIFEST.json").read_text())
    return _digest({name: _sha(d / name) if (d / name).exists() else "missing" for name in man.files})


def uncovered_files(release_dir: str | Path) -> list[str]:
    """Release files present in the directory but not covered by its manifest (e.g. the viewer of a
    bundle made before the manifest covered it): nothing vouches for them."""
    d = Path(release_dir)
    man = Manifest.model_validate_json((d / "MANIFEST.json").read_text())
    return [n for n in (*FILES, HTML) if (d / n).exists() and n not in man.files]


def verify(release_dir: str | Path, digest: str | None = None) -> bool:
    """True iff the files match the manifest, and - given the ``digest`` published at release time -
    the manifest is the one committed to.

    Without ``digest`` this only shows the bundle is internally consistent: whoever edits the files
    can rewrite ``MANIFEST.json`` as well, so compare against the published digest.
    """
    d = Path(release_dir)
    man = Manifest.model_validate_json((d / "MANIFEST.json").read_text())
    if not set(FILES) <= set(man.files) or not all((d / name).exists() for name in man.files):
        return False
    files = {name: _sha(d / name) for name in man.files}
    ok = files == man.files and _digest(files) == man.digest
    return ok and (digest is None or man.digest == digest.strip().lower())


class Resolution(BaseModel):
    release_digest: str
    resolved_at: str
    n_resolved: int
    n_unresolved: int
    metrics: dict[str, Any] = Field(default_factory=dict)


Truth = dict[str, Any] | Callable[[list[TaskItem]], dict[str, Any]]


def _option_label(item: TaskItem, label: Any) -> str:
    """The answer option ``label`` names: itself, else its unique case-insensitive match (Manifold's
    ``YES`` -> ``yes``). A label that is no option would score every answer as wrong."""
    labels = item.labels
    if not labels or label in labels:  # items without options accept any label
        return label
    matches = [lab for lab in labels if lab.casefold() == str(label).strip().casefold()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"item {item.id!r}: {label!r} is not one of its answer options {labels}")


def _apply_truth(item: TaskItem, t: Any) -> TaskItem:
    if t is None:
        return item
    if isinstance(t, GroundTruth):
        upd: dict[str, Any] = {}
        if t.correct is not None:
            upd["correct"] = _option_label(item, t.correct)
        if t.values:
            upd["values"] = {_option_label(item, k): v for k, v in t.values.items()}
        return item.model_copy(update={"ground_truth": t.model_copy(update=upd)})
    if isinstance(t, str):
        t = _option_label(item, t)
        answers = [a.model_copy(update={"value": 1.0 if a.label == t else -1.0}) for a in item.answers or []]
        return item.model_copy(update={"ground_truth": GroundTruth(status="known", correct=t), "answers": answers or item.answers})
    if isinstance(t, dict):
        vals = {_option_label(item, k): float(v) for k, v in t.items()}
        answers = [a.model_copy(update={"value": vals[a.label]}) for a in item.answers or [] if a.label in vals]
        return item.model_copy(update={"ground_truth": GroundTruth(status="known", values=vals),
                                       "answers": answers or item.answers})
    raise TypeError(f"unsupported ground truth {t!r}")


def resolve(release_dir: str | Path, truth: Truth, *, out_dir: str | Path | None = None,
            ground_truth: Sequence[GroundTruthScorer] | None = None, reward_rule: RewardRule | None = None,
            require_valid: bool = True, html: bool = True, digest: str | None = None) -> Resolution:
    """Resolve a release with ground truth (labels, value dicts, GroundTruth objects, or a callable).

    Labels must name answer options (matched case-insensitively if unambiguous); anything else is
    refused rather than scored as a wrong answer. Pass the domain's ground-truth scorers (e.g.
    ``dom.ground_truth_scorers()``) where the defaults do not measure what matters (forecast scores).
    Deferred rewards are recomputed with ``reward_rule`` (defaults to each episode's stored rewards
    unless ``outcome.data`` needs a resolution, in which case pass e.g. ``MarketScoringReward()``).
    With ``digest`` (the published commitment) the release must also match it.
    """
    d = Path(release_dir)
    if require_valid and not verify(d, digest):
        raise ValueError("release files do not match MANIFEST.json" + (" or the published digest" if digest else "")
                         + " - refusing to resolve an altered release")
    man = Manifest.model_validate_json((d / "MANIFEST.json").read_text())
    items = [TaskItem.model_validate_json(x) for x in (d / "items.jsonl").read_text().splitlines() if x.strip()]
    eps = [Episode.model_validate_json(x) for x in (d / "episodes.jsonl").read_text().splitlines() if x.strip()]
    truths = truth(items) if callable(truth) else truth
    if unknown := [k for k in truths if k not in {it.id for it in items}]:
        log.warning("ground truth for %d item(s) not in the release ignored (e.g. %s)", len(unknown), unknown[:3])
    resolved_items, errors = {}, []
    for it in items:
        try:
            resolved_items[it.id] = _apply_truth(it, truths.get(it.id))
        except ValueError as e:
            errors.append(str(e))
    if errors:
        raise ValueError(f"{len(errors)} ground-truth label(s) are not answer options: " + "; ".join(errors[:5]))
    scorers = list(ground_truth) if ground_truth is not None else default_scorers()

    async def go() -> list[Episode]:
        out = []
        for ep in eps:
            it = resolved_items.get(ep.item_id)
            if it is None or truths.get(ep.item_id) is None:
                out.append(ep)
                continue
            e = ep.model_copy(deep=True)
            if it.ground_truth is not None and it.ground_truth.correct is not None:
                e.outcome.data["resolution"] = it.ground_truth.correct
            if reward_rule is not None:
                e.rewards = await reward_rule.acompute(e, None)
                e.reward_status = "pending" if any(v is None for r, v in e.rewards.items() if r in e.trainable_roles) else "final"
            out.append(await score_episode(e, it, scorers))
        return out

    scored = run_sync(go())
    n_res = sum(1 for e in scored if e.gt_status == "known")
    metrics: dict[str, Any] = {}
    from so_arena.analysis.frames import role_frame
    from so_arena.analysis.metrics import asd, incentive_alignment, judge_accuracy

    df = role_frame([e for e in scored if e.gt_status == "known"])
    if not df.empty:
        for name, fn in (("asd", asd), ("alignment", incentive_alignment), ("judge_accuracy", judge_accuracy)):
            try:
                t = fn(df)
                metrics[name] = json.loads(t.to_json(orient="records"))
            except Exception as e:  # metrics that need arms that are absent are skipped
                metrics[name] = {"error": repr(e)}
    res = Resolution(release_digest=man.digest, resolved_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                     n_resolved=n_res, n_unresolved=len(scored) - n_res, metrics=metrics)
    out = Path(out_dir) if out_dir is not None else d / "resolved"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "episodes.jsonl", "w") as f:
        for e in scored:
            f.write(e.model_dump_json() + "\n")
    (out / "resolution.json").write_text(res.model_dump_json(indent=2))
    if html:
        from so_arena.analysis.report import build_report

        build_report(scored, out / "report.html", title=f"{man.title} (resolved)",
                     subtitle=f"release {man.digest[:12]} · {n_res} episodes resolved")
    return res
