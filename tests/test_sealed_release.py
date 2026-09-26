"""Sealed releases: commit-then-reveal with per-item salted commitments and Merkle inclusion proofs."""

import json

import pytest

import so_arena as soa
from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
from so_arena.mechanisms import Debate, DirectJudge
from so_arena.release import (
    COMMITMENTS,
    OPENINGS,
    Manifest,
    inclusion_proof,
    release,
    release_digest,
    resolve,
    reveal,
    uncovered_files,
    verify,
)
from so_arena.release.commit import commitment, merkle_proof, merkle_root, verify_opening, verify_proof
from so_arena.samplers.arms import ASDExperiment


@pytest.fixture(scope="module")
def run():
    dom = SyntheticPersuasion(n_items=5, seed=4)
    items = dom.load()
    mechs = [DirectJudge(), Debate(rounds=1, affordances={"agents": ["answer_key"]})]
    eps = ASDExperiment(mechs, items, agent=synthetic_arguer(), fixtures={"judge": synthetic_judge()},
                        ctx=dom.context()).run()
    return items, eps


def _sealed(run, tmp_path, **kw):
    items, eps = run
    rel = tmp_path / "rel"
    man = release(eps, items, rel, title="Sealed", sealed=True, salt="kept-private", **kw)
    return items, eps, rel, man


# ----------------------------------------------------------------------------- the commitment scheme

def test_merkle_proofs_and_commitments():
    for n in range(1, 10):
        leaves = [commitment({"i": i}, f"{i:064x}") for i in range(n)]
        root = merkle_root(leaves)
        for i in range(n):
            proof = merkle_proof(leaves, i)
            assert verify_proof(leaves[i], proof, root)
            assert not verify_proof(leaves[(i + 1) % n], proof, root) or n == 1
            if proof:
                bad = [list(p) for p in proof]
                bad[0][1] = "L" if bad[0][1] == "R" else "R"
                assert not verify_proof(leaves[i], bad, root)
    a, b, c = (commitment(x, "00" * 32) for x in "abc")
    assert merkle_root([a, b, c]) != merkle_root([a, b, c, c])  # an unpaired node is promoted, not duplicated
    # hiding: the same content under two salts; binding: other content under the same salt
    assert commitment({"decision": "A"}, "01" * 32) != commitment({"decision": "A"}, "02" * 32)
    assert commitment({"decision": "A"}, "01" * 32) != commitment({"decision": "B"}, "01" * 32)


# ----------------------------------------------------------------------------- sealing

def test_sealed_bundle_publishes_only_commitments(run, tmp_path):
    items, eps, rel, man = _sealed(run, tmp_path)
    assert man.sealed and man.digest == release_digest(rel) and man.mechanisms == [] and man.n_items == len(items)
    assert {p.name for p in rel.iterdir()} == {"MANIFEST.json", COMMITMENTS}
    assert (tmp_path / "rel.private" / OPENINGS).exists()  # the openings stay outside the bundle
    published = "".join(p.read_text() for p in rel.iterdir())
    for needle in [*(it.id for it in items), *(it.question[:40] for it in items),
                   "arm", "argue_true", "synthetic", "debate", "direct", "judge", '"A"', '"B"', "answer_key"]:
        assert needle not in published, needle
    rows = [json.loads(x) for x in (rel / COMMITMENTS).read_text().splitlines()]
    assert len(rows) == len(items) + 1 and set(rows[0]) == {"leaf", "commitment", "proof"}  # + the summary
    assert verify(rel) and verify(rel, man.digest) and not verify(rel, "0" * 64)
    assert uncovered_files(rel) == []
    with pytest.raises(ValueError, match="not been revealed"):
        resolve(rel, {it.id: it.true_label for it in items})


def test_private_directory_must_lie_outside_the_bundle(run, tmp_path):
    items, eps = run
    with pytest.raises(ValueError, match="outside"):
        release(eps, items, tmp_path / "rel", sealed=True, private_dir=tmp_path / "rel" / "keys")
    with pytest.raises(ValueError, match="outside"):
        release(eps, items, tmp_path / "a" / "rel", sealed=True, private_dir=tmp_path / "a")


def test_reveal_item_by_item_then_everything(run, tmp_path):
    items, eps, rel, man = _sealed(run, tmp_path)
    first = items[0].id
    out = reveal(rel, items=[first])
    assert out == {"revealed": 1, "sealed": len(items), "complete": False}
    assert verify(rel, man.digest)
    shown = [json.loads(x) for x in (rel / "items.jsonl").read_text().splitlines()]
    assert [it["id"] for it in shown] == [first] and not (rel / "rankings.json").exists()  # the summary waits
    assert {json.loads(x)["item_id"] for x in (rel / "episodes.jsonl").read_text().splitlines()} == {first}
    # one item checks against the published digest alone
    p = inclusion_proof(rel, first)
    assert p["root"] == man.digest and verify_opening(p["opening"], p["proof"], man.digest)
    with pytest.raises(KeyError):
        inclusion_proof(rel, items[1].id)

    assert reveal(rel)["complete"] and verify(rel, man.digest)
    # what was committed is what an unsealed release publishes (same pseudonym salt)
    plain = tmp_path / "plain"
    release(eps, items, plain, salt="kept-private", html=False)
    for name in ("items.jsonl", "episodes.jsonl"):
        parse = lambda d: sorted(json.dumps(json.loads(x), sort_keys=True) for x in (d / name).read_text().splitlines())  # noqa: E731
        assert parse(rel) == parse(plain), name
    board = json.loads((rel / "rankings.json").read_text())
    assert board == json.loads((plain / "rankings.json").read_text())
    # leak-proofing is intact: no private information, ground truth or arm names in the revealed data
    revealed = "".join((rel / n).read_text() for n in ("items.jsonl", "episodes.jsonl", "rankings.json", OPENINGS))
    assert all(json.loads(x)["private"] == {} for x in (rel / "items.jsonl").read_text().splitlines())
    assert "argue_true" not in revealed and '"ground_truth":{}' in revealed.replace(" ", "")
    res = resolve(rel, {it.id: it.true_label for it in items}, html=False)
    assert res.release_digest == man.digest and res.n_resolved == len(eps)


def test_tampering_is_detected(run, tmp_path):
    items, eps, rel, man = _sealed(run, tmp_path)
    reveal(rel)
    backup = {p.name: p.read_text() for p in rel.iterdir()}

    def restore():
        for name, text in backup.items():
            (rel / name).write_text(text)
        assert verify(rel, man.digest)

    # an edited episode in the data files
    p = rel / "episodes.jsonl"
    p.write_text(p.read_text().replace('"judgment"', '"judgement"', 1))
    assert not verify(rel)
    with pytest.raises(ValueError):
        resolve(rel, {it.id: it.true_label for it in items})
    restore()
    # an edited opening, with the data files rewritten to match it
    rows = [json.loads(x) for x in (rel / OPENINGS).read_text().splitlines()]
    item_row = next(r for r in rows if r["content"]["kind"] == "item")
    item_row["content"]["episodes"][0]["outcome"]["decision"] = "Z"
    (rel / OPENINGS).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    from so_arena.release import _revealed_files

    for name, text in _revealed_files(rows).items():
        (rel / name).write_text(text)
    assert not verify(rel)
    restore()
    # another salt for the same content
    rows = [json.loads(x) for x in (rel / OPENINGS).read_text().splitlines()]
    rows[0]["salt"] = "ab" * 32
    (rel / OPENINGS).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    assert not verify(rel)
    restore()
    # re-committed contents with a rewritten manifest: consistent, but not what the published digest commits to
    c = [json.loads(x) for x in (rel / COMMITMENTS).read_text().splitlines()]
    c[0]["commitment"] = "11" * 32
    hashes = [r["commitment"] for r in c]
    root = merkle_root(hashes)
    (rel / COMMITMENTS).write_text("".join(json.dumps({**r, "proof": merkle_proof(hashes, r["leaf"])}) + "\n" for r in c))
    import hashlib

    forged = Manifest.model_validate_json(backup["MANIFEST.json"]).model_copy(update={
        "digest": root, "files": {COMMITMENTS: hashlib.sha256((rel / COMMITMENTS).read_bytes()).hexdigest()}})
    (rel / "MANIFEST.json").write_text(forged.model_dump_json())
    assert not verify(rel, man.digest)
    restore()


def test_a_tampered_private_opening_is_not_revealed(run, tmp_path):
    items, eps, rel, man = _sealed(run, tmp_path)
    priv = tmp_path / "rel.private" / OPENINGS
    rows = [json.loads(x) for x in priv.read_text().splitlines()]
    target = next(r for r in rows if r["content"]["kind"] == "item")
    target["content"]["item"]["question"] = "something else"
    priv.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    with pytest.raises(ValueError, match="does not match"):
        reveal(rel)
    assert not (rel / "items.jsonl").exists() and verify(rel, man.digest)


def test_cli_sealed_release_and_reveal(run, tmp_path, capsys):
    from so_arena import cli
    from so_arena.core.store import RunStore

    items, eps = run
    store = RunStore(tmp_path / "run")
    store.save_items(items)
    for e in eps:
        store.append(e)
    out = tmp_path / "pub"
    assert cli.main(["release", str(tmp_path / "run"), str(out), "--sealed"]) == 0
    digest = Manifest.model_validate_json((out / "MANIFEST.json").read_text()).digest
    assert digest in capsys.readouterr().out and set(p.name for p in out.iterdir()) == {"MANIFEST.json", COMMITMENTS}
    assert cli.main(["reveal", str(out), "--item", items[0].id]) == 0
    assert cli.main(["verify", str(out), "--digest", digest]) == 0
    assert cli.main(["reveal", str(out)]) == 0 and (out / "rankings.json").exists()
    assert cli.main(["verify", str(out), "--digest", digest]) == 0
