import csv
import json
import math
import urllib.error

import pytest

import so_arena as soa
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.domains import get_domain
from so_arena.domains import qa
from so_arena.domains.qa import (
    DISTRACTOR_KINDS,
    GPQA,
    GSM8K,
    MMLU,
    DatasetUnavailable,
    GatedDatasetError,
    QuALITY,
    TruthfulQA,
    gsm8k_wrong_answers,
)
from so_arena.mechanisms import Debate


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SO_ARENA_DATA", str(tmp_path / "cache"))


def offline(*a, **k):
    raise urllib.error.URLError("no network in this test")


def values(item):
    return sorted(a.value for a in item.answers)


# ----------------------------------------------------------------------------- GSM8K

def test_gsm8k_sample_binary_items():
    dom = GSM8K(offline=True)
    items = dom.load()
    assert dom.used_source == "sample" and len(items) == 50
    assert items[0].id == "gsm8k-test-0000" and len({it.id for it in items}) == 50
    for it in items:
        assert values(it) == [-1.0, 1.0] and "####" not in it.question
        correct = it.answer(it.true_label).text
        wrong = it.answer(it.false_labels[0]).text
        assert correct == it.ground_truth.data["answer"] and wrong != correct
        assert it.ground_truth.data["wrong_answers"] == [wrong]
        assert set(it.metadata["distractor_kinds"]) <= set(DISTRACTOR_KINDS)
        assert it.private["solution"].endswith(f"Answer: {correct}")
    assert {it.true_label for it in items} == {"A", "B"}
    first = GSM8K(offline=True, kinds=["intermediate"]).load()[0]
    assert first.answer(first.false_labels[0]).text == "9"  # dropped the final x2 step: 18 -> 9


def test_gsm8k_deterministic_by_seed_and_limit_subset():
    def sig(items):
        return [(it.id, [a.text for a in it.answers]) for it in items]

    a, b = GSM8K(offline=True).load(seed=3), GSM8K(offline=True).load(seed=3)
    assert sig(a) == sig(b)
    assert sig(a) != sig(GSM8K(offline=True).load(seed=4))
    sub = GSM8K(offline=True).load(limit=10, seed=1)
    ids = [it.id for it in sub]
    assert len(ids) == 10 and ids == sorted(ids) and ids != [f"gsm8k-test-{i:04d}" for i in range(10)]
    assert ids == [it.id for it in GSM8K(offline=True).load(limit=10, seed=1)]
    full = {it.id: [x.text for x in it.answers] for it in GSM8K(offline=True).load(seed=1)}
    assert all(full[it.id] == [x.text for x in it.answers] for it in sub)  # option order independent of limit


def test_gsm8k_wrong_answer_kinds():
    sol = "3 bags of 4 apples make 3*4=<<3*4=12>>12 apples.\nWith 5 more he has 12+5=<<12+5=17>>17.\n#### 17"
    c = gsm8k_wrong_answers(sol, "17")
    assert c["intermediate"] == ["12"]  # a dropped step
    assert set(c["wrong_op"]) == {"7", "60"}  # 12-5, 12*5 (12/5 is not an integer)
    assert set(c["factor"]) == {"34", "170"}  # x2, x10 (halves are not integers)
    assert set(c["perturb"]) == {"18", "16", "19", "15"}
    assert c["transpose"] == ["71"]
    assert "17" not in sum(c.values(), [])
    only = GSM8K(offline=True, kinds=["transpose"]).load()
    assert only and all(it.metadata["distractor_kinds"] == ["transpose"] for it in only)
    with pytest.raises(ValueError):
        GSM8K(kinds=["nonsense"])


def test_gsm8k_multiple_choice_and_verifier():
    items = GSM8K(binary=False, offline=True).load()
    for it in items:
        assert 2 <= len(it.answers) <= 4 and values(it).count(1.0) == 1
        kinds = it.metadata["distractor_kinds"]
        assert len(kinds) == len(it.answers) - 1
    assert all(len(it.answers) == 4 for it in items)
    assert sum(len(set(it.metadata["distractor_kinds"])) == 3 for it in items) > 40  # distinct slips first
    assert "python" in GSM8K().context().verifiers


def test_gsm8k_falls_back_to_sample_when_download_fails(monkeypatch, caplog):
    monkeypatch.setattr(qa, "download", offline)
    dom = GSM8K()
    items = dom.load(limit=5)
    assert dom.used_source == "sample" and len(items) == 5 and "bundled" in caplog.text
    with pytest.raises(DatasetUnavailable, match="no bundled"):
        GSM8K().load(split="train")


# ----------------------------------------------------------------------------- MMLU / TruthfulQA (canned rows)

MMLU_ROWS = [  # made up, in the cais/mmlu schema
    {"question": "What is 2 + 2?", "subject": "elementary_mathematics", "choices": ["3", "4", "5", "22"], "answer": 1},
    {"question": "Which gas do plants mostly absorb for photosynthesis?", "subject": "elementary_mathematics",
     "choices": ["Oxygen", "Carbon dioxide", "Helium", "Neon"], "answer": 1},
]


def test_mmlu_from_canned_rows(monkeypatch):
    calls = []

    def fake_rows(repo, config, split, parquet_path=None):
        calls.append((repo, config, split, parquet_path))
        return MMLU_ROWS

    monkeypatch.setattr(qa, "hf_rows", fake_rows)
    items = MMLU(subjects="elementary_mathematics").load()
    assert calls == [("cais/mmlu", "elementary_mathematics", "test", "elementary_mathematics/test-00000-of-00001.parquet")]
    assert [it.id for it in items] == ["mmlu-elementary_mathematics-test-0000", "mmlu-elementary_mathematics-test-0001"]
    it = items[0]
    assert it.question.startswith("Subject: elementary mathematics.") and len(it.answers) == 2
    assert it.answer(it.true_label).text == "4" and it.metadata["subject"] == "elementary_mathematics"
    mc = MMLU(subjects=["elementary_mathematics"], binary=False).load(split="val")
    assert [len(x.answers) for x in mc] == [4, 4] and mc[0].metadata["split"] == "validation"
    assert mc[0].answer(mc[0].true_label).text == "4" and values(mc[0]) == [-1.0, -1.0, -1.0, 1.0]
    with pytest.raises(ValueError):
        MMLU(subjects=["underwater_basket_weaving"])
    with pytest.raises(ValueError):
        MMLU().load(split="nope")


def test_offline_without_sample_is_a_clear_error(monkeypatch):
    with pytest.raises(DatasetUnavailable, match="no bundled sample"):
        MMLU(subjects="anatomy", offline=True).load()
    monkeypatch.setattr(qa, "hf_rows", offline)
    with pytest.raises(DatasetUnavailable, match="could not download"):
        MMLU(subjects="anatomy").load()


TRUTHFULQA_ROWS = [  # made up, in the mc_task.json schema
    {"question": "Can a goldfish remember things for more than three seconds?",
     "mc0_targets": {"Yes, goldfish can remember things for months": 1, "No, goldfish forget everything after three seconds": 0},
     "mc1_targets": {"Yes, goldfish can remember things for months": 1, "No, goldfish forget everything after three seconds": 0,
                     "No, goldfish have no memory": 0, "Only if they are trained": 0},
     "mc2_targets": {"Yes, goldfish can remember things for months": 1, "Yes": 1, "No": 0}},
]


def test_truthfulqa_binary_uses_best_incorrect_answer(monkeypatch, tmp_path):
    path = tmp_path / "mc_task.json"
    path.write_text(json.dumps(TRUTHFULQA_ROWS))
    monkeypatch.setattr(qa, "download", lambda url, *a, **k: path)
    [it] = TruthfulQA().load()
    assert it.id == "truthfulqa-0000" and it.metadata["distractor"] == "best"
    assert {a.text for a in it.answers} == set(TRUTHFULQA_ROWS[0]["mc0_targets"])
    assert it.answer(it.true_label).text.startswith("Yes, goldfish")
    [rnd] = TruthfulQA(distractor="random").load()
    assert rnd.metadata["distractor"] == "random" and len(rnd.answers) == 2
    [mc] = TruthfulQA(binary=False).load(split="test")
    assert len(mc.answers) == 4 and values(mc).count(1.0) == 1
    assert mc.ground_truth.data["true_answers"] == ["Yes, goldfish can remember things for months", "Yes"]


# ----------------------------------------------------------------------------- GPQA (gated; never bundled)

def test_gpqa_requires_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(GatedDatasetError, match="HF_TOKEN"):
        GPQA().load()

    def refuse(url, name, headers=None, **k):
        assert headers == {"Authorization": "Bearer hf_fake"}
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    monkeypatch.setattr(qa, "_fetch_to_cache", refuse)
    with pytest.raises(GatedDatasetError, match="accepted the dataset's terms"):
        GPQA(subset="main").load()
    with pytest.raises(ValueError):
        GPQA(subset="tiny")


def test_gpqa_parsing_with_made_up_row(monkeypatch, tmp_path):
    path = tmp_path / "gpqa_diamond.csv"
    cols = ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3",
            "Explanation", "Subdomain", "High-level domain", "Record ID"]
    with open(path, "w", newline="") as f:  # a made-up question, NOT a GPQA example
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow(dict(zip(cols, ["What colour is a clear daytime sky, typically? ", "Blue\n", "Green", "Red", "Purple",
                                   "Rayleigh scattering.", "Optics", "Physics", "recFAKE001"])))
    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    monkeypatch.setattr(qa, "_fetch_to_cache", lambda url, name, headers=None, **k: path)
    [it] = GPQA().load()
    assert it.id == "gpqa-diamond-recFAKE001" and it.metadata["do_not_publish"] is True
    assert it.answer(it.true_label).text == "Blue" and it.question == "What colour is a clear daytime sky, typically?"
    assert it.private["explanation"] == "Rayleigh scattering." and len(it.answers) == 2
    [mc] = GPQA(binary=False).load()
    assert sorted(a.text for a in mc.answers) == ["Blue", "Green", "Purple", "Red"]


# ----------------------------------------------------------------------------- QuALITY

PASSAGE = ("The lighthouse keeper, Mara Venn, had not seen a ship in forty days.\n\n On the forty-first morning she "
           "climbed the stairs and wrote in her log: “The fog has lifted, and the sea is empty.” Then she lit "
           "the lamp anyway, because the rules said so.")
ARTICLE = {  # a synthetic article in the QuALITY schema
    "article_id": "syn1", "title": "The Empty Sea", "author": "Nobody", "source": "synthetic", "article": PASSAGE,
    "questions": [
        {"question": "Why does Mara light the lamp?", "question_unique_id": "syn1_q1", "gold_label": 2, "writer_label": 2,
         "options": ["Because a ship is coming", "Because the rules require it", "To signal for help", "To warm the room"],
         "difficult": 1,
         "validation": [{"untimed_answer": 2, "untimed_best_distractor": 1}, {"untimed_answer": 2, "untimed_best_distractor": 1},
                        {"untimed_answer": 2, "untimed_best_distractor": 3}, {"untimed_answer": 2, "untimed_best_distractor": 2}],
         "speed_validation": [{"speed_answer": 1}, {"speed_answer": 2}, {"speed_answer": 3}, {"speed_answer": 1}]},
        {"question": "How long had Mara gone without seeing a ship?", "question_unique_id": "syn1_q2", "gold_label": 1,
         "options": ["Forty days", "Four days", "A year", "A week"], "difficult": 0,
         "validation": [{"untimed_answer": 1}, {"untimed_answer": 3}], "speed_validation": [{"speed_answer": 1}]},
        {"question": "An unlabelled (test-split) question?", "question_unique_id": "syn1_q3",
         "options": ["a", "b", "c", "d"]},
    ],
}


def test_quality_parsing_and_filters(monkeypatch):
    recs = QuALITY.parse_article(ARTICLE)
    assert [r.id for r in recs] == ["quality-syn1_q1", "quality-syn1_q2"]  # unlabelled questions dropped
    r1, r2 = recs
    assert r1.best_distractor == "Because a ship is coming"  # most-voted distractor (votes for the answer ignored)
    assert r1.metadata["untimed_accuracy"] == 1.0 and r1.metadata["speed_accuracy"] == 0.25 and r1.metadata["difficult"]
    assert r2.best_distractor is None and r2.metadata["untimed_accuracy"] == 0.5
    item = QuALITY().to_item(r1)
    assert item.private["passage"] == PASSAGE and "lighthouse" not in item.question
    assert item.question.startswith("This question is about an article.")  # source is not Gutenberg fiction
    assert {a.text for a in item.answers} == {"Because the rules require it", "Because a ship is coming"}
    assert QuALITY.expert_affordances == ["passage"]
    monkeypatch.setattr(QuALITY, "records", lambda self, split: QuALITY.parse_article(ARTICLE))
    assert [it.id for it in QuALITY(hard_only=True).load()] == ["quality-syn1_q1"]
    assert [it.id for it in QuALITY(min_untimed_accuracy=1.0).load()] == ["quality-syn1_q1"]
    assert [it.id for it in QuALITY(max_speed_accuracy=0.5).load()] == ["quality-syn1_q1"]
    assert len(QuALITY(binary=False).load(split="test")[0].answers) == 4  # "test" -> dev (labels are hidden)


def test_quality_debate_verified_and_fabricated_quotes():
    dom = get_domain("quality")
    item = dom.to_item(QuALITY.parse_article(ARTICLE)[0])
    saw_passage = {}

    def prompt_text(req):
        return "\n".join(m.content for m in req.prompt)

    def honest(req, ctx):
        saw_passage[ctx.role] = "forty-first morning" in prompt_text(req)
        return ('She follows procedure: <claim kind="quote">then she lit the lamp anyway -- because the rules said '
                'so</claim>.')

    def liar(req, ctx):
        saw_passage[ctx.role] = "forty-first morning" in prompt_text(req)
        return 'A ship was coming: <claim kind="quote">A ship appeared on the horizon at dawn</claim>.'

    def judge(req, ctx):
        text = prompt_text(req)
        saw_passage["judge"] = "forty-first morning" in text
        failed = {t.role for t in req.view.transcript if "<failed" in t.text}
        trusted = [r for r in ("debater_a", "debater_b") if r not in failed]
        good = req.view.positions[trusted[0]]
        return {lab: (0.9 if lab == good else 0.1) for lab in req.options}

    mech = Debate(rounds=1, verification=soa.VerificationPolicy(verifiers=["quote"]), affordances={"agents": ["passage"]})
    prof = Profile(name="p", players={"debater_a": PlayerSpec(policy=soa.FunctionPolicy(honest), stance="true"),
                                      "debater_b": PlayerSpec(policy=soa.FunctionPolicy(liar), stance="false"),
                                      "judge": soa.FunctionPolicy(judge)})
    ep = run_sync(run_episodes(mech, [item], [prof], ctx=dom.context(), ground_truth=dom.ground_truth_scorers()))[0]
    assert ep.error is None
    assert [v.status for v in ep.verifications("debater_a")] == ["verified"]
    assert [v.status for v in ep.verifications("debater_b")] == ["refuted"]
    shown = {t.role: t.shown for t in ep.turns if t.phase == "round1"}
    assert '<verified kind="quote">' in shown["debater_a"] and '<failed kind="quote">' in shown["debater_b"]
    assert saw_passage == {"debater_a": True, "debater_b": True, "judge": False}  # information asymmetry
    assert ep.outcome.probs[item.true_label] == pytest.approx(0.9)
    assert ep.rewards["debater_a"] == pytest.approx(math.log(0.9))
    assert ep.ground_truth["role_values"] == {"debater_a": 1.0, "debater_b": -1.0}
    assert ep.ground_truth["judge_correct"] == 1.0


# ----------------------------------------------------------------------------- live download

@pytest.mark.network
def test_live_gsm8k_download():
    dom = GSM8K()
    items = dom.load()
    assert dom.used_source == "download" and len(items) == 1319
    assert [it.model_dump() for it in items[:50]] == [it.model_dump() for it in GSM8K(offline=True).load()]
