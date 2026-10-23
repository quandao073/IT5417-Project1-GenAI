"""H1: the one source of every template string the image modes train on, benchmark with and
query with (spec 2026-10-23-label-aware-image-retrieval-design.md §4)."""

from __future__ import annotations

import pytest

from src.common.config import load_config
from src.data.labels import CHEXPERT_LABELS
from src.retrieval import templates as tp

CFG = load_config("concepts")
TRAIN = load_config("training")
PAPER = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Fracture", "Lung Lesion",
         "Lung Opacity", "Pleural Effusion", "Pneumonia", "Pneumothorax"]


def test_phrase_is_en_first():
    assert tp.phrase("Edema", CFG) == "pulmonary edema"
    assert tp.phrase("No Finding", CFG) == "no finding"


def test_render_orders_by_chexpert_labels():
    text = tp.render(["Edema", "Cardiomegaly"], ["Pleural Effusion"], CFG)
    assert text == "cardiomegaly and pulmonary edema and no pleural effusion"
    swapped = tp.render(["Cardiomegaly", "Edema"], [], CFG)
    assert swapped == tp.render(["Edema", "Cardiomegaly"], [], CFG)
    assert tp.render([], ["Pneumothorax", "Edema"], CFG) == "no pulmonary edema and no pneumothorax"


def test_render_exclude_only():
    assert tp.render([], ["Pneumothorax"], CFG) == "no pneumothorax"


def test_render_rejects_empty_and_negated_no_finding():
    with pytest.raises(ValueError):
        tp.render([], [], CFG)
    with pytest.raises(ValueError):
        tp.render(["Edema"], ["No Finding"], CFG)


def test_render_paraphrase_pair_and_negation():
    assert tp.render_paraphrase(["Cardiomegaly"], ["Pleural Effusion"], CFG) == (
        "an image with cardiomegaly but without pleural effusion")
    assert tp.render_paraphrase(["Edema", "Cardiomegaly"], [], CFG) == (
        "an image with cardiomegaly and pulmonary edema")
    assert tp.render_paraphrase(["Cardiomegaly"], [], CFG) == "an image with cardiomegaly"


def test_generate_counts_for_paper_concepts():
    queries = tp.generate(PAPER, ["single", "pair", "negation", "paraphrase"], CFG)
    kinds = [q.kind for q in queries]
    assert kinds.count("single") == 10
    assert kinds.count("pair") == 45
    assert kinds.count("negation") == 90
    assert kinds.count("paraphrase") == 135
    assert len({q.id for q in queries}) == len(queries)
    for q in queries:
        if q.kind == "paraphrase":
            assert q.text == tp.render_paraphrase(q.include, q.exclude, CFG)
        else:
            assert q.text == tp.render(q.include, q.exclude, CFG)


def test_generate_excludes_no_finding_from_pairs():
    concepts = tp.training_concepts(TRAIN)
    assert len(concepts) == 12 and "No Finding" in concepts
    queries = tp.generate(concepts, ["single", "pair", "negation"], CFG)
    kinds = [q.kind for q in queries]
    assert kinds.count("single") == 12
    assert kinds.count("pair") == 55 and kinds.count("negation") == 110
    for q in queries:
        if q.kind != "single":
            assert "No Finding" not in q.include and "No Finding" not in q.exclude


def test_generate_multi_is_seeded_and_disjoint():
    concepts = tp.training_concepts(TRAIN)
    a = tp.generate(concepts, ["triple", "pair_negation"], CFG, seed=1, n_multi=60)
    b = tp.generate(concepts, ["triple", "pair_negation"], CFG, seed=1, n_multi=60)
    c = tp.generate(concepts, ["triple", "pair_negation"], CFG, seed=2, n_multi=60)
    assert len(a) == 60 and [q.text for q in a] == [q.text for q in b]
    assert [q.text for q in a] != [q.text for q in c]
    assert sum(q.kind == "triple" for q in a) == 30
    assert sum(q.kind == "pair_negation" for q in a) == 30
    assert len({q.text for q in a}) == 60
    for q in a:
        assert len(set(q.include) | set(q.exclude)) == 3
        assert "No Finding" not in q.include and "No Finding" not in q.exclude


def test_gold_matches_text():
    (q,) = tp.generate(["Edema", "Pleural Effusion"], ["negation"], CFG)[:1]
    assert [c.concept for c in q.gold.include] == list(q.include)
    assert all(c.status == "PRESENT" for c in q.gold.include)
    assert q.gold.exclude == list(q.exclude)
    assert q.gold.language == "en" and q.gold.views == []


def test_training_concepts_follow_chexpert_order():
    concepts = tp.training_concepts(TRAIN)
    assert concepts == [c for c in CHEXPERT_LABELS if c in set(concepts)]
    assert set(CHEXPERT_LABELS) - set(concepts) == {"Enlarged Cardiomediastinum", "Pleural Other"}
    assert TRAIN["paper_concepts"] == PAPER
