"""H5: captions rendered from resolved labels for the label-aware loss (plan Task 8, D32)."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import load_config
from src.data.labels import CHEXPERT_LABELS
from src.retrieval.templates import training_concepts
from training.captions import (
    CODE_ABSENT,
    CODE_PRESENT,
    CODE_UNCERTAIN,
    CaptionSampler,
    label_vector,
    status_codes,
)

CFG = load_config("concepts")
CONCEPTS = training_concepts(load_config("training"))
MIX = {"single": 0.5, "pair": 0.25, "negation": 0.25}
COL = {name: i for i, name in enumerate(CHEXPERT_LABELS)}
J = {name: i for i, name in enumerate(CONCEPTS)}


def row(statuses: dict[str, str]) -> np.ndarray:
    out = np.array(["NOT_MENTIONED"] * len(CHEXPERT_LABELS), dtype=object)
    for concept, status in statuses.items():
        out[COL[concept]] = status
    return out


def sampler(rows: list[np.ndarray], mix=MIX, **kwargs) -> CaptionSampler:
    return CaptionSampler(np.stack(rows), CONCEPTS, CFG, mix=mix, seed=7, **kwargs)


def test_eligible_requires_a_present_training_concept():
    s = sampler([
        row({"Edema": "UNCERTAIN"}),
        row({"Enlarged Cardiomediastinum": "PRESENT"}),
        row({"No Finding": "PRESENT"}),
        row({"Edema": "PRESENT", "Pleural Other": "PRESENT"}),
        row({"Pleural Effusion": "ABSENT"}),
    ])
    assert s.eligible().tolist() == [False, False, True, True, False]


def test_single_caption_is_rendered_phrase():
    s = sampler([row({"Edema": "PRESENT", "Cardiomegaly": "ABSENT"})], mix={"single": 1.0})
    text, y = s.sample(0)
    assert text == "pulmonary edema"
    assert y.dtype == np.int8 and y.shape == (len(CONCEPTS),)
    expected = np.zeros(len(CONCEPTS), dtype=np.int8)
    expected[J["Edema"]] = 1
    assert y.tolist() == expected.tolist()


def test_pair_falls_back_to_single_when_one_present():
    s = sampler([
        row({"Edema": "PRESENT"}),
        row({"Edema": "PRESENT", "Cardiomegaly": "PRESENT"}),
    ], mix={"pair": 1.0})
    text, y = s.sample(0)
    assert text == "pulmonary edema"
    assert int(y.sum()) == 1
    text, y = s.sample(1)
    assert text == "cardiomegaly and pulmonary edema"
    assert y[J["Edema"]] == 1 and y[J["Cardiomegaly"]] == 1 and int(np.abs(y).sum()) == 2


def test_negation_only_negates_absent():
    s = sampler([
        row({"Edema": "PRESENT", "Pleural Effusion": "ABSENT", "Pneumothorax": "ABSENT",
             "Cardiomegaly": "NOT_MENTIONED", "Atelectasis": "NOT_MENTIONED"}),
        row({"Edema": "PRESENT", "Cardiomegaly": "NOT_MENTIONED"}),
    ], mix={"negation": 1.0})
    seen = set()
    for _ in range(500):
        text, y = s.sample(0)
        assert text.startswith("pulmonary edema and no ")
        negated = text.split(" and no ", 1)[1]
        assert negated in {"pleural effusion", "pneumothorax"}
        seen.add(negated)
        concept = "Pleural Effusion" if negated == "pleural effusion" else "Pneumothorax"
        assert y[J["Edema"]] == 1 and y[J[concept]] == -1
        assert int(np.abs(y).sum()) == 2
    assert seen == {"pleural effusion", "pneumothorax"}
    # No ABSENT concept to negate: back to the single caption.
    text, y = s.sample(1)
    assert text == "pulmonary edema" and int(np.abs(y).sum()) == 1


def test_uncertain_never_in_caption():
    s = sampler([row({"Edema": "PRESENT", "Cardiomegaly": "PRESENT", "Pneumonia": "UNCERTAIN",
                      "Pleural Effusion": "ABSENT"})])
    for _ in range(500):
        text, y = s.sample(0)
        assert "pneumonia" not in text
        assert y[J["Pneumonia"]] == 0


def test_no_finding_with_pathology_prefers_pathology():
    s = sampler([
        row({"No Finding": "PRESENT", "Fracture": "PRESENT", "Edema": "ABSENT"}),
        row({"No Finding": "PRESENT", "Edema": "ABSENT"}),
    ])
    for _ in range(500):
        text, y = s.sample(0)
        assert "no finding" not in text
        assert y[J["No Finding"]] == 0 and y[J["Fracture"]] == 1
    # Alone, No Finding is the single caption; it is never paired or negated (D32).
    for _ in range(100):
        text, y = s.sample(1)
        assert text == "no finding"
        assert y[J["No Finding"]] == 1 and int(np.abs(y).sum()) == 1


def test_mix_within_two_points_on_10k_draws():
    s = sampler([row({"Edema": "PRESENT", "Cardiomegaly": "PRESENT",
                      "Pleural Effusion": "ABSENT"})])
    counts = {"single": 0, "pair": 0, "negation": 0}
    for _ in range(10_000):
        counts[s.sample_kind(0)] += 1
    for kind, share in MIX.items():
        assert abs(counts[kind] / 10_000 - share) < 0.02, counts


def test_impression_fraction_and_label_vector():
    labels = row({"Edema": "PRESENT", "Pleural Effusion": "ABSENT", "Pneumonia": "UNCERTAIN",
                  "Enlarged Cardiomediastinum": "PRESENT"})
    s = sampler([labels], impressions=["  Edema present.\n"], impression_fraction=0.2)
    expected = np.zeros(len(CONCEPTS), dtype=np.int8)
    expected[J["Edema"]] = 1
    expected[J["Pleural Effusion"]] = -1
    assert label_vector(labels, CONCEPTS).tolist() == expected.tolist()
    n_impression = 0
    for _ in range(10_000):
        text, y = s.sample(0)
        if text == "Edema present.":
            n_impression += 1
            assert y.tolist() == expected.tolist()
        else:
            assert text == "pulmonary edema" or text.startswith("pulmonary edema and no ")
    assert abs(n_impression / 10_000 - 0.2) < 0.02


def test_rejects_bad_mix_and_missing_impressions():
    with pytest.raises(ValueError):
        sampler([row({"Edema": "PRESENT"})], mix={"single": 0.5, "pair": 0.25})
    with pytest.raises(ValueError):
        sampler([row({"Edema": "PRESENT"})], mix={"single": 1.0, "triple": 0.0})
    with pytest.raises(ValueError):
        sampler([row({"Edema": "PRESENT"})], impression_fraction=0.2)
    with pytest.raises(ValueError):
        sampler([row({"Edema": "UNCERTAIN"})]).sample(0)


def test_status_codes_keep_uncertain_apart_from_not_mentioned():
    labels = row({"Edema": "PRESENT", "Pleural Effusion": "ABSENT", "Pneumonia": "UNCERTAIN",
                  "Enlarged Cardiomediastinum": "PRESENT"})
    codes = status_codes(labels, CONCEPTS)
    assert codes.dtype == np.int8 and codes.shape == (len(CONCEPTS),)
    assert codes[J["Edema"]] == CODE_PRESENT == 1
    assert codes[J["Pleural Effusion"]] == CODE_ABSENT == -1
    assert codes[J["Pneumonia"]] == CODE_UNCERTAIN == 2
    assert codes[J["Cardiomegaly"]] == 0
    assert sampler([labels]).image_codes(0).tolist() == codes.tolist()
