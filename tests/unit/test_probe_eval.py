"""probe_eval: the three text-scoring rows and the paired probe-vs-zero-shot comparison."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import load_config
from src.data.labels import CHEXPERT_LABELS
from src.embeddings import spike
from src.evaluation import probe_eval
from tests.unit.fakes import FakeEncoder

TEMPLATE = "chest x-ray showing {}"


def _unit(rng, n, d):
    v = rng.normal(size=(n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_text_scores_shapes_and_contrast_definition():
    cfg, encoder = load_config("concepts"), FakeEncoder(dim=8)
    vectors = _unit(np.random.default_rng(0), 20, 8)

    scores = probe_eval.text_scores(vectors, encoder, cfg, TEMPLATE)

    assert list(scores) == ["zero_shot", "ensemble", "contrast"]
    assert all(s.shape == (20, 14) for s in scores.values())
    prompts = spike.build_prompts(cfg, TEMPLATE)
    np.testing.assert_allclose(scores["zero_shot"], vectors @ encoder.encode_texts(prompts).T,
                               atol=1e-6)
    negative = encoder.encode_texts(
        [TEMPLATE.format(f"no {cfg['concepts'][c]['en'][0]}") for c in CHEXPERT_LABELS])
    np.testing.assert_allclose(scores["contrast"], scores["zero_shot"] - vectors @ negative.T,
                               atol=1e-6)


def test_ensemble_is_the_renormalised_mean_over_every_english_form():
    cfg, encoder = load_config("concepts"), FakeEncoder(dim=8)
    vectors = _unit(np.random.default_rng(1), 5, 8)

    ensemble = probe_eval.text_scores(vectors, encoder, cfg, TEMPLATE)["ensemble"]

    forms = cfg["concepts"]["Edema"]["en"]
    assert len(forms) > 1
    mean = encoder.encode_texts([TEMPLATE.format(f) for f in forms]).mean(axis=0)
    expected = vectors @ (mean / np.linalg.norm(mean))
    np.testing.assert_allclose(ensemble[:, CHEXPERT_LABELS.index("Edema")], expected, atol=1e-6)


def _synthetic(n=400, seed=3):
    rng = np.random.default_rng(seed)
    statuses = np.full((n, 14), "ABSENT", dtype=object)
    statuses[: n // 2, :] = "PRESENT"
    rng.shuffle(statuses, axis=0)
    labels = (statuses == "PRESENT").astype(float)
    return statuses, labels, rng


def test_compare_reports_paired_difference_and_s1():
    statuses, labels, rng = _synthetic()
    zero_shot = rng.normal(size=labels.shape)             # chance
    probe = labels + 0.3 * rng.normal(size=labels.shape)  # near-perfect

    result = probe_eval.compare({"zero_shot": zero_shot, "probe": probe}, statuses, n_boot=200)

    assert result["probe"]["macro_auroc_strict"] > 0.95
    assert 0.4 < result["zero_shot"]["macro_auroc_strict"] < 0.6
    diff = result["probe_minus_zero_shot"]
    assert diff["mean"] > 0.3 and diff["ci95"][0] > 0
    assert result["s1_passed"] is True
    assert set(result["probe"]["per_concept"]) == set(CHEXPERT_LABELS)


def test_compare_with_identical_scores_is_a_zero_difference_that_fails_s1():
    statuses, _, rng = _synthetic()
    scores = rng.normal(size=(400, 14))
    statuses[:, 5] = "NOT_MENTIONED"                      # one concept is not eligible

    result = probe_eval.compare({"zero_shot": scores, "probe": scores.copy()}, statuses,
                                n_boot=200)

    assert result["probe_minus_zero_shot"]["mean"] == 0.0
    assert result["s1_passed"] is False
    assert result["probe"]["per_concept"][CHEXPERT_LABELS[5]] is None
    assert result["probe"]["per_concept"][CHEXPERT_LABELS[0]] is not None


def test_compare_needs_both_zero_shot_and_probe():
    statuses, _, rng = _synthetic()
    with pytest.raises(ValueError, match="zero_shot"):
        probe_eval.compare({"probe": rng.normal(size=(400, 14))}, statuses, n_boot=10)
