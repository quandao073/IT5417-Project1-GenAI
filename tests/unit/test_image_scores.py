"""The V2 score matrix: cosine with each label's caption, calibrated on pool-val into a logit,
saved in the probe format (probe.npz, corpus_scores.parquet, probe.json)."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.data.labels import CHEXPERT_LABELS
from src.retrieval.image_scores import MAX_LOGIT, MIN_CLASS, calibrate, corpus_scores

D = 4


def _units(n: int, seed: int) -> np.ndarray:
    x = np.random.default_rng(seed).normal(size=(n, D))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _world(n=400, seed=0):
    """Captions t_j = e_(j mod D); concept 0's status follows cos with t_0 plus noise, every
    other concept has random PRESENT/ABSENT with enough of both."""
    rng = np.random.default_rng(seed)
    text = np.eye(D, dtype=np.float32)[[j % D for j in range(14)]]
    pool = _units(n, seed + 1)
    statuses = np.where(rng.random((n, 14)) < 0.5, "PRESENT", "ABSENT").astype(object)
    noisy = pool[:, 0] + rng.normal(scale=0.3, size=n)
    statuses[:, 0] = np.where(noisy > 0, "PRESENT", "ABSENT")
    return text, pool, statuses


def test_calibrate_recovers_a_monotone_mapping():
    text, pool, statuses = _world()
    _, fit = calibrate(text, pool, statuses)
    first = fit["calibration"][CHEXPERT_LABELS[0]]
    assert first["a"] > 1.0 and first["val_auroc"] > 0.85
    assert first["n_pos"] + first["n_neg"] == 400
    assert fit["uncalibrated"] == []
    assert set(fit["calibration"]) == set(CHEXPERT_LABELS)


def test_calibrated_probe_equals_affine_cosine():
    text, pool, statuses = _world()
    probe, fit = calibrate(text, pool, statuses)
    X = _units(30, 7)
    logits = probe.logits(X)
    for j, concept in enumerate(CHEXPERT_LABELS):
        c = fit["calibration"][concept]
        expected = c["a"] * (X.astype(np.float64) @ text[j]) + c["b"]
        assert np.allclose(logits[:, j], expected, atol=1e-5)


def test_small_class_concept_is_left_uncalibrated():
    text, pool, statuses = _world()
    statuses[:, 3] = "ABSENT"
    statuses[: MIN_CLASS - 1, 3] = "PRESENT"
    statuses[MIN_CLASS - 1: 2 * MIN_CLASS, 3] = "UNCERTAIN"    # never a class of its own
    probe, fit = calibrate(text, pool, statuses)
    small = fit["calibration"][CHEXPERT_LABELS[3]]
    assert (small["a"], small["b"], small["val_auroc"]) == (1.0, 0.0, None)
    assert small["n_pos"] == MIN_CLASS - 1
    assert fit["uncalibrated"] == [CHEXPERT_LABELS[3]]
    assert np.allclose(probe.W[3], text[3]) and probe.b[3] == 0.0


def test_corpus_scores_has_one_column_per_concept_in_study_order():
    text, pool, statuses = _world()
    probe, _ = calibrate(text, pool, statuses)
    studies = pl.DataFrame({"study_id": ["s2", "s1"], "selected_image_id": ["i2", "i1"]})
    vectors = _units(2, 9)
    frame = corpus_scores(probe, vectors, studies)
    assert frame.columns == ["study_id", "image_id", *CHEXPERT_LABELS]
    assert frame["study_id"].to_list() == ["s2", "s1"]
    assert np.allclose(frame.select(CHEXPERT_LABELS).to_numpy(), probe.logits(vectors))


def test_a_separable_concept_is_flagged_saturated():
    """Newton at lambda 0 diverges on separable data: a in the thousands would swamp every other
    term of a decomposed sum. The fit stays, the concept is listed (likelier after fine-tuning)."""
    text, pool, statuses = _world()
    statuses[:, 0] = np.where(pool[:, 0] > 0, "PRESENT", "ABSENT")      # no noise: separable
    _, fit = calibrate(text, pool, statuses)
    first = fit["calibration"][CHEXPERT_LABELS[0]]
    assert abs(first["a"]) * 2 > MAX_LOGIT and first["val_auroc"] == 1.0
    assert fit["saturated"] == [CHEXPERT_LABELS[0]]
    _, noisy = calibrate(*_world())
    assert noisy["saturated"] == []
