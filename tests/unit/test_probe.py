"""Logistic probes on frozen image embeddings: fitting, lambda selection, artifact."""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from src.common.config import load_config
from src.data.labels import CHEXPERT_LABELS
from src.embeddings import probe as probe_module
from src.embeddings.probe import (
    META_FILE,
    PROBE_FILE,
    SCORES_FILE,
    Probe,
    fit_logistic,
    load_probe,
    make_targets,
    patient_split,
    save_probe,
    select_lambda,
    train_probe,
)
from src.evaluation import zero_shot as zs

STATES = np.array(["PRESENT", "ABSENT", "UNCERTAIN", "NOT_MENTIONED"], dtype=object)


def separable(n: int = 200, d: int = 4):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    direction = np.array([1.0, -2.0, 0.5, 1.5])[:d]
    score = X @ direction
    X = X[np.abs(score) > 0.5]  # a margin, so the classes are strictly separable
    y = (X @ direction > 0).astype(np.float64)
    return X.astype(np.float32), y


def status_matrix(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return STATES[rng.integers(0, 4, size=(n, len(CHEXPERT_LABELS)))].astype(object)


def test_make_targets_maps_four_states():
    y, keep = make_targets(
        np.array(["PRESENT", "ABSENT", "NOT_MENTIONED", "UNCERTAIN"], dtype=object)
    )
    assert y.tolist() == [1, 0, 0, 0] and keep.tolist() == [True, True, True, False]


def test_fit_logistic_separates_synthetic_data():
    X, y = separable()
    assert len(y) > 100
    w = fit_logistic(X, y, lam=1.0)
    assert w.shape == (X.shape[1] + 1,) and w.dtype == np.float64
    Xb = np.hstack([X.astype(np.float64), np.ones((len(X), 1))])
    assert zs.auroc(Xb @ w, y == 1, y == 0) == 1.0


def test_fit_logistic_is_deterministic():
    X, y = separable()
    assert np.array_equal(fit_logistic(X, y, lam=1.0), fit_logistic(X, y, lam=1.0))


def test_larger_lambda_shrinks_weights_but_not_bias():
    X, y = separable()
    small = fit_logistic(X, y, lam=0.1)
    large = fit_logistic(X, y, lam=100.0)
    assert np.linalg.norm(large[:-1]) < np.linalg.norm(small[:-1])
    # A huge penalty flattens the weights; the bias, if unpenalised, lands on the
    # log-odds of the class prior instead of being shrunk to 0.
    y_imbalanced = np.zeros(len(X))
    y_imbalanced[:20] = 1.0
    heavy = fit_logistic(X, y_imbalanced, lam=1e8)
    prior = y_imbalanced.mean()
    assert np.linalg.norm(heavy[:-1]) < 1e-3
    assert np.isclose(heavy[-1], np.log(prior / (1 - prior)), atol=1e-2)


def test_fit_logistic_with_no_positives_is_finite():
    X, _ = separable()
    w = fit_logistic(X, np.zeros(len(X)), lam=1.0)
    assert np.isfinite(w).all()


def test_patient_split_keeps_a_patient_on_one_side_and_hits_the_fraction():
    patients = [f"patient{i:05d}" for i in range(10_000)]
    val = patient_split(patients, val_fraction=0.1, seed=20260101)
    assert val.dtype == bool and 0.08 < val.mean() < 0.12
    doubled = patient_split(patients + patients, val_fraction=0.1, seed=20260101)
    assert np.array_equal(doubled[:10_000], doubled[10_000:])
    other = patient_split(patients, val_fraction=0.1, seed=1)
    assert not np.array_equal(val, other)


def test_select_lambda_prefers_larger_on_tie_and_skips_thin_concepts(monkeypatch):
    n = 4000
    patients = [f"p{i}" for i in range(n)]
    statuses = status_matrix(n)
    val = patient_split(patients, val_fraction=0.1, seed=7)
    thin = 1
    column = np.where(val, "PRESENT", "NOT_MENTIONED").astype(object)
    column[np.flatnonzero(val)[:3]] = "ABSENT"  # only 3 ABSENT in val
    statuses[:, thin] = column
    X = np.random.default_rng(2).normal(size=(n, 5)).astype(np.float32)

    monkeypatch.setattr(
        probe_module, "fit_logistic", lambda X, y, lam, **kw: np.zeros(X.shape[1] + 1)
    )
    lam, report = select_lambda(
        X, statuses, patients, [0.1, 1, 10, 100], val_fraction=0.1, seed=7,
        max_iter=5, tol=1e-6,
    )
    assert lam == 100
    assert set(report["per_lambda"]) == {"0.1", "1", "10", "100"}
    assert all(v == 0.5 for v in report["per_lambda"].values())
    assert CHEXPERT_LABELS[thin] not in report["eligible"]
    assert CHEXPERT_LABELS[0] in report["eligible"]
    assert report["val_auroc"][CHEXPERT_LABELS[thin]] is None
    assert report["val_auroc"][CHEXPERT_LABELS[0]] == 0.5
    # Counts of the val slice for every concept, the thin one included: PRESENT and strict ABSENT.
    assert list(report["n_val"]) == list(CHEXPERT_LABELS)
    assert report["n_val"][CHEXPERT_LABELS[thin]] == {"pos": int(val.sum()) - 3, "neg": 3}
    assert report["n_val"][CHEXPERT_LABELS[0]] == {
        "pos": int((statuses[val, 0] == "PRESENT").sum()),
        "neg": int((statuses[val, 0] == "ABSENT").sum()),
    }


def test_select_lambda_picks_the_best_macro_when_lambdas_differ(monkeypatch):
    n = 4000
    patients = [f"p{i}" for i in range(n)]
    statuses = status_matrix(n)
    X = np.random.default_rng(2).normal(size=(n, 5)).astype(np.float32)

    def fake(X, y, lam, **kw):  # only lam == 1 produces a useful score
        w = np.zeros(X.shape[1] + 1)
        if lam == 1:
            w[0] = 1.0
        return w

    # Column 0 of X is informative for every concept, so lam == 1 beats the rest.
    statuses[:, :] = np.where(X[:, :1] > 0, "PRESENT", "ABSENT")
    monkeypatch.setattr(probe_module, "fit_logistic", fake)
    lam, report = select_lambda(
        X, statuses, patients, [0.1, 1, 10, 100], val_fraction=0.1, seed=7,
        max_iter=5, tol=1e-6,
    )
    assert lam == 1
    assert report["per_lambda"]["1"] == 1.0 and report["per_lambda"]["100"] == 0.5


def test_train_probe_refits_on_everything_and_reports_counts(monkeypatch):
    n, d = 3000, 8
    patients = [f"p{i // 2}" for i in range(n)]
    statuses = status_matrix(n, seed=3)
    X = np.random.default_rng(4).normal(size=(n, d)).astype(np.float32)
    statuses[:, 0] = np.where(X[:, 0] > 0, "PRESENT", "ABSENT")
    statuses[::10, 0] = "UNCERTAIN"
    cfg = {"lambdas": [0.1, 1, 10, 100], "val_fraction": 0.1, "seed": 20260101,
           "max_iter": 50, "tol": 1e-6}

    sizes: list[int] = []
    real = probe_module.fit_logistic

    def spy(X, y, lam, **kw):
        sizes.append(len(y))
        return real(X, y, lam, **kw)

    monkeypatch.setattr(probe_module, "fit_logistic", spy)
    probe, report = train_probe(X, statuses, patients, cfg)

    kept = [int((statuses[:, j] != "UNCERTAIN").sum()) for j in range(len(CHEXPERT_LABELS))]
    assert sizes[-len(CHEXPERT_LABELS):] == kept  # the final refit sees the val slice too
    assert report["lambda"] in cfg["lambdas"]
    assert set(report["selection"]["per_lambda"]) == {"0.1", "1", "10", "100"}
    first = report["n_train"][CHEXPERT_LABELS[0]]
    assert first["dropped_uncertain"] == 300
    assert first["pos"] + first["neg"] == n - 300
    assert first["pos"] == int((statuses[:, 0] == "PRESENT").sum())
    assert probe.concepts == tuple(CHEXPERT_LABELS)
    assert probe.W.shape == (14, d) and probe.W.dtype == np.float32
    assert probe.b.shape == (14,) and probe.b.dtype == np.float32
    logits = probe.logits(X)
    assert logits.shape == (n, 14) and logits.dtype == np.float32
    pos, strict, _ = zs.label_masks(statuses[:, 0])
    assert zs.auroc(logits[:, 0], pos, strict) > 0.95


def test_save_and_load_round_trip(tmp_path):
    rng = np.random.default_rng(5)
    original = Probe(
        W=rng.normal(size=(14, 6)).astype(np.float32),
        b=rng.normal(size=14).astype(np.float32),
        concepts=tuple(CHEXPERT_LABELS),
    )
    scores = pl.DataFrame({"image_id": ["a", "b"], "Cardiomegaly": [0.1, 0.9]})
    out = tmp_path / "nested" / "probe"
    save_probe(out, original, {"lambda": 10.0, "seed": 1}, scores)
    save_probe(out, original, {"lambda": 10.0, "seed": 1}, scores)  # replaces cleanly

    assert sorted(p.name for p in out.iterdir()) == sorted([PROBE_FILE, SCORES_FILE, META_FILE])
    loaded, meta = load_probe(out)
    assert np.array_equal(loaded.W, original.W) and loaded.W.dtype == np.float32
    assert np.array_equal(loaded.b, original.b)
    assert loaded.concepts == tuple(CHEXPERT_LABELS)
    assert all(isinstance(c, str) for c in loaded.concepts)
    assert meta["lambda"] == 10.0
    assert json.loads((out / META_FILE).read_text(encoding="utf-8"))["seed"] == 1
    assert pl.read_parquet(out / SCORES_FILE).equals(scores)


def test_models_yaml_has_the_probe_section():
    cfg = load_config("models")["image_probe"]
    assert cfg["lambdas"] == [0.1, 1, 10, 100] and cfg["seed"] == 20260101
