"""Decomposed concept scores for `image_decomposed` (spec H6, plan Task 16).

Concept c is scored by the cosine between an image and its own H1 caption `render([c], [])`,
calibrated on pool-val by a one-variable logistic fit: logit_c(x) = a_c · (x · t_c) + b_c. That
is a probe with W[c] = a_c · t_c and b[c] = b_c, so the artifact is the same probe.npz /
corpus_scores.parquet / probe.json that `image_concepts.open_image_probe` reads and mode
`image` uses. A concept with fewer than MIN_CLASS positives or negatives on pool-val keeps
a = 1, b = 0 (the raw cosine) and is listed as `uncalibrated`. Targets and the strict AUROC are
those of the probe: PRESENT against ABSENT or NOT_MENTIONED, UNCERTAIN dropped.

The fit has no penalty (lambda 0, spec H6), so a caption that separates its slice perfectly
drives `a` into the thousands, and that one term would swamp a decomposed sum. Such a concept
(a calibrated logit beyond MAX_LOGIT on the slice) is kept but listed as `saturated`.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.data.labels import CHEXPERT_LABELS
from src.embeddings.probe import Probe, fit_logistic, make_targets
from src.evaluation import zero_shot as zs

MIN_CLASS = 10
MAX_LOGIT = 20.0    # sigma(20) = 1 - 2e-9: beyond it the term only measures separation


def calibrate(text_vectors: np.ndarray, pool_vectors: np.ndarray, pool_statuses: np.ndarray, *,
              lam: float = 0.0, max_iter: int = 50, tol: float = 1e-6) -> tuple[Probe, dict]:
    """(probe, {"calibration": {concept: {a, b, n_pos, n_neg, val_auroc}}, "uncalibrated",
    "saturated"}).

    `text_vectors` [14, D] are the L2-normalised captions in CHEXPERT_LABELS order;
    `pool_vectors` [n, D] and `pool_statuses` [n, 14] are the pool-val slice."""
    X = np.asarray(pool_vectors, dtype=np.float64)
    W, b, calibration, uncalibrated, saturated = [], [], {}, [], []
    for j, concept in enumerate(CHEXPERT_LABELS):
        t = np.asarray(text_vectors[j], dtype=np.float64)
        cos = X @ t
        y, keep = make_targets(pool_statuses[:, j])
        n_pos = int(y[keep].sum())
        n_neg = int(keep.sum()) - n_pos
        if n_pos < MIN_CLASS or n_neg < MIN_CLASS:
            a, bias, val_auroc = 1.0, 0.0, None
            uncalibrated.append(concept)
        else:
            w = fit_logistic(cos[keep, None], y[keep], lam, max_iter=max_iter, tol=tol)
            a, bias = float(w[0]), float(w[1])
            pos, strict, _ = zs.label_masks(pool_statuses[:, j])
            logits = a * cos + bias
            value = zs.auroc(logits, pos, strict)
            val_auroc = None if np.isnan(value) else value
            if np.max(np.abs(logits)) > MAX_LOGIT:
                saturated.append(concept)
        W.append(a * t)
        b.append(bias)
        calibration[concept] = {"a": a, "b": bias, "n_pos": n_pos, "n_neg": n_neg,
                                "val_auroc": val_auroc}
    probe = Probe(W=np.asarray(W, dtype=np.float32), b=np.asarray(b, dtype=np.float32),
                  concepts=tuple(CHEXPERT_LABELS))
    return probe, {"calibration": calibration, "uncalibrated": uncalibrated,
                   "saturated": saturated}


def corpus_scores(probe: Probe, corpus_vectors: np.ndarray, studies: pl.DataFrame) -> pl.DataFrame:
    """One row per study (its selected image's vector, in `studies` order), one logit column
    per concept: the corpus_scores.parquet of the artifact."""
    logits = probe.logits(corpus_vectors)
    return pl.DataFrame({
        "study_id": studies["study_id"], "image_id": studies["selected_image_id"],
        **{c: logits[:, j] for j, c in enumerate(CHEXPERT_LABELS)},
    })
