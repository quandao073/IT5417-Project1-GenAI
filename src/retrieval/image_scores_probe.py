"""Linear-probe columns for the score matrix, kept per column only where they beat the caption.

A column of the matrix is a calibrated caption cosine (`image_scores.calibrate`): one weight on
one direction. A probe instead fits all 512 weights, an L2-penalised logistic regression on the
pool-train image vectors (`probe.fit_logistic`), against the column's own targets: the resolved
CheXpert statuses for the 14 labels, the RadGraph silver statuses for the extended columns
(D62); PRESENT against ABSENT or NOT_MENTIONED, UNCERTAIN dropped. The penalty is chosen per
column on pool-val by the lenient AUROC (D55), and the chosen probe's logit is calibrated on
pool-val by the one-variable fit of the caption columns, so both kinds of column are a row of
W and b and the matrix format does not change.

A column takes the probe only when the paired bootstrap of (probe - caption) lenient AUROC on
pool-val has its 95% interval above 0. The corpus is only scored, never labelled.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from src.embeddings.probe import fit_logistic, make_targets
from src.evaluation import zero_shot as zs


def lenient(logits: np.ndarray, statuses: np.ndarray) -> float:
    pos, _, neg = zs.label_masks(statuses)
    return zs.auroc(logits, pos, neg)


def fit_column(X_train: np.ndarray, s_train: np.ndarray, X_val: np.ndarray, s_val: np.ndarray,
               lambdas: Sequence[float], *, max_iter: int, tol: float) -> dict:
    """{"w": [d], "b", "lambda", "per_lambda": {lam: pool-val lenient AUROC}, "calibration":
    {a, b}}: the probe of one column, its logit calibrated on pool-val (w, b include the
    calibration). A tie goes to the larger lambda."""
    y, keep = make_targets(s_train)
    per_lambda, fits = {}, {}
    for lam in lambdas:
        w = fit_logistic(X_train[keep], y[keep], lam, max_iter=max_iter, tol=tol)
        fits[lam] = w
        per_lambda[lam] = lenient(X_val @ w[:-1] + w[-1], s_val)
    best = lambdas[0]
    for lam in lambdas[1:]:
        if math.isclose(per_lambda[lam], per_lambda[best], rel_tol=1e-9):
            best = max(best, lam)
        elif per_lambda[lam] > per_lambda[best]:
            best = lam
    w = fits[best]
    raw = X_val @ w[:-1] + w[-1]
    yv, kv = make_targets(s_val)
    a, c = fit_logistic(raw[kv, None], yv[kv], 0.0, max_iter=max_iter, tol=tol)
    return {"w": a * w[:-1], "b": float(a * w[-1] + c), "lambda": best,
            "per_lambda": {str(k): float(v) for k, v in per_lambda.items()},
            "calibration": {"a": float(a), "b": float(c)}}


def paired_delta(probe_logits: np.ndarray, caption_logits: np.ndarray, statuses: np.ndarray, *,
                 n_boot: int, seed: int) -> dict:
    """Lenient AUROC of both columns on the same images, their difference (probe - caption)
    and its 95% percentile interval over `n_boot` paired resamples."""
    pos, _, neg = zs.label_masks(statuses)
    keep = np.flatnonzero(pos | neg)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        pick = rng.choice(keep, size=keep.size, replace=True)
        d = (zs.auroc(probe_logits[pick], pos[pick], neg[pick])
             - zs.auroc(caption_logits[pick], pos[pick], neg[pick]))
        if not np.isnan(d):
            reps.append(d)
    lo, hi = np.percentile(reps, [2.5, 97.5]) if reps else (np.nan, np.nan)
    p, z = lenient(probe_logits, statuses), lenient(caption_logits, statuses)
    return {"probe": float(p), "caption": float(z), "delta": float(p - z),
            "ci95": [float(lo), float(hi)]}


def use_probe(delta: dict) -> bool:
    return delta["delta"] > 0 and delta["ci95"][0] > 0
