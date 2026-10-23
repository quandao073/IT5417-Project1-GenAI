"""Linear probes on the frozen image embeddings, one per CheXpert concept.

The probes are trained on studies outside the corpus, so scoring the corpus with
them is out-of-sample. Logistic regression is plain Newton/IRLS in numpy: the
project carries no scikit-learn, and 512 weights per concept is a small solve.

A row's target is PRESENT against ABSENT or NOT_MENTIONED; UNCERTAIN rows are
dropped from fitting (their label is not a negative). Evaluation is the strict
convention of `zero_shot`: PRESENT against ABSENT only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src.data.labels import CHEXPERT_LABELS
from src.embeddings.embed import write_json_atomic
from src.evaluation import zero_shot as zs

PROBE_FILE = "probe.npz"
SCORES_FILE = "corpus_scores.parquet"
META_FILE = "probe.json"


def make_targets(status_col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(y, keep): y is 1 for PRESENT; keep drops UNCERTAIN rows."""
    return (status_col == "PRESENT").astype(np.float64), status_col != "UNCERTAIN"


def fit_logistic(
    X: np.ndarray, y: np.ndarray, lam: float, *, max_iter: int = 50, tol: float = 1e-6
) -> np.ndarray:
    """L2-penalised logistic regression by Newton's method; returns [d + 1], bias last.

    The loss is summed over samples, not averaged, so `lam` means the same thing
    whatever the training-set size. The bias is not penalised.
    """
    n, d = X.shape
    Xb = np.hstack([np.asarray(X, dtype=np.float64), np.ones((n, 1))])
    y = np.asarray(y, dtype=np.float64)
    penalty = lam * np.eye(d + 1)
    penalty[-1, -1] = 0.0
    w = np.zeros(d + 1)
    for _ in range(max_iter):
        p = np.exp(-np.logaddexp(0.0, -(Xb @ w)))
        grad = Xb.T @ (p - y) + penalty @ w
        hess = (Xb * (p * (1.0 - p))[:, None]).T @ Xb + penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.solve(hess + 1e-8 * np.eye(d + 1), grad)
        w = w - step
        if np.max(np.abs(step)) < tol:
            break
    return w


def patient_split(
    patient_ids: Sequence[str], *, val_fraction: float, seed: int
) -> np.ndarray:
    """True = validation. A pure function of (seed, patient), so every image of a
    patient lands on one side and the split does not depend on row order."""
    return np.array(
        [
            int(hashlib.sha256(f"{seed}:{patient}".encode()).hexdigest(), 16) % 1000
            < val_fraction * 1000
            for patient in patient_ids
        ],
        dtype=bool,
    )


def select_lambda(
    X: np.ndarray,
    statuses: np.ndarray,
    patient_ids: Sequence[str],
    lambdas: Sequence[float],
    *,
    val_fraction: float,
    seed: int,
    max_iter: int,
    tol: float,
    min_samples: int = zs.MIN_SAMPLES,
) -> tuple[float, dict]:
    """Pick the lambda with the best macro strict AUROC on held-out patients.

    Every concept is fitted on the train slice at every lambda. Only concepts
    with at least `min_samples` PRESENT and ABSENT rows in the validation slice
    count. A tie goes to the larger lambda: the stronger regulariser.
    """
    val = patient_split(patient_ids, val_fraction=val_fraction, seed=seed)
    X_train, X_val = X[~val], X[val]
    eligible = zs.eligible_concepts(statuses[val], min_samples)
    if not eligible:
        raise ValueError(
            f"no concept has {min_samples} PRESENT and {min_samples} ABSENT validation rows"
        )

    per_concept: dict[float, dict[int, float]] = {}
    for lam in lambdas:
        per_concept[lam] = {}
        for j in eligible:
            y, keep = make_targets(statuses[~val, j])
            w = fit_logistic(X_train[keep], y[keep], lam, max_iter=max_iter, tol=tol)
            logits = X_val.astype(np.float64) @ w[:-1] + w[-1]
            pos, strict, _ = zs.label_masks(statuses[val, j])
            per_concept[lam][j] = zs.auroc(logits, pos, strict)

    macros = {lam: zs.macro(list(per_concept[lam].values())) for lam in lambdas}
    best = lambdas[0]
    for lam in lambdas[1:]:
        if math.isclose(macros[lam], macros[best], rel_tol=1e-9):
            best = max(best, lam)
        elif macros[lam] > macros[best]:
            best = lam
    chosen = {j: per_concept[best][j] for j in eligible}
    return best, {
        "per_lambda": {str(lam): macros[lam] for lam in lambdas},
        "eligible": [CHEXPERT_LABELS[j] for j in eligible],
        "val_auroc": {
            name: chosen.get(j) for j, name in enumerate(CHEXPERT_LABELS)
        },
    }


@dataclass(frozen=True)
class Probe:
    W: np.ndarray  # [14, d] float32
    b: np.ndarray  # [14] float32
    concepts: tuple[str, ...]

    def logits(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float32) @ self.W.T + self.b).astype(np.float32)


def train_probe(
    X: np.ndarray, statuses: np.ndarray, patient_ids: Sequence[str], cfg: dict
) -> tuple[Probe, dict]:
    """Choose lambda on held-out patients, then refit every concept on all rows."""
    lam, selection = select_lambda(
        X, statuses, patient_ids, cfg["lambdas"],
        val_fraction=cfg["val_fraction"], seed=cfg["seed"],
        max_iter=cfg["max_iter"], tol=cfg["tol"],
    )
    W, b, n_train = [], [], {}
    for j, name in enumerate(CHEXPERT_LABELS):
        y, keep = make_targets(statuses[:, j])
        w = fit_logistic(X[keep], y[keep], lam, max_iter=cfg["max_iter"], tol=cfg["tol"])
        W.append(w[:-1])
        b.append(w[-1])
        n_train[name] = {
            "pos": int(y[keep].sum()),
            "neg": int((1.0 - y[keep]).sum()),
            "dropped_uncertain": int((~keep).sum()),
        }
    probe = Probe(
        W=np.asarray(W, dtype=np.float32),
        b=np.asarray(b, dtype=np.float32),
        concepts=tuple(CHEXPERT_LABELS),
    )
    return probe, {"lambda": lam, "selection": selection, "n_train": n_train}


def save_probe(out_dir: Path, probe: Probe, meta: dict, corpus_scores: pl.DataFrame) -> None:
    """Each file goes through <name>.part and a rename, so a reader never sees half a file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    part = out_dir / (PROBE_FILE + ".part")
    with open(part, "wb") as handle:
        np.savez(handle, W=probe.W, b=probe.b, concepts=np.asarray(probe.concepts))
    part.replace(out_dir / PROBE_FILE)
    part = out_dir / (SCORES_FILE + ".part")
    corpus_scores.write_parquet(part)
    part.replace(out_dir / SCORES_FILE)
    write_json_atomic(out_dir / META_FILE, meta)


def load_probe(out_dir: Path) -> tuple[Probe, dict]:
    with np.load(out_dir / PROBE_FILE) as archive:
        probe = Probe(
            W=archive["W"].astype(np.float32, copy=False),
            b=archive["b"].astype(np.float32, copy=False),
            concepts=tuple(str(c) for c in archive["concepts"]),
        )
    meta = json.loads((out_dir / META_FILE).read_text(encoding="utf-8"))
    return probe, meta
