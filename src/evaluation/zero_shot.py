"""Zero-shot metrics and the model gate for the Phase 2 VLM spike.

A score is the cosine between an image and one positive prompt - the exact
operation vector mode will run - so the gate measures what gets deployed.

Negatives come in two flavours. Strict counts only ABSENT and is what the gate
uses; lenient adds NOT_MENTIONED, the usual CheXpert convention, and is reported
for reference only. NOT_MENTIONED is never a strict negative (WORKLOG T7).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

MIN_SAMPLES = 10


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, ties sharing the mean of the ranks they span."""
    order = np.argsort(values, kind="mergesort")
    _, first, counts = np.unique(values[order], return_index=True, return_counts=True)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.repeat(first + (counts + 1) / 2, counts)
    return ranks


def auroc(scores: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC over the rows in `pos` or `neg`; nan if a class is empty."""
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    keep = pos | neg
    ranks = _average_ranks(np.asarray(scores, dtype=float)[keep])
    rank_sum = ranks[pos[keep]].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def label_masks(statuses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(positive, strict negative, lenient negative). UNCERTAIN is in none."""
    pos = statuses == "PRESENT"
    strict = statuses == "ABSENT"
    lenient = strict | (statuses == "NOT_MENTIONED")
    return pos, strict, lenient


def _summary(values: np.ndarray) -> dict | None:
    if values.size == 0:
        return None
    return {
        "mean": float(values.mean()),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def concept_metrics(
    scores: np.ndarray, statuses: np.ndarray, min_samples: int = MIN_SAMPLES
) -> dict:
    """Every per-concept number the spike reports, for one model."""
    scores = np.asarray(scores, dtype=float)
    pos, strict, lenient = label_masks(statuses)
    n_pos, n_strict, n_lenient = int(pos.sum()), int(strict.sum()), int(lenient.sum())

    def gated(neg: np.ndarray, n_neg: int) -> float | None:
        if n_pos < min_samples or n_neg < min_samples:
            return None
        return auroc(scores, pos, neg)

    top10 = np.argsort(-scores, kind="mergesort")[:10]
    return {
        "n_pos": n_pos,
        "n_neg_strict": n_strict,
        "n_neg_lenient": n_lenient,
        "auroc_strict": gated(strict, n_strict),
        "auroc_lenient": gated(lenient, n_lenient),
        "p_at_10": float(pos[top10].mean()),
        "prevalence": n_pos / len(statuses),
        "cosine": {"present": _summary(scores[pos]), "absent": _summary(scores[strict])},
    }


def eligible_concepts(statuses: np.ndarray, min_samples: int = MIN_SAMPLES) -> list[int]:
    """Columns with enough positives and strict negatives. Labels only, so the
    set is identical for every model and their macro scores compare."""
    pos, strict, _ = label_masks(statuses)
    enough = (pos.sum(axis=0) >= min_samples) & (strict.sum(axis=0) >= min_samples)
    return [int(j) for j in np.flatnonzero(enough)]


def macro(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def bootstrap_macro(
    scores: Mapping[str, np.ndarray],
    statuses: np.ndarray,
    eligible: Sequence[int],
    n_boot: int = 1000,
    seed: int = 20260101,
) -> dict[str, np.ndarray]:
    """Macro strict AUROC per resample. The same row draw serves every model
    (paired), so replicate differences between models are meaningful."""
    rng = np.random.default_rng(seed)
    pos, strict, _ = label_masks(statuses)
    n = len(statuses)
    out = {name: np.empty(n_boot) for name in scores}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for name, matrix in scores.items():
            values = [auroc(matrix[idx, j], pos[idx, j], strict[idx, j]) for j in eligible]
            out[name][b] = np.nanmean(values)
    return out


def ci95(values: np.ndarray) -> list[float]:
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


# Lower is freer. CheXzero's weights carry no license of their own.
LICENSE_RANK = {"mit": 0, "unspecified": 1, "cc-by-nc-4.0": 2}


def _hard_gate_failures(r: Mapping, min_macro: float, max_hours: float) -> list[str]:
    fails = []
    if r["device"] != "cpu":
        fails.append(f"ran on {r['device']}, not cpu")
    if r["errors"]:
        fails.append(f"{r['errors']} image errors")
    if r["dim_image"] != r["dim_text"]:
        fails.append(f"image dim {r['dim_image']} != text dim {r['dim_text']}")
    if r["macro_auroc_strict"] is None or r["macro_auroc_strict"] <= min_macro:
        fails.append(f"macro strict AUROC {r['macro_auroc_strict']} <= {min_macro}")
    if r["forecast_25k_hours"] > max_hours:
        fails.append(f"25k forecast {r['forecast_25k_hours']:.1f} h > {max_hours} h")
    return fails


def gate(
    results: Mapping[str, Mapping],
    *,
    min_macro: float = 0.55,
    max_hours: float = 8.0,
    tie_margin: float = 0.02,
    cost_tie: float = 0.10,
) -> tuple[str | None, list[str]]:
    """Pick the model, or None. Returns the winner and every reason, in order.

    Hard gates first. Among survivors the highest macro strict AUROC wins; if the
    top two are within `tie_margin`, the cheaper one (CPU s/image) wins, and if
    their costs are within `cost_tie` of each other, the freer license does.
    """
    reasons: list[str] = []
    passed = []
    for name in sorted(results):
        if fails := _hard_gate_failures(results[name], min_macro, max_hours):
            reasons.append(f"{name}: rejected ({'; '.join(fails)})")
        else:
            passed.append(name)
    if not passed:
        return None, [*reasons, "no candidate passed the hard gates"]

    ranked = sorted(passed, key=lambda n: -results[n]["macro_auroc_strict"])
    best = ranked[0]
    if len(ranked) > 1:
        first, second = results[ranked[0]], results[ranked[1]]
        gap = first["macro_auroc_strict"] - second["macro_auroc_strict"]
        if gap < tie_margin:
            slow = max(first["s_per_image"], second["s_per_image"])
            fast = min(first["s_per_image"], second["s_per_image"])
            if (slow - fast) / slow >= cost_tie:
                best = min(ranked[:2], key=lambda n: results[n]["s_per_image"])
                reasons.append(f"top two within {gap:.4f} AUROC; {best} is cheaper")
            else:
                rank = {n: LICENSE_RANK.get(results[n]["license"], 99) for n in ranked[:2]}
                if rank[ranked[1]] < rank[ranked[0]]:
                    best = ranked[1]
                reasons.append(
                    f"top two within {gap:.4f} AUROC and similar cost; "
                    f"{best} by license ({results[best]['license']})"
                )
    reasons.append(
        f"selected {best} (macro strict AUROC {results[best]['macro_auroc_strict']:.3f})"
    )
    return best, reasons
