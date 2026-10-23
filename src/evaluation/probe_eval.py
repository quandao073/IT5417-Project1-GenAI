"""Image-concept probe against the zero-shot text prompts, on the corpus.

Every row is a [n, 14] score matrix over the same studies, so all rows share the
eligible concepts and one bootstrap draw (paired). The probe was fitted on
patients outside the corpus; the text rows use no labels at all.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.data.labels import CHEXPERT_LABELS
from src.embeddings import spike
from src.evaluation import zero_shot as zs


def text_scores(
    vectors: np.ndarray, encoder, concepts_cfg: dict, template: str
) -> dict[str, np.ndarray]:
    """The three prompt-based rows: zero_shot (en[0], what Phase 3 measured), ensemble
    (mean of every English form) and contrast (zero_shot minus a "no <concept>" prompt)."""
    forms = [concepts_cfg["concepts"][c]["en"] for c in CHEXPERT_LABELS]
    zero_shot = vectors @ encoder.encode_texts(spike.build_prompts(concepts_cfg, template)).T

    ensemble_text = []
    for concept_forms in forms:
        mean = encoder.encode_texts([template.format(f) for f in concept_forms]).mean(axis=0)
        ensemble_text.append(mean / np.linalg.norm(mean))
    negative = encoder.encode_texts([template.format(f"no {f[0]}") for f in forms])
    return {
        "zero_shot": zero_shot,
        "ensemble": vectors @ np.stack(ensemble_text).T,
        "contrast": zero_shot - vectors @ negative.T,
    }


def compare(
    scores: Mapping[str, np.ndarray], statuses: np.ndarray, *, n_boot: int, seed: int = 20260101
) -> dict:
    """Macro and per-concept strict AUROC per row, and the paired probe - zero_shot gap."""
    for needed in ("zero_shot", "probe"):
        if needed not in scores:
            raise ValueError(f"compare needs a {needed!r} row; got {sorted(scores)}")
    eligible = zs.eligible_concepts(statuses)
    pos, strict, _ = zs.label_masks(statuses)

    result: dict = {}
    for name, matrix in scores.items():
        per_concept = {
            c: (zs.auroc(matrix[:, j], pos[:, j], strict[:, j]) if j in eligible else None)
            for j, c in enumerate(CHEXPERT_LABELS)
        }
        result[name] = {
            "macro_auroc_strict": zs.macro([per_concept[CHEXPERT_LABELS[j]] for j in eligible]),
            "per_concept": per_concept,
        }
    reps = zs.bootstrap_macro(scores, statuses, eligible, n_boot=n_boot, seed=seed)
    diff = reps["probe"] - reps["zero_shot"]
    ci = zs.ci95(diff)
    result["probe_minus_zero_shot"] = {"mean": float(diff.mean()), "ci95": ci}
    result["s1_passed"] = bool(ci[0] > 0)
    result["eligible_concepts"] = [CHEXPERT_LABELS[j] for j in eligible]
    result["n_boot"] = n_boot
    return result
