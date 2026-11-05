"""`quality` of each scorable concept (WORKLOG D55): the lenient AUROC of its score column on
pool-val with a 95% bootstrap interval, `weak` below the gate (point estimate, 0.65) and
`borderline` when the interval's lower end is below it. The 14 label columns are measured against
the resolved CheXpert statuses of pool-val; an extended column keeps what `build-image-scores-ext`
measured against its RadGraph silver labels. Keys are concept ids (CUI or PRJ id), as in the KG.
"""

from __future__ import annotations

import numpy as np

from src.retrieval import image_scores_ext as iext


def base_columns(logits: np.ndarray, statuses: np.ndarray, labels: list[str], *, n_boot: int,
                 seed: int) -> dict[str, dict]:
    """{label: lenient AUROC, ci95, n_present, n_negative} for the first len(labels) columns of
    `logits` ([n, k], pool-val) against `statuses` ([n, len(labels)])."""
    return {label: iext.lenient_auroc(logits[:, j], statuses[:, j], n_boot=n_boot, seed=seed)
            for j, label in enumerate(labels)}


def table(base: dict[str, dict], label_ids: dict[str, str], extended: dict[str, dict],
          gate: float) -> dict[str, dict]:
    """{concept id: {column, auroc_lenient, ci95, n_present, n_negative, weak, borderline}}."""
    rows = {label_ids[label]: {"column": label, **res} for label, res in base.items()}
    rows |= {cui: {"column": cui, **{k: v[k] for k in ("auroc_lenient", "ci95", "n_present")}}
             for cui, v in extended.items()}
    for row in rows.values():
        row["weak"] = bool(row["auroc_lenient"] < gate)
        row["borderline"] = bool(row["ci95"][0] < gate)
    return dict(sorted(rows.items()))
