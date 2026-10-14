"""Label leg of retrieval (plan 11.1, tool contract 12.2; WORKLOG G1, D9, D18, T7).

labels.parquet keeps every source; this module resolves it once, at API startup,
into one int8 status per (study, concept) plus the winning source for
provenance, and filters that matrix with numpy. NOT_MENTIONED is never ABSENT:
it passes `exclude_present` (no evidence of presence) and fails
`explicit_absence_required` (no evidence of absence). Label mode has no ranking
signal, so matches come back in one permutation fixed by a seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from src.data.labels import CHEXPERT_LABELS, resolve
from src.query.schema import Constraint

STATUS_CODE = {"NOT_MENTIONED": 0, "ABSENT": 1, "UNCERTAIN": 2, "PRESENT": 3}
CODE_STATUS = {code: status for status, code in STATUS_CODE.items()}
AbsencePolicy = Literal["exclude_present", "explicit_absence_required"]
UncertaintyPolicy = Literal["EXCLUDE", "INCLUDE"]

_NOT_MENTIONED, _ABSENT, _UNCERTAIN, _PRESENT = 0, 1, 2, 3


@dataclass(frozen=True)
class LabelMatrix:
    study_ids: np.ndarray                 # str, in the order given to build_label_matrix
    row_of: dict[str, int]
    concepts: tuple[str, ...]
    col_of: dict[str, int]
    status: np.ndarray                    # int8 [n, len(concepts)], STATUS_CODE values
    source: np.ndarray                    # object [n, len(concepts)], str or None
    order: np.ndarray                     # int64 permutation of range(n)


def build_label_matrix(
    labels_long: pl.DataFrame,
    study_ids: Sequence[str],
    priority: Sequence[str],
    *,
    seed: int,
    concepts: Sequence[str] = CHEXPERT_LABELS,
) -> LabelMatrix:
    ids = list(study_ids)
    resolved = resolve(labels_long.filter(pl.col("study_id").is_in(ids)), priority)
    frame = pl.DataFrame({"study_id": ids})
    status = np.full((len(ids), len(concepts)), _NOT_MENTIONED, dtype=np.int8)
    source = np.full((len(ids), len(concepts)), None, dtype=object)
    for j, concept in enumerate(concepts):
        one = resolved.filter(pl.col("concept") == concept).select("study_id", "status", "source")
        joined = frame.join(one, on="study_id", how="left")
        codes = joined["status"].fill_null("NOT_MENTIONED").replace_strict(
            STATUS_CODE, return_dtype=pl.Int8)
        status[:, j] = codes.to_numpy()
        source[:, j] = joined["source"].to_list()
    return LabelMatrix(
        study_ids=np.asarray(ids, dtype=object),
        row_of={s: i for i, s in enumerate(ids)},
        concepts=tuple(concepts),
        col_of={c: j for j, c in enumerate(concepts)},
        status=status,
        source=source,
        order=np.random.default_rng(seed).permutation(len(ids)),
    )


def filter_studies_by_labels(
    matrix: LabelMatrix,
    include: Sequence[Constraint],
    exclude: Sequence[str],
    *,
    absence_policy: AbsencePolicy,
    uncertainty_policy: UncertaintyPolicy,
) -> np.ndarray:
    """Boolean mask over matrix rows; every constraint must hold (spec 5.2)."""
    if not include and not exclude:
        raise ValueError("at least one constraint is required; refusing to return the corpus")
    keep_uncertain = uncertainty_policy == "INCLUDE"
    mask = np.ones(len(matrix.study_ids), dtype=bool)
    for c in include:
        col = matrix.status[:, matrix.col_of[c.concept]]
        if c.status == "PRESENT":
            ok = col == _PRESENT
            if keep_uncertain:
                ok |= col == _UNCERTAIN
        else:
            ok = col == _UNCERTAIN
        mask &= ok
    for concept in exclude:
        col = matrix.status[:, matrix.col_of[concept]]
        if absence_policy == "explicit_absence_required":
            ok = col == _ABSENT
        else:
            ok = col != _PRESENT
            if not keep_uncertain:
                ok &= col != _UNCERTAIN
        mask &= ok
    return mask


def ordered_matches(matrix: LabelMatrix, mask: np.ndarray) -> list[str]:
    """Matching study ids in the fixed pseudo-random order (D18)."""
    order = matrix.order
    return matrix.study_ids[order][mask[order]].tolist()


def resolved_labels(matrix: LabelMatrix, study_id: str, concepts: Sequence[str]) -> list[dict]:
    row = matrix.row_of[study_id]
    return [{"concept": c, "status": CODE_STATUS[int(matrix.status[row, matrix.col_of[c]])],
             "source": matrix.source[row, matrix.col_of[c]]} for c in concepts]
