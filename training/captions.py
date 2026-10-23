"""Captions for the label-aware fine-tune, rendered from resolved labels (H5, WORKLOG D32).

Every training sentence is a template of `src.retrieval.templates.render`, so a query rendered
from a parsed SearchQuery is in the training distribution by construction. The FT1 mix follows
CXR-RETRIEVE: one PRESENT concept, two PRESENT concepts, or one PRESENT and one ABSENT concept
("a and no b"). UNCERTAIN never enters a caption and counts as 0 in `y`; NOT_MENTIONED is never
negated (T7). A study whose only PRESENT training concept is No Finding gets the single caption
"no finding"; when a pathology is PRESENT as well (label noise) the pathology wins and No Finding
is never mentioned. FT2 swaps a fraction of samples for the study's impression text, whose `y`
comes from the labels themselves (+1 PRESENT, -1 ABSENT, 0 otherwise).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from src.data.labels import CHEXPERT_LABELS
from src.retrieval.templates import NO_FINDING, render

KINDS = ("single", "pair", "negation")
IMPRESSION = "impression"
_COL = {name: i for i, name in enumerate(CHEXPERT_LABELS)}


def label_vector(statuses_row: np.ndarray, concepts: Sequence[str]) -> np.ndarray:
    """int8 [L]: +1 where PRESENT, -1 where ABSENT, 0 for UNCERTAIN and NOT_MENTIONED."""
    y = np.zeros(len(concepts), dtype=np.int8)
    for j, concept in enumerate(concepts):
        status = statuses_row[_COL[concept]]
        if status == "PRESENT":
            y[j] = 1
        elif status == "ABSENT":
            y[j] = -1
    return y


class CaptionSampler:
    """Draws one caption and its label vector for a pool row.

    `statuses` is the object [N, 14] matrix of `probe_pool.resolved_status_matrix` (CHEXPERT_LABELS
    order); `concepts` are the L training concepts of `training.yaml`. `mix` weights the template
    kinds and must sum to one; `impression_fraction` of the draws return the impression instead.
    """

    def __init__(
        self,
        statuses: np.ndarray,
        concepts: Sequence[str],
        concepts_cfg: dict,
        *,
        mix: Mapping[str, float],
        impressions: Sequence[str | None] | None = None,
        impression_fraction: float = 0.0,
        seed: int,
    ):
        statuses = np.asarray(statuses, dtype=object)
        if statuses.ndim != 2 or statuses.shape[1] != len(CHEXPERT_LABELS):
            raise ValueError(f"statuses must be [N, {len(CHEXPERT_LABELS)}], got {statuses.shape}")
        if unknown := set(mix) - set(KINDS):
            raise ValueError(f"unknown caption kinds {sorted(unknown)}; known: {KINDS}")
        if not math.isclose(sum(mix.values()), 1.0, abs_tol=1e-6):
            raise ValueError(f"caption mix must sum to 1, got {dict(mix)}")
        if impression_fraction and impressions is None:
            raise ValueError("impression_fraction > 0 needs the impressions")
        if impressions is not None and len(impressions) != statuses.shape[0]:
            raise ValueError("one impression per statuses row is required")

        self.concepts = tuple(concepts)
        self._cfg = concepts_cfg
        self._statuses = statuses
        self._impressions = impressions
        self._impression_fraction = float(impression_fraction)
        self._kinds = [k for k in KINDS if mix.get(k, 0.0) > 0]
        self._p = np.array([mix[k] for k in self._kinds], dtype=np.float64)
        self._p /= self._p.sum()
        sub = statuses[:, [_COL[c] for c in self.concepts]]
        self._present = sub == "PRESENT"
        self._absent = sub == "ABSENT"
        self._no_finding = self.concepts.index(NO_FINDING) if NO_FINDING in self.concepts else -1
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        """Fresh stream, so DataLoader workers do not replay one another's draws."""
        self._rng = np.random.default_rng(seed)

    def eligible(self) -> np.ndarray:
        """bool [N]: the row has a PRESENT training concept (No Finding counts as one)."""
        return self._present.any(axis=1)

    def sample_kind(self, row: int) -> str:
        """The kind drawn for this row, before any fallback to `single`."""
        if self._impressions is not None and self._rng.random() < self._impression_fraction:
            return IMPRESSION
        return str(self._kinds[self._rng.choice(len(self._kinds), p=self._p)])

    def sample(self, row: int) -> tuple[str, np.ndarray]:
        """(text, y int8 [L]); y is +1 for affirmed, -1 for negated, 0 for the rest."""
        kind = self.sample_kind(row)
        if kind == IMPRESSION:
            text = (self._impressions[row] or "").strip()
            if not text:
                raise ValueError(f"row {row} has an empty impression")
            return text, label_vector(self._statuses[row], self.concepts)

        present = np.flatnonzero(self._present[row])
        if present.size == 0:
            raise ValueError(f"row {row} has no PRESENT training concept; filter by eligible()")
        pathologies = present[present != self._no_finding]
        y = np.zeros(len(self.concepts), dtype=np.int8)
        if pathologies.size == 0:
            y[self._no_finding] = 1
            return render([NO_FINDING], [], self._cfg), y

        include = [int(self._rng.choice(pathologies))]
        exclude: list[int] = []
        if kind == "pair" and pathologies.size >= 2:
            include = [int(j) for j in self._rng.choice(pathologies, 2, replace=False)]
        elif kind == "negation":
            absent = np.flatnonzero(self._absent[row])
            absent = absent[absent != self._no_finding]
            if absent.size:
                exclude = [int(self._rng.choice(absent))]
        y[include] = 1
        y[exclude] = -1
        text = render([self.concepts[j] for j in include],
                      [self.concepts[j] for j in exclude], self._cfg)
        return text, y
