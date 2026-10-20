"""Hybrid retrieval (spec 8): reciprocal-rank fusion of the vector and report legs.

Pure: the two legs arrive as callables already bound to the query, the label-allowed set and
the view facet, so this module never touches Qdrant or SQLite. Label constraints are applied
before it runs (WORKLOG D23): `label_order` is the allowed set in the fixed D18 order, and
results never leave it because both legs only ever return studies inside it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from src.retrieval.image_search import ImageCandidate
from src.retrieval.report_search import ReportCandidate, ReportSearchError

_NO_SIGNAL = "no ranking signal; results in label order"


@dataclass(frozen=True)
class FusionParams:
    rrf_k: int
    leg_weights: Mapping[str, float]
    vector_top_k: int
    report_top_k: int

    @classmethod
    def from_config(cls, retrieval_cfg: dict) -> FusionParams:
        fusion, candidates = retrieval_cfg["fusion"], retrieval_cfg["candidates"]
        return cls(rrf_k=int(fusion["rrf_k"]),
                   leg_weights={leg: float(w) for leg, w in fusion["leg_weights"].items()},
                   vector_top_k=int(candidates["vector_top_k"]),
                   report_top_k=int(candidates["report_top_k"]))


@dataclass(frozen=True)
class HybridResult:
    study_id: str
    score: float                  # RRF score; a ranking signal, not a probability
    vector_rank: int | None
    report_rank: int | None
    rank: int                     # 1-based, final


@dataclass(frozen=True)
class HybridOutput:
    results: list[HybridResult]
    total_matches: int | None     # len(label_order) when the query has constraints, else None
    warnings: list[str]
    legs: dict[str, int]          # candidates each leg contributed


def rrf(rankings: Mapping[str, Sequence[str]], weights: Mapping[str, float],
        k: int) -> dict[str, float]:
    """score(s) = sum over legs containing s of weight / (k + 1-based rank in that leg)."""
    scores: dict[str, float] = {}
    for leg, ranked in rankings.items():
        for rank, study_id in enumerate(ranked, start=1):
            scores[study_id] = scores.get(study_id, 0.0) + weights[leg] / (k + rank)
    return scores


def _vector_ranking(candidates: Sequence[ImageCandidate]) -> list[str]:
    # A study owns several images in Qdrant; its best image sets its rank.
    return list(dict.fromkeys(c.study_id for c in candidates))


def _order_key(study_id: str, score: float, vector_rank: int | None) -> tuple:
    """Best score first; ties go to the study the vector leg ranked higher, then to the id."""
    return (-score, vector_rank is None, vector_rank or 0, study_id)


def search_hybrid(
    *, top_k: int,
    label_order: Sequence[str] | None,
    vector_leg: Callable[[int], list[ImageCandidate]] | None,
    report_leg: Callable[[int], list[ReportCandidate]] | None,
    fusion: FusionParams,
) -> HybridOutput:
    warnings: list[str] = []
    vector = _vector_ranking(vector_leg(fusion.vector_top_k)) if vector_leg else []
    report: list[str] = []
    if report_leg is not None:
        try:
            report = [c.study_id for c in report_leg(fusion.report_top_k)]
        except ReportSearchError as exc:
            warnings.append(f"report leg failed: {exc}")
    if label_order is not None:
        # The legs are bound to the allowed set, but nothing here may rely on it.
        allowed = set(label_order)
        vector = [s for s in vector if s in allowed]
        report = [s for s in report if s in allowed]
    legs = {"vector": len(vector), "report": len(report)}
    total = None if label_order is None else len(label_order)

    if not vector and not report:
        if label_order is None:
            return HybridOutput([], total, warnings, legs)
        warnings.append(_NO_SIGNAL)
        results = [HybridResult(s, 0.0, None, None, rank)
                   for rank, s in enumerate(label_order[:top_k], start=1)]
        return HybridOutput(results, total, warnings, legs)

    scores = rrf({"vector": vector, "report": report}, fusion.leg_weights, fusion.rrf_k)
    vector_rank = {s: r for r, s in enumerate(vector, start=1)}
    report_rank = {s: r for r, s in enumerate(report, start=1)}
    ordered = sorted(scores, key=lambda s: _order_key(s, scores[s], vector_rank.get(s)))
    results = [HybridResult(s, scores[s], vector_rank.get(s), report_rank.get(s), rank)
               for rank, s in enumerate(ordered[:top_k], start=1)]
    return HybridOutput(results, total, warnings, legs)
