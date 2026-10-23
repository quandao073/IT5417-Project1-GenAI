"""The search behind /api/v1/search (spec 8.2, 9.2): rank in one of four modes, then quote.

`label` and `vector` keep their Phase 4 ranking, because they are the baselines `hybrid` has
to beat: `label` returns the label-filtered studies in the fixed D18 order, `vector` ranks
images by cosine with only the view facet (G3) and now flags each hit's constraint violations.
`image` ranks the whole corpus by per-concept probe logits (`image_concepts`), no labels, no
reports. `hybrid` filters on labels first (D23) and fuses the vector, report and image legs
inside the allowed set (`hybrid.search_hybrid`). Every mode then quotes report sentences as
evidence; without the report index search still answers, with a warning instead of evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Literal

from src.query.schema import SearchQuery
from src.retrieval.evidence import Evidence
from src.retrieval.hybrid import FusionParams, search_hybrid
from src.retrieval.image_concepts import rank_studies
from src.retrieval.image_search import ImageCandidate, search_image_embeddings
from src.retrieval.label_filter import (
    filter_studies_by_labels,
    ordered_matches,
    resolved_labels,
    satisfies,
)
from src.retrieval.report_search import (
    ReportSearchError,
    build_match,
    fetch_texts,
    free_text_tokens,
    search_reports,
)

if TYPE_CHECKING:
    from src.api.dependencies import AppState

_VECTOR_SKIPPED = "semantic query is empty; vector leg skipped"
_NO_EVIDENCE = "report index unavailable; no evidence"
_UNCERTAIN_AS_PRESENT = "image score treats UNCERTAIN as PRESENT"


class VectorLegError(RuntimeError):
    """The encoder or Qdrant failed; the route answers 503, as Phase 4 did."""


@dataclass(frozen=True)
class Hit:
    rank: int
    study_id: str
    image_id: str
    score: float | None = None      # vector: cosine; image: probe score; hybrid: RRF; label: None
    vector_rank: int | None = None
    report_rank: int | None = None
    image_rank: int | None = None
    rrf_score: float | None = None
    label_match: bool | None = None
    constraints_satisfied: bool | None = None
    labels: list[dict] = field(default_factory=list)   # status of each constrained concept
    evidence: list[Evidence] = field(default_factory=list)


@dataclass(frozen=True)
class SearchOutcome:
    hits: list[Hit]
    total_matches: int | None
    warnings: list[str]


def run_search(state: AppState, query: SearchQuery,
               mode: Literal["label", "vector", "image", "hybrid"], *, top_k: int,
               views: Sequence[str], absence_policy: str, uncertainty_policy: str,
               fusion: FusionParams | None = None) -> SearchOutcome:
    policies = {"absence_policy": absence_policy, "uncertainty_policy": uncertainty_policy}
    warnings: list[str] = []
    if mode == "vector":
        hits, total = _vector_hits(state, query, views, top_k, policies, warnings), None
    elif mode == "image":
        hits, total = _image_hits(state, query, views, top_k, policies, warnings), None
    elif mode == "label":
        hits, total = _label_hits(state, query, views, top_k, policies, warnings)
    else:
        hits, total = _hybrid_hits(state, query, views, top_k, policies,
                                   fusion or state.fusion, warnings)
    quotes = _evidence(state, query, hits, warnings)
    concepts = query.constrained_concepts()
    hits = [replace(h, labels=resolved_labels(state.labels, h.study_id, concepts),
                    evidence=quotes.get(h.study_id, [])) for h in hits]
    return SearchOutcome(hits, total, warnings)


def _vector_search(state: AppState, semantic_query: str, views: Sequence[str], limit: int,
                   *, point_ids: Sequence[str] | None = None) -> list[ImageCandidate]:
    try:
        return search_image_embeddings(
            semantic_query, views or None, limit, encoder=state.encoder, client=state.client,
            collection=state.collection, lock=state.encode_lock, point_ids=point_ids,
        )
    except Exception as exc:  # noqa: BLE001 - a dead dependency is a 503, not a traceback
        raise VectorLegError(str(exc)) from exc


def _allowed(state: AppState, query: SearchQuery, views: Sequence[str],
             policies: dict) -> list[str]:
    """Studies passing the constraints, in D18 order, whose selected image is in `views`."""
    mask = filter_studies_by_labels(state.labels, query.include, query.exclude, **policies)
    studies = ordered_matches(state.labels, mask)
    if views:
        wanted = set(views)
        studies = [s for s in studies
                   if state.images[state.study_image[s]].projection in wanted]
    return studies


def _vector_hits(state: AppState, query: SearchQuery, views: Sequence[str], top_k: int,
                 policies: dict, warnings: list[str]) -> list[Hit]:
    if not query.semantic_query:
        warnings.append(_VECTOR_SKIPPED)
        return []
    constrained = bool(query.include or query.exclude)
    return [
        Hit(c.rank, c.study_id, c.image_id, c.score, vector_rank=c.rank,
            constraints_satisfied=satisfies(state.labels, c.study_id, query.include,
                                            query.exclude, **policies) if constrained else None)
        for c in _vector_search(state, query.semantic_query, views, top_k)
    ]


def _image_hits(state: AppState, query: SearchQuery, views: Sequence[str], top_k: int,
                policies: dict, warnings: list[str]) -> list[Hit]:
    if not (query.include or query.exclude):
        if query.semantic_query:
            warnings.append("no concept; ranked by text-image cosine")
        return _vector_hits(state, query, views, top_k, policies, warnings)
    if state.image_status != "ok":
        warnings.append("image probe unavailable")
        return []
    if any(c.status == "UNCERTAIN" for c in query.include):
        warnings.append(_UNCERTAIN_AS_PRESENT)
    wanted = set(views)
    allowed = [s for s, image_id in state.study_image.items()
               if state.images[image_id].projection in wanted] if views else None
    ranked = rank_studies(state.image_scores, query.include, query.exclude,
                          allowed=allowed, limit=top_k)
    return [Hit(rank, study, state.study_image[study], score, image_rank=rank,
                constraints_satisfied=satisfies(state.labels, study, query.include,
                                                query.exclude, **policies))
            for rank, (study, score) in enumerate(ranked, start=1)]


def _label_hits(state: AppState, query: SearchQuery, views: Sequence[str], top_k: int,
                policies: dict, warnings: list[str]) -> tuple[list[Hit], int]:
    if not query.include and not query.exclude:
        warnings.append("label mode needs at least one concept")
        return [], 0
    studies = _allowed(state, query, views, policies)
    hits = [Hit(i, s, state.study_image[s], label_match=True, constraints_satisfied=True)
            for i, s in enumerate(studies[:top_k], start=1)]
    return hits, len(studies)


def _hybrid_hits(state: AppState, query: SearchQuery, views: Sequence[str], top_k: int,
                 policies: dict, fusion: FusionParams,
                 warnings: list[str]) -> tuple[list[Hit], int | None]:
    constrained = bool(query.include or query.exclude)
    if constrained:
        label_order = _allowed(state, query, views, policies)
        if not label_order:
            return [], 0   # nothing passes; "results in label order" would mislead
        point_ids = [state.point_of_study[s] for s in label_order]
        report_allowed: Sequence[str] | None = label_order
    else:
        # The vector leg applies views as a Qdrant filter (G3); the report leg needs ids.
        label_order, point_ids, report_allowed = None, None, None
        if views:
            wanted = set(views)
            report_allowed = sorted({r.study_id for r in state.images.values()
                                     if r.projection in wanted})

    vector_leg = None
    if query.semantic_query:
        vector_leg = partial(_vector_search, state, query.semantic_query, views,
                             point_ids=point_ids)
    else:
        warnings.append(_VECTOR_SKIPPED)
    report_leg = None
    if state.report_status != "ok":
        warnings.append("report index unavailable; hybrid ran without the report leg")
    elif (match := build_match(query, state.report_terms,
                               min_token_chars=state.report_min_token_chars)) is None:
        warnings.append("report leg skipped: no English term to match")
    else:
        report_leg = partial(search_reports, state.report_db, state.report_lock, match,
                             report_allowed)

    image_leg = None
    if constrained and fusion.weight("image") > 0:   # at weight 0 the leg is silent, like main
        if state.image_status != "ok":
            warnings.append("image probe unavailable; hybrid ran without the image leg")
        else:
            if any(c.status == "UNCERTAIN" for c in query.include):
                warnings.append(_UNCERTAIN_AS_PRESENT)

            def image_leg(k: int) -> list[tuple[str, float]]:
                return rank_studies(state.image_scores, query.include, query.exclude,
                                    allowed=label_order, limit=k)

    out = search_hybrid(top_k=top_k, label_order=label_order, vector_leg=vector_leg,
                        report_leg=report_leg, image_leg=image_leg, fusion=fusion)
    warnings.extend(out.warnings)
    hits = [Hit(r.rank, r.study_id, state.study_image[r.study_id], r.score,
                vector_rank=r.vector_rank, report_rank=r.report_rank, image_rank=r.image_rank,
                rrf_score=r.score,
                label_match=True if constrained else None,   # None: no label filter ran
                constraints_satisfied=True if constrained else None)
            for r in out.results]
    return hits, out.total_matches


def _evidence(state: AppState, query: SearchQuery, hits: list[Hit],
              warnings: list[str]) -> dict[str, list[Evidence]]:
    """{study_id: quotes}, one quote per constrained concept, free text only when nothing is
    included. A study is quoted once even when several of its images are vector hits."""
    if state.report_status != "ok":
        warnings.append(_NO_EVIDENCE)
        return {}
    try:
        texts = fetch_texts(state.report_db, state.report_lock,
                            list(dict.fromkeys(h.study_id for h in hits)))
    except ReportSearchError:
        warnings.append(_NO_EVIDENCE)   # the report index never takes search down
        return {}
    tokens = () if query.include else free_text_tokens(
        query, min_token_chars=state.report_min_token_chars)
    return {s: state.evidence.build(t, query, free_text_tokens=tokens) for s, t in texts.items()}
