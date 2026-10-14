"""POST /api/v1/search - modes `label` and `vector`; `hybrid` arrives in Phase 5.

Both modes run the same parser. `vector` encodes the parsed semantic_query and
applies only the view facet (WORKLOG G3); `label` filters the resolved label
matrix and returns matches in the fixed order (D18). Every result lists the
resolved status of each constrained concept, so vector mode shows its own
constraint violations.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import AppState, require_ready
from src.api.schemas.search import (
    LabelAssertion,
    ParsedQuery,
    ScoreBreakdown,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from src.query.schema import SearchQuery
from src.retrieval.image_search import search_image_embeddings
from src.retrieval.label_filter import filter_studies_by_labels, ordered_matches, resolved_labels

router = APIRouter()


@router.post("/api/v1/search", response_model=SearchResponse)
def search(
    body: SearchRequest, state: Annotated[AppState, Depends(require_ready)]
) -> SearchResponse:
    if body.mode == "hybrid":
        raise HTTPException(status_code=400,
                            detail="mode 'hybrid' is not available in this build")
    started = time.perf_counter()
    parsed = state.parser.parse(body.query)
    query = parsed.query
    warnings = list(parsed.warnings)
    views = body.views or query.views
    constrained = query.constrained_concepts()
    total: int | None = None
    if body.mode == "vector":
        hits = _vector_hits(state, query.semantic_query, views, body.top_k, warnings)
        results = [
            _result(state, rank=h.rank, study_id=h.study_id, image_id=h.image_id,
                    score=h.score, breakdown=ScoreBreakdown(vector_rank=h.rank),
                    constrained=constrained)
            for h in hits
        ]
    else:
        studies, total = _label_matches(state, query, body, views, warnings)
        results = [
            _result(state, rank=i, study_id=s, image_id=state.study_image[s], score=None,
                    breakdown=ScoreBreakdown(label_match=True), constrained=constrained)
            for i, s in enumerate(studies[: body.top_k], start=1)
        ]
    return SearchResponse(
        request_id=uuid.uuid4().hex,
        build_id=state.manifest["build_id"],
        parsed_query=ParsedQuery.from_result(parsed),
        results=results,
        total_matches=total,
        warnings=warnings,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def _vector_hits(state: AppState, semantic_query: str, views: list[str], top_k: int,
                 warnings: list[str]):
    if not semantic_query:
        warnings.append("semantic query is empty; vector leg skipped")
        return []
    try:
        return search_image_embeddings(
            semantic_query, views or None, top_k, encoder=state.encoder, client=state.client,
            collection=state.collection, lock=state.encode_lock,
        )
    except Exception as exc:  # noqa: BLE001 - a dead dependency is a 503, not a traceback
        raise HTTPException(status_code=503,
                            detail={"status": "not_ready", "reason": f"qdrant: {exc}"}) from exc


def _label_matches(state: AppState, query: SearchQuery, body: SearchRequest,
                   views: list[str], warnings: list[str]) -> tuple[list[str], int]:
    if not query.include and not query.exclude:
        warnings.append("label mode needs at least one concept")
        return [], 0
    mask = filter_studies_by_labels(
        state.labels, query.include, query.exclude,
        absence_policy=body.absence_policy, uncertainty_policy=body.uncertainty_policy,
    )
    studies = ordered_matches(state.labels, mask)
    if views:
        allowed = set(views)
        studies = [s for s in studies
                   if state.images[state.study_image[s]].projection in allowed]
    return studies, len(studies)


def _result(state: AppState, *, rank: int, study_id: str, image_id: str, score: float | None,
            breakdown: ScoreBreakdown, constrained: list[str]) -> SearchResult:
    return SearchResult(
        rank=rank, study_id=study_id, image_id=image_id,
        image_url=f"/api/v1/images/{image_id}", score=score, score_breakdown=breakdown,
        labels=[LabelAssertion(**a) for a in resolved_labels(state.labels, study_id, constrained)],
    )
