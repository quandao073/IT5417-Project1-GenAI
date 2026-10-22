"""POST /api/v1/search - modes `label`, `vector` and `hybrid` (spec 9.2).

A thin adapter: parse the request, run `src.retrieval.pipeline.run_search`, map its hits to
the response. The modes, the evidence and the degradation warnings live in the pipeline. A
dead encoder or Qdrant is a 503 in every mode that needs the vector leg.
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
from src.retrieval import pipeline

router = APIRouter()


@router.post("/api/v1/search", response_model=SearchResponse)
def search(
    body: SearchRequest, state: Annotated[AppState, Depends(require_ready)]
) -> SearchResponse:
    started = time.perf_counter()
    parsed = state.parser.parse(body.query)
    query = parsed.query
    try:
        outcome = pipeline.run_search(
            state, query, body.mode, top_k=body.top_k, views=body.views or query.views,
            absence_policy=body.absence_policy, uncertainty_policy=body.uncertainty_policy,
        )
    except pipeline.VectorLegError as exc:
        raise HTTPException(status_code=503,
                            detail={"status": "not_ready", "reason": f"qdrant: {exc}"}) from exc
    return SearchResponse(
        request_id=uuid.uuid4().hex,
        build_id=state.manifest["build_id"],
        parsed_query=ParsedQuery.from_result(parsed),
        results=[_result(hit) for hit in outcome.hits],
        total_matches=outcome.total_matches,
        warnings=[*parsed.warnings, *outcome.warnings],
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def _result(hit: pipeline.Hit) -> SearchResult:
    return SearchResult(
        rank=hit.rank, study_id=hit.study_id, image_id=hit.image_id,
        image_url=f"/api/v1/images/{hit.image_id}", score=hit.score,
        score_breakdown=ScoreBreakdown(
            vector_rank=hit.vector_rank, report_rank=hit.report_rank, image_rank=hit.image_rank,
            rrf_score=hit.rrf_score,
            label_match=hit.label_match, constraints_satisfied=hit.constraints_satisfied),
        labels=[LabelAssertion(**a) for a in hit.labels],
        report_evidence=[e.model_dump() for e in hit.evidence],
    )
