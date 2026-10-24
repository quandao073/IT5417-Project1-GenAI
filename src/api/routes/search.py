"""POST /api/v1/search - one path: template -> text encoder -> cosine (WORKLOG D38).

A thin adapter: parse the request, run `src.retrieval.pipeline.run_search`, map its hits to
the response. The template, the evidence and the degradation warnings live in the pipeline.
A dead encoder or Qdrant is a 503.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import AppState, require_ready
from src.api.schemas.search import ParsedQuery, SearchRequest, SearchResponse, SearchResult
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
        outcome = pipeline.run_search(state, query, top_k=body.top_k,
                                      views=body.views or query.views)
    except pipeline.VectorLegError as exc:
        raise HTTPException(status_code=503,
                            detail={"status": "not_ready", "reason": f"qdrant: {exc}"}) from exc
    return SearchResponse(
        request_id=uuid.uuid4().hex,
        build_id=state.manifest["build_id"],
        parsed_query=ParsedQuery.from_result(parsed),
        query_text=outcome.text,
        results=[_result(hit) for hit in outcome.hits],
        warnings=[*parsed.warnings, *outcome.warnings],
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def _result(hit: pipeline.Hit) -> SearchResult:
    return SearchResult(
        rank=hit.rank, study_id=hit.study_id, image_id=hit.image_id,
        image_url=f"/api/v1/images/{hit.image_id}", score=hit.score,
        report_evidence=[e.model_dump() for e in hit.evidence],
    )
