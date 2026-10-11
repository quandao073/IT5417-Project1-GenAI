"""POST /api/v1/search - Phase 3 serves mode `vector` only."""

from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import AppState, require_ready
from src.api.schemas.search import (
    ParsedQueryStub,
    ScoreBreakdown,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from src.retrieval.image_search import search_image_embeddings

router = APIRouter()


@router.post("/api/v1/search", response_model=SearchResponse)
def search(
    body: SearchRequest, state: Annotated[AppState, Depends(require_ready)]
) -> SearchResponse:
    if body.mode != "vector":
        raise HTTPException(status_code=400,
                            detail=f"mode '{body.mode}' is not available in this build")
    started = time.perf_counter()
    try:
        hits = search_image_embeddings(
            body.query, body.views, body.top_k, encoder=state.encoder, client=state.client,
            collection=state.collection, lock=state.encode_lock,
        )
    except Exception as exc:  # noqa: BLE001 - a dead dependency is a 503, not a traceback
        raise HTTPException(status_code=503,
                            detail={"status": "not_ready", "reason": f"qdrant: {exc}"}) from exc
    return SearchResponse(
        request_id=uuid.uuid4().hex,
        build_id=state.manifest["build_id"],
        parsed_query=ParsedQueryStub(semantic_query=body.query, views=body.views),
        results=[
            SearchResult(rank=h.rank, study_id=h.study_id, image_id=h.image_id,
                         image_url=f"/api/v1/images/{h.image_id}", score=h.score,
                         score_breakdown=ScoreBreakdown(vector_rank=h.rank))
            for h in hits
        ],
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
