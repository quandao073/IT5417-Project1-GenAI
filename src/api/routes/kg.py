"""POST /api/v1/search/kg - the knowledge-graph search (docs/v2-kg-design.md, N2, N8, N9).

The reply is newline-delimited JSON: first a `results` event as soon as the images are ranked,
then an `explanation` event when the language model's page explanation has been written and
checked, so the results never wait for it. `stream: false` returns both in one JSON object.
Every reply names the data that produced it (KG build, score matrix, models, prompt hashes) and
the same record goes to the log as one JSON line.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.api.dependencies import AppState, require_ready
from src.kg.pipeline import KGResult

router = APIRouter()
logger = logging.getLogger("cxr.kg")
if not logger.handlers:                         # one JSON line per request, on stderr
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class KGSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    views: list[Literal["AP", "PA"]] = Field(default_factory=list,
                                             description="empty: the views the question names")
    stream: bool = Field(True, description="NDJSON events; false: one JSON object")
    explain: bool = Field(True, description="write the page explanation (one model call)")

    @field_validator("query")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


def results_event(state: AppState, result: KGResult) -> dict:
    kg = state.kg
    intents = [{**i.to_dict(), "warnings": [w.render(result.language) for w in i.warnings]}
               for i in result.intents]
    return {
        "event": "results", "request_id": result.request_id,
        "kg_build_id": kg.versions.get("kg_build_id"),
        "matrix_version": kg.versions.get("matrix_version"),
        "build_id": kg.versions.get("build_id"), "versions": kg.versions,
        "language": result.language,
        "parsed_query": {"include": [c.model_dump() for c in result.query.include],
                         "exclude": result.query.exclude, "views": result.query.views,
                         "terms": [t.model_dump() for t in result.query.terms]},
        "intents": intents, "warnings": result.warnings_text(),
        "degraded": result.degraded,
        "results": [{"rank": r["rank"], "study_id": r["study_id"], "image_id": r["image_id"],
                     "image_url": f"/api/v1/images/{r['image_id']}", "view": r["view"],
                     "score": r["score"]} for r in result.rows],
        "timings_ms": result.timings, "trace": result.trace,
        "disclaimer": state.settings.disclaimer_text}


def explanation_event(result: KGResult, explanation) -> dict:
    if explanation is None:
        return {"event": "explanation", "request_id": result.request_id, "status": "disabled",
                "fallback": "path"}
    return {"event": "explanation", "request_id": result.request_id,
            "status": explanation.status, "summary": explanation.summary,
            "images": [{"rank": k, "line": v} for k, v in sorted(explanation.images.items())],
            "dropped": explanation.dropped, "first_token_ms": explanation.first_token_ms,
            "total_ms": round(explanation.total_ms, 1), "error": explanation.error,
            "fallback": None if explanation.status in ("ok", "partial") else "path"}


def log_record(state: AppState, result: KGResult, explanation, question: str) -> dict:
    """The structured line of one request: ids, versions, timings and what degraded."""
    return {"event": "kg_search", "request_id": result.request_id,
            **{k: state.kg.versions.get(k) for k in ("kg_build_id", "matrix_version",
                                                     "build_id", "encoder_revision")},
            "parser_model": state.kg.versions.get("parser_model"),
            "prompt_sha256": {k.removesuffix("_prompt_sha256"): v
                              for k, v in state.kg.versions.items()
                              if k.endswith("_prompt_sha256")},
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
            "language": result.language, "results": len(result.rows),
            "timings_ms": result.timings, "degraded": result.degraded,
            "explanation": None if explanation is None else {
                "status": explanation.status, "total_ms": round(explanation.total_ms, 1),
                "first_token_ms": explanation.first_token_ms, "dropped": len(explanation.dropped)}}


@router.post("/api/v1/search/kg")
def search_kg(body: KGSearchRequest, state: Annotated[AppState, Depends(require_ready)]):
    if state.kg is None:
        raise HTTPException(status_code=503, detail={
            "status": "kg_unavailable", "reason": state.kg_reason or "the KG search is disabled"})
    result = state.kg.search(body.query, top_k=body.top_k, views=tuple(body.views))
    first = results_event(state, result)

    def explained():
        started = time.perf_counter()
        explanation = state.kg.explain(result) if body.explain else None
        event = explanation_event(result, explanation)
        logger.info(json.dumps(log_record(state, result, explanation, body.query),
                               ensure_ascii=False))
        return event, (time.perf_counter() - started) * 1000

    if not body.stream:
        event, _ = explained()
        return {**first, "explanation": event}

    def events() -> Iterator[bytes]:
        yield (json.dumps(first, ensure_ascii=False) + "\n").encode("utf-8")
        event, _ = explained()
        yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(events(), media_type="application/x-ndjson")
