"""Request/response models for /api/v1/search (plan section 13.3, Phase 3 subset)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

View = Literal["AP", "PA"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500,
                       description="English semantic query; used verbatim until the parser lands")
    mode: Literal["vector", "label", "hybrid"] = "vector"
    top_k: int = Field(10, ge=1, le=50)
    views: list[View] | None = Field(None, description="AP/PA facet filter; empty = no filter")

    @field_validator("query")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class ParsedQueryStub(BaseModel):
    semantic_query: str
    views: list[View] | None
    parser: Literal["passthrough"] = "passthrough"


class ScoreBreakdown(BaseModel):
    vector_rank: int


class SearchResult(BaseModel):
    rank: int
    study_id: str
    image_id: str
    image_url: str
    score: float = Field(description=(
        "Cosine similarity between the query and the image embedding, used for ranking. "
        "NOT a probability; on this corpus it falls roughly in 0.2-0.5."))
    score_breakdown: ScoreBreakdown


class SearchResponse(BaseModel):
    request_id: str
    build_id: str
    parsed_query: ParsedQueryStub
    results: list[SearchResult]
    latency_ms: int


class HealthResponse(BaseModel):
    status: Literal["ready"]
    build_id: str
    model_id: str
    revision: str
    dimension: int
    vectors: int
    qdrant: Literal["ok"]
    report_index: Literal["disabled"]
    llm: Literal["disabled"]
