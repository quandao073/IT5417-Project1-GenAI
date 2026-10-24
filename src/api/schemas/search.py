"""Request/response models for /api/v1/search (plan section 13.3, Phase 3 subset)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.query.schema import ParseResult, SearchQuery
from src.retrieval.evidence import Evidence

View = Literal["AP", "PA"]


def _non_blank(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("query must not be blank")
    return value


class ParseRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)

    @field_validator("query")
    @classmethod
    def _strip(cls, value: str) -> str:
        return _non_blank(value)


class ParsedQuery(SearchQuery):
    """SearchQuery plus how it was obtained (plan 13.2, flat)."""

    parser: Literal["llm", "rule"]
    fallback_used: bool
    warnings: list[str]
    cached: bool
    llm_ms: float | None

    @classmethod
    def from_result(cls, result: ParseResult) -> ParsedQuery:
        return cls(**result.query.model_dump(), parser=result.parser,
                   fallback_used=result.fallback_used, warnings=list(result.warnings),
                   cached=result.cached, llm_ms=result.llm_ms)


_MODE_REMOVED = ("`mode` was removed: search has one path (template -> text encoder -> "
                 "cosine); the encoder is chosen on the server by BUILD_ID")


class SearchRequest(BaseModel):
    # An unknown field is a 422, never silently ignored: a client still sending `mode` or a
    # label policy would otherwise believe it was applied.
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500,
                       description="English or Vietnamese request; parsed by /query/parse rules")
    top_k: int = Field(10, ge=1, le=50)
    views: list[View] | None = Field(
        None, description="AP/PA facet; overrides the views the parser found when given")

    @model_validator(mode="before")
    @classmethod
    def _no_mode(cls, data: Any) -> Any:
        if isinstance(data, dict) and "mode" in data:
            raise ValueError(_MODE_REMOVED)
        return data

    @field_validator("query")
    @classmethod
    def _strip(cls, value: str) -> str:
        return _non_blank(value)


class SearchResult(BaseModel):
    rank: int
    study_id: str
    image_id: str
    image_url: str
    score: float = Field(description=(
        "cosine similarity between the query text and the image embedding, a ranking signal, "
        "NOT a probability"))
    report_evidence: list[Evidence] = Field(default_factory=list, description=(
        "exact sentences of the study's report, quoted after ranking for display only: they "
        "never change a rank. One per constrained concept the report mentions, or the best "
        "free-text match when nothing is included; empty for a study without a report and "
        "without the report index"))


class SearchResponse(BaseModel):
    request_id: str
    build_id: str
    parsed_query: ParsedQuery
    query_text: str | None = Field(description=(
        "what the text encoder read: the template of the parsed constraints, or the free text "
        "when none was recognised; null when nothing was searched"))
    results: list[SearchResult]
    warnings: list[str]
    latency_ms: int


class HealthResponse(BaseModel):
    status: Literal["ready"]
    build_id: str
    encoder: str
    model_id: str
    revision: str
    dimension: int
    vectors: int
    studies: int
    qdrant: Literal["ok"]
    report_index: Literal["ok", "disabled"]
    report_studies: int | None
    llm: Literal["ok", "disabled", "unreachable", "missing_model"]
    llm_model: str
