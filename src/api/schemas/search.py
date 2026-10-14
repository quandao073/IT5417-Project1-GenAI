"""Request/response models for /api/v1/search (plan section 13.3, Phase 3 subset)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.query.schema import ParseResult, SearchQuery

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


AbsencePolicy = Literal["exclude_present", "explicit_absence_required"]
UncertaintyPolicy = Literal["EXCLUDE", "INCLUDE"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500,
                       description="English or Vietnamese request; parsed by /query/parse rules")
    mode: Literal["vector", "label", "hybrid"] = "vector"
    top_k: int = Field(10, ge=1, le=50)
    views: list[View] | None = Field(
        None, description="AP/PA facet; overrides the views the parser found when given")
    # Defaults mirror configs/retrieval.yaml:constraints (a test ties them together).
    absence_policy: AbsencePolicy = "exclude_present"
    uncertainty_policy: UncertaintyPolicy = "EXCLUDE"

    @field_validator("query")
    @classmethod
    def _strip(cls, value: str) -> str:
        return _non_blank(value)


class LabelAssertion(BaseModel):
    concept: str
    status: Literal["PRESENT", "ABSENT", "UNCERTAIN", "NOT_MENTIONED"]
    source: str | None = Field(description="winning label source (WORKLOG G1); null if silent")


class ScoreBreakdown(BaseModel):
    vector_rank: int | None = None
    label_match: bool | None = None


class SearchResult(BaseModel):
    rank: int
    study_id: str
    image_id: str
    image_url: str
    score: float | None = Field(description=(
        "vector mode: cosine similarity between the query and the image embedding, a ranking "
        "signal, NOT a probability (roughly 0.2-0.5 on this corpus). label mode: null."))
    score_breakdown: ScoreBreakdown
    labels: list[LabelAssertion] = Field(
        description="resolved status of every constrained concept, in both modes")


class SearchResponse(BaseModel):
    request_id: str
    build_id: str
    parsed_query: ParsedQuery
    results: list[SearchResult]
    total_matches: int | None = Field(
        description="label mode: studies matching after the view filter; vector mode: null")
    warnings: list[str]
    latency_ms: int


class HealthResponse(BaseModel):
    status: Literal["ready"]
    build_id: str
    model_id: str
    revision: str
    dimension: int
    vectors: int
    studies: int
    qdrant: Literal["ok"]
    report_index: Literal["disabled"]
    llm: Literal["ok", "disabled", "unreachable", "missing_model"]
    llm_model: str
