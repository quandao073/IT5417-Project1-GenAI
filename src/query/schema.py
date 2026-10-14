"""The structured query every retrieval mode consumes (plan 10.2, slimmed per WORKLOG D17).

The parser - LLM or rule - produces a SearchQuery; `label` mode filters on its
include/exclude lists, `vector` mode encodes its semantic_query. Concept names
are canonicalised to the 14 CheXpert labels here, so nothing downstream ever
sees a name labels.parquet does not know. Absence and uncertainty policies are
request parameters, not parse output; `explicit_absence`, `top_k` and `intent`
from the plan's schema are therefore absent (D17).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.data.labels import CHEXPERT_LABELS

Status = Literal["PRESENT", "UNCERTAIN"]
View = Literal["AP", "PA"]
Language = Literal["en", "vi"]
ParserName = Literal["llm", "rule"]

_CANONICAL = {name.casefold(): name for name in CHEXPERT_LABELS}


def canonical_concept(name: str) -> str:
    """'cardiomegaly' -> 'Cardiomegaly'; anything outside the 14 labels raises."""
    key = " ".join(str(name).split()).casefold()
    if key not in _CANONICAL:
        raise ValueError(f"unknown concept {name!r}; expected one of {list(CHEXPERT_LABELS)}")
    return _CANONICAL[key]


class Constraint(BaseModel):
    concept: str
    status: Status

    @field_validator("concept")
    @classmethod
    def _canonical(cls, value: str) -> str:
        return canonical_concept(value)


class SearchQuery(BaseModel):
    semantic_query: str = Field(
        "", max_length=300,
        description="English text handed to the text encoder; '' = nothing to encode")
    include: list[Constraint] = Field(default_factory=list)
    exclude: list[str] = Field(
        default_factory=list,
        description="negated concepts; the request's absence_policy decides how they apply")
    views: list[View] = Field(default_factory=list, description="empty = no view filter")
    language: Language

    @field_validator("semantic_query")
    @classmethod
    def _collapse_whitespace(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("exclude")
    @classmethod
    def _canonical_exclude(cls, value: list[str]) -> list[str]:
        return [canonical_concept(v) for v in value]

    @model_validator(mode="after")
    def _dedupe_and_check(self) -> SearchQuery:
        first: dict[str, Constraint] = {}
        for c in self.include:
            first.setdefault(c.concept, c)
        self.include = list(first.values())
        self.exclude = list(dict.fromkeys(self.exclude))
        self.views = list(dict.fromkeys(self.views))
        if both := sorted(first.keys() & set(self.exclude)):
            raise ValueError(f"concepts in both include and exclude: {both}")
        return self

    def constrained_concepts(self) -> list[str]:
        """Include concepts then exclude concepts, in order, each once."""
        return list(dict.fromkeys([c.concept for c in self.include] + self.exclude))


class ParseError(ValueError):
    """The LLM produced nothing usable and the caller asked for no fallback."""


@dataclass(frozen=True)
class ParseResult:
    query: SearchQuery
    parser: ParserName
    fallback_used: bool = False
    warnings: tuple[str, ...] = ()
    llm_ms: float | None = None
    cached: bool = False
