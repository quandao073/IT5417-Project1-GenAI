"""The search behind /api/v1/search: one path from the parsed query to ranked images.

The parsed constraints become one template (`templates.render`), the build's text encoder
reads it, and Qdrant ranks every image of the build by cosine, with only the AP/PA view facet
as a filter. Nothing here reads a label: the label filter comes back as a separate knob later.
Report sentences are quoted after the ranking, for display only, so they never move a hit;
without the report index search still answers, with a warning instead of evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from src.query.schema import SearchQuery
from src.retrieval.evidence import Evidence
from src.retrieval.image_search import search_image_embeddings
from src.retrieval.report_search import ReportSearchError, fetch_texts, free_text_tokens
from src.retrieval.templates import NO_FINDING, render

if TYPE_CHECKING:
    from src.api.dependencies import AppState

NO_CONCEPT = "no concept recognised; ranked by the free text"
NOTHING_TO_SEARCH = "no concept and no free text; nothing searched"
NO_FINDING_DROPPED = "No Finding cannot be negated in the query template; dropped"
UNCERTAIN_AS_PRESENT = "UNCERTAIN is searched as PRESENT"
ONLY_NEGATIONS = ("only negated findings ({}): the current model tends to return images "
                  "that show them")
MORE_THAN_TWO = "more than two constraints ({}) in one template"
NO_EVIDENCE = "report index unavailable; no evidence"


class VectorLegError(RuntimeError):
    """The encoder or Qdrant failed; the route answers 503."""


@dataclass(frozen=True)
class Hit:
    rank: int
    study_id: str
    image_id: str
    score: float   # cosine between the query text and the image; a ranking signal
    evidence: list[Evidence] = field(default_factory=list)


@dataclass(frozen=True)
class SearchOutcome:
    hits: list[Hit]
    text: str | None   # what the text encoder read; None when nothing was searched
    warnings: list[str]


def query_text(query: SearchQuery, concepts_cfg: dict, warnings: list[str]) -> str | None:
    """The template of the parsed constraints, else the parser's free text, else None.

    A negated No Finding is dropped (no template can say it). UNCERTAIN is searched as
    PRESENT. A concept both included and excluded never gets here: SearchQuery refuses it."""
    include = [c.concept for c in query.include]
    exclude = [c for c in query.exclude if c != NO_FINDING]
    if len(exclude) != len(query.exclude):
        warnings.append(NO_FINDING_DROPPED)
    if not include and not exclude:
        if not query.semantic_query:
            warnings.append(NOTHING_TO_SEARCH)
            return None
        warnings.append(NO_CONCEPT)
        return query.semantic_query
    if any(c.status == "UNCERTAIN" for c in query.include):
        warnings.append(UNCERTAIN_AS_PRESENT)
    if not include:
        warnings.append(ONLY_NEGATIONS.format(", ".join(exclude)))
    if len(include) + len(exclude) > 2:
        warnings.append(MORE_THAN_TWO.format(len(include) + len(exclude)))
    return render(include, exclude, concepts_cfg)


def run_search(state: AppState, query: SearchQuery, *, top_k: int,
               views: Sequence[str]) -> SearchOutcome:
    warnings: list[str] = []
    text = query_text(query, state.concepts_cfg, warnings)
    hits: list[Hit] = []
    if text is not None:
        try:
            candidates = search_image_embeddings(
                text, views or None, top_k, encoder=state.encoder, client=state.client,
                collection=state.collection, lock=state.encode_lock)
        except Exception as exc:  # noqa: BLE001 - a dead dependency is a 503, not a traceback
            raise VectorLegError(str(exc)) from exc
        hits = [Hit(c.rank, c.study_id, c.image_id, c.score) for c in candidates]
    quotes = _evidence(state, query, hits, warnings) if hits else {}
    hits = [replace(h, evidence=quotes.get(h.study_id, [])) for h in hits]
    return SearchOutcome(hits, text, warnings)


def _evidence(state: AppState, query: SearchQuery, hits: list[Hit],
              warnings: list[str]) -> dict[str, list[Evidence]]:
    """{study_id: quotes}, one quote per constrained concept, free text only when nothing is
    included. A study is quoted once even when several of its images are hits."""
    if state.report_status != "ok":
        warnings.append(NO_EVIDENCE)
        return {}
    try:
        texts = fetch_texts(state.report_db, state.report_lock,
                            list(dict.fromkeys(h.study_id for h in hits)))
    except ReportSearchError:
        warnings.append(NO_EVIDENCE)   # the report index never takes search down
        return {}
    tokens = () if query.include else free_text_tokens(
        query, min_token_chars=state.report_min_token_chars)
    return {s: state.evidence.build(t, query, free_text_tokens=tokens) for s, t in texts.items()}
