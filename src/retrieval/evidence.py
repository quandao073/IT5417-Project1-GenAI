"""Exact-sentence evidence for a search result (Phase 5, spec 7).

`sentences` cuts a report into spans that are verbatim slices of it (`text[start:end] ==
span.text`). `EvidenceMatcher` reuses the rule parser, built over `en + report` terms, to give
each sentence a polarity per concept, then picks one sentence per constrained concept (the first
with the wanted polarity, else the first that mentions it). Nothing is ever composed: a concept
no sentence mentions yields no evidence.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from src.query.rule_parser import RuleParser, tokenize
from src.query.schema import SearchQuery

Polarity = Literal["affirmed", "uncertain", "negated", "mention"]

# Split after . ! ? only when whitespace follows, so "2.5 cm" stays whole. Never at "\n":
# reports hard-wrap lines in the middle of a sentence.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_ENUMERATOR = re.compile(r"\d{1,2}\.")   # "1." list numbering (T9) is not a sentence


class EvidenceError(ValueError):
    """A span is not an exact slice of the report it claims to quote."""


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str


class Evidence(BaseModel):
    concept: str | None   # None = free-text match
    text: str             # exact slice of the report's search_text
    start: int
    end: int
    polarity: Polarity


def _trimmed(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) past surrounding whitespace by moving offsets, not the string."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def sentences(text: str) -> list[Span]:
    """Sentence spans in report order; each one is an exact slice of `text`."""
    cuts = [(m.start(), m.end()) for m in _BOUNDARY.finditer(text)]
    starts = [0] + [end for _, end in cuts]
    ends = [start for start, _ in cuts] + [len(text)]
    spans: list[Span] = []
    pending: int | None = None   # start of an enumerator waiting for its sentence
    for raw_start, raw_end in zip(starts, ends, strict=True):
        start, end = _trimmed(text, raw_start, raw_end)
        if start == end:
            continue
        if _ENUMERATOR.fullmatch(text[start:end]):
            pending = start if pending is None else pending
            continue
        if pending is not None:
            start, pending = pending, None
        spans.append(Span(start, end, text[start:end]))
    return spans


def wanted_polarity(query: SearchQuery) -> dict[str, str]:
    """What the query asks of each constrained concept: include PRESENT -> affirmed,
    include UNCERTAIN -> uncertain, exclude -> negated."""
    wanted = {c.concept: "affirmed" if c.status == "PRESENT" else "uncertain"
              for c in query.include}
    wanted.update({concept: "negated" for concept in query.exclude})
    return wanted


class EvidenceMatcher:
    def __init__(self, concepts_cfg: dict):
        cfg = copy.deepcopy(concepts_cfg)
        for forms in cfg["concepts"].values():
            forms["en"] = forms["en"] + [t for t in forms.get("report", []) if t not in forms["en"]]
        self._parser = RuleParser(cfg)

    def _polarities(self, sentence: str) -> dict[str, str]:
        """{concept: polarity} for the concepts this one sentence mentions."""
        parsed = self._parser.parse(sentence)
        found = {c.concept: "affirmed" if c.status == "PRESENT" else "uncertain"
                 for c in parsed.include}
        found.update({concept: "negated" for concept in parsed.exclude})
        return found

    def build(self, text: str, query: SearchQuery, *, free_text_tokens: Sequence[str] = (),
              verify: bool = True) -> list[Evidence]:
        spans = sentences(text)
        if verify:
            # Faithfulness guard (spec 7.1): every quoted span must be a real slice.
            for span in spans:
                if text[span.start:span.end] != span.text:
                    raise EvidenceError(f"span [{span.start}:{span.end}] is not in the report")
        found = [self._polarities(span.text) for span in spans]
        wanted = wanted_polarity(query)
        out: list[Evidence] = []
        for concept in query.constrained_concepts():
            mentions = [(span, f[concept]) for span, f in zip(spans, found, strict=True)
                        if concept in f]
            preferred = [m for m in mentions if m[1] == wanted[concept]]
            if chosen := next(iter(preferred or mentions), None):
                span, polarity = chosen
                out.append(Evidence(concept=concept, text=span.text, start=span.start,
                                    end=span.end, polarity=polarity))
        best = self._best_mention(spans, free_text_tokens) if not query.include else None
        if best:
            out.append(Evidence(concept=None, text=best.text, start=best.start, end=best.end,
                                polarity="mention"))
        return out

    @staticmethod
    def _best_mention(spans: Sequence[Span], free_text_tokens: Sequence[str]) -> Span | None:
        """The sentence matching the most distinct tokens; the first one on a tie."""
        wanted = {t.casefold() for t in free_text_tokens}
        best, best_count = None, 0
        for span in spans:
            count = len(wanted & set(tokenize(span.text)))
            if count > best_count:
                best, best_count = span, count
        return best
