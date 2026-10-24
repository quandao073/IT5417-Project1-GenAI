"""What evidence needs from the report index: the verbatim text and the free-text tokens.

The report index no longer ranks anything (the FTS leg left with hybrid mode). `fetch_texts`
returns the verbatim `search_text` the evidence matcher quotes from, and `free_text_tokens`
the words it looks for when a query names no concept.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence

from src.query.rule_parser import tokenize
from src.query.schema import SearchQuery

_CLAUSE_TOKENS = frozenset({",", ";", "."})
# They carry no information as words to look for: "and" matches nearly every report.
_FTS_KEYWORDS = frozenset({"and", "or", "not", "near"})


class ReportSearchError(RuntimeError):
    """The report index could not be read (unreadable or closed); callers degrade."""


def free_text_tokens(query: SearchQuery, *, min_token_chars: int = 3) -> list[str]:
    """The tokens of `semantic_query` evidence looks for: casefolded, in order, deduplicated."""
    tokens = (t.replace('"', "") for t in tokenize(query.semantic_query)
              if t not in _CLAUSE_TOKENS)
    return list(dict.fromkeys(
        t for t in tokens if len(t) >= min_token_chars and t not in _FTS_KEYWORDS))


def fetch_texts(con: sqlite3.Connection, lock: threading.Lock,
                study_ids: Sequence[str]) -> dict[str, str]:
    """Verbatim search_text for the ids that exist; unknown ids are skipped."""
    if not study_ids:
        return {}
    marks = ",".join("?" * len(study_ids))
    try:
        with lock:
            rows = con.execute(
                f"SELECT study_id, search_text FROM reports WHERE study_id IN ({marks})",
                list(study_ids)).fetchall()
    except sqlite3.Error as exc:
        raise ReportSearchError(str(exc)) from exc
    return dict(rows)
