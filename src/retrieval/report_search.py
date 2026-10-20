"""Report leg of retrieval (Phase 5): FTS5 bm25 over the canonical reports.

`build_match` turns a SearchQuery into an FTS5 MATCH string in which every term is a quoted
phrase joined by OR, so no FTS syntax from user text can reach the engine. `search_reports`
ranks inside the label-filtered set; it is a ranking signal for hybrid mode, not a filter.
`fetch_texts` returns the verbatim `search_text` the evidence matcher needs.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.query.rule_parser import tokenize
from src.query.schema import SearchQuery

_CLAUSE_TOKENS = frozenset({",", ";", "."})
# Quoting already makes these inert; they are dropped because as ranking terms they carry
# no information ("and" matches nearly every report) and "AND OR *" must count as no term.
_FTS_KEYWORDS = frozenset({"and", "or", "not", "near"})


class ReportSearchError(RuntimeError):
    """The FTS5 query failed (malformed MATCH, unreadable index); callers degrade."""


@dataclass(frozen=True)
class ReportCandidate:
    study_id: str
    score: float   # -bm25(reports_fts): higher is better; a ranking signal, not a probability
    rank: int      # 1-based


def report_terms_from_config(concepts_cfg: dict) -> dict[str, list[str]]:
    """{concept: its English report terms} in YAML order; `en`/`vi` are not used here."""
    return {concept: list(forms["report"]) for concept, forms in concepts_cfg["concepts"].items()}


def free_text_tokens(query: SearchQuery, *, min_token_chars: int = 3) -> list[str]:
    """The tokens the free-text branch quotes: casefolded, in order, deduplicated."""
    tokens = (t.replace('"', "") for t in tokenize(query.semantic_query)
              if t not in _CLAUSE_TOKENS)
    return list(dict.fromkeys(
        t for t in tokens if len(t) >= min_token_chars and t not in _FTS_KEYWORDS))


def _quoted(terms: Sequence[str]) -> str | None:
    return " OR ".join(f'"{t}"' for t in terms) if terms else None


def build_match(query: SearchQuery, report_terms: Mapping[str, Sequence[str]], *,
                min_token_chars: int = 3) -> str | None:
    """OR of quoted phrases. None means there is nothing English to match.

    With an include, only the include concepts' curated report terms enter (they are written
    the way reports phrase findings); the `exclude` concepts' terms do not, because the label
    filter owns them. Without an include, the free-text branch quotes every token of
    `semantic_query` (`free_text_tokens`), the words of an excluded finding too: for an English
    exclude-only query that is the raw text (WORKLOG D26), so "show me films with no evidence
    of pneumothorax" becomes `"show" OR "films" OR "with" OR "evidence" OR "pneumothorax"` and
    ranks first, inside the label-allowed set, the reports that mention the excluded finding,
    often to deny it. Kept on purpose for now (WORKLOG P11).
    """
    if query.include:
        terms = [term.replace('"', "") for c in query.include for term in report_terms[c.concept]]
        return _quoted(terms)
    return _quoted(free_text_tokens(query, min_token_chars=min_token_chars))


def search_reports(con: sqlite3.Connection, lock: threading.Lock, match: str,
                   allowed: Sequence[str] | None, candidate_k: int) -> list[ReportCandidate]:
    if allowed is not None and not allowed:
        return []
    sql = "SELECT study_id, bm25(reports_fts) FROM reports_fts WHERE reports_fts MATCH ?"
    params: list = [match]
    if allowed is not None:
        # json_each keeps the read-only connection stateless (no TEMP table to clear) and
        # needs no 999-variable batching for up to the 10,000 ids the label filter yields.
        sql += " AND study_id IN (SELECT value FROM json_each(?))"
        params.append(json.dumps(list(allowed)))
    sql += " ORDER BY bm25(reports_fts), study_id LIMIT ?"
    params.append(candidate_k)
    try:
        with lock:
            rows = con.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        raise ReportSearchError(str(exc)) from exc
    return [ReportCandidate(study_id=sid, score=-float(bm25), rank=rank)
            for rank, (sid, bm25) in enumerate(rows, start=1)]


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
