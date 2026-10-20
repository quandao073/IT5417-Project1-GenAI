"""SQLite FTS5 index over the canonical reports (Phase 5 report leg).

One file, three tables: `reports` (verbatim search_text, so evidence offsets match
reports.parquet), `reports_fts` (the full-text index) and `meta` (provenance). The index is
tied to the canonical corpus, not to an embedding build: reports do not depend on the encoder.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.data import contracts
from src.embeddings.validate import Check

TOKENIZER = "porter unicode61"
SCHEMA_VERSION = "1"
META_KEYS = ("studies", "corpus_sha256", "tokenizer", "sqlite_version", "concepts_sha256",
             "git_commit", "built_at", "schema_version")
BATCH = 1_000

_SCHEMA = f"""
CREATE TABLE reports (study_id TEXT PRIMARY KEY, search_text TEXT NOT NULL,
                      report_checksum TEXT NOT NULL);
CREATE VIRTUAL TABLE reports_fts USING fts5(study_id UNINDEXED, text,
                                            tokenize = '{TOKENIZER}');
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class ReportIndexError(RuntimeError):
    """The report index is missing, unreadable or of an unknown schema version."""


def corpus_sha256(reports: pl.DataFrame) -> str:
    """Order-independent fingerprint of (study_id, report_checksum) over the whole corpus."""
    pairs = sorted(zip(reports["study_id"].to_list(), reports["report_checksum"].to_list(),
                       strict=True))
    return hashlib.sha256("".join(f"{sid}\t{chk}\n" for sid, chk in pairs).encode()).hexdigest()


def _batches(reports: pl.DataFrame):
    rows = list(zip(reports["study_id"].to_list(), reports["search_text"].to_list(),
                    reports["report_checksum"].to_list(), strict=True))
    for start in range(0, len(rows), BATCH):
        yield rows[start:start + BATCH]


def build_report_index(reports: pl.DataFrame, path: Path, *, concepts_sha256: str,
                       git_commit: str | None) -> dict:
    """Write the index to `<path>.part`, then atomically replace `path`. Returns `meta`."""
    contracts.validate(reports, "reports")
    meta = {
        "studies": str(reports.height),
        "corpus_sha256": corpus_sha256(reports),
        "tokenizer": TOKENIZER,
        "sqlite_version": sqlite3.sqlite_version,
        "concepts_sha256": concepts_sha256,
        "git_commit": git_commit or "",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.unlink(missing_ok=True)
    con = sqlite3.connect(part)
    try:
        with con:  # one transaction: commit on success, rollback on error
            con.executescript(_SCHEMA)
            for batch in _batches(reports):
                con.executemany("INSERT INTO reports VALUES (?, ?, ?)", batch)
                con.executemany("INSERT INTO reports_fts (study_id, text) VALUES (?, ?)",
                                [(sid, text) for sid, text, _ in batch])
            con.executemany("INSERT INTO meta VALUES (?, ?)", list(meta.items()))
    finally:
        con.close()
    os.replace(part, path)
    return meta


def read_meta(con: sqlite3.Connection) -> dict:
    return dict(con.execute("SELECT key, value FROM meta"))


def open_report_index(path: Path) -> sqlite3.Connection:
    """Open read-only. The caller serialises use of the shared connection with a lock."""
    if not path.is_file():
        raise ReportIndexError(f"{path} not found; run cxr build-report-index")
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    try:
        version = read_meta(con).get("schema_version")
    except sqlite3.DatabaseError as exc:
        con.close()
        raise ReportIndexError(f"{path} is unreadable: {exc}") from exc
    if version != SCHEMA_VERSION:
        con.close()
        raise ReportIndexError(
            f"{path} has schema_version {version!r}, expected {SCHEMA_VERSION!r}; "
            "run cxr build-report-index")
    return con


def _first_failure(con: sqlite3.Connection, reports: pl.DataFrame) -> str | None:
    meta = read_meta(con)
    n_rows = con.execute("SELECT count(*) FROM reports").fetchone()[0]
    if not meta.get("studies") == str(n_rows) == str(reports.height):
        return f"studies: meta {meta.get('studies')}, rows {n_rows}, parquet {reports.height}"
    indexed = {sid for (sid,) in con.execute("SELECT study_id FROM reports")}
    if indexed != set(reports["study_id"].to_list()):
        return "study_id set differs from reports.parquet"
    expected = corpus_sha256(reports)
    if meta.get("corpus_sha256") != expected:
        return (f"corpus_sha256: index {str(meta.get('corpus_sha256'))[:12]} "
                f"vs parquet {expected[:12]}; run cxr build-report-index")
    return None


def validate_report_index(path: Path, reports: pl.DataFrame) -> Check:
    """The seventh `cxr validate-indexes` check; an exception is a FAIL line, not a traceback."""
    try:
        con = open_report_index(path)
        try:
            failure = _first_failure(con, reports)
        finally:
            con.close()
        if failure:
            return Check("report_index", False, failure)
        return Check("report_index", True, f"{reports.height} studies, corpus_sha256 matches")
    except Exception as exc:
        return Check("report_index", False, f"{type(exc).__name__}: {exc}")
