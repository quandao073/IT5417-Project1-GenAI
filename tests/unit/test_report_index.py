"""The SQLite FTS5 report index: build, open read-only, and validate against the parquet."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence

import polars as pl
import pytest

from src.retrieval.report_index import (
    ReportIndexError,
    build_report_index,
    corpus_sha256,
    open_report_index,
    read_meta,
    validate_report_index,
)

ROWS = [("patient00000/study1", "Mild pulmonary edema.  No \npleural effusion."),
        ("patient00001/study1", "Moderate pulmonary edema and a small right pleural effusion."),
        ("patient00002/study1", "Possible mild edema.  1. Lines and tubes unchanged."),
        ("patient00003/study1", "Stable cardiomegaly.  No acute cardiopulmonary abnormality.")]


def reports_frame(rows: Sequence[tuple[str, str]]) -> pl.DataFrame:
    """rows of (study_id, search_text); the checksum is the sha256 canonical.py computes."""
    return pl.DataFrame(
        [(sid, sid.split("/")[0], None, text, text, i,
          hashlib.sha256(text.encode("utf-8")).hexdigest(), "t")
         for i, (sid, text) in enumerate(rows)],
        schema={"study_id": pl.Utf8, "patient_id": pl.Utf8, "findings": pl.Utf8,
                "impression": pl.Utf8, "search_text": pl.Utf8, "source_row_index": pl.Int64,
                "report_checksum": pl.Utf8, "dataset_version": pl.Utf8},
        orient="row",
    )


def _build(tmp_path, rows=ROWS):
    path = tmp_path / "r.sqlite"
    meta = build_report_index(reports_frame(rows), path, concepts_sha256="c" * 64,
                              git_commit="abc")
    return path, meta


def test_build_writes_meta_and_rows(tmp_path):
    path, meta = _build(tmp_path)
    assert meta["studies"] == "4" and meta["tokenizer"] == "porter unicode61"
    assert meta["schema_version"] == "1"
    assert meta["corpus_sha256"] == corpus_sha256(reports_frame(ROWS))
    assert meta["concepts_sha256"] == "c" * 64 and meta["git_commit"] == "abc"
    assert not (tmp_path / "r.sqlite.part").exists()
    con = open_report_index(path)
    assert read_meta(con) == meta
    assert con.execute("select count(*) from reports").fetchone()[0] == 4
    hits = con.execute("select study_id from reports_fts "
                       "where reports_fts match '\"pleural effusion\"' order by study_id")
    assert hits.fetchall() == [("patient00000/study1",), ("patient00001/study1",)]


def test_build_without_git_commit_stores_empty_string(tmp_path):
    meta = build_report_index(reports_frame(ROWS), tmp_path / "r.sqlite",
                              concepts_sha256="c" * 64, git_commit=None)
    assert meta["git_commit"] == ""


def test_search_text_is_stored_verbatim(tmp_path):
    path, _ = _build(tmp_path)
    got = open_report_index(path).execute(
        "select search_text from reports where study_id = 'patient00000/study1'").fetchone()[0]
    assert got == "Mild pulmonary edema.  No \npleural effusion."


def test_build_replaces_an_existing_file(tmp_path):
    _build(tmp_path)
    path, meta = _build(tmp_path, ROWS[:2])
    assert meta["studies"] == "2"
    assert open_report_index(path).execute("select count(*) from reports").fetchone()[0] == 2


def test_open_is_read_only(tmp_path):
    path, _ = _build(tmp_path)
    con = open_report_index(path)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("insert into meta values ('x','y')")


def test_open_rejects_missing_file_and_wrong_schema(tmp_path):
    with pytest.raises(ReportIndexError, match="not found"):
        open_report_index(tmp_path / "nope.sqlite")
    path, _ = _build(tmp_path)
    with sqlite3.connect(path) as con:
        con.execute("update meta set value = '0' where key = 'schema_version'")
    with pytest.raises(ReportIndexError, match="schema_version"):
        open_report_index(path)


def test_validate_passes_then_catches_count_and_checksum_drift(tmp_path):
    path, _ = _build(tmp_path)
    frame = reports_frame(ROWS)
    assert validate_report_index(path, frame).ok
    short = validate_report_index(path, frame.head(3))
    assert not short.ok and "studies" in short.detail
    drifted = frame.with_columns(pl.lit("0" * 64).alias("report_checksum"))
    check = validate_report_index(path, drifted)
    assert not check.ok and "corpus_sha256" in check.detail
    missing = validate_report_index(tmp_path / "nope.sqlite", frame)
    assert not missing.ok and "run cxr build-report-index" in missing.detail


def test_validate_catches_a_different_study_id_set(tmp_path):
    path, _ = _build(tmp_path)
    swapped = reports_frame([("patient09999/study1", ROWS[0][1]), *ROWS[1:]])
    check = validate_report_index(path, swapped)
    assert not check.ok and "study_id set" in check.detail


def test_corpus_sha256_is_order_independent():
    assert corpus_sha256(reports_frame(ROWS)) == corpus_sha256(reports_frame(ROWS[::-1]))
