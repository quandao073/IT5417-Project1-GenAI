"""CLI wiring for Phase 5: build-report-index and the seventh validate-indexes check."""

from __future__ import annotations

import json

import polars as pl

from src.cli.main import main
from src.embeddings.validate import Check
from src.retrieval.report_index import build_report_index, open_report_index, read_meta
from tests.unit.test_embed import images_frame
from tests.unit.test_report_index import ROWS, reports_frame


def _canonical(tmp_path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    reports_frame(ROWS).write_parquet(canonical / "reports.parquet")
    return canonical


def test_build_report_index_writes_the_file_and_prints_ascii(tmp_path, capsys):
    canonical = _canonical(tmp_path)
    out = tmp_path / "idx" / "report_search.sqlite"
    assert main(["build-report-index", "--canonical-dir", str(canonical),
                 "--out", str(out)]) == 0
    assert out.is_file() and read_meta(open_report_index(out))["studies"] == "4"
    text = capsys.readouterr().out
    text.encode("ascii")
    assert "rows=4 sha=" in text


def test_validate_indexes_prints_seven_checks(tmp_path, capsys, monkeypatch):
    canonical = _canonical(tmp_path)
    indexes = tmp_path / "indexes"
    indexes.mkdir()
    (indexes / "build_manifest.json").write_text(json.dumps({"build_id": "b"}), encoding="utf-8")
    images_frame(8).write_parquet(canonical / "images.parquet")
    build_report_index(pl.read_parquet(canonical / "reports.parquet"),
                       indexes / "report_search.sqlite", concepts_sha256="c" * 64,
                       git_commit=None)
    six = [Check(f"check{i}", True, "ok") for i in range(6)]
    monkeypatch.setattr("src.cli.main.validate.validate_indexes", lambda *a, **k: six)
    monkeypatch.setattr("src.cli.main.validate.expected_config_hash", lambda *a, **k: "x")
    monkeypatch.setattr("src.cli.main._qdrant_client", lambda *a, **k: None)
    assert main(["validate-indexes", "--indexes-dir", str(indexes),
                 "--canonical-dir", str(canonical), "--embeddings", str(tmp_path / "emb")]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 7 and lines[-1].startswith("PASS report_index")
