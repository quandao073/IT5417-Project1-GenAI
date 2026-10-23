"""CLI wiring for the image-concept probe: build-probe-pool and embed --images."""

from __future__ import annotations

import json

import polars as pl

from src.cli.main import build_parser, main
from src.embeddings.validate import Check
from src.retrieval.report_index import build_report_index
from tests.unit.test_api_state import BUILD, studies_frame, write_probe_artifact
from tests.unit.test_cli_phase5 import _canonical
from tests.unit.test_embed import images_frame


def test_embed_accepts_images_override():
    args = build_parser().parse_args(["embed", "--images", "x.parquet"])
    assert args.images == "x.parquet"


def test_embed_default_images_is_none():
    assert build_parser().parse_args(["embed"]).images is None


def test_build_probe_pool_defaults_do_not_touch_main_paths():
    args = build_parser().parse_args(["build-probe-pool"])
    assert args.out == "data/probe_pool"
    assert args.canonical_dir == "data/canonical"
    assert args.plus_csv == "data/raw/chexpert-plus/df_chexpert_plus_240401.csv"
    assert args.small_root == "data/raw/CheXpert-v1.0-small"
    assert args.chexbert_dir == "data/raw/chexpert-plus/labels"


def test_bench_vector_accepts_image_mode():
    assert build_parser().parse_args(["bench-vector", "--mode", "image"]).mode == "image"


def _eight_check_setup(tmp_path, monkeypatch):
    canonical = _canonical(tmp_path)
    indexes = tmp_path / "indexes"
    indexes.mkdir()
    (indexes / "build_manifest.json").write_text(json.dumps({"build_id": BUILD}), encoding="utf-8")
    images = images_frame(8)
    images.write_parquet(canonical / "images.parquet")
    studies = studies_frame(images)
    studies.write_parquet(canonical / "studies.parquet")
    build_report_index(pl.read_parquet(canonical / "reports.parquet"),
                       indexes / "report_search.sqlite", concepts_sha256="c" * 64,
                       git_commit=None)
    six = [Check(f"check{i}", True, "ok") for i in range(6)]
    monkeypatch.setattr("src.cli.main.validate.validate_indexes", lambda *a, **k: six)
    monkeypatch.setattr("src.cli.main.validate.expected_config_hash", lambda *a, **k: "x")
    monkeypatch.setattr("src.cli.main._qdrant_client", lambda *a, **k: None)
    monkeypatch.setattr("src.cli.main.load_config",
                        lambda name: {"image_encoder": {"revision": "fakerev0"}})
    argv = ["validate-indexes", "--indexes-dir", str(indexes), "--canonical-dir", str(canonical),
            "--embeddings", str(tmp_path / "emb")]
    return argv, indexes, studies


def test_validate_indexes_prints_eight_checks(tmp_path, capsys, monkeypatch):
    argv, _, _ = _eight_check_setup(tmp_path, monkeypatch)
    assert main(argv) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 8 and lines[-2].startswith("PASS report_index")
    assert lines[-1].startswith("FAIL image_probe")
    lines[-1].encode("ascii")


def test_validate_indexes_passes_the_image_probe_check(tmp_path, capsys, monkeypatch):
    argv, indexes, studies = _eight_check_setup(tmp_path, monkeypatch)
    write_probe_artifact(indexes / "image_probe", studies)
    assert main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 8 and lines[-1].startswith("PASS image_probe")


def test_validate_indexes_takes_an_image_probe_dir(tmp_path, capsys, monkeypatch):
    argv, _, studies = _eight_check_setup(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    write_probe_artifact(elsewhere, studies)
    assert main([*argv, "--image-probe-dir", str(elsewhere)]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("PASS image_probe")
