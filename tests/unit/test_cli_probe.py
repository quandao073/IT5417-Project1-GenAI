"""CLI wiring kept from the image-concept probe (build-probe-pool, embed --images) and
the checks of validate-indexes."""

from __future__ import annotations

import polars as pl
import pytest

from src.cli.main import build_parser, main
from src.embeddings import index
from src.embeddings.validate import Check
from src.retrieval.report_index import build_report_index
from tests.unit.test_api_state import BUILD, studies_frame
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


def _validate_setup(tmp_path, monkeypatch):
    canonical = _canonical(tmp_path)
    indexes = tmp_path / "indexes"
    indexes.mkdir()
    index.write_manifest(index.manifest_path(indexes, BUILD),
                         {"build_id": BUILD, "encoder": "fake"})
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
    monkeypatch.setattr("src.cli.main.load_config", lambda name: {"image_encoder": {
        "selected": "fake", "candidates": {"fake": {"revision": "fakerev0"}}}})
    argv = ["validate-indexes", "--indexes-dir", str(indexes), "--canonical-dir", str(canonical),
            "--embeddings", str(tmp_path / "emb")]
    return argv


def test_validate_indexes_prints_the_build_checks_and_the_report_index(tmp_path, capsys,
                                                                     monkeypatch):
    """Six build checks and the report index; the image probe check left with mode `image`."""
    assert main(_validate_setup(tmp_path, monkeypatch)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 7 and lines[-1].startswith("PASS report_index")
    lines[-1].encode("ascii")


def test_validate_indexes_has_no_image_probe_flag():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["validate-indexes", "--image-probe-dir", "x"])


def test_train_probe_is_gone():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train-probe"])
