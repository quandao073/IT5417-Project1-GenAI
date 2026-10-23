"""CLI wiring for the image-concept probe: build-probe-pool and embed --images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.cli.main import build_parser, main
from src.data.labels import CHEXPERT_LABELS
from src.embeddings import embed, probe
from src.embeddings.validate import Check
from src.retrieval.report_index import build_report_index
from tests.unit.fakes import FakeEncoder
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


def test_eval_retrieval_accepts_image_mode_and_ablation_flag():
    parse = build_parser().parse_args
    assert parse(["eval-retrieval", "--modes", "image"]).modes == ["image"]
    assert parse(["eval-retrieval"]).modes == ["label", "vector", "hybrid"]
    assert parse(["eval-retrieval"]).ablation is False
    assert parse(["eval-retrieval", "--ablation"]).ablation is True
    with pytest.raises(SystemExit):
        parse(["eval-retrieval", "--grid", "--ablation"])


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


def test_train_probe_defaults_use_new_paths_only():
    args = build_parser().parse_args(["train-probe"])
    assert args.out == "data/indexes/image_probe"
    assert args.report == "artifacts/metrics/image_probe.json"
    assert args.pool_dir == "data/probe_pool"
    assert args.pool_build_id == "20261023_pool88k_biomedclip_v1"
    assert args.embeddings == "data/indexes/embeddings"
    assert args.indexes_dir == "data/indexes" and args.canonical_dir == "data/canonical"
    assert args.n_boot == 1000


POOL_BUILD = "pool_fake_v1"


def _train_world(tmp_path, *, shared_patient=False, valid_path=False, revision="fakerev0"):
    """A 20-study corpus, a 12-study pool and a manifest, all embedded with FakeEncoder."""
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    corpus = images_frame(40)
    corpus.write_parquet(canonical / "images.parquet")
    studies_frame(corpus).write_parquet(canonical / "studies.parquet")
    rows = [(s, "Edema", "PRESENT" if i % 2 else "ABSENT", "CHEXPERT_V1")
            for i, s in enumerate(studies_frame(corpus)["study_id"])]
    pl.DataFrame(rows, schema=["study_id", "concept", "status", "source"], orient="row"
                 ).write_parquet(canonical / "labels.parquet")

    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    n = 12
    pool = pl.DataFrame({
        "image_id": [f"{1000 + i:064x}" for i in range(n)],
        "study_id": [f"poolpatient{i:05d}/study1" for i in range(n)],
        "patient_id": [f"poolpatient{i:05d}" for i in range(n)],
        "path_to_image": [f"train/poolpatient{i:05d}/study1/view1_frontal.jpg" for i in range(n)],
        "projection": ["AP"] * n,
    })
    if valid_path:
        pool = pool.with_columns(
            pl.when(pl.col("image_id") == pool["image_id"][0])
            .then(pl.lit("valid/poolpatient00000/study1/view1_frontal.jpg"))
            .otherwise(pl.col("path_to_image")).alias("path_to_image"))
    if shared_patient:
        pool = pool.with_columns(
            pl.when(pl.col("image_id") == pool["image_id"][0]).then(pl.lit("patient00000"))
            .otherwise(pl.col("patient_id")).alias("patient_id"))
    pool.write_parquet(pool_dir / "pool_images.parquet")
    pl.DataFrame([(s, "Edema", "PRESENT" if i % 2 else "ABSENT")
                  for i, s in enumerate(pool["study_id"])],
                 schema=["study_id", "concept", "status"], orient="row"
                 ).write_parquet(pool_dir / "pool_labels.parquet")
    (pool_dir / "pool.json").write_text(json.dumps({"pool_sha256": "p" * 64, "studies": n}))
    embed.run_embed(pool, FakeEncoder(), Path("root"), pool_dir / "embeddings" / POOL_BUILD,
                    POOL_BUILD, shard_size=5, batch_size=4, library_versions={}, git_commit=None)

    embeddings = tmp_path / "embeddings"
    embed.run_embed(corpus, FakeEncoder(), Path("root"), embeddings / BUILD, BUILD,
                    shard_size=16, batch_size=4, library_versions={}, git_commit=None)
    indexes = tmp_path / "indexes"
    indexes.mkdir()
    (indexes / "build_manifest.json").write_text(json.dumps({
        "build_id": BUILD,
        "embedding_model": {"model_id": "fake/model", "revision": revision, "dimension": 4}}))
    return ["train-probe", "--pool-dir", str(pool_dir), "--pool-build-id", POOL_BUILD,
            "--embeddings", str(embeddings), "--indexes-dir", str(indexes),
            "--canonical-dir", str(canonical), "--out", str(tmp_path / "probe"),
            "--report", str(tmp_path / "report.json"), "--n-boot", "20"]


def test_train_probe_refuses_a_pool_that_overlaps_the_corpus(tmp_path, capsys):
    argv = _train_world(tmp_path, shared_patient=True)
    assert main(argv) == 1
    assert "POOL OVERLAPS CORPUS: 1 patients" in capsys.readouterr().err
    assert not (tmp_path / "probe").exists() and not (tmp_path / "report.json").exists()


def test_train_probe_refuses_a_pool_embedded_by_another_revision(tmp_path, capsys):
    argv = _train_world(tmp_path, revision="otherrev")
    assert main(argv) == 1
    assert "POOL BUILD MISMATCH" in capsys.readouterr().err
    assert not (tmp_path / "probe").exists() and not (tmp_path / "report.json").exists()


def test_train_probe_refuses_a_pool_with_a_valid_study(tmp_path, capsys):
    argv = _train_world(tmp_path, valid_path=True)
    assert main(argv) == 1
    assert "valid/" in capsys.readouterr().err
    assert not (tmp_path / "probe").exists() and not (tmp_path / "report.json").exists()


def test_train_probe_writes_the_artifact_and_merges_the_report(tmp_path, capsys, monkeypatch):
    argv = _train_world(tmp_path)
    seen = {}

    def fake_train(X, statuses, patients, cfg):
        seen.update(n=len(X), patients=list(patients), statuses=statuses)
        zero = probe.Probe(np.zeros((14, 4), np.float32), np.zeros(14, np.float32),
                           tuple(CHEXPERT_LABELS))
        return zero, {"lambda": 10, "selection": {"per_lambda": {"10": 0.7}}, "n_train": {}}

    monkeypatch.setattr("src.cli.main.train_probe", fake_train)
    monkeypatch.setattr("src.cli.main._selected_encoder", lambda device: ("fake", FakeEncoder()))
    (tmp_path / "report.json").write_text('{"quality": {"keep": 1}}')

    assert main(argv) == 0

    # rows follow the shard order (by image_id), patients and statuses aligned to it
    assert seen["n"] == 12 and seen["patients"][0] == "poolpatient00000"
    assert seen["statuses"][1, CHEXPERT_LABELS.index("Edema")] == "PRESENT"
    loaded, meta = probe.load_probe(tmp_path / "probe")
    assert meta["probe_id"] == f"{BUILD}_probe_lam10" and meta["build_id"] == BUILD
    assert meta["encoder"] == {"model_id": "fake/model", "revision": "fakerev0", "dim": 4}
    assert meta["pool_sha256"] == "p" * 64 and meta["pool_build_id"] == POOL_BUILD
    scores = pl.read_parquet(tmp_path / "probe" / probe.SCORES_FILE)
    assert scores.height == 20 and scores.columns == ["study_id", "image_id", *CHEXPERT_LABELS]
    assert scores["image_id"].to_list() == pl.read_parquet(
        tmp_path / "canonical" / "studies.parquet")["selected_image_id"].to_list()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["quality"] == {"keep": 1}
    assert set(report["probe"]["comparison"]) >= {"zero_shot", "ensemble", "contrast", "probe",
                                                  "probe_minus_zero_shot", "s1_passed"}
    assert report["probe"]["pool"]["studies"] == 12
    out = capsys.readouterr().out
    out.encode("ascii")
    assert "lambda" in out and "probe - zero_shot" in out and "S1 " in out
    assert loaded.W.shape == (14, 4)
