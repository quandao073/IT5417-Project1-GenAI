"""`cxr build-image-scores` (TIP V2-5): the V2 score matrix, on a corpus and a pool embedded
with FakeEncoder."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from src.cli.main import main
from src.common.config import load_config
from src.data.labels import CHEXPERT_LABELS
from src.embeddings import embed, probe
from src.retrieval.templates import render
from tests.unit.fakes import FakeEncoder
from tests.unit.test_api_state import BUILD, studies_frame
from tests.unit.test_embed import images_frame

POOL_BUILD = "pool_fake_v1"


def _encoder(revision="fakerev0"):
    encoder = FakeEncoder()
    encoder.revision = revision
    return encoder


def _world(tmp_path, *, shared_patient=False, valid_path=False, pool_revision="fakerev0"):
    """A 20-study corpus and a 12-study pool, both embedded with FakeEncoder; returns the argv."""
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    corpus = images_frame(40)
    corpus.write_parquet(canonical / "images.parquet")
    studies_frame(corpus).write_parquet(canonical / "studies.parquet")

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
    first = pl.col("image_id") == pool["image_id"][0]
    if valid_path:
        pool = pool.with_columns(
            pl.when(first).then(pl.lit("valid/poolpatient00000/study1/view1_frontal.jpg"))
            .otherwise(pl.col("path_to_image")).alias("path_to_image"))
    if shared_patient:
        pool = pool.with_columns(
            pl.when(first).then(pl.lit("patient00000")).otherwise(pl.col("patient_id"))
            .alias("patient_id"))
    pool.write_parquet(pool_dir / "pool_images.parquet")
    pl.DataFrame([(s, "Edema", "PRESENT" if i % 2 else "ABSENT")
                  for i, s in enumerate(pool["study_id"])],
                 schema=["study_id", "concept", "status"], orient="row"
                 ).write_parquet(pool_dir / "pool_labels.parquet")
    (pool_dir / "pool.json").write_text(json.dumps({"pool_sha256": "p" * 64, "studies": n}))
    embed.run_embed(pool, _encoder(pool_revision), Path("root"),
                    pool_dir / "embeddings" / POOL_BUILD, POOL_BUILD, shard_size=5, batch_size=4,
                    library_versions={}, git_commit=None)

    embeddings = tmp_path / "embeddings"
    embed.run_embed(corpus, FakeEncoder(), Path("root"), embeddings / BUILD, BUILD,
                    shard_size=16, batch_size=4, library_versions={}, git_commit=None)
    return ["build-image-scores", "--encoder", "fake2", "--corpus-build-id", BUILD,
            "--pool-build-id", POOL_BUILD, "--pool-dir", str(pool_dir),
            "--embeddings", str(embeddings), "--canonical-dir", str(canonical),
            "--out", str(tmp_path / "scores"), "--report", str(tmp_path / "report.json")]


def _serve(monkeypatch, encoder):
    """The models.yaml candidate the build's encoder gate (index.build_encoder) loads."""
    monkeypatch.setattr("src.embeddings.index._load_candidate",
                        lambda name, candidate, device: encoder)


def test_cli_writes_the_matrix_with_fourteen_label_columns(tmp_path, monkeypatch):
    _serve(monkeypatch, _encoder())
    assert main(_world(tmp_path)) == 0
    loaded, meta = probe.load_probe(tmp_path / "scores")
    assert meta["build_id"] == BUILD and meta["pool_build_id"] == POOL_BUILD
    assert meta["encoder"] == {"name": "fake2", "model_id": "fake/model",
                               "revision": "fakerev0", "dim": 4}
    concepts = load_config("concepts")
    assert meta["prompts"] == {c: render([c], [], concepts) for c in CHEXPERT_LABELS}
    assert set(meta["calibration"]) == set(CHEXPERT_LABELS)
    assert loaded.W.shape == (14, 4)
    scores = pl.read_parquet(tmp_path / "scores" / probe.SCORES_FILE)
    studies = pl.read_parquet(tmp_path / "canonical" / "studies.parquet")
    assert scores.columns == ["study_id", "image_id", *CHEXPERT_LABELS]
    assert scores["study_id"].to_list() == studies["study_id"].to_list()
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["image_scores"]["build_id"] == BUILD


def test_cli_refuses_an_encoder_whose_revision_differs_from_the_build(tmp_path, monkeypatch,
                                                                      capsys):
    _serve(monkeypatch, _encoder("otherrev"))
    assert main(_world(tmp_path)) == 1
    assert "ENCODER MISMATCH" in capsys.readouterr().err
    assert not (tmp_path / "scores").exists()


@pytest.mark.parametrize("world, message", [
    ({"shared_patient": True}, "POOL OVERLAPS CORPUS: 1 patients"),
    ({"valid_path": True}, "POOL CONTAINS valid/ STUDIES"),
    ({"pool_revision": "otherrev"}, "POOL BUILD MISMATCH"),
])
def test_cli_refuses_a_pool_it_must_not_calibrate_on(tmp_path, monkeypatch, capsys, world,
                                                     message):
    _serve(monkeypatch, _encoder())
    assert main(_world(tmp_path, **world)) == 1
    assert message in capsys.readouterr().err
    assert not (tmp_path / "scores").exists() and not (tmp_path / "report.json").exists()
