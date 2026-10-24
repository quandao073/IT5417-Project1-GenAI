"""`cxr build-image-scores` (plan Task 16): the H6 artifact the image_decomposed mode opens."""

from __future__ import annotations

import json

import polars as pl

from src.cli.main import main
from src.data.labels import CHEXPERT_LABELS
from src.embeddings import probe
from src.retrieval.image_concepts import open_image_probe
from tests.unit.fakes import FakeEncoder
from tests.unit.test_api_state import BUILD
from tests.unit.test_cli_probe import POOL_BUILD, _train_world


def _argv(tmp_path):
    train = _train_world(tmp_path)
    paths = dict(zip(train[1::2], train[2::2], strict=False))
    return ["build-image-scores", "--encoder", "fake2", "--corpus-build-id", BUILD,
            "--pool-build-id", POOL_BUILD, "--pool-dir", paths["--pool-dir"],
            "--embeddings", paths["--embeddings"], "--canonical-dir", paths["--canonical-dir"],
            "--out", str(tmp_path / "scores"), "--report", str(tmp_path / "report.json")]


def test_cli_writes_h6_artifact_with_fourteen_columns(tmp_path, monkeypatch):
    monkeypatch.setattr("src.cli.main._encoder_by_name",
                        lambda name, device: (name, FakeEncoder()))
    assert main(_argv(tmp_path)) == 0
    _, meta = probe.load_probe(tmp_path / "scores")
    assert meta["source"] == "decomposed-fake2" and meta["build_id"] == BUILD
    assert meta["encoder"] == {"name": "fake2", "model_id": "fake/model",
                               "revision": "fakerev0", "dim": 4}
    assert meta["prompts"]["Edema"] == "pulmonary edema"
    assert meta["prompts"]["No Finding"] == "no finding"
    assert set(meta["calibration"]) == set(CHEXPERT_LABELS)
    scores = pl.read_parquet(tmp_path / "scores" / probe.SCORES_FILE)
    assert scores.columns == ["study_id", "image_id", *CHEXPERT_LABELS] and scores.height == 20
    studies = pl.read_parquet(tmp_path / "canonical" / "studies.parquet")["study_id"].to_list()
    opened, _ = open_image_probe(tmp_path / "scores", build_id=BUILD, revision="fakerev0",
                                 study_ids=studies, encoder_name="fake2",
                                 command="cxr build-image-scores")
    assert opened.logits.shape == (20, 14)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["image_scores"]["source"] == "decomposed-fake2"


def test_cli_refuses_an_encoder_whose_revision_differs_from_the_build(tmp_path, monkeypatch,
                                                                      capsys):
    other = FakeEncoder()
    other.revision = "otherrev"
    monkeypatch.setattr("src.cli.main._encoder_by_name", lambda name, device: (name, other))
    assert main(_argv(tmp_path)) == 1
    assert "ENCODER MISMATCH" in capsys.readouterr().err
    assert not (tmp_path / "scores").exists()
