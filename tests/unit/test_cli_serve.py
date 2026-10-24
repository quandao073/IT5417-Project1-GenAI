"""CLI wiring of WP5: `index --role image`, the image checks of `validate-indexes`."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from qdrant_client import QdrantClient

from src.cli.main import build_parser, main
from src.embeddings import embed, index
from src.embeddings.validate import Check
from tests.unit.fakes import FakeEncoder
from tests.unit.test_api_state import BUILD, write_probe_artifact
from tests.unit.test_cli_probe import _eight_check_setup
from tests.unit.test_embed import images_frame

IMAGE_BUILD = "20261023_chexpert25k_fake2_v1"


def _names(lines):
    """Check names of validate-indexes lines; a 12-character name touches its colon."""
    return [re.match(r"(?:PASS|FAIL) (\S+?)\s*: ", line).group(1) for line in lines]


def test_index_role_defaults_to_vector():
    assert build_parser().parse_args(["index"]).role == "vector"
    assert build_parser().parse_args(["index", "--role", "image"]).role == "image"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["index", "--role", "graph"])


def _shards(tmp_path, build_id):
    images = images_frame(8)
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    images.write_parquet(canonical / "images.parquet")
    embed.run_embed(images, FakeEncoder(), Path("root"), tmp_path / "emb" / build_id, build_id,
                    shard_size=4, batch_size=4, library_versions={}, git_commit=None,
                    encoder_name="fake2")
    return canonical


def _configs_with_fake2(tmp_path, monkeypatch):
    """The real configs plus a `fake2` candidate: `cxr index` hashes the build's own entry."""
    import shutil

    out = tmp_path / "configs"
    shutil.copytree("configs", out)
    models = yaml.safe_load((out / "models.yaml").read_text(encoding="utf-8"))
    models["image_encoder"]["candidates"]["fake2"] = {"model_id": "fake/model",
                                                      "revision": "fakerev0"}
    (out / "models.yaml").write_text(yaml.safe_dump(models, sort_keys=False), encoding="utf-8")
    monkeypatch.setenv("CONFIG_DIR", str(out))
    return out


def test_index_role_image_writes_the_image_manifest_only(tmp_path, monkeypatch):
    configs = _configs_with_fake2(tmp_path, monkeypatch)
    canonical = _shards(tmp_path, IMAGE_BUILD)
    client = QdrantClient(":memory:")
    monkeypatch.setattr("src.cli.main._qdrant_client", lambda url: client)
    indexes = tmp_path / "indexes"
    argv = ["index", "--role", "image", "--embeddings", str(tmp_path / "emb"),
            "--build-id", IMAGE_BUILD, "--canonical-dir", str(canonical),
            "--indexes-dir", str(indexes)]
    assert main(argv) == 0
    assert not (indexes / index.MANIFEST_FILE).exists()
    manifest = json.loads((indexes / index.IMAGE_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["build_id"] == IMAGE_BUILD and manifest["encoder"] == "fake2"
    assert manifest["collection"] == "cxr_images_" + IMAGE_BUILD
    assert client.collection_exists(manifest["collection"])
    from src.common.config import config_hash, index_config_scope

    assert manifest["config_scope"] == index_config_scope("fake2", configs)
    assert manifest["config_hash"] == config_hash(manifest["config_scope"], configs)


def test_index_default_role_writes_the_vector_manifest_with_the_encoder(tmp_path, monkeypatch):
    _configs_with_fake2(tmp_path, monkeypatch)
    canonical = _shards(tmp_path, BUILD)
    monkeypatch.setattr("src.cli.main._qdrant_client", lambda url: QdrantClient(":memory:"))
    indexes = tmp_path / "indexes"
    assert main(["index", "--embeddings", str(tmp_path / "emb"), "--build-id", BUILD,
                 "--canonical-dir", str(canonical), "--indexes-dir", str(indexes)]) == 0
    manifest = json.loads((indexes / index.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["encoder"] == "fake2" and not (indexes / index.IMAGE_MANIFEST_FILE).exists()


MODELS = {"image_encoder": {"revision": "fakerev0",
                            "candidates": {"fake2": {"revision": "fakerev2"}}}}


def _image_world(tmp_path, monkeypatch, *, scores_encoder="fake2"):
    """The eight-check world plus an image manifest and a matching H6 scores artifact."""
    argv, indexes, studies = _eight_check_setup(tmp_path, monkeypatch)
    write_probe_artifact(indexes / "image_probe", studies)
    (indexes / index.IMAGE_MANIFEST_FILE).write_text(json.dumps({
        "build_id": IMAGE_BUILD, "encoder": "fake2",
        "embedding_model": {"model_id": "fake/model", "revision": "fakerev2", "dimension": 4}}),
        encoding="utf-8")
    write_probe_artifact(indexes / "image_scores", studies, build_id=IMAGE_BUILD,
                         encoder={"name": scores_encoder, "model_id": "fake/model",
                                  "revision": "fakerev2", "dim": 4})
    monkeypatch.setattr("src.cli.main.load_config", lambda name: MODELS)
    seen = []

    def fake_validate(client, manifest, shards_dir, images, **kw):
        seen.append((manifest["build_id"], shards_dir.name, kw["expected_revision"]))
        return [Check(f"check{i}", True, "ok") for i in range(6)]

    monkeypatch.setattr("src.cli.main.validate.validate_indexes", fake_validate)
    return argv, seen


def test_validate_indexes_adds_seven_image_checks_when_the_image_manifest_exists(
        tmp_path, capsys, monkeypatch):
    argv, seen = _image_world(tmp_path, monkeypatch)
    assert main(argv) == 0
    out = capsys.readouterr().out
    out.encode("ascii")
    lines = out.splitlines()
    assert len(lines) == 15 and all(line.startswith("PASS ") for line in lines)
    assert _names(lines[:8]) == [*(f"check{i}" for i in range(6)), "report_index", "image_probe"]
    assert _names(lines[8:]) == [*(f"image:check{i}" for i in range(6)), "image_scores"]
    assert f"build {IMAGE_BUILD}" in lines[-1]
    # the six checks ran once per manifest, each against its own shards and encoder pin
    assert seen == [(BUILD, BUILD, "fakerev0"), (IMAGE_BUILD, IMAGE_BUILD, "fakerev2")]


def test_validate_indexes_image_scores_check_names_the_encoder(tmp_path, capsys, monkeypatch):
    argv, _ = _image_world(tmp_path, monkeypatch, scores_encoder="fake")
    assert main(argv) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 15 and lines[-1].startswith("FAIL image_scores")
    assert "fake" in lines[-1] and "fake2" in lines[-1]


def test_validate_indexes_takes_an_image_scores_dir(tmp_path, capsys, monkeypatch):
    argv, _ = _image_world(tmp_path, monkeypatch)
    assert main([*argv, "--image-scores-dir", str(tmp_path / "nowhere")]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1].startswith("FAIL image_scores") and "nowhere" in lines[-1]


def test_index_refuses_shards_of_an_encoder_that_is_no_candidate(tmp_path, monkeypatch, capsys):
    canonical = _shards(tmp_path, BUILD)       # encoder fake2, absent from the real models.yaml
    monkeypatch.setattr("src.cli.main._qdrant_client", lambda url: QdrantClient(":memory:"))
    assert main(["index", "--embeddings", str(tmp_path / "emb"), "--build-id", BUILD,
                 "--canonical-dir", str(canonical), "--indexes-dir", str(tmp_path / "ix")]) == 1
    assert "no models.yaml image_encoder candidate" in capsys.readouterr().err
