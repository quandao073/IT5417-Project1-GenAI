"""`cxr index` registers each build under its own id; the commands that read a build pick
one and query it with the encoder that embedded it."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src.cli.main import build_parser, main
from src.common import config
from src.embeddings import embed, index
from src.embeddings.validate import Check
from tests.unit.fakes import FakeEncoder
from tests.unit.test_api_state import BUILD, studies_frame
from tests.unit.test_embed import images_frame
from tests.unit.test_label_filter import labels_frame
from tests.unit.test_report_index import ROWS, reports_frame

MODELS = """image_encoder:
  selected: fake
  normalize: true
  text_prompt_template: "chest x-ray showing {}"
  candidates:
    fake: {model_id: fake/model, revision: fakerev0}
    other: {model_id: other/model, revision: otherrev}
"""


@pytest.fixture
def world(tmp_path, monkeypatch):
    """configs/ with two candidates, a canonical images table and one build's shards."""
    configs = tmp_path / "configs"
    configs.mkdir()
    for name in ("data", "concepts", "retrieval"):
        shutil.copy(config.default_config_dir() / f"{name}.yaml", configs)
    (configs / "models.yaml").write_text(MODELS, encoding="utf-8")
    monkeypatch.setenv("CONFIG_DIR", str(configs))
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    images = images_frame(6)
    images.write_parquet(canonical / "images.parquet")
    embed.run_embed(images, FakeEncoder(), Path("root"), tmp_path / "emb" / BUILD, BUILD,
                    shard_size=4, batch_size=4, library_versions={}, git_commit=None)
    client = QdrantClient(":memory:")
    monkeypatch.setattr("src.cli.main._qdrant_client", lambda *a, **k: client)
    return tmp_path


def _index(root: Path) -> int:
    return main(["index", "--embeddings", str(root / "emb"), "--build-id", BUILD,
                 "--canonical-dir", str(root / "canonical"), "--indexes-dir", str(root / "idx")])


def test_index_registers_the_build_under_its_id_with_its_encoder(world):
    assert _index(world) == 0
    manifest = json.loads(index.manifest_path(world / "idx", BUILD).read_text(encoding="utf-8"))
    assert manifest["encoder"] == "fake"
    assert manifest["config_scope"] == config.index_config_scope("fake")
    assert manifest["config_hash"] == config.config_hash(manifest["config_scope"])
    assert list((world / "idx").iterdir()) == [world / "idx" / index.BUILDS_DIR]


def test_index_refuses_shards_that_no_candidate_pins(world, capsys):
    (world / "configs" / "models.yaml").write_text(
        MODELS.replace("fakerev0", "elsewhere"), encoding="utf-8")
    assert _index(world) == 1
    assert "fakerev0" in capsys.readouterr().err
    assert not (world / "idx").exists()


class TestValidateOneBuild:
    @pytest.fixture
    def seen(self, world, monkeypatch):
        """Two registered builds; validate_indexes records what it was asked to check."""
        reports_frame(ROWS).write_parquet(world / "canonical" / "reports.parquet")
        studies_frame(images_frame(6)).write_parquet(world / "canonical" / "studies.parquet")
        for build_id, encoder in ((BUILD, "fake"), ("other_build", "other")):
            index.write_manifest(index.manifest_path(world / "idx", build_id),
                                 {"build_id": build_id, "encoder": encoder})
        calls = {}

        def fake_validate(client, manifest, shards_dir, images, **kw):
            calls.update(kw, build_id=manifest["build_id"], shards_dir=shards_dir)
            return [Check("checks", True, "ok")]

        monkeypatch.setattr("src.cli.main.validate.validate_indexes", fake_validate)
        monkeypatch.setattr("src.cli.main.validate.expected_config_hash", lambda *a, **k: "x")
        monkeypatch.setattr("src.cli.main.report_index.validate_report_index",
                            lambda *a, **k: Check("report_index", True, "ok"))
        return calls

    def _validate(self, world, *extra):
        return main(["validate-indexes", "--indexes-dir", str(world / "idx"),
                     "--canonical-dir", str(world / "canonical"),
                     "--embeddings", str(world / "emb"), *extra])

    def test_defaults_to_the_selected_encoder_s_build(self, world, seen):
        assert self._validate(world) == 0
        assert seen["build_id"] == BUILD and seen["expected_revision"] == "fakerev0"

    def test_checks_the_named_build_against_its_own_candidate(self, world, seen):
        assert self._validate(world, "--build-id", "other_build") == 0
        assert seen["build_id"] == "other_build" and seen["expected_revision"] == "otherrev"
        assert seen["shards_dir"] == world / "emb" / "other_build"

    def test_an_unregistered_build_is_one_fail_line(self, world, seen, capsys):
        assert self._validate(world, "--build-id", "ghost") == 1
        assert "build ghost is not registered" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["validate-indexes", "score-corpus", "eval-retrieval"])
def test_every_command_that_reads_a_build_takes_a_build_id_or_an_encoder(command):
    args = build_parser().parse_args([command, "--build-id", "b1", "--encoder", "fake"])
    assert (args.build_id, args.encoder) == ("b1", "fake")


class TestScoreCorpus:
    @pytest.fixture
    def loaded(self, world, monkeypatch):
        """BUILD (embedded by `fake`) registered while `other` is selected; records which
        encoder score_corpus is given."""
        assert _index(world) == 0
        models = world / "configs" / "models.yaml"
        models.write_text(MODELS.replace("selected: fake", "selected: other"), encoding="utf-8")
        labels_frame([("patient00000/study1", "Edema", "PRESENT", "CHEXBERT_IMPRESSION")]
                     ).write_parquet(world / "canonical" / "labels.parquet")
        names, seen = [], {}
        monkeypatch.setattr("src.embeddings.base.load_encoder",
                            lambda name, candidate, device: names.append(name) or FakeEncoder())

        def fake_score(shards_dir, encoder, *args, **kw):
            seen.update(shards_dir=shards_dir, encoder=encoder)
            return {"macro_auroc_strict": None, "ci95": None, "n_images": 6, "n_studies": 3,
                    "eligible_concepts": [], "concepts": {}}

        monkeypatch.setattr("src.cli.main.vb.score_corpus", fake_score)
        return names, seen

    def _score(self, world, *extra):
        return main(["score-corpus", "--indexes-dir", str(world / "idx"),
                     "--embeddings", str(world / "emb"),
                     "--canonical-dir", str(world / "canonical"),
                     "--out", str(world / "out.json"), "--n-boot", "2", *extra])

    def test_scores_a_build_with_the_encoder_that_embedded_it(self, world, loaded):
        names, seen = loaded
        assert self._score(world, "--build-id", BUILD) == 0
        assert names == ["fake"] and seen["shards_dir"] == world / "emb" / BUILD

    def test_refuses_an_encoder_whose_revision_differs_from_the_build(self, world, loaded,
                                                                      monkeypatch, capsys):
        stale = FakeEncoder()
        stale.revision = "stale"
        monkeypatch.setattr("src.embeddings.base.load_encoder", lambda *a: stale)
        assert self._score(world, "--build-id", BUILD) == 1
        assert f"stale != build {BUILD} revision fakerev0" in capsys.readouterr().err
        assert not (world / "out.json").exists()

    def test_refuses_a_build_id_and_an_encoder_that_disagree(self, world, loaded, capsys):
        assert self._score(world, "--build-id", BUILD, "--encoder", "other") == 1
        assert f"build {BUILD} was embedded by 'fake', not 'other'" in capsys.readouterr().err
        assert not (world / "out.json").exists()
