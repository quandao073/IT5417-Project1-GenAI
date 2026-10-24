"""Startup checks decide whether the API may serve (plan section 16)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from src.api import dependencies as deps
from src.api.main import create_app
from src.common.config import Settings, load_config
from src.common.ids import qdrant_point_id
from src.data.labels import CHEXPERT_LABELS
from src.embeddings import embed, index, probe
from src.retrieval.hybrid import FusionParams
from src.retrieval.report_index import build_report_index
from tests.unit.fakes import FakeEncoder, FakeOllama
from tests.unit.test_embed import images_frame
from tests.unit.test_label_filter import labels_frame
from tests.unit.test_report_index import ROWS, reports_frame

BUILD = "20261011_chexpert25k_fake_v1"
IMAGE_BUILD = "20261023_chexpert25k_fake2_v1"   # the image modes' second collection (D34)
IMAGE_ENCODER = {"name": "fake2", "model_id": "fake/model", "revision": FakeEncoder.revision,
                 "dim": 4}

WORLD_LABELS = [  # (study_id, concept, status, source) for the 4 studies of images_frame(8)
    ("patient00000/study1", "Edema", "PRESENT", "CHEXBERT_IMPRESSION"),
    ("patient00000/study1", "Pleural Effusion", "ABSENT", "CHEXPERT_V1"),
    ("patient00001/study1", "Edema", "PRESENT", "CHEXBERT_IMPRESSION"),
    ("patient00001/study1", "Pleural Effusion", "PRESENT", "CHEXBERT_IMPRESSION"),
    ("patient00002/study1", "Edema", "UNCERTAIN", "CHEXBERT_IMPRESSION"),
    ("patient00003/study1", "Cardiomegaly", "PRESENT", "CHEXPERT_V1"),
]


def studies_frame(images: pl.DataFrame) -> pl.DataFrame:
    """One row per study; the selected image is the lowest image_id (view1)."""
    first = images.sort("image_id").group_by("study_id", maintain_order=True).first()
    return first.select(
        "study_id", "patient_id", pl.col("image_id").alias("selected_image_id"),
        pl.lit(True).alias("has_report"), pl.lit(True).alias("has_impression"),
        pl.lit(False).alias("has_findings"), pl.lit("ok").alias("ingestion_status"),
    ).sort("study_id")


def write_probe_artifact(out_dir, studies, *, build_id=BUILD, drop=0, encoder=None):
    """A probe artifact for `studies` (a studies_frame): zero weights, zero logits,
    `drop` rows of corpus_scores left out; `encoder` replaces the meta's encoder block."""
    meta = {"probe_id": "p1", "build_id": build_id,
            "encoder": encoder or {"model_id": "fake", "revision": FakeEncoder().revision,
                                   "dim": 4}}
    scores = pl.DataFrame({
        "study_id": studies["study_id"], "image_id": studies["selected_image_id"],
        **{c: pl.Series([0.0] * studies.height, dtype=pl.Float32) for c in CHEXPERT_LABELS},
    })
    probe.save_probe(
        out_dir, probe.Probe(np.zeros((14, 4), np.float32), np.zeros(14, np.float32),
                             tuple(CHEXPERT_LABELS)),
        meta, scores.head(studies.height - drop))


def write_image_build(tmp_path, client, images, studies, indexes, *, scores_build=IMAGE_BUILD,
                      scores_encoder=IMAGE_ENCODER):
    """The image modes' build: shards by `fake2`, a second collection, its manifest and the
    decomposed scores artifact (H6) for `scores_build`."""
    out = tmp_path / "embeddings" / IMAGE_BUILD
    embed.run_embed(images, FakeEncoder(), Path("root"), out, IMAGE_BUILD, shard_size=4,
                    batch_size=4, library_versions={}, git_commit=None, encoder_name="fake2")
    manifest = index.run_index(client, out, images, "cxr_images", recreate=True,
                               config_hash="c", git_commit=None, dataset={}, library_versions={})
    index.write_manifest(indexes / index.IMAGE_MANIFEST_FILE, manifest)
    write_probe_artifact(indexes / "image_scores", studies, build_id=scores_build,
                         encoder=scores_encoder)
    return manifest


@pytest.fixture
def world(tmp_path):
    """Canonical dir, shards, an in-memory Qdrant with the collection, a manifest, the
    report index (ROWS are keyed by the same four studies), an image probe and the image
    modes' build (second collection, image manifest, decomposed scores)."""
    images = images_frame(8)
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    images.write_parquet(canonical / "images.parquet")
    studies = studies_frame(images)
    studies.write_parquet(canonical / "studies.parquet")
    labels_frame(WORLD_LABELS).write_parquet(canonical / "labels.parquet")
    reports_frame(ROWS).write_parquet(canonical / "reports.parquet")
    out = tmp_path / "embeddings" / BUILD
    embed.run_embed(images, FakeEncoder(), Path("root"), out, BUILD, shard_size=4,
                    batch_size=4, library_versions={}, git_commit=None)
    client = QdrantClient(":memory:")
    manifest = index.run_index(client, out, images, "cxr_images", recreate=False,
                               config_hash="c", git_commit=None, dataset={}, library_versions={})
    indexes = tmp_path / "indexes"
    index.write_manifest(indexes / index.MANIFEST_FILE, manifest)
    settings = Settings(_env_file=None, canonical_dir=canonical, image_root=tmp_path / "root",
                        fts_db_path=indexes / "x.sqlite", indexes_dir=indexes,
                        image_probe_dir=indexes / "image_probe", llm_disabled=True,
                        image_build_manifest=indexes / index.IMAGE_MANIFEST_FILE,
                        image_scores_dir=indexes / "image_scores")
    build_report_index(reports_frame(ROWS), settings.fts_db_path, concepts_sha256="x",
                       git_commit=None)
    write_probe_artifact(settings.image_probe_dir, studies)
    write_image_build(tmp_path, client, images, studies, indexes)
    return settings, client, manifest


def _state(world, encoder=None, client=None, settings=None, llm_factory=None,
           image_encoder=None, image_loader=None):
    s, c, _ = world
    return deps.build_state(
        settings or s,
        encoder_loader=lambda name, cand, device: encoder or FakeEncoder(),
        client_factory=lambda url: client or c,
        llm_factory=llm_factory,
        image_encoder_loader=image_loader or (
            lambda name, cand, device: image_encoder or FakeEncoder()),
    )


class TestBuildState:
    def test_ready_when_everything_agrees(self, world):
        state = _state(world)
        assert state.ready and state.not_ready_reason is None
        assert state.collection == "cxr_images_" + BUILD
        assert len(state.images) == 8
        assert state.images[next(iter(state.images))].path_to_image.startswith("train/")

    def test_not_ready_without_a_manifest(self, world, tmp_path):
        settings, client, _ = world
        settings = settings.model_copy(update={"indexes_dir": tmp_path / "nowhere"})
        state = _state(world, settings=settings)
        assert not state.ready and "build_manifest.json" in state.not_ready_reason

    def test_not_ready_when_the_collection_is_missing(self, world):
        state = _state(world, client=QdrantClient(":memory:"))
        assert not state.ready and "collection" in state.not_ready_reason

    def test_not_ready_when_vector_count_differs(self, world):
        settings, client, manifest = world
        client.upsert(manifest["collection"], points=[models.PointStruct(
            id="00000000-0000-0000-0000-000000000009", vector=[1.0, 0, 0, 0], payload={})])
        state = _state(world)
        assert not state.ready and "9 vectors" in state.not_ready_reason

    def test_not_ready_when_encoder_revision_differs(self, world):
        bad = FakeEncoder()
        bad.revision = "otherrev"
        state = _state(world, encoder=bad)
        assert not state.ready and "revision" in state.not_ready_reason

    def test_not_ready_when_qdrant_is_unreachable(self, world):
        class Dead:
            def collection_exists(self, name):
                raise ConnectionError("refused")
        state = _state(world, client=Dead())
        assert not state.ready and "refused" in state.not_ready_reason

    def test_ready_state_carries_the_label_matrix_and_study_map(self, world):
        state = _state(world)
        assert state.labels is not None and state.labels.status.shape == (4, 14)
        assert state.study_image["patient00000/study1"] == f"{0:064x}"
        assert state.parser is not None and state.llm_status == "disabled"
        assert state.images[f"{0:064x}"].projection == "PA"

    def test_not_ready_when_studies_disagree_with_the_manifest(self, world):
        settings, _, _ = world
        path = settings.canonical_dir / "studies.parquet"
        pl.read_parquet(path).head(3).write_parquet(path)
        state = _state(world)
        assert not state.ready and "studies.parquet has 3 rows" in state.not_ready_reason

    def test_not_ready_without_labels(self, world):
        settings, _, _ = world
        (settings.canonical_dir / "labels.parquet").unlink()
        assert not _state(world).ready

    def test_llm_unreachable_degrades_to_the_rule_parser_but_stays_ready(self, world):
        settings = world[0].model_copy(update={"llm_disabled": False})
        state = _state(world, settings=settings,
                       llm_factory=lambda *a, **k: FakeOllama("unreachable"))
        assert state.ready and state.llm_status == "unreachable"
        r = state.parser.parse("cardiomegaly")
        assert r.parser == "rule" and "unreachable" in r.warnings[0]

    def test_llm_ok_wires_the_client_in(self, world):
        settings = world[0].model_copy(update={"llm_disabled": False})
        fake = FakeOllama("ok")
        state = _state(world, settings=settings, llm_factory=lambda *a, **k: fake)
        assert state.ready and state.llm_status == "ok" and state.parser.llm is fake
        assert "Request:" in state.parser.system_prompt

    def test_a_hosted_provider_without_a_key_degrades_to_the_rule_parser(self, world):
        """A missing LLM_API_KEY is a configuration slip, not a reason to refuse search."""
        settings = world[0].model_copy(update={"llm_disabled": False, "llm_provider": "openai",
                                               "llm_api_key": None})
        state = _state(world, settings=settings)
        assert state.ready and state.llm_status == "unreachable"
        r = state.parser.parse("cardiomegaly")
        assert r.parser == "rule" and "LLM_API_KEY" in r.warnings[0]


class TestReportIndex:
    def test_ready_state_opens_the_report_index(self, world):
        state = _state(world)
        assert state.report_status == "ok" and state.report_reason is None
        assert state.report_meta["studies"] == "4" and len(state.point_of_study) == 4
        selected = state.images[state.study_image["patient00001/study1"]]
        assert state.point_of_study["patient00001/study1"] == str(
            qdrant_point_id(selected.path_to_image))
        assert state.report_terms["Edema"][0] == "pulmonary edema"
        assert state.evidence is not None

    def test_missing_report_index_degrades_but_stays_ready(self, world):
        world[0].fts_db_path.unlink()
        state = _state(world)
        assert state.ready and state.report_status == "disabled" and state.report_db is None
        assert "not found" in state.report_reason

    def test_meta_studies_mismatch_disables_report_index(self, world):
        build_report_index(reports_frame(ROWS[:3]), world[0].fts_db_path, concepts_sha256="x",
                           git_commit=None)
        state = _state(world)
        assert state.ready and state.report_status == "disabled" and state.report_db is None
        assert "studies" in state.report_reason

    def test_state_carries_fusion_params_from_config(self, world):
        state = _state(world)
        assert state.fusion == FusionParams.from_config(load_config("retrieval"))
        assert state.fusion.rrf_k == 10 and state.fusion.vector_top_k == 100
        assert state.report_min_token_chars == 3


class TestImageProbe:
    def test_state_loads_the_image_probe(self, world):
        state = _state(world)
        assert state.image_status == "ok" and state.image_reason is None
        assert state.image_meta["probe_id"] == "p1"
        assert state.image_scores.logits.shape == (4, 14)
        # aligned to the label matrix, which is what the pipeline indexes both by
        assert state.image_scores.row_of == {s: i for i, s in enumerate(state.labels.study_ids)}

    def test_state_is_ready_without_the_probe(self, world, tmp_path):
        settings = world[0].model_copy(update={"image_probe_dir": tmp_path / "nowhere"})
        state = _state(world, settings=settings)
        assert state.ready and state.image_status == "disabled" and state.image_scores is None
        assert "not found" in state.image_reason

    def test_state_disables_the_probe_on_build_id_mismatch(self, world):
        settings = world[0]
        write_probe_artifact(settings.image_probe_dir,
                             pl.read_parquet(settings.canonical_dir / "studies.parquet"),
                             build_id="another_build")
        state = _state(world)
        assert state.ready and state.image_status == "disabled"
        assert "another_build" in state.image_reason and "run cxr train-probe" in state.image_reason

    def test_state_disables_the_probe_when_a_corpus_study_has_no_score(self, world, tmp_path):
        settings = world[0]
        short = tmp_path / "short_probe"
        write_probe_artifact(short, pl.read_parquet(settings.canonical_dir / "studies.parquet"),
                             drop=1)
        state = _state(world, settings=settings.model_copy(update={"image_probe_dir": short}))
        assert state.ready and state.image_status == "disabled"
        assert "1 corpus studies" in state.image_reason

    def test_a_mismatched_encoder_revision_disables_the_probe_only(self, world):
        settings = world[0]
        meta_path = settings.image_probe_dir / probe.META_FILE
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["encoder"]["revision"] = "oldrev"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        state = _state(world)
        assert state.ready and state.image_status == "disabled" and "oldrev" in state.image_reason

    def test_a_corrupt_probe_file_disables_the_probe_only(self, world):
        (world[0].image_probe_dir / probe.PROBE_FILE).write_bytes(b"not an npz")
        state = _state(world)
        assert state.ready is True and state.image_status == "disabled"
        assert "probe.npz" in state.image_reason

    def test_probe_meta_without_build_id_disables_the_probe_only(self, world):
        meta_path = world[0].image_probe_dir / probe.META_FILE
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        del meta["build_id"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        state = _state(world)
        assert state.ready is True and state.image_status == "disabled"
        assert "build_id" in state.image_reason


def test_a_broken_training_yaml_leaves_the_api_ready(world, monkeypatch, caplog):
    """training.yaml is edited for every fine-tune variant; it only feeds a warning (H2)."""
    real = deps.load_config

    def load(name, *args, **kwargs):
        if name == "training":
            raise ValueError("training.yaml:concepts must follow CHEXPERT_LABELS order")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(deps, "load_config", load)
    state = _state(world)
    assert state.ready and state.joint_status == state.decomposed_status == "ok"
    assert state.trained_concepts is None
    assert any("training concepts unknown" in r.message for r in caplog.records)


class TestImageModes:
    """The image modes open their own manifest, collection, encoder and scores; nothing
    about them ever touches readiness (Review Focus 5)."""

    def test_ready_with_both_image_modes(self, world):
        state = _state(world)
        assert state.ready and state.collection == "cxr_images_" + BUILD
        assert state.joint_status == "ok" and state.joint_reason is None
        assert state.joint_collection == "cxr_images_" + IMAGE_BUILD
        assert state.joint_manifest["build_id"] == IMAGE_BUILD
        assert state.joint_manifest["encoder"] == "fake2"
        assert state.joint_encoder is not None and state.joint_encoder is not state.encoder
        assert state.decomposed_status == "ok" and state.decomposed_reason is None
        assert state.decomposed_scores.logits.shape == (4, 14)
        assert state.decomposed_scores.row_of == {
            s: i for i, s in enumerate(state.labels.study_ids)}
        assert state.decomposed_meta["probe_id"] == "p1"

    def test_image_modes_disabled_without_manifest(self, world, tmp_path):
        settings = world[0].model_copy(update={"image_build_manifest": tmp_path / "nowhere.json"})
        state = _state(world, settings=settings)
        assert state.ready and state.joint_status == state.decomposed_status == "disabled"
        assert "nowhere.json" in state.joint_reason
        assert "cxr index --role image" in state.joint_reason
        assert state.decomposed_reason == state.joint_reason
        assert state.joint_encoder is None and state.decomposed_scores is None

    def test_image_modes_disabled_when_collection_missing_but_api_ready(self, world):
        settings, client, _ = world
        client.delete_collection("cxr_images_" + IMAGE_BUILD)
        state = _state(world)
        assert state.ready and state.not_ready_reason is None
        assert state.joint_status == state.decomposed_status == "disabled"
        assert "cxr_images_" + IMAGE_BUILD in state.joint_reason
        assert "does not exist" in state.joint_reason

    def test_image_modes_disabled_when_vector_count_differs(self, world):
        settings, client, _ = world
        client.upsert("cxr_images_" + IMAGE_BUILD, points=[models.PointStruct(
            id="00000000-0000-0000-0000-000000000009", vector=[1.0, 0, 0, 0], payload={})])
        state = _state(world)
        assert state.ready and state.joint_status == "disabled"
        assert "9 vectors" in state.joint_reason and state.decomposed_status == "disabled"

    def test_image_modes_disabled_when_encoder_revision_differs(self, world):
        bad = FakeEncoder()
        bad.revision = "otherrev"
        state = _state(world, image_encoder=bad)
        assert state.ready and state.joint_status == state.decomposed_status == "disabled"
        assert "otherrev" in state.joint_reason and "fake2" in state.joint_reason

    def test_an_unknown_encoder_candidate_disables_the_image_modes(self, world):
        # the production loader: `fake2` is no models.yaml candidate
        state = _state(world, image_loader=deps._default_encoder)
        assert state.ready and state.joint_status == "disabled"
        assert "fake2" in state.joint_reason and "candidate" in state.joint_reason

    def test_decomposed_disabled_when_scores_are_for_another_build(self, world):
        settings, _, _ = world
        studies = pl.read_parquet(settings.canonical_dir / "studies.parquet")
        write_probe_artifact(settings.image_scores_dir, studies, build_id="another_build",
                             encoder=IMAGE_ENCODER)
        state = _state(world)
        assert state.ready and state.joint_status == "ok"
        assert state.decomposed_status == "disabled" and state.decomposed_scores is None
        assert "another_build" in state.decomposed_reason
        assert "cxr build-image-scores" in state.decomposed_reason

    def test_decomposed_disabled_when_scores_name_another_encoder(self, world):
        settings, _, _ = world
        studies = pl.read_parquet(settings.canonical_dir / "studies.parquet")
        write_probe_artifact(settings.image_scores_dir, studies, build_id=IMAGE_BUILD,
                             encoder={**IMAGE_ENCODER, "name": "fake"})
        state = _state(world)
        assert state.joint_status == "ok" and state.decomposed_status == "disabled"
        assert "fake2" in state.decomposed_reason

    def test_decomposed_disabled_without_scores_but_joint_stays(self, world, tmp_path):
        settings = world[0].model_copy(update={"image_scores_dir": tmp_path / "nowhere"})
        state = _state(world, settings=settings)
        assert state.ready and state.joint_status == "ok"
        assert state.decomposed_status == "disabled" and "nowhere" in state.decomposed_reason

    def test_health_reports_image_modes(self, world, tmp_path):
        body = TestClient(create_app(_state(world))).get("/api/v1/health").json()
        assert body["status"] == "ready" and body["build_id"] == BUILD
        assert body["image_joint"] == "ok" and body["image_decomposed"] == "ok"
        assert body["image_encoder"] == {
            "name": "fake2", "model_id": "fake/model", "revision": "fakerev0",
            "build_id": IMAGE_BUILD, "collection": "cxr_images_" + IMAGE_BUILD}
        settings = world[0].model_copy(update={"image_build_manifest": tmp_path / "no.json"})
        off = TestClient(create_app(_state(world, settings=settings))).get("/api/v1/health")
        assert off.status_code == 200
        assert off.json()["image_joint"] == off.json()["image_decomposed"] == "disabled"
        assert off.json()["image_encoder"] is None
        assert off.json()["image_probe"] == "ok"          # the probe is a separate artifact


class TestHealth:
    def test_ready_health_reports_the_build(self, world):
        client = TestClient(create_app(_state(world)))
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ready" and body["build_id"] == BUILD
        assert body["dimension"] == 4 and body["vectors"] == 8
        assert body["report_index"] == "ok" and body["report_studies"] == 4
        assert body["llm"] == "disabled"
        assert body["image_probe"] == "ok" and body["image_probe_id"] == "p1"
        assert body["studies"] == 4 and body["llm_model"] == "gpt-5-mini-2025-08-07"

    def test_not_ready_health_is_503_with_reason(self, world, tmp_path):
        settings, _, _ = world
        settings = settings.model_copy(update={"indexes_dir": tmp_path / "nowhere"})
        client = TestClient(create_app(_state(world, settings=settings)))
        response = client.get("/api/v1/health")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready" and response.json()["reason"]
