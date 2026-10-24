"""Startup checks decide whether the API may serve (plan section 16)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from src.api import dependencies as deps
from src.api.main import create_app
from src.common.config import Settings
from src.embeddings import embed, index
from src.retrieval.report_index import build_report_index
from tests.unit.fakes import FakeEncoder, FakeOllama
from tests.unit.test_embed import images_frame
from tests.unit.test_label_filter import labels_frame
from tests.unit.test_report_index import ROWS, reports_frame

BUILD = "20261011_chexpert25k_fake_v1"

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


@pytest.fixture
def world(tmp_path):
    """Canonical dir, shards, an in-memory Qdrant with the collection, a manifest and the
    report index (ROWS are keyed by the same four studies)."""
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
                               config_hash="c", git_commit=None, dataset={}, library_versions={},
                               encoder="fake")
    indexes = tmp_path / "indexes"
    index.write_manifest(index.manifest_path(indexes, BUILD), manifest)
    settings = Settings(_env_file=None, canonical_dir=canonical, image_root=tmp_path / "root",
                        fts_db_path=indexes / "x.sqlite", indexes_dir=indexes, build_id=BUILD,
                        llm_disabled=True)
    build_report_index(reports_frame(ROWS), settings.fts_db_path, concepts_sha256="x",
                       git_commit=None)
    return settings, client, manifest


def _state(world, encoder=None, client=None, settings=None, llm_factory=None):
    s, c, _ = world
    return deps.build_state(
        settings or s,
        encoder_loader=lambda name, cand, device: encoder or FakeEncoder(),
        client_factory=lambda url: client or c,
        llm_factory=llm_factory,
    )


class TestBuildState:
    def test_ready_when_everything_agrees(self, world):
        state = _state(world)
        assert state.ready and state.not_ready_reason is None
        assert state.collection == "cxr_images_" + BUILD
        assert len(state.images) == 8
        assert state.images[next(iter(state.images))].path_to_image.startswith("train/")

    def test_not_ready_when_the_build_is_not_registered(self, world, tmp_path):
        settings, client, _ = world
        settings = settings.model_copy(update={"indexes_dir": tmp_path / "nowhere"})
        state = _state(world, settings=settings)
        assert not state.ready and f"build {BUILD} is not registered" in state.not_ready_reason

    def test_loads_the_encoder_the_build_s_manifest_names(self, world):
        """Builds of several encoders share the registry; `selected` does not pick the model."""
        settings, client, _ = world
        seen = []
        state = deps.build_state(
            settings, client_factory=lambda url: client,
            encoder_loader=lambda name, cand, device: seen.append(name) or FakeEncoder())
        assert state.ready and seen == ["fake"]

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

    def test_ready_state_carries_the_images_and_the_concepts_it_renders(self, world):
        state = _state(world)
        assert state.parser is not None and state.llm_status == "disabled"
        assert state.images[f"{0:064x}"].projection == "PA"
        assert state.concepts_cfg["concepts"]["Edema"]["en"][0] == "pulmonary edema"

    def test_not_ready_when_studies_disagree_with_the_manifest(self, world):
        settings, _, _ = world
        path = settings.canonical_dir / "studies.parquet"
        pl.read_parquet(path).head(3).write_parquet(path)
        state = _state(world)
        assert not state.ready and "studies.parquet has 3 rows" in state.not_ready_reason

    def test_never_loads_the_labels(self, world):
        """Search reads no label; the label filter comes back as its own knob later."""
        settings, _, _ = world
        (settings.canonical_dir / "labels.parquet").unlink()
        state = _state(world)
        assert state.ready and not hasattr(state, "labels")

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
        assert state.report_meta["studies"] == "4" and state.report_min_token_chars == 3
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


class TestHealth:
    def test_ready_health_reports_the_build(self, world):
        client = TestClient(create_app(_state(world)))
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ready" and body["build_id"] == BUILD
        assert body["dimension"] == 4 and body["vectors"] == 8
        assert body["report_index"] == "ok" and body["report_studies"] == 4
        assert body["llm"] == "disabled" and body["encoder"] == "fake"
        assert body["studies"] == 4 and body["llm_model"] == "gpt-5-mini-2025-08-07"

    def test_not_ready_health_is_503_with_reason(self, world, tmp_path):
        settings, _, _ = world
        settings = settings.model_copy(update={"indexes_dir": tmp_path / "nowhere"})
        client = TestClient(create_app(_state(world, settings=settings)))
        response = client.get("/api/v1/health")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready" and response.json()["reason"]
