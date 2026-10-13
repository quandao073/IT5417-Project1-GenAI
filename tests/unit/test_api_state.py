"""Startup checks decide whether the API may serve (plan section 16)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from src.api import dependencies as deps
from src.api.main import create_app
from src.common.config import Settings
from src.embeddings import embed, index
from tests.unit.fakes import FakeEncoder
from tests.unit.test_embed import images_frame

BUILD = "20261011_chexpert25k_fake_v1"


@pytest.fixture
def world(tmp_path):
    """Canonical dir, shards, an in-memory Qdrant with the collection, a manifest."""
    images = images_frame(8)
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    images.write_parquet(canonical / "images.parquet")
    out = tmp_path / "embeddings" / BUILD
    embed.run_embed(images, FakeEncoder(), Path("root"), out, BUILD, shard_size=4,
                    batch_size=4, library_versions={}, git_commit=None)
    client = QdrantClient(":memory:")
    manifest = index.run_index(client, out, images, "cxr_images", recreate=False,
                               config_hash="c", git_commit=None, dataset={}, library_versions={})
    indexes = tmp_path / "indexes"
    index.write_manifest(indexes / index.MANIFEST_FILE, manifest)
    settings = Settings(_env_file=None, canonical_dir=canonical, image_root=tmp_path / "root",
                        fts_db_path=indexes / "x.sqlite", indexes_dir=indexes)
    return settings, client, manifest


def _state(world, encoder=None, client=None, settings=None):
    s, c, _ = world
    return deps.build_state(
        settings or s,
        encoder_loader=lambda name, cand, device: encoder or FakeEncoder(),
        client_factory=lambda url: client or c,
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


class TestHealth:
    def test_ready_health_reports_the_build(self, world):
        client = TestClient(create_app(_state(world)))
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ready" and body["build_id"] == BUILD
        assert body["dimension"] == 4 and body["vectors"] == 8
        assert body["report_index"] == "disabled" and body["llm"] == "disabled"

    def test_not_ready_health_is_503_with_reason(self, world, tmp_path):
        settings, _, _ = world
        settings = settings.model_copy(update={"indexes_dir": tmp_path / "nowhere"})
        client = TestClient(create_app(_state(world, settings=settings)))
        response = client.get("/api/v1/health")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready" and response.json()["reason"]
