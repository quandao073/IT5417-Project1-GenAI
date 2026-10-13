"""/search in vector mode and /images, against a fake state."""

from __future__ import annotations

import polars as pl
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import create_app
from tests.unit.test_api_state import _state
from tests.unit.test_api_state import world as _world_fixture

# Re-exported under its own name so pytest finds it here (an import would trip ruff F811).
world = _world_fixture


@pytest.fixture
def client(world):
    state = _state(world)
    settings = state.settings
    # one real jpeg so /images can serve it
    some_id, record = next(iter(state.images.items()))
    target = settings.image_root / record.path_to_image
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), 128).save(target)
    return TestClient(create_app(state)), state, some_id


class TestSearch:
    def test_vector_search_returns_ranked_results_with_urls(self, client):
        c, state, _ = client
        body = c.post("/api/v1/search", json={"query": "pleural effusion", "top_k": 3}).json()
        assert body["build_id"] == state.manifest["build_id"]
        assert body["parsed_query"] == {"semantic_query": "pleural effusion", "views": None,
                                        "parser": "passthrough"}
        assert [r["rank"] for r in body["results"]] == [1, 2, 3]
        first = body["results"][0]
        assert first["image_url"] == f"/api/v1/images/{first['image_id']}"
        assert first["score_breakdown"] == {"vector_rank": 1}
        assert first["study_id"] == state.images[first["image_id"]].study_id
        assert isinstance(body["latency_ms"], int) and len(body["request_id"]) == 32

    def test_query_is_stripped_and_blank_query_is_rejected(self, client):
        c, _, _ = client
        ok = c.post("/api/v1/search", json={"query": "  edema  "}).json()
        assert ok["parsed_query"]["semantic_query"] == "edema"
        assert c.post("/api/v1/search", json={"query": "   "}).status_code == 422

    def test_top_k_bounds(self, client):
        c, _, _ = client
        assert c.post("/api/v1/search", json={"query": "x", "top_k": 51}).status_code == 422
        assert c.post("/api/v1/search", json={"query": "x", "top_k": 0}).status_code == 422
        assert len(c.post("/api/v1/search", json={"query": "x"}).json()["results"]) == 8

    def test_views_filter_and_empty_views(self, client):
        c, state, _ = client
        images = pl.read_parquet(state.settings.canonical_dir / "images.parquet")
        pa = set(images.filter(pl.col("projection") == "PA")["image_id"])
        hits = c.post("/api/v1/search",
                      json={"query": "x", "views": ["PA"], "top_k": 50}).json()
        assert {r["image_id"] for r in hits["results"]} == pa
        all_hits = c.post("/api/v1/search", json={"query": "x", "views": [], "top_k": 50}).json()
        assert len(all_hits["results"]) == 8
        lateral = c.post("/api/v1/search", json={"query": "x", "views": ["LATERAL"]})
        assert lateral.status_code == 422

    def test_other_modes_are_400_in_this_build(self, client):
        c, _, _ = client
        r = c.post("/api/v1/search", json={"query": "x", "mode": "hybrid"})
        assert r.status_code == 400 and "hybrid" in r.json()["detail"]

    def test_search_is_503_when_not_ready(self, world, tmp_path):
        settings, _, _ = world
        settings = settings.model_copy(update={"indexes_dir": tmp_path / "nowhere"})
        c = TestClient(create_app(_state(world, settings=settings)))
        r = c.post("/api/v1/search", json={"query": "x"})
        assert r.status_code == 503 and r.json()["detail"]["status"] == "not_ready"

    def test_search_returns_503_when_qdrant_fails_mid_flight(self, client):
        c, state, _ = client

        class Dying:
            def query_points(self, *a, **k):
                raise ConnectionError("qdrant went away")

        state.client = Dying()
        r = c.post("/api/v1/search", json={"query": "x"})
        assert r.status_code == 503 and "qdrant" in r.json()["detail"]["reason"]


class TestImages:
    def test_serves_the_jpeg_with_immutable_cache(self, client):
        c, _, some_id = client
        r = c.get(f"/api/v1/images/{some_id}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert "immutable" in r.headers["cache-control"]

    def test_unknown_id_is_404(self, client):
        c, _, _ = client
        assert c.get("/api/v1/images/" + "f" * 64).status_code == 404

    def test_path_like_ids_never_touch_the_filesystem(self, client):
        c, _, _ = client
        assert c.get("/api/v1/images/..%2F..%2Fetc%2Fpasswd").status_code == 404
        path_like = "train%2Fpatient00001%2Fstudy1%2Fview1_frontal.jpg"
        assert c.get("/api/v1/images/" + path_like).status_code == 404
