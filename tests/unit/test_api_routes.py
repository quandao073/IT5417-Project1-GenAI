"""/search in vector mode and /images, against a fake state."""

from __future__ import annotations

import polars as pl
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import create_app
from src.api.schemas.search import SearchRequest
from src.common.config import load_config
from src.query import llm_parser as llm
from tests.unit.fakes import FakeOllama
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
        parsed = body["parsed_query"]
        assert parsed["semantic_query"] == "chest x-ray showing pleural effusion"
        assert parsed["include"] == [{"concept": "Pleural Effusion", "status": "PRESENT"}]
        assert parsed["parser"] == "rule" and parsed["fallback_used"] is False
        assert [r["rank"] for r in body["results"]] == [1, 2, 3]
        first = body["results"][0]
        assert first["image_url"] == f"/api/v1/images/{first['image_id']}"
        assert first["score_breakdown"] == {"vector_rank": 1, "label_match": None}
        assert first["study_id"] == state.images[first["image_id"]].study_id
        assert isinstance(first["score"], float)
        assert [a["concept"] for a in first["labels"]] == ["Pleural Effusion"]
        assert body["total_matches"] is None and body["warnings"] == []
        assert isinstance(body["latency_ms"], int) and len(body["request_id"]) == 32

    def test_query_is_stripped_and_blank_query_is_rejected(self, client):
        c, _, _ = client
        ok = c.post("/api/v1/search", json={"query": "  edema  "}).json()
        assert ok["parsed_query"]["semantic_query"] == "chest x-ray showing pulmonary edema"
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


    def test_vector_mode_encodes_the_semantic_query_not_the_raw_text(self, client):
        c, _, _ = client
        a = c.post("/api/v1/search", json={"query": "pleural effusion"}).json()["results"]
        b = c.post("/api/v1/search", json={"query": "Any effusion, please"}).json()["results"]
        assert [r["image_id"] for r in a] == [r["image_id"] for r in b]

    def test_parsed_views_apply_unless_the_request_overrides_them(self, client):
        c, state, _ = client
        images = pl.read_parquet(state.settings.canonical_dir / "images.parquet")
        pa = set(images.filter(pl.col("projection") == "PA")["image_id"])
        ap = set(images.filter(pl.col("projection") == "AP")["image_id"])
        parsed = c.post("/api/v1/search", json={"query": "PA view with edema", "top_k": 50}).json()
        assert {r["image_id"] for r in parsed["results"]} == pa
        assert parsed["parsed_query"]["views"] == ["PA"]
        forced = c.post("/api/v1/search",
                        json={"query": "PA view with edema", "views": ["AP"], "top_k": 50}).json()
        assert {r["image_id"] for r in forced["results"]} == ap

    def test_vietnamese_query_without_a_concept_returns_nothing_in_vector_mode(self, client):
        c, _, _ = client
        body = c.post("/api/v1/search", json={"query": "vùng mờ gần rốn phổi"}).json()
        assert body["results"] == [] and body["parsed_query"]["semantic_query"] == ""
        assert any("vector leg skipped" in w for w in body["warnings"])

    def test_vector_results_expose_the_constraint_labels(self, client):
        c, state, _ = client
        body = c.post("/api/v1/search",
                      json={"query": "edema with no pleural effusion", "top_k": 50}).json()
        by_study = {r["study_id"]: r["labels"] for r in body["results"]}
        assert by_study["patient00001/study1"] == [
            {"concept": "Edema", "status": "PRESENT", "source": "CHEXBERT_IMPRESSION"},
            {"concept": "Pleural Effusion", "status": "PRESENT", "source": "CHEXBERT_IMPRESSION"}]
        assert by_study["patient00003/study1"] == [
            {"concept": "Edema", "status": "NOT_MENTIONED", "source": None},
            {"concept": "Pleural Effusion", "status": "NOT_MENTIONED", "source": None}]

    def test_llm_dying_mid_flight_falls_back_in_search(self, world):
        settings = world[0].model_copy(update={"llm_disabled": False})
        fake = FakeOllama("ok", [llm.LLMError("ConnectError: gone")])
        c = TestClient(create_app(_state(world, settings=settings,
                                         llm_factory=lambda *a, **k: fake)))
        body = c.post("/api/v1/search", json={"query": "edema", "mode": "label"}).json()
        assert body["parsed_query"]["parser"] == "rule"
        assert body["parsed_query"]["fallback_used"] is True
        assert body["warnings"] == ["llm unreachable: ConnectError: gone"]
        assert body["total_matches"] == 2


class TestLabelMode:
    def post(self, c, query, **extra):
        return c.post("/api/v1/search", json={"query": query, "mode": "label", **extra}).json()

    def studies(self, body):
        return {r["study_id"] for r in body["results"]}

    def test_filters_counts_and_explains(self, client):
        c, state, _ = client
        body = self.post(c, "edema with no pleural effusion")
        assert self.studies(body) == {"patient00000/study1"} and body["total_matches"] == 1
        r = body["results"][0]
        assert r["rank"] == 1 and r["score"] is None
        assert r["score_breakdown"] == {"vector_rank": None, "label_match": True}
        assert r["image_id"] == state.study_image["patient00000/study1"]
        assert r["image_url"] == f"/api/v1/images/{r['image_id']}"
        assert r["labels"] == [
            {"concept": "Edema", "status": "PRESENT", "source": "CHEXBERT_IMPRESSION"},
            {"concept": "Pleural Effusion", "status": "ABSENT", "source": "CHEXPERT_V1"}]

    def test_uncertainty_policy_include_keeps_uncertain_studies(self, client):
        c, _, _ = client
        body = self.post(c, "edema with no pleural effusion", uncertainty_policy="INCLUDE")
        assert self.studies(body) == {"patient00000/study1", "patient00002/study1"}

    def test_explicit_absence_required_needs_a_recorded_absent(self, client):
        c, _, _ = client
        lenient = self.post(c, "cardiomegaly without effusion")
        strict = self.post(c, "cardiomegaly without effusion",
                           absence_policy="explicit_absence_required")
        assert self.studies(lenient) == {"patient00003/study1"}
        assert self.studies(strict) == set() and strict["total_matches"] == 0

    def test_explicit_absence_required_can_match_nothing(self, client):
        c, _, _ = client
        body = self.post(c, "fracture", absence_policy="explicit_absence_required", top_k=50)
        assert body["results"] == [] and body["total_matches"] == 0

    def test_no_constraint_is_empty_with_a_warning(self, client):
        c, _, _ = client
        body = self.post(c, "dense white area near the hilum")
        assert body["results"] == [] and body["total_matches"] == 0
        assert "label mode needs at least one concept" in body["warnings"]

    def test_views_filter_applies_to_label_mode(self, client):
        c, _, _ = client
        # studies 0 and 1 have Edema PRESENT; their selected images are i=0 (PA) and i=2 (AP)
        assert self.studies(self.post(c, "edema")) == {"patient00000/study1",
                                                       "patient00001/study1"}
        pa = self.post(c, "edema", views=["PA"])
        assert self.studies(pa) == {"patient00000/study1"} and pa["total_matches"] == 1
        parsed = self.post(c, "AP films with edema")
        assert self.studies(parsed) == {"patient00001/study1"}

    def test_top_k_truncates_but_total_counts_all(self, client):
        c, _, _ = client
        body = self.post(c, "edema", uncertainty_policy="INCLUDE", top_k=2)
        assert len(body["results"]) == 2 and body["total_matches"] == 3
        assert [r["rank"] for r in body["results"]] == [1, 2]

    def test_order_is_stable_across_calls(self, client):
        c, _, _ = client
        a = [r["study_id"] for r in self.post(c, "edema", uncertainty_policy="INCLUDE")["results"]]
        b = [r["study_id"] for r in self.post(c, "edema", uncertainty_policy="INCLUDE")["results"]]
        assert a == b and len(a) == 3

    def test_label_mode_never_touches_qdrant(self, client):
        c, state, _ = client

        class Dying:
            def query_points(self, *a, **k):
                raise ConnectionError("qdrant went away")

        state.client = Dying()
        assert c.post("/api/v1/search", json={"query": "edema", "mode": "label"}).status_code == 200

    def test_unknown_policies_are_422(self, client):
        c, _, _ = client
        assert c.post("/api/v1/search", json={"query": "edema", "mode": "label",
                                              "absence_policy": "strict"}).status_code == 422
        assert c.post("/api/v1/search", json={"query": "edema", "mode": "label",
                                              "uncertainty_policy": "MAYBE"}).status_code == 422


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


def test_request_defaults_match_retrieval_yaml():
    cfg = load_config("retrieval")
    req = SearchRequest(query="x")
    assert req.absence_policy == cfg["constraints"]["absence_policy_default"]
    assert req.uncertainty_policy == cfg["constraints"]["uncertainty_policy_default"]
    assert req.top_k == cfg["results"]["default_top_k"]
