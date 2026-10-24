"""/search (one path), /health and /images, against a fake state."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

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


@pytest.fixture
def client_without_index(world, tmp_path):
    settings = world[0].model_copy(update={"fts_db_path": tmp_path / "missing.sqlite"})
    return TestClient(create_app(_state(world, settings=settings)))


class TestSearch:
    def test_returns_ranked_images_with_urls_and_nothing_from_the_labels(self, client):
        c, state, _ = client
        body = c.post("/api/v1/search", json={"query": "pleural effusion", "top_k": 3}).json()
        assert body["build_id"] == state.manifest["build_id"]
        parsed = body["parsed_query"]
        assert parsed["include"] == [{"concept": "Pleural Effusion", "status": "PRESENT"}]
        assert parsed["parser"] == "rule" and parsed["fallback_used"] is False
        assert body["query_text"] == "pleural effusion"
        assert [r["rank"] for r in body["results"]] == [1, 2, 3]
        first = body["results"][0]
        assert set(first) == {"rank", "study_id", "image_id", "image_url", "score",
                              "report_evidence"}
        assert first["image_url"] == f"/api/v1/images/{first['image_id']}"
        assert first["study_id"] == state.images[first["image_id"]].study_id
        assert isinstance(first["score"], float) and isinstance(first["report_evidence"], list)
        assert set(body) == {"request_id", "build_id", "parsed_query", "query_text", "results",
                             "warnings", "latency_ms"}
        assert body["warnings"] == []
        assert isinstance(body["latency_ms"], int) and len(body["request_id"]) == 32

    def test_a_request_with_a_mode_is_422_and_says_why(self, client):
        c, _, _ = client
        r = c.post("/api/v1/search", json={"query": "edema", "mode": "vector"})
        assert r.status_code == 422
        message = str(r.json()["detail"])
        assert "mode" in message and "BUILD_ID" in message

    @pytest.mark.parametrize("field", ["absence_policy", "uncertainty_policy", "colour"])
    def test_any_other_unknown_field_is_422(self, client, field):
        c, _, _ = client
        assert c.post("/api/v1/search", json={"query": "edema", field: "x"}).status_code == 422

    def test_query_is_stripped_and_blank_query_is_rejected(self, client):
        c, _, _ = client
        ok = c.post("/api/v1/search", json={"query": "  edema  "}).json()
        assert ok["query_text"] == "pulmonary edema"
        assert c.post("/api/v1/search", json={"query": "   "}).status_code == 422

    def test_top_k_bounds(self, client):
        c, _, _ = client
        assert c.post("/api/v1/search", json={"query": "x", "top_k": 51}).status_code == 422
        assert c.post("/api/v1/search", json={"query": "x", "top_k": 0}).status_code == 422
        assert len(c.post("/api/v1/search", json={"query": "x"}).json()["results"]) == 8

    def test_view_filter_and_empty_views(self, client):
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

    def test_two_phrasings_of_one_constraint_read_the_same_template(self, client):
        c, _, _ = client
        a = c.post("/api/v1/search", json={"query": "pleural effusion"}).json()
        b = c.post("/api/v1/search", json={"query": "Any effusion, please"}).json()
        assert a["query_text"] == b["query_text"] == "pleural effusion"
        assert [r["image_id"] for r in a["results"]] == [r["image_id"] for r in b["results"]]

    def test_vietnamese_query_without_a_concept_searches_nothing(self, client):
        c, _, _ = client
        body = c.post("/api/v1/search", json={"query": "vùng mờ gần rốn phổi"}).json()
        assert body["results"] == [] and body["query_text"] is None
        assert "no concept and no free text; nothing searched" in body["warnings"]

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

    def test_llm_dying_mid_flight_falls_back_in_search(self, world):
        settings = world[0].model_copy(update={"llm_disabled": False})
        fake = FakeOllama("ok", [llm.LLMError("ConnectError: gone")])
        c = TestClient(create_app(_state(world, settings=settings,
                                         llm_factory=lambda *a, **k: fake)))
        body = c.post("/api/v1/search", json={"query": "edema"}).json()
        assert body["parsed_query"]["parser"] == "rule"
        assert body["parsed_query"]["fallback_used"] is True
        assert body["warnings"] == ["llm unreachable: ConnectError: gone"]
        assert body["results"]


def _search(c, query, **extra):
    return c.post("/api/v1/search", json={"query": query, **extra}).json()


def _concepts_and_polarities(result):
    return [(e["concept"], e["polarity"]) for e in result["report_evidence"]]


class TestEvidence:
    def test_quotes_each_constrained_concept_with_its_polarity(self, client):
        c, _, _ = client
        body = _search(c, "edema with no pleural effusion", top_k=8)
        by_study = {r["study_id"]: r for r in body["results"]}
        assert _concepts_and_polarities(by_study["patient00000/study1"]) == [
            ("Edema", "affirmed"), ("Pleural Effusion", "negated")]
        # a hit whose report affirms the excluded finding shows it in the quote
        assert _concepts_and_polarities(by_study["patient00001/study1"]) == [
            ("Edema", "affirmed"), ("Pleural Effusion", "affirmed")]

    def test_free_text_quotes_the_matching_sentence(self, client):
        c, _, _ = client
        body = _search(c, "unchanged", top_k=8)
        by_study = {r["study_id"]: r for r in body["results"]}
        assert by_study["patient00002/study1"]["report_evidence"] == [
            {"concept": None, "text": "1. Lines and tubes unchanged.", "start": 22, "end": 51,
             "polarity": "mention"}]
        assert by_study["patient00000/study1"]["report_evidence"] == []

    def test_without_the_report_index_the_same_ranking_comes_back_unquoted(
            self, client, client_without_index):
        c, _, _ = client
        quoted = _search(c, "edema", top_k=8)
        bare = _search(client_without_index, "edema", top_k=8)
        assert client_without_index.get("/api/v1/health").json()["report_index"] == "disabled"
        assert [r["image_id"] for r in bare["results"]] == [
            r["image_id"] for r in quoted["results"]]
        assert all(r["report_evidence"] == [] for r in bare["results"])
        assert bare["warnings"] == ["report index unavailable; no evidence"]

    def test_a_study_without_a_report_comes_back_with_empty_evidence(self, world):
        """The index lacks one study's report (meta still counts four): that hit is ranked as
        usual, with no evidence, no error and no warning; the others are still quoted."""
        settings = world[0]
        with sqlite3.connect(settings.fts_db_path) as con:
            con.execute("DELETE FROM reports WHERE study_id = 'patient00001/study1'")
        c = TestClient(create_app(_state(world)))
        body = _search(c, "edema with no pleural effusion", top_k=8)
        by_study = {r["study_id"]: r["report_evidence"] for r in body["results"]}
        assert len(body["results"]) == 8 and body["warnings"] == []
        assert by_study["patient00001/study1"] == []
        assert [(e["concept"], e["polarity"]) for e in by_study["patient00000/study1"]] == [
            ("Edema", "affirmed"), ("Pleural Effusion", "negated")]

    def test_a_failing_report_index_never_fails_search(self, client):
        c, state, _ = client
        state.report_db.close()
        body = _search(c, "edema")
        assert body["results"] and "report index unavailable; no evidence" in body["warnings"]

    def test_parallel_requests_share_the_connection(self, client):
        c, _, _ = client
        with ThreadPoolExecutor(8) as ex:
            responses = list(ex.map(
                lambda _: c.post("/api/v1/search", json={"query": "edema"}), range(40)))
        assert [r.status_code for r in responses] == [200] * 40
        # A sqlite error degrades to a warning with a 200, so the status alone proves nothing.
        for body in (r.json() for r in responses):
            assert "report index unavailable; no evidence" not in body["warnings"]
            assert any(r["report_evidence"] for r in body["results"])


class TestHealth:
    def test_health_names_the_encoder_and_the_report_index(self, client):
        c, _, _ = client
        h = c.get("/api/v1/health").json()
        assert h["encoder"] == "fake"
        assert h["report_index"] == "ok" and h["report_studies"] == 4
        assert "image_probe" not in h

    def test_health_without_the_report_index_is_still_ready(self, client_without_index):
        r = client_without_index.get("/api/v1/health")
        assert r.status_code == 200 and r.json()["report_studies"] is None


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


def test_request_default_top_k_matches_retrieval_yaml():
    assert SearchRequest(query="x").top_k == load_config("retrieval")["results"]["default_top_k"]
