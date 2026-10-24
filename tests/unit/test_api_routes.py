"""/search in its three modes, /health and /images, against a fake state."""

from __future__ import annotations

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
from tests.unit.test_report_index import ROWS

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
        # the fake encoder ranks study 3 first, whose effusion is NOT_MENTIONED: a violation
        assert first["score_breakdown"] == {"vector_rank": 1, "report_rank": None,
                                            "image_rank": None, "rrf_score": None,
                                            "label_match": None,
                                            "constraints_satisfied": False,
                                            "concept_scores": None}
        assert first["labels"][0]["status"] == "NOT_MENTIONED"
        assert isinstance(first["report_evidence"], list)
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
        assert r["score_breakdown"] == {"vector_rank": None, "report_rank": None,
                                        "image_rank": None, "rrf_score": None,
                                        "label_match": True,
                                        "constraints_satisfied": True,
                                        "concept_scores": None}
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

    def test_view_filter_applies_to_label_mode(self, client):
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


def _search(c, query, mode, **extra):
    return c.post("/api/v1/search", json={"query": query, "mode": mode, **extra}).json()


def _concepts_and_polarities(result):
    return [(e["concept"], e["polarity"]) for e in result["report_evidence"]]


class TestHybrid:
    def test_hybrid_returns_only_constraint_satisfying_studies(self, client):
        c, state, _ = client
        body = _search(c, "edema with no pleural effusion", "hybrid")
        ids = [r["study_id"] for r in body["results"]]
        # only study 0 has Edema PRESENT and Pleural Effusion not PRESENT
        assert ids == ["patient00000/study1"] and body["total_matches"] == 1
        r = body["results"][0]
        breakdown = r["score_breakdown"]
        assert breakdown["constraints_satisfied"] is True and breakdown["label_match"] is True
        assert breakdown["vector_rank"] == 1 and breakdown["report_rank"] == 1
        # both legs rank it first: 0.5/11 + 1/11 with rrf_k 10 and weights (0.5, 1)
        assert r["score"] == pytest.approx(1.5 / 11) and breakdown["rrf_score"] == r["score"]
        assert r["image_id"] == state.study_image["patient00000/study1"]
        assert _concepts_and_polarities(r) == [("Edema", "affirmed"),
                                               ("Pleural Effusion", "negated")]
        assert r["report_evidence"][1]["text"] == "No \npleural effusion."
        assert body["warnings"] == []

    def test_hybrid_small_allowed_set(self, client):
        c, _, _ = client
        body = _search(c, "cardiomegaly", "hybrid", top_k=10)
        assert [r["study_id"] for r in body["results"]] == ["patient00003/study1"]
        assert body["total_matches"] == 1

    def test_hybrid_with_an_empty_allowed_set_returns_nothing(self, client):
        c, _, _ = client
        body = _search(c, "pneumothorax", "hybrid")
        assert body["results"] == [] and body["total_matches"] == 0
        assert "no ranking signal; results in label order" not in body["warnings"]

    def test_hybrid_without_constraints_ranks_the_corpus(self, client):
        c, _, _ = client
        body = _search(c, "hiatal hernia", "hybrid")
        assert body["total_matches"] is None and len(body["results"]) == 4
        assert {r["study_id"] for r in body["results"]} == {
            f"patient{i:05d}/study1" for i in range(4)}
        breakdowns = [r["score_breakdown"] for r in body["results"]]
        assert [b["vector_rank"] for b in breakdowns] == [1, 2, 3, 4]
        assert all(b["report_rank"] is None and b["constraints_satisfied"] is None
                   and b["label_match"] is None for b in breakdowns)   # no label filter ran
        assert all(r["report_evidence"] == [] for r in body["results"])  # no report says it

    def test_hybrid_free_text_quotes_the_matching_sentence(self, client):
        c, _, _ = client
        body = _search(c, "unchanged", "hybrid")
        by_study = {r["study_id"]: r for r in body["results"]}
        r = by_study["patient00002/study1"]
        assert r["score_breakdown"]["report_rank"] == 1
        assert r["report_evidence"] == [{"concept": None, "text": "1. Lines and tubes unchanged.",
                                         "start": 22, "end": 51, "polarity": "mention"}]
        assert by_study["patient00000/study1"]["report_evidence"] == []

    def test_hybrid_vietnamese_exclude_only_uses_label_order(self, client):
        c, _, _ = client
        query = "không có tràn dịch màng phổi"
        body = _search(c, query, "hybrid")
        assert "no ranking signal; results in label order" in body["warnings"]
        assert "report leg skipped: no English term to match" in body["warnings"]
        assert "semantic query is empty; vector leg skipped" in body["warnings"]
        assert body["results"][0]["score"] == 0.0 and body["total_matches"] == 3
        label = _search(c, query, "label")
        assert ([r["study_id"] for r in body["results"]]
                == [r["study_id"] for r in label["results"]])

    def test_qdrant_failure_is_503_in_hybrid_too(self, client):
        c, state, _ = client

        class Dying:
            def query_points(self, *a, **k):
                raise ConnectionError("qdrant went away")

        state.client = Dying()
        r = c.post("/api/v1/search", json={"query": "edema", "mode": "hybrid"})
        assert r.status_code == 503 and "qdrant" in r.json()["detail"]["reason"]

    def test_parallel_hybrid_requests_share_the_connection(self, client):
        c, _, _ = client
        with ThreadPoolExecutor(8) as ex:
            responses = list(ex.map(
                lambda _: c.post("/api/v1/search", json={"query": "edema", "mode": "hybrid"}),
                range(40)))
        assert [r.status_code for r in responses] == [200] * 40
        # A sqlite error degrades to a warning with a 200, so the status alone proves nothing:
        # every response must have used the shared connection successfully.
        for body in (r.json() for r in responses):
            assert not any(w.startswith("report leg failed") for w in body["warnings"])
            assert "report index unavailable; no evidence" not in body["warnings"]
            assert body["results"] and all(
                r["score_breakdown"]["report_rank"] is not None and r["report_evidence"]
                for r in body["results"])


class TestEvidence:
    def test_vector_mode_flags_violations_and_shows_negated_evidence(self, client):
        c, _, _ = client
        body = _search(c, "edema with no pleural effusion", "vector", top_k=8)
        flags = {r["study_id"]: r["score_breakdown"]["constraints_satisfied"]
                 for r in body["results"]}
        assert flags["patient00001/study1"] is False and flags["patient00000/study1"] is True
        by_study = {r["study_id"]: r for r in body["results"]}
        assert _concepts_and_polarities(by_study["patient00000/study1"]) == [
            ("Edema", "affirmed"), ("Pleural Effusion", "negated")]
        # the violation shows in the quote: study 1's report affirms the effusion
        assert _concepts_and_polarities(by_study["patient00001/study1"]) == [
            ("Edema", "affirmed"), ("Pleural Effusion", "affirmed")]
        assert all(r["score_breakdown"]["rrf_score"] is None for r in body["results"])

    def test_vector_mode_without_constraints_has_no_flag(self, client):
        c, _, _ = client
        body = _search(c, "hiatal hernia", "vector")
        assert all(r["score_breakdown"]["constraints_satisfied"] is None
                   for r in body["results"])

    def test_label_mode_now_carries_evidence(self, client):
        c, _, _ = client
        body = _search(c, "edema", "label")
        by_study = {r["study_id"]: r["report_evidence"] for r in body["results"]}
        assert by_study == {
            "patient00000/study1": [{"concept": "Edema", "text": "Mild pulmonary edema.",
                                     "start": 0, "end": 21, "polarity": "affirmed"}],
            "patient00001/study1": [{"concept": "Edema", "polarity": "affirmed", "start": 0,
                                     "end": 60, "text": "Moderate pulmonary edema and a small "
                                                        "right pleural effusion."}]}

    def test_report_index_disabled_degrades_hybrid(self, client_without_index):
        c = client_without_index
        assert c.get("/api/v1/health").json()["report_index"] == "disabled"
        body = _search(c, "edema", "hybrid")
        assert body["results"]
        assert body["warnings"].count(
            "report index unavailable; hybrid ran without the report leg") == 1
        assert all(r["report_evidence"] == [] for r in body["results"])
        assert body["warnings"].count("report index unavailable; no evidence") == 1
        assert all(r["score_breakdown"]["report_rank"] is None for r in body["results"])
        label = _search(c, "edema", "label")
        assert label["total_matches"] == 2
        assert label["warnings"] == ["report index unavailable; no evidence"]

    def test_a_failing_report_index_never_fails_search(self, client):
        c, state, _ = client
        state.report_db.close()
        body = _search(c, "edema", "hybrid")
        assert len(body["results"]) == 2
        assert any(w.startswith("report leg failed: ") for w in body["warnings"])
        assert "report index unavailable; no evidence" in body["warnings"]


class TestImageMode:
    def test_image_mode_returns_image_rank_and_labels(self, client):
        c, _, _ = client
        body = _search(c, "edema without pleural effusion", "image", top_k=4)
        results = body["results"]
        assert [r["rank"] for r in results] == [1, 2, 3, 4]
        assert [r["score_breakdown"]["image_rank"] for r in results] == [1, 2, 3, 4]
        assert all(r["score_breakdown"]["vector_rank"] is None for r in results)
        assert all(isinstance(r["score"], float) for r in results)
        assert [a["concept"] for a in results[0]["labels"]] == ["Edema", "Pleural Effusion"]

    def test_image_mode_rejects_nothing_new_in_schema(self, client):
        c, _, _ = client
        assert c.post("/api/v1/search", json={"query": "edema", "mode": "image"}).status_code == 200
        assert c.post("/api/v1/search", json={"query": "edema", "mode": "graph"}).status_code == 422


class TestHealth:
    def test_health_reports_the_image_probe(self, client):
        c, _, _ = client
        h = c.get("/api/v1/health").json()
        assert h["image_probe"] == "ok" and h["image_probe_id"] == "p1"

    def test_health_without_the_probe_is_still_ready(self, world, tmp_path):
        settings = world[0].model_copy(update={"image_probe_dir": tmp_path / "nowhere"})
        r = TestClient(create_app(_state(world, settings=settings))).get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["image_probe"] == "disabled" and r.json()["image_probe_id"] is None

    def test_health_reports_the_report_index(self, client):
        c, _, _ = client
        h = c.get("/api/v1/health").json()
        assert h["report_index"] == "ok" and h["report_studies"] == 4

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


def test_request_defaults_match_retrieval_yaml():
    cfg = load_config("retrieval")
    req = SearchRequest(query="x")
    assert req.absence_policy == cfg["constraints"]["absence_policy_default"]
    assert req.uncertainty_policy == cfg["constraints"]["uncertainty_policy_default"]
    assert req.top_k == cfg["results"]["default_top_k"]


class TestImageModesRoute:
    def test_request_modes_match_retrieval_yaml(self):
        modes = SearchRequest.model_fields["mode"].annotation.__args__
        assert set(modes) == set(load_config("retrieval")["modes"])
        assert {"image_joint", "image_decomposed"} <= set(modes)

    def test_image_decomposed_returns_concept_scores(self, client):
        c, _, _ = client
        body = _search(c, "edema without pleural effusion", "image_decomposed", top_k=4)
        breakdown = body["results"][0]["score_breakdown"]
        assert set(breakdown["concept_scores"]) == {"Edema", "Pleural Effusion"}
        assert breakdown["image_rank"] == 1 and breakdown["vector_rank"] is None
        joint = _search(c, "edema without pleural effusion", "image_joint", top_k=4)
        assert joint["results"][0]["score_breakdown"]["concept_scores"] is None

    def test_search_result_has_projection(self, client):
        c, state, _ = client
        for mode in ("vector", "image_joint"):
            results = _search(c, "edema", mode, top_k=8)["results"]
            assert results and all(
                r["projection"] == state.images[r["image_id"]].projection for r in results)


class TestStudies:
    def test_studies_endpoint_returns_labels_and_report(self, client):
        c, state, _ = client
        study = "patient00000/study1"
        for path in (study, study.replace("/", "%2F")):
            r = c.get(f"/api/v1/studies/{path}")
            assert r.status_code == 200, path
            body = r.json()
            assert body["study_id"] == study and body["build_id"] == state.manifest["build_id"]
            assert body["image_id"] == state.study_image[study]
            assert body["projection"] == state.images[body["image_id"]].projection
            labels = {a["concept"]: a for a in body["labels"]}
            assert len(body["labels"]) == 14
            assert labels["Edema"] == {"concept": "Edema", "status": "PRESENT",
                                       "source": "CHEXBERT_IMPRESSION"}
            assert labels["Pleural Effusion"]["status"] == "ABSENT"
            assert labels["Pneumonia"]["status"] == "NOT_MENTIONED"
            assert body["report"] == {"impression": ROWS[0][1], "findings": None,
                                      "search_text": ROWS[0][1]}

    def test_studies_unknown_is_404(self, client):
        c, _, _ = client
        r = c.get("/api/v1/studies/patient99999/study9")
        assert r.status_code == 404
        assert r.json()["detail"] == {"reason": "unknown study patient99999/study9"}

    def test_studies_without_reports_parquet_has_no_report(self, client):
        c, state, _ = client
        (state.settings.canonical_dir / "reports.parquet").unlink()
        body = c.get("/api/v1/studies/patient00000/study1").json()
        assert body["report"] is None and len(body["labels"]) == 14


class TestCors:
    def test_the_web_origin_may_call_the_api(self, client):
        c, _, _ = client
        r = c.options("/api/v1/search", headers={
            "Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"})
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == "http://localhost:3000"

    def test_another_origin_gets_no_cors_header(self, client):
        c, _, _ = client
        r = c.get("/api/v1/health", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in r.headers
