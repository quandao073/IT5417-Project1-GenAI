"""POST /api/v1/query/parse against a fake state."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.query import llm_parser as llm
from tests.unit.fakes import FakeOllama
from tests.unit.test_api_state import _state
from tests.unit.test_api_state import world as _world_fixture

world = _world_fixture


@pytest.fixture
def client(world):
    return TestClient(create_app(_state(world)))


def llm_client(world, *replies) -> TestClient:
    settings = world[0].model_copy(update={"llm_disabled": False})
    fake = FakeOllama("ok", replies)
    return TestClient(create_app(_state(world, settings=settings,
                                        llm_factory=lambda *a, **k: fake)))


def reply(**fields) -> str:
    base = {"findings": "", "include": [], "exclude": [], "views": [], "language": "en"}
    return json.dumps({**base, **fields})


class TestParseEndpoint:
    def test_returns_the_parsed_query_flat(self, client):
        body = client.post("/api/v1/query/parse",
                           json={"query": "Find edema with no pleural effusion"}).json()
        assert body["include"] == [{"concept": "Edema", "status": "PRESENT"}]
        assert body["exclude"] == ["Pleural Effusion"] and body["views"] == []
        assert body["language"] == "en" and body["parser"] == "rule"
        assert body["fallback_used"] is False and body["warnings"] == []
        assert body["semantic_query"] == "chest x-ray showing pulmonary edema"
        assert body["cached"] is False and body["llm_ms"] is None

    def test_vietnamese_use_case_six(self, client):
        q = "Tìm ảnh X-quang thẳng có tim to và phù phổi nhưng không có tràn dịch màng phổi."
        body = client.post("/api/v1/query/parse", json={"query": q}).json()
        assert [c["concept"] for c in body["include"]] == ["Cardiomegaly", "Edema"]
        assert body["exclude"] == ["Pleural Effusion"] and body["language"] == "vi"
        assert body["semantic_query"] == "chest x-ray showing cardiomegaly and pulmonary edema"

    def test_blank_and_overlong_queries_are_422(self, client):
        assert client.post("/api/v1/query/parse", json={"query": "   "}).status_code == 422
        assert client.post("/api/v1/query/parse", json={"query": "x" * 501}).status_code == 422

    def test_second_call_is_cached(self, client):
        client.post("/api/v1/query/parse", json={"query": "edema"})
        assert client.post("/api/v1/query/parse", json={"query": "edema"}).json()["cached"]

    def test_503_when_not_ready(self, world, tmp_path):
        settings = world[0].model_copy(update={"indexes_dir": tmp_path / "nowhere"})
        c = TestClient(create_app(_state(world, settings=settings)))
        r = c.post("/api/v1/query/parse", json={"query": "edema"})
        assert r.status_code == 503 and r.json()["detail"]["status"] == "not_ready"

    def test_llm_path_is_reported(self, world):
        c = llm_client(world, reply(include=[{"concept": "Edema", "status": "PRESENT"}]))
        body = c.post("/api/v1/query/parse", json={"query": "edema"}).json()
        assert body["parser"] == "llm" and body["llm_ms"] is not None

    def test_llm_failure_falls_back_and_says_so(self, world):
        c = llm_client(world, llm.LLMError("ReadTimeout: slow"))
        body = c.post("/api/v1/query/parse", json={"query": "edema"}).json()
        assert body["parser"] == "rule" and body["fallback_used"] is True
        assert body["warnings"] == ["llm unreachable: ReadTimeout: slow"]
        assert body["include"] == [{"concept": "Edema", "status": "PRESENT"}]

    def test_health_reports_llm_ok(self, world):
        c = llm_client(world)
        assert c.get("/api/v1/health").json()["llm"] == "ok"
