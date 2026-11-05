"""POST /api/v1/search/kg and the KG fields of /health, over the tiny KG."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from tests.kg.search_kit import Chat, make_search, parse_reply
from tests.unit.test_api_state import _state
from tests.unit.test_api_state import world as _world_fixture

world = _world_fixture

EXPLANATION = {"summary": "Câu hỏi tìm tim to.", "images": [{"rank": 1, "line": "Khớp tim to."}]}


def kg_client(world, **search_kwargs):
    state = _state(world)
    state.kg = make_search(**search_kwargs)
    return TestClient(create_app(state)), state


def ndjson(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line]


@pytest.fixture
def client(world):
    return kg_client(world, parser_chat=Chat(parse_reply(include=["Cardiomegaly"])),
                     explain_chat=Chat(EXPLANATION))


def test_the_results_event_comes_first_then_the_explanation(client):
    c, _ = client
    r = c.post("/api/v1/search/kg", json={"query": "tim to", "top_k": 3})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    first, second = ndjson(r)
    assert (first["event"], second["event"]) == ("results", "explanation")
    assert first["request_id"] == second["request_id"]
    assert [x["rank"] for x in first["results"]] == [1, 2, 3]
    assert first["results"][0]["image_url"].startswith("/api/v1/images/")
    assert first["intents"][0]["term"] == "Cardiomegaly" and first["disclaimer"]
    assert first["kg_build_id"] == "kg_test" and first["matrix_version"] == "m@0"
    assert {"parse_ms", "score_ms", "total_ms"} <= set(first["timings_ms"])
    assert second["status"] in ("ok", "partial") and second["summary"]


def test_stream_false_returns_one_object(client):
    c, _ = client
    body = c.post("/api/v1/search/kg", json={"query": "tim to", "stream": False}).json()
    assert body["event"] == "results" and body["explanation"]["event"] == "explanation"


def test_explain_false_skips_the_model_call(world):
    chat = Chat(EXPLANATION)
    c, _ = kg_client(world, parser_chat=Chat(parse_reply(include=["Cardiomegaly"])),
                     explain_chat=chat)
    second = ndjson(c.post("/api/v1/search/kg", json={"query": "tim to", "explain": False}))[1]
    assert second["status"] == "disabled" and second["fallback"] == "path" and chat.calls == []


def test_a_dead_explainer_still_returns_the_results_and_says_to_show_the_path(world):
    c, _ = kg_client(world, parser_chat=Chat(parse_reply(include=["Cardiomegaly"])),
                     explain_chat=Chat(ConnectionError("refused")))
    first, second = ndjson(c.post("/api/v1/search/kg", json={"query": "tim to"}))
    assert first["results"] and second["status"] == "failed" and second["fallback"] == "path"


def test_each_request_leaves_one_json_log_line_with_the_versions(client, caplog):
    c, _ = client
    with caplog.at_level(logging.INFO, logger="cxr.kg"):
        c.post("/api/v1/search/kg", json={"query": "tim to"})
    record = json.loads(next(r.getMessage() for r in caplog.records if r.name == "cxr.kg"))
    assert record["event"] == "kg_search" and record["kg_build_id"] == "kg_test"
    assert record["matrix_version"] == "m@0" and len(record["request_id"]) == 32
    assert "tim to" not in json.dumps(record) and record["explanation"]["status"]


@pytest.mark.parametrize("body", [{"query": " "}, {"query": "x", "top_k": 0},
                                  {"query": "x", "views": ["LATERAL"]},
                                  {"query": "x", "mode": "a"}])
def test_a_bad_request_is_422(client, body):
    assert client[0].post("/api/v1/search/kg", json=body).status_code == 422


def test_503_when_the_kg_runtime_is_missing(world):
    state = _state(world)
    state.kg, state.kg_reason = None, "score matrix missing"
    r = TestClient(create_app(state)).post("/api/v1/search/kg", json={"query": "tim to"})
    assert r.status_code == 503 and r.json()["detail"]["reason"] == "score matrix missing"


def test_health_reports_neo4j_and_the_kg_versions(client, world):
    c, state = client
    h = c.get("/api/v1/health").json()
    assert (h["neo4j"], h["kg_build_id"], h["matrix_version"]) == ("ok", "kg_test", "m@0")
    state.kg.store = None
    assert c.get("/api/v1/health").json()["neo4j"] == "unreachable"
    plain = TestClient(create_app(_state(world))).get("/api/v1/health").json()
    assert plain["neo4j"] == "disabled" and plain["kg_build_id"] is None
