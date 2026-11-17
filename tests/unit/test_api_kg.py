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


def test_each_result_carries_the_probability_of_each_intent_and_column(client):
    c, _ = client
    first = ndjson(c.post("/api/v1/search/kg", json={"query": "tim to", "top_k": 3}))[0]
    intent = first["intents"][0]
    for r in first["results"]:
        (entry,) = r["intents"]
        assert entry["intent"] == 0 and 0 <= entry["p"] <= 1
        assert [col["cui"] for col in entry["columns"]] == [c["cui"] for c in intent["columns"]]


def test_intents_and_columns_carry_the_vietnamese_name_when_the_kg_has_one(client):
    c, state = client
    cui = ndjson(c.post("/api/v1/search/kg", json={"query": "tim to"}))[0]["intents"][0]["cui"]
    state.kg.names_vi = {cui: "tim to"}
    intent = ndjson(c.post("/api/v1/search/kg", json={"query": "tim to"}))[0]["intents"][0]
    assert intent["name_vi"] == "tim to" and intent["columns"][0]["name_vi"] == "tim to"


def test_a_page_ranked_by_text_has_no_column_numbers(client):
    from types import SimpleNamespace

    from src.api.routes.kg import result_scores

    result = SimpleNamespace(trace={"fallback": {"route": "text"}}, rows=[{"row": 0}, {"row": 1}])
    assert result_scores(None, result) == [[], []]


def test_vi_names_prefers_hand_made_then_concepts_yaml_then_the_shortest():
    import polars as pl

    from src.kg.runtime import vi_names

    aliases = pl.DataFrame({
        "cui": ["C1", "C1", "C1", "C2", "C2", "C3"],
        "text": ["dài hơn nữa", "ngắn", "tay", "một hai", "một", "heart"],
        "lang": ["vi", "vi", "vi", "vi", "vi", "en"],
        "source": ["concepts_yaml", "llm", "manual", "llm", "llm", "mrconso"]})
    assert vi_names(aliases) == {"C1": "tay", "C2": "một"}


def test_vi_ascii_phrases_keep_only_vietnamese_forms_that_are_no_english_word():
    import polars as pl

    from src.kg.runtime import vi_ascii_phrases

    aliases = pl.DataFrame({
        "norm": ["suy tim", "tràn dịch", "copd", "sheath catheter", "catheter", "copd", "tdmp"],
        "lang": ["vi", "vi", "vi", "vi", "en", "en", "vi"]})
    assert vi_ascii_phrases(aliases) == {"suy tim", "tdmp"}
