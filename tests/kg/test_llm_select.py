"""The V2-3 selection step (D56): schema, validation, cache, both clients, without a network."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from src.kg import llm_select as ls
from src.kg import sapbert

ITEM = {"kind": "Observation", "string": "effusion",
        "input": {"string": "effusion", "kind": "Observation", "mentions": 5,
                  "located_at": [["pleural", 4]],
                  "candidates": [{"cui": "C0032227", "name": "Pleural effusion"},
                                 {"cui": "C0013687", "name": "Effusion"}]}}


def test_the_schema_is_strict_for_both_apis():
    assert ls.CHOICE_SCHEMA == {
        "type": "object", "additionalProperties": False,
        "required": ["decision", "cui", "reason"],
        "properties": {"decision": {"enum": ["cui", "skip", "out_of_scope", "none"],
                                    "type": "string"},
                       "cui": {"type": "string"}, "reason": {"type": "string"}}}


def test_a_cui_outside_the_candidates_is_rejected():
    allowed = {"C0032227"}
    assert ls.validate('{"decision": "cui", "cui": "C0032227", "reason": "r"}', allowed) == {
        "decision": "cui", "cui": "C0032227", "reason": "r"}
    assert ls.validate('{"decision": "cui", "cui": "C0000001", "reason": "r"}', allowed)[
        "decision"] == "rejected"
    assert ls.validate('{"decision": "skip", "cui": "C0032227", "reason": "r"}', allowed) == {
        "decision": "skip", "cui": "", "reason": "r"}
    with pytest.raises(ls.LLMSelectError):
        ls.validate('{"decision": "maybe", "cui": "", "reason": "r"}', allowed)


def test_context_and_inputs_carry_strings_counts_and_candidates_only():
    occ = pl.DataFrame({"obs": ["effusion"] * 3 + ["opacity"],
                        "anat": ["pleural"] * 2 + ["lung"] * 2})
    context = ls.located_context(occ)
    assert context[("Observation", "effusion")] == [["pleural", 2], ["lung", 1]]
    assert context[("Anatomy", "lung")] == [["effusion", 1], ["opacity", 1]]
    links = pl.DataFrame({"kind": ["Observation", "Observation"], "string": ["effusion", "normal"],
                          "mentions": [3, 9], "route": ["exact", None]})
    cands = pl.DataFrame({"kind": ["Observation"], "string": ["effusion"], "rank": [1],
                          "cui": ["C0013687"], "cosine": [1.0], "exact": [True]})
    info = pl.DataFrame({"cui": ["C0013687"], "name": ["Effusion"],
                         "semantic_types": ["T046 Pathologic Function"], "sabs": ["MSH,NCI"]})
    items = ls.inputs(links, cands, info, context)
    assert [i["string"] for i in items] == ["effusion"]           # rule-decided strings stay home
    assert items[0]["input"] == {
        "string": "effusion", "kind": "Observation", "mentions": 3,
        "located_at": [["pleural", 2], ["lung", 1]],
        "candidates": [{"cui": "C0013687", "name": "Effusion",
                        "semantic_types": "T046 Pathologic Function", "sources": "MSH,NCI",
                        "cosine": 1.0, "exact_name": True}]}


class Scripted:
    name, model = "fake", "m1"

    def __init__(self, replies):
        self.replies, self.asked = list(replies), []

    def ask(self, system, user, schema):
        self.asked.append(json.loads(user)["string"])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, {"input_tokens": 50, "output_tokens": 5}, "m1"


def test_answers_are_cached_errors_are_not(tmp_path):
    other = {**ITEM, "string": "edema", "input": {**ITEM["input"], "string": "edema"}}
    cache = ls.Cache(tmp_path / "c.jsonl")
    bot = Scripted([ls.LLMSelectError("HTTP 500"),
                    '{"decision": "cui", "cui": "C0032227", "reason": "pleural"}'])
    stats = ls.run([ITEM, other], bot, "sys", "p" * 64, cache, limit=None, workers=1)
    assert (stats["called"], stats["answered"], stats["errors"], stats["remaining"]) == (
        2, 1, ["HTTP 500"], 1)
    assert stats["input_tokens"] == 50                    # usage of the answered call only
    reloaded = ls.Cache(tmp_path / "c.jsonl")
    assert ls.choices([ITEM, other], reloaded, "m1", "p" * 64) == {
        ("Observation", "edema"): {"decision": "cui", "cui": "C0032227", "reason": "pleural"}}
    bot = Scripted(['{"decision": "none", "cui": "", "reason": "x"}'])
    ls.run([ITEM, other], bot, "sys", "p" * 64, reloaded, limit=5, workers=1)
    assert bot.asked == ["effusion"]                      # only the failed one is asked again
    assert ls.choices([ITEM], reloaded, "m2", "p" * 64) == {}     # another model, another key


def test_anthropic_request_uses_output_format_and_no_fallback():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=stop.pop(0), model="claude-x",
            content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="{}")],
            usage=SimpleNamespace(input_tokens=200, output_tokens=40))

    stop = ["end_turn", "refusal"]
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    cfg = {"model": "claude-x", "effort": "medium", "max_tokens": 100, "timeout_seconds": 5}
    chooser = ls.AnthropicChooser("opus", cfg, "k", client=client)
    assert chooser.ask("sys", "user", ls.CHOICE_SCHEMA) == (
        "{}", {"input_tokens": 200, "output_tokens": 40}, "claude-x")
    sent = calls[0]
    assert sent["output_config"] == {"effort": "medium", "format": {
        "type": "json_schema", "schema": ls.CHOICE_SCHEMA}}
    assert sent["system"] == "sys" and "fallbacks" not in sent
    with pytest.raises(ls.Refused) as refused:
        chooser.ask("sys", "user", ls.CHOICE_SCHEMA)
    assert refused.value.usage == {"input_tokens": 200, "output_tokens": 40}


def test_a_refusal_is_cached_as_none_refused(tmp_path):
    cache = ls.Cache(tmp_path / "c.jsonl")
    bot = Scripted([ls.Refused({"input_tokens": 7, "output_tokens": 0})])
    stats = ls.run([ITEM], bot, "sys", "p" * 64, cache, limit=None, workers=1)
    assert (stats["answered"], stats["refused"], stats["input_tokens"]) == (1, 1, 7)
    assert ls.choices([ITEM], cache, "m1", "p" * 64)[("Observation", "effusion")] == {
        "decision": "none", "cui": "", "reason": "refused"}


def test_nearest_returns_the_k_best_allowed_rows_across_chunks():
    vectors = np.eye(5, dtype=np.float32)
    query = np.array([[0.1, 0.5, 0.2, 0.9, 0.4]], dtype=np.float32)
    allowed = np.array([True, True, True, False, True])
    rows, scores = sapbert.nearest(query, vectors, allowed, k=3, chunk=2)
    assert rows.tolist() == [[1, 4, 2]]                   # row 3 scores best but is not allowed
    assert np.allclose(scores, [[0.5, 0.4, 0.2]])
