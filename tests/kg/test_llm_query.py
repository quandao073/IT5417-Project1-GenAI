"""Guard (timeout, circuit breaker), ConceptChooser and Explainer (grounding) of the KG search."""

from __future__ import annotations

import time

import polars as pl
import pytest

from src.common.config import load_config
from src.kg import expand, lookup
from src.kg.expand import PRESENT
from src.kg.guard import Guard
from src.kg.llm_query import ConceptChooser, Explainer, GroundingIndex, build_facts
from src.query.llm_parser import LLMError, LLMInvalidOutput
from tests.kg import tiny_kg
from tests.kg.fake_store import FakeStore
from tests.kg.search_kit import Chat, index

QCFG = load_config("kg")["query"]
EXPAND = {**QCFG["expand"], "weak_or_children": ["C10"]}


def raising(exc):
    def fn():
        raise exc
    return fn


# ---- guard -----------------------------------------------------------------------------------

def test_a_hanging_call_ends_at_the_limit_and_a_5xx_or_connection_error_is_an_llm_error():
    guard = Guard(3, 60)
    started = time.perf_counter()
    with pytest.raises(LLMError, match="timeout"):
        guard.call(lambda: time.sleep(2), 0.1)
    assert time.perf_counter() - started < 0.6
    with pytest.raises(LLMError, match="503"):
        guard.call(raising(LLMError("HTTP 503")), 1)
    with pytest.raises(LLMError, match="ConnectionError"):
        guard.call(raising(ConnectionError("refused")), 1)


def test_three_failures_in_a_row_open_the_circuit_and_it_closes_after_the_cooldown():
    now = [0.0]
    guard = Guard(3, 60, clock=lambda: now[0])
    calls = []

    def boom():
        calls.append(1)
        raise LLMError("down")

    for _ in range(3):
        with pytest.raises(LLMError, match="down"):
            guard.call(boom, 1)
    assert guard.is_open
    with pytest.raises(LLMError, match="circuit open"):
        guard.call(boom, 1)
    assert len(calls) == 3 and guard.stats["short_circuited"] == 1
    now[0] = 61
    assert not guard.is_open and guard.call(lambda: "ok", 1) == "ok"


def test_a_success_resets_the_count_and_an_invalid_reply_is_not_an_outage():
    guard = Guard(2, 60)
    for _ in range(5):
        with pytest.raises(LLMInvalidOutput):
            guard.call(raising(LLMInvalidOutput("bad json")), 1)
    assert not guard.is_open
    with pytest.raises(LLMError):
        guard.call(raising(LLMError("x")), 1)
    guard.call(lambda: 1, 1)
    with pytest.raises(LLMError):
        guard.call(raising(LLMError("x")), 1)
    assert not guard.is_open


# ---- concept choice --------------------------------------------------------------------------

def found(term):
    return lookup.lookup_concept(FakeStore(), term, QCFG)


def chooser(chat, timeout=1.0):
    return ConceptChooser(chat, Guard(3, 60), "choose", timeout)


def test_one_exact_candidate_needs_no_model_and_no_candidate_means_none():
    chat = Chat({"choice": "C1", "reason": ""})
    got = chooser(chat).choose("q", "tràn dịch màng phổi", PRESENT, found("tràn dịch màng phổi"))
    assert (got.cui, got.via, chat.calls) == ("C1", "single", [])
    assert chooser(chat).choose("q", "xyz", PRESENT, found("xyz qwerty")).via == "none"


def test_the_model_sees_the_whole_question_and_picks_one_of_the_candidates():
    chat = Chat({"choice": "C13", "reason": "both lungs"})
    got = chooser(chat).choose("giảm thể tích phổi hai bên", "giảm thể tích phổi", PRESENT,
                               found("giảm thể tích phổi"))
    assert (got.cui, got.via) == ("C13", "llm")
    assert "giảm thể tích phổi hai bên" in chat.calls[0][1]["content"]


@pytest.mark.parametrize("reply", [{"choice": "C99", "reason": ""}, "not json",
                                   LLMError("HTTP 500"), 5])
def test_an_answer_outside_the_list_or_a_failure_falls_back_to_the_first_candidate(reply):
    got = chooser(Chat(reply), 0.2).choose("q", "giảm thể tích phổi", PRESENT,
                                           found("giảm thể tích phổi"))
    assert (got.cui, got.via) == ("C12", "deterministic") and got.error


def test_the_model_may_say_none_of_the_candidates_fits():
    got = chooser(Chat({"choice": "none", "reason": "different thing"})).choose(
        "q", "giảm thể tích phổi", PRESENT, found("giảm thể tích phổi"))
    assert (got.cui, got.via) == (None, "llm")


# ---- explanation grounding -------------------------------------------------------------------

def page():
    """Intent: pleural effusion (C1); two result rows with their percents."""
    intents = [expand.expand_concepts(FakeStore(), "C1", PRESENT, EXPAND, "tràn dịch màng phổi")]
    rows = [{"rank": 1, "view": "PA"}, {"rank": 2, "view": "AP"}]
    probs = [{0: {"intent": 0.91, "columns": {"C1": 0.91}}},
             {0: {"intent": 0.64, "columns": {"C1": 0.64}}}]
    return intents, build_facts("tràn dịch màng phổi", "vi", intents, rows, probs)


def explainer(reply):
    return Explainer(Chat(reply), Guard(3, 60), "explain", 1.0, index())


GOOD = {"summary": "Câu hỏi tìm tràn dịch màng phổi. Ảnh hạng 1 khớp 91%.",
        "images": [{"rank": 1, "line": "Khớp 91% với tràn dịch màng phổi."},
                   {"rank": 2, "line": "Khớp 64% với tràn dịch màng phổi."}]}


def test_a_grounded_explanation_is_kept_whole():
    intents, facts = page()
    got = explainer(GOOD).explain(facts, intents)
    assert got.status == "ok" and got.dropped == [] and set(got.images) == {1, 2}
    assert got.first_token_ms == 12.0


def test_a_sentence_with_an_unknown_number_or_concept_or_diagnosis_is_dropped():
    intents, facts = page()
    reply = {"summary": "Câu hỏi tìm tràn dịch màng phổi. Có 4821 ảnh phù hợp. "
                        "Ảnh hạng 1 cũng gợi ý pneumonia. Bệnh nhân bị tràn dịch.",
             "images": [{"rank": 1, "line": "Khớp 91%."}, {"rank": 9, "line": "Khớp 91%."}]}
    got = explainer(reply).explain(facts, intents)
    assert got.status == "partial" and got.summary == "Câu hỏi tìm tràn dịch màng phổi."
    why = {d["text"]: d["why"] for d in got.dropped}
    assert why["Có 4821 ảnh phù hợp."].startswith("number not in the facts")
    assert why["Ảnh hạng 1 cũng gợi ý pneumonia."] == "concept not in the facts: pneumonia"
    assert why["Bệnh nhân bị tràn dịch."].startswith("diagnostic wording")
    assert any(d.get("rank") == 9 and d["why"] == "rank not in the results"
               for d in got.dropped) and got.images == {1: "Khớp 91%."}


def test_nothing_grounded_is_reported_so_the_caller_shows_the_raw_path():
    intents, facts = page()
    reply = {"summary": "Bệnh nhân bị suy tim.", "images": []}
    assert explainer(reply).explain(facts, intents).status == "ungrounded"


@pytest.mark.parametrize("reply", [LLMError("HTTP 529"), "not json", {"summary": 1}])
def test_a_failed_call_is_reported_as_failed(reply):
    intents, facts = page()
    got = explainer(reply).explain(facts, intents)
    assert got.status == "failed" and got.error


def test_markup_that_leaked_into_the_text_is_not_an_explanation():
    intents, facts = page()
    reply = {"summary": "Câu hỏi tìm tràn dịch màng phổi.</summary><parameter name=\"images\">[]",
             "images": []}
    got = explainer(reply).explain(facts, intents)
    assert got.status == "ungrounded" and got.dropped[0]["why"] == "markup in the text"


def test_a_generic_phrase_of_two_words_is_not_a_mention_of_a_concept():
    nodes = tiny_kg.nodes().filter(pl.col("cui") == "C3")
    aliases = pl.DataFrame({"cui": ["C3", "C3"], "norm": ["bất thường", "phù phổi nặng"]})
    cfg = {**load_config("kg")["explain"], "generic_words": ["bất thường"]}
    got = GroundingIndex.build(nodes, aliases, cfg)
    assert "bất thường" not in got.aliases and "phù phổi nặng" in got.aliases
