"""Knowledge for the explanation: the concept set, the gloss table, the K# items of a path and
the grounding of the knowledge part. Glosses and definitions here are written for the tests."""

from __future__ import annotations

import json

import polars as pl
import pytest

from src.common.config import load_config
from src.kg import expand, knowledge, llm_select
from src.kg.expand import PRESENT
from src.kg.guard import Guard
from src.kg.llm_query import Explainer, build_facts
from tests.kg import tiny_kg
from tests.kg.fake_store import FakeStore
from tests.kg.search_kit import Chat, index

CFG = load_config("kg")
EXPAND = CFG["query"]["expand"] | {"min_n": CFG["graph"]["min_n"]}
KCFG = CFG["knowledge"]
DIAGNOSTIC = CFG["explain"]["diagnostic"]


def graph() -> knowledge.Graph:
    return knowledge.Graph(tiny_kg.edges())


def test_the_concept_set_is_the_columns_the_routed_diseases_and_the_regions_of_a_column():
    got = dict(knowledge.concept_set(tiny_kg.nodes(), graph(), tiny_kg.quality(), EXPAND)
               .select("cui", "route").iter_rows())
    assert {c for c, r in got.items() if r == "column"} == set(tiny_kg.COLUMNS)
    assert got["C7"] == "wider"         # a scorable concept below it
    assert got["C6"] == "suggests"      # suggested by columns that are not weak
    assert got["C14"] == "suggests"     # by a manual edge
    assert got["C5"] == "narrower"      # a core disease below a column
    assert got["C8"] == "anatomy" and got["C9"] == "anatomy"   # C1 is typical of C9, part of C8
    assert "C12" not in got and "C13" not in got                # findings without a column


def test_a_narrower_route_counts_only_for_core_diseases():
    nodes = tiny_kg.nodes().with_columns(
        pl.when(pl.col("cui") == "C5").then(False).otherwise(pl.col("in_core")).alias("in_core"))
    got = knowledge.concept_set(nodes, graph(), tiny_kg.quality(), EXPAND)
    assert "C5" not in set(got["cui"])


def test_a_reply_must_answer_every_concept_of_the_batch_once():
    item = {"input": {"concepts": [{"cui": "C1"}, {"cui": "C2"}]}}
    one = {"cui": "C1", "en": "a", "vi": "b", "note": ""}
    with pytest.raises(llm_select.LLMSelectError):
        knowledge.parse_glosses(json.dumps({"glosses": [one]}), item)
    got = knowledge.parse_glosses(json.dumps({"glosses": [{**one, "cui": "C2"}, one]}), item)
    assert [g["cui"] for g in got["glosses"]] == ["C1", "C2"]


def test_flags_catch_a_long_a_copied_and_a_diagnostic_gloss():
    definition = ["Fluid that gathers in the space between the two layers around the lung."]
    ok = {"en": "Liquid between the pleural layers that blunts the angle at the lung base.",
          "vi": "Chất lỏng tụ quanh phổi, góc đáy phổi mất nhọn."}
    assert knowledge.flags(ok, definition, KCFG, DIAGNOSTIC) == ""
    long = {**ok, "en": " ".join(["word"] * 26)}
    assert knowledge.flags(long, definition, KCFG, DIAGNOSTIC) == "too_long_en"
    copy = {**ok, "en": "Fluid that gathers in the space between the two layers."}
    assert knowledge.flags(copy, definition, KCFG, DIAGNOSTIC) == "copied_en"
    diag = {**ok, "vi": "Hình ảnh giúp chẩn đoán tràn dịch."}
    assert knowledge.flags(diag, definition, KCFG, DIAGNOSTIC) == "diagnostic_vi"


def _table(verdict: str, adjudicated: dict | None = None) -> dict:
    concepts = pl.DataFrame({"cui": ["C1"], "name": ["Pleural effusion"], "label": ["Finding"],
                             "route": ["column"]})
    defs = pl.DataFrame(schema={"cui": pl.String, "sab": pl.String, "definition": pl.String})
    glosses = {"C1": {"cui": "C1", "en": "Fluid around the lung.", "vi": "Dịch quanh phổi.",
                      "note": ""}}
    verdicts = {"C1": {"cui": "C1", "verdict": verdict, "reason": "r"}}
    row = knowledge.table(concepts, defs, glosses, verdicts, adjudicated or {}, KCFG,
                          DIAGNOSTIC).row(0, named=True)
    return row


def test_a_gloss_is_used_when_both_models_agree_or_the_chat_accepts_it():
    agreed = _table("agree")
    assert agreed["usable"] and agreed["source"] == "llm"
    assert agreed["final_vi"] == "Dịch quanh phổi."
    assert not _table("disagree")["usable"]
    fixed = _table("disagree", {"C1": {"decision": "accept", "fixed_vi": "Dịch trong khoang."}})
    assert fixed["usable"] and fixed["final_vi"] == "Dịch trong khoang."
    assert not _table("agree", {"C1": {"decision": "reject"}})["usable"]


def test_the_review_csv_keeps_a_decision_and_puts_the_disagreements_first():
    concepts = pl.DataFrame({"cui": ["C1", "C2"], "name": ["a", "b"], "label": ["Finding"] * 2,
                             "route": ["column"] * 2})
    defs = pl.DataFrame(schema={"cui": pl.String, "sab": pl.String, "definition": pl.String})
    glosses = {c: {"cui": c, "en": "x y.", "vi": "x y.", "note": ""} for c in ("C1", "C2")}
    verdicts = {"C1": {"verdict": "agree", "reason": ""}, "C2": {"verdict": "disagree",
                                                               "reason": "too broad"}}
    adjudicated = {"C2": {"decision": "reject", "note": "chat"}}
    table = knowledge.table(concepts, defs, glosses, verdicts, adjudicated, KCFG, DIAGNOSTIC)
    review = knowledge.review_rows(table, adjudicated)
    assert review["cui"].to_list() == ["C2", "C1"]
    assert review.row(0, named=True)["decision"] == "reject"


# ---- K# items -----------------------------------------------------------------------------------

GLOSSES = {"C6": {"en": "Weak heart pumping, often with a big heart shadow.",
                  "vi": "Tim bơm yếu, trên phim bóng tim lớn."},
           "C2": {"en": "A heart shadow wider than usual.", "vi": "Bóng tim rộng hơn bình thường."}}


def base(max_items: int = 10) -> knowledge.KnowledgeBase:
    names = {"C2": {"vi": "tim to"}, "C3": {"vi": "phù phổi"}, "C6": {"vi": "suy tim"}}
    return knowledge.KnowledgeBase(GLOSSES, graph(), names, dict(tiny_kg.NAMES), EXPAND,
                                   max_items)


def test_a_disease_gives_its_gloss_the_suggests_edges_with_p_and_the_glosses_of_its_signs():
    intent = expand.expand_concepts(FakeStore(), "C6", PRESENT, EXPAND, "suy tim")
    items = base().items([intent], "vi")
    texts = [k["text"] for k in items]
    assert [k["id"] for k in items] == [f"K{i}" for i in range(1, len(items) + 1)]
    assert texts[0] == "suy tim: Tim bơm yếu, trên phim bóng tim lớn."
    assert "tim to thường được ghi nhận cùng suy tim trong report, p = 0,30." in texts
    assert "phù phổi thường được ghi nhận cùng suy tim trong report, p = 0,10." in texts
    assert texts[-1] == "tim to: Bóng tim rộng hơn bình thường."
    assert base(max_items=2).items([intent], "vi") == items[:2]


def test_the_other_routes_give_is_a_typically_at_and_part_of_sentences():
    kb = base()
    wider = expand.expand_concepts(FakeStore(), "C7", PRESENT, EXPAND, "x")
    narrower = expand.expand_concepts(FakeStore(), "C5", PRESENT, EXPAND, "x")
    region = expand.expand_concepts(FakeStore(), "C8", PRESENT, EXPAND, "x")
    manual = expand.expand_concepts(FakeStore(), "C14", PRESENT, EXPAND, "x")
    assert [k["text"] for k in kb.items([wider], "en")] == ["Pleural effusion is a kind of "
                                                            "Pleural disease."]
    assert [k["text"] for k in kb.items([narrower], "en")] == ["Lobar pneumonia is a kind of "
                                                               "Pneumonia."]
    assert [k["text"] for k in kb.items([region], "en")] == [
        "Pleural effusion is usually seen in the region Pleural space.",
        "Pleural space is part of Pleura."]
    assert any("added by hand" in k["text"] for k in kb.items([manual], "en"))


def test_a_label_is_named_by_its_label_and_another_concept_by_its_longest_vietnamese_alias(
        tmp_path):
    aliases = pl.concat([tiny_kg.aliases(), pl.DataFrame(
        [{"cui": "C6", "text": "st", "norm": "st", "lang": "vi", "source": "llm"}])])
    kb = knowledge.KnowledgeBase.load(tmp_path / "none.parquet", tiny_kg.edges(), aliases,
                                      tiny_kg.nodes(), EXPAND, 10,
                                      {"C1": {"en": "Pleural Effusion", "vi": "tdm"}})
    assert kb.glosses == {}                                  # no definitions yet: edges only
    assert (kb.name("C1", "en"), kb.name("C1", "vi")) == ("Pleural Effusion", "tdm")
    assert kb.name("C6", "vi") == "suy tim" and kb.name("C6", "en") == "Heart failure"

def test_every_path_edge_matches_the_path_expand_wrote():
    """The rebuilt edges walk the same concepts as intent.path (names joined by arrows)."""
    kb = base()
    for cui in ("C7", "C5"):
        intent = expand.expand_concepts(FakeStore(), cui, PRESENT, EXPAND, "x")
        names = [tiny_kg.NAMES[c] for rel, a, b, _ in kb.path_edges(intent) for c in (a, b)]
        assert set(names) == set(intent.path[0].replace(" ← ", "|").replace(" → ", "|")
                                 .split("|"))


# ---- the knowledge part of the explanation ------------------------------------------------------

def _page():
    intents = [expand.expand_concepts(FakeStore(), "C6", PRESENT, EXPAND, "suy tim")]
    rows = [{"rank": 1, "view": "PA"}]
    probs = [{0: {"intent": 0.8, "columns": {c.cui: 0.8 for c in intents[0].columns}}}]
    items = base().items(intents, "vi")
    return intents, items, build_facts("suy tim", "vi", intents, rows, probs, knowledge=items)


def _explain(knowledge_text: str):
    reply = {"summary": "Câu hỏi tìm suy tim.", "images": [{"rank": 1, "line": "Khớp 80%."}],
             "knowledge": knowledge_text}
    intents, items, facts = _page()
    return Explainer(Chat(reply), Guard(3, 60), "explain", 1.0, index()).explain(facts, intents,
                                                                                 items)


def test_a_cited_sentence_is_kept_and_its_citations_counted():
    got = _explain("Tim to thường được ghi nhận cùng suy tim, p = 0,30 [K2]. "
                   "Suy tim là tim bơm yếu [K1][K2].")
    assert got.status == "ok" and got.citations == 3
    assert got.knowledge.startswith("Tim to") and got.knowledge_items[0]["id"] == "K1"


@pytest.mark.parametrize("sentence, why", [
    ("Tim to hay đi cùng suy tim.", "no citation"),
    ("Tim to hay đi cùng suy tim [K9].", "citation not in the knowledge: K9"),
    ("Tim to đi cùng suy tim, p = 0,55 [K2].", "number not in the cited items: 0,55"),
    ("Suy tim hay kèm viêm phổi [K1].", "concept not in the cited items: viêm phổi"),
    ("Phim giúp chẩn đoán suy tim [K1].", "diagnostic wording: chẩn đoán"),
])
def test_a_knowledge_sentence_without_a_valid_citation_or_with_outside_content_is_dropped(
        sentence, why):
    got = _explain(sentence)
    assert got.knowledge == "" and got.status == "partial"
    assert got.dropped == [{"text": sentence, "why": why, "part": "knowledge"}]


def test_without_knowledge_the_facts_and_the_reply_are_those_of_prompt_v3():
    intents = [expand.expand_concepts(FakeStore(), "C1", PRESENT, EXPAND, "x")]
    facts = build_facts("x", "en", intents, [{"rank": 1, "view": "PA"}],
                        [{0: {"intent": 0.9, "columns": {"C1": 0.9}}}])
    assert "knowledge" not in facts
    chat = Chat({"summary": "Pleural effusion fits 90%.", "images": []})
    got = Explainer(chat, Guard(3, 60), "explain", 1.0, index()).explain(facts, intents)
    assert got.knowledge_items is None and got.citations == 0


def test_the_search_sends_the_items_of_its_path_and_the_event_carries_them():
    from src.api.routes.kg import explanation_event
    from tests.kg.search_kit import make_search, parse_reply

    reply = {"summary": "Câu hỏi tìm tim to.", "images": [],
             "knowledge": "Tim to là bóng tim rộng hơn bình thường [K1]."}
    chat = Chat(reply)
    search = make_search(parser_chat=Chat(parse_reply(include=["Cardiomegaly"])),
                         explain_chat=chat)
    search.explainer.knowledge = base()
    result = search.search("tim to", top_k=2)
    got = search.explain(result)
    sent = json.loads(chat.calls[0][1]["content"])
    assert sent["knowledge"] == [{"id": "K1", "text": "tim to: Bóng tim rộng hơn bình thường."}]
    event = explanation_event(result, got)
    assert event["knowledge"] == reply["knowledge"] and event["citations"] == 1
    assert event["knowledge_items"] == sent["knowledge"]
    search.explainer.knowledge = None                       # explain.knowledge: false (v3)
    assert "knowledge" not in explanation_event(result, search.explain(result))
