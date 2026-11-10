"""Knowledge for the explanation: the concept set, the gloss table, the K# items of a path and
the grounding of the knowledge part. Glosses and definitions here are written for the tests."""

from __future__ import annotations

import json

import polars as pl
import pytest

from src.common.config import load_config
from src.kg import expand, knowledge, llm_select
from src.kg.expand import PRESENT
from tests.kg import tiny_kg
from tests.kg.fake_store import FakeStore

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
    ok = {"en": "Fluid in the pleural space, seen as a blunted angle at the lung base.",
          "vi": "Dịch trong khoang màng phổi, thấy là góc sườn hoành tù."}
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

GLOSSES = {"C6": {"en": "Heart pump weakness; the film may show a large heart.",
                  "vi": "Tim bơm kém; phim có thể thấy bóng tim to."},
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
    assert texts[0] == "suy tim: Tim bơm kém; phim có thể thấy bóng tim to."
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


def test_every_path_edge_matches_the_path_expand_wrote():
    """The rebuilt edges walk the same concepts as intent.path (names joined by arrows)."""
    kb = base()
    for cui in ("C7", "C5"):
        intent = expand.expand_concepts(FakeStore(), cui, PRESENT, EXPAND, "x")
        names = [tiny_kg.NAMES[c] for rel, a, b, _ in kb.path_edges(intent) for c in (a, b)]
        assert set(names) == set(intent.path[0].replace(" ← ", "|").replace(" → ", "|")
                                 .split("|"))

