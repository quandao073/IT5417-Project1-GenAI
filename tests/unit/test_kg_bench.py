"""The KG benchmark builder's pure parts: rules of a group, pair ids, the split, the ledger."""

from __future__ import annotations

import pytest

from src.evaluation import kg_bench as kb
from src.evaluation import kg_llm


def q(lang, translation_of=-1, cui="C1", views=(), text="x", unscorable=False):
    terms = [{"text": "a", "cui": cui, "polarity": "PRESENT"}]
    if unscorable:
        terms.append({"text": "fever", "cui": "", "polarity": "PRESENT"})
    return {"question": text, "lang": lang, "translation_of": translation_of, "query_en": text,
            "views": list(views), "terms": terms, "expected_scorable": [{"cui": cui, "weak": False}],
            "exclude": [], "expect_warnings": [], "note": ""}


def good_group():
    return [q("en", text="a1", views=["PA"]), q("vi", 0, text="a2"), q("en", text="b1"),
            q("vi", 2, text="b2"), q("en", text="c", unscorable=True), q("vi", text="d"),
            q("en", text="e"), q("vi", text="f"), q("en", text="g")]


def test_a_group_that_keeps_the_rules_has_no_problem_and_breaks_are_named():
    assert kb.structure_problems(good_group(), 5, 4, {"C1"}, set()) == []
    bad = good_group()
    bad[1]["translation_of"] = 3                      # points at a question of its own language
    bad[5]["terms"][0]["cui"] = "C9"
    found = kb.structure_problems(bad, 5, 4, {"C1"}, {"g"})
    assert any("translation_of 3" in p for p in found)
    assert any("C9" in p for p in found) and any("reuses" in p for p in found)


def test_both_questions_of_a_pair_carry_the_id_of_the_first():
    rows = kb.assign_ids("control", good_group())
    assert rows[0]["pair"] == rows[1]["pair"] == "kg-control-01"
    assert rows[4]["pair"] is None and rows[2]["pair"] == "kg-control-03"


def test_the_split_keeps_pairs_together_and_alternates_the_dev_share():
    rows = kb.assign_ids("a", good_group()) + kb.assign_ids("b", good_group())
    got = kb.split(rows, [5, 4], seed=1)
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        if r["pair"]:
            assert got[r["id"]] == got[r["pair"]]
    dev_a = [i for i in got if i.startswith("kg-a") and got[i] == "dev"]
    dev_b = [i for i in got if i.startswith("kg-b") and got[i] == "dev"]
    assert (len(dev_a), len(dev_b)) == (5, 4)
    en = sum(by_id[i]["lang"] == "en" for i in dev_a)
    assert en in (2, 3)                                   # as close to half as 5 allows
    assert kb.split(rows, [5, 4], seed=1) == got          # the seed fixes it


def test_language_counts_alternate_to_half_and_half():
    counts = [kb.lang_counts(i, 9) for i in range(8)]
    assert sum(c[0] for c in counts) == sum(c[1] for c in counts) == 36


def test_the_ledger_prices_usage_and_stops_at_the_budget(tmp_path):
    pricing = {"m": {"input": 4.0, "output": 20.0, "cache_read": 0.2, "cache_write": 5.0}}
    ledger = kg_llm.Ledger(tmp_path / "l.jsonl", pricing, budget=1.0)
    cost = ledger.add("s", "m", {"input_tokens": 100_000, "output_tokens": 10_000})
    assert cost == pytest.approx(0.6)
    ledger.check()
    ledger.add("s", "m", {"output_tokens": 30_000})
    with pytest.raises(kg_llm.BudgetExceeded):
        ledger.check()
    again = kg_llm.Ledger(tmp_path / "l.jsonl", pricing, budget=1.0)   # read back from disk
    assert again.total == pytest.approx(1.2) and again.by("step") == {"s": 1.2}
