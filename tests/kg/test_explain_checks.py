"""The checks added after the layer-4 pair comparison (WORKLOG P22), on sentences of the 24
explanations with knowledge that Opus found less faithful (dev, session D)."""

from __future__ import annotations

from src.common.config import load_config
from src.kg.llm_query import combine_problem, hedge_problem, range_problem

CFG = load_config("kg")["explain"]
WORDS, HEDGES = CFG["combine_words"], CFG["hedge_words"]


def facts(combines, results):
    """`results`: per rank, per intent (percent, {concept: percent})."""
    return {"intents": [{"id": i + 1, "combine": c} for i, c in enumerate(combines)],
            "results": [{"rank": r + 1, "view": "PA", "match": [
                {"intent": i + 1, "percent": p, "concepts": [
                    {"concept": k, "percent": v} for k, v in cs.items()]}
                for i, (p, cs) in enumerate(row)]} for r, row in enumerate(results)]}


SINGLE2 = facts(["single", "single"], [[(85, {"edema": 85}), (95, {"effusion": 95})],
                                       [(87, {"edema": 87}), (91, {"effusion": 91})]])


def test_a_way_of_combining_the_facts_do_not_have_is_dropped():
    # kg-control-02: "weighted" where both intents are single
    s = ('Truy vấn tìm ảnh PA có phù phổi và tràn dịch màng phổi, điểm được tính theo cách '
         '"weighted" (kết hợp hai khái niệm).')
    assert combine_problem(s, SINGLE2, WORDS) == "combine not in the facts: weighted"
    # kg-anatomy-13 (VI) and kg-new_column-14 (EN): any_of / "best fit among them" for single
    assert combine_problem("chấm theo kiểu any_of nên điểm là mức khớp tốt nhất.", SINGLE2,
                           WORDS) == "combine not in the facts: any_of"
    assert combine_problem("scoring atelectasis as wanted, using the best fit among them.",
                           SINGLE2, WORDS) == "combine not in the facts: any_of"


def test_a_way_of_combining_the_facts_have_is_kept():
    mixed = facts(["single", "weighted"], [[(87, {"c": 87}), (4, {"a": 1, "b": 4})]])
    # kg-negation_broad-03: the excluded intent is weighted, 33% each
    assert combine_problem("interstitial signs, weighted 33% each.", mixed, WORDS) is None
    assert combine_problem("Each intent is scored on its own.", SINGLE2, WORDS) is None
    any_w = facts(["any_of_weighted"], [[(50, {"a": 50})]])
    assert combine_problem("the best fit, each weighted by how typical it is.", any_w,
                           WORDS) is None


def test_a_range_that_is_not_the_lowest_and_highest_of_a_series_is_dropped():
    # kg-disease-14: heart-failure fit "74% to 87%" while ranks 5 and 7 are at 71% and 73%
    fits = [74, 80, 86, 87, 71, 85, 73, 82, 86, 79]
    f = facts(["single", "weighted"], [[(14, {"effusion": 14}), (p, {"cm": p})] for p in fits])
    assert range_problem("high heart-failure fit (74% to 87%).", f) == (
        "range not in the facts: 74-87%")
    assert range_problem("high heart-failure fit (71% to 87%).", f) is None
    # kg-anatomy-03 (VI wording) and a dash range
    g = facts(["single"], [[(p, {"ptx": p})] for p in (59, 55, 49)])
    assert range_problem("độ khớp từ 50% đến 59%.", g) == "range not in the facts: 50-59%"
    assert range_problem("độ khớp từ 49% đến 59%.", g) is None
    assert range_problem("fits of 49–59 %.", g) is None
    assert range_problem("fits of 49-58%.", g) == "range not in the facts: 49-58%"


def test_a_concept_series_inside_an_intent_counts_and_text_without_a_range_passes():
    f = facts(["weighted"], [[(30, {"a": 10, "b": 40})], [(20, {"a": 5, "b": 35})]])
    assert range_problem("signs from 5% to 10%.", f) is None
    assert range_problem("Rank 1 fits 30%; ranks 1 to 2 are PA.", f) is None


def test_a_knowledge_sentence_must_keep_the_hedge_of_its_item():
    # kg-negation_broad-04: K1 says "thường khi …", the sentence states it as a rule
    item = ("tim to: Bóng tim to, thường khi chiều ngang tim vượt quá một nửa đường kính trong "
            "lồng ngực trên phim X-quang ngực thẳng.")
    bare = ("Tim to là bóng tim có chiều ngang vượt quá một nửa đường kính trong lồng ngực "
            "trên phim X-quang ngực thẳng [K1].")
    assert hedge_problem(bare, [item], HEDGES) == "hedge of the cited item missing"
    kept = "Tim to thường là khi chiều ngang tim vượt quá một nửa lồng ngực [K1]."
    assert hedge_problem(kept, [item], HEDGES) is None
    en = "Cardiomegaly: an enlarged heart shadow, usually wider than half the chest."
    assert hedge_problem("Cardiomegaly is a heart wider than half the chest [K1].", [en],
                         HEDGES) == "hedge of the cited item missing"
    assert hedge_problem("Cardiomegaly is often a heart wider than half the chest [K1].", [en],
                         HEDGES) is None


def test_an_item_without_a_hedge_and_the_word_abnormal_need_none():
    item = "dày mô kẽ phổi: khung mô nâng đỡ của phổi dày lên, thấy là đường mờ bất thường."
    assert hedge_problem("Dày mô kẽ phổi là khung mô nâng đỡ dày lên [K2].", [item],
                         HEDGES) is None
