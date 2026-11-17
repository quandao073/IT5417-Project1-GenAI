"""The checks added after the layer-4 pair comparison (WORKLOG P22), on sentences of the 24
explanations with knowledge that Opus found less faithful (dev, session D)."""

from __future__ import annotations

from src.common.config import load_config
from src.kg.llm_query import (
    build_facts,
    combine_problem,
    hedge_problem,
    range_problem,
    ranking_problem,
)

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


# -- comparing and ranking (demo review 2026-11-17) ----------------------------------------------

RANKING = load_config("kg")["explain"]["ranking_words"]
# "suy tim": the intent's whole percent per rank; ranks 2 and 7 tie at 80 %, rank 5 is the top
HF = [78, 80, 75, 79, 82, 70, 80, 77, 76, 74]


def hf_facts(percents=HF) -> dict:
    return {"results": [{"rank": r, "view": "AP", "match": [
        {"intent": 1, "percent": p, "concepts": [{"concept": "Cardiomegaly", "percent": 60 + r}]}]}
        for r, p in enumerate(percents, start=1)]}


def test_the_suy_tim_sentence_is_dropped_rank_7_ties_rank_2():
    s = "Ảnh hạng 2 và hạng 5 có độ khớp cao nhất (80% và 82%)."
    assert ranking_problem(s, hf_facts(), RANKING) == "max claim not true in the facts"


def test_a_true_top_claim_is_kept_in_both_languages():
    assert ranking_problem("Ảnh hạng 5 có độ khớp cao nhất (82%).", hf_facts(), RANKING) is None
    assert ranking_problem("Rank 5 has the highest fit, 82%.", hf_facts(), RANKING) is None
    assert ranking_problem("The highest fit is 82%.", hf_facts(), RANKING) is None


def test_a_tie_must_be_named_whole():
    ties = [80, 82, 75, 79, 82, 70, 80, 77, 76, 74]          # ranks 2 and 5 share the top
    assert ranking_problem("Hạng 5 khớp cao nhất (82%).", hf_facts(ties), RANKING)
    assert ranking_problem("Hạng 2 và hạng 5 khớp cao nhất (82%).", hf_facts(ties), RANKING) is None
    assert ranking_problem("Hạng 2 và hạng 7 có độ khớp ngang nhau (80%).", hf_facts(),
                           RANKING) is None
    assert ranking_problem("Hạng 2 và hạng 5 có độ khớp bằng nhau.", hf_facts(), RANKING)


def test_lowest_and_comparisons_follow_the_facts():
    assert ranking_problem("Rank 6 has the lowest fit (70%).", hf_facts(), RANKING) is None
    assert ranking_problem("Rank 1 has the lowest fit.", hf_facts(), RANKING)
    assert ranking_problem("Hạng 5 khớp cao hơn hạng 2.", hf_facts(), RANKING) is None
    assert ranking_problem("Hạng 2 khớp cao hơn hạng 5.", hf_facts(), RANKING)
    assert ranking_problem("Rank 2 fits higher than rank 7.", hf_facts(), RANKING)   # equal
    # another quantity of the facts can make it true: Cardiomegaly is 70 % at rank 10, the top
    top = "Rank 10 has the highest Cardiomegaly (70%)."
    assert ranking_problem(top, hf_facts(), RANKING) is None


def test_a_sentence_without_ranking_words_is_not_checked_and_one_with_nothing_to_check_drops():
    assert ranking_problem("Mười ảnh đứng đầu đều khớp suy tim.", hf_facts(), RANKING) is None
    assert ranking_problem("Ảnh này khớp tốt nhất.", hf_facts(), RANKING)


def test_the_facts_carry_the_two_terms_of_a_fused_score():
    from src.kg.expand import Column, Intent

    it = Intent("suy tim", "C1", "Heart failure", "PRESENT", "disease", "weighted",
                [Column("C2", "Cardiomegaly", 1.0, 0.7, False, False, "Cardiomegaly")])
    rows = [{"rank": 1, "view": "AP", "kg_z": 1.934, "clip_part": 0.196},
            {"rank": 2, "view": "PA"}]
    probs = [{0: {"intent": 0.8, "columns": {"C2": 0.8}}}] * 2
    results = build_facts("suy tim", "vi", [it], rows, probs)["results"]
    assert results[0]["rank_score"] == {"kg_z": 1.93, "clip": 0.2}
    assert "rank_score" not in results[1]
    assert ranking_problem("Rank 1 has the highest kg_z.", {"results": results}, RANKING) is None
