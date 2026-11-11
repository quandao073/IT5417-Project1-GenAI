"""Metrics, judge agreement, layer 2 and the B2 ablation transforms of the KG evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import kg_eval as ke
from src.evaluation import kg_eval_run as kr
from src.kg.expand import ABSENT, NOISY_OR, PRESENT, WEIGHTED, Column, Intent
from src.kg.scoring import ScoreMatrix


def col(cui, weight=1.0):
    return Column(cui=cui, column=cui, weight=weight, quality=0.8, weak=False, borderline=False)


def test_ndcg_uses_the_whole_judged_pool_as_the_ideal():
    assert ke.ndcg([2, 2, 0], [2, 2, 0, 0], 10) == pytest.approx(1.0)
    assert ke.ndcg([0, 2], [2, 0], 10) < 1.0
    assert ke.ndcg([0, 0], [0, 0], 10) == 0.0


def test_query_metrics_count_strict_and_lenient_relevance_and_the_view_rule():
    grades = {"a": 2, "b": 1, "c": 0, "d": 2}
    judged = {s: {"grade": g, "violates_exclusion": s == "c"} for s, g in grades.items()}
    views = {"a": "AP", "b": "PA", "c": "PA", "d": "PA"}
    m = ke.query_metrics(["a", "b", "c"], judged.get, lambda s: views[s] == "PA",
                         ke.gains(list(grades), judged.get, lambda s: views[s] == "PA"), 10, True)
    assert m["p10"] == 0.0 and m["p10_lenient"] == pytest.approx(0.1)   # a is off the view
    assert m["excl_viol10"] == pytest.approx(0.1) and not m["no_result"]
    empty = ke.query_metrics([], judged.get, lambda s: True, [2], 10, False)
    assert empty["no_result"] and empty["excl_viol10"] is None and empty["ndcg10"] == 0.0


def test_kappa_is_one_on_agreement_and_zero_by_chance():
    assert ke.cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == pytest.approx(1.0)
    assert ke.cohen_kappa([0, 0, 2, 2], [0, 2, 0, 2]) == pytest.approx(0.0)
    x, y = [0, 1, 2], [0, 2, 2]
    assert ke.cohen_kappa(x, y, weighted=True) > ke.cohen_kappa(x, y)


def test_the_bootstrap_interval_brackets_the_mean_difference():
    d = ke.bootstrap_mean_diff([0.5, 0.7, 0.9, 0.4], [0.4, 0.5, 0.6, 0.6], 2000, seed=1)
    assert d["lo"] <= d["mean"] <= d["hi"] and (d["wins"], d["losses"]) == (3, 1)


def test_layer2_compares_signed_column_sets_and_chosen_concepts():
    row = {"terms": [{"text": "x", "cui": "C1", "polarity": PRESENT},
                     {"text": "y", "cui": "C2", "polarity": ABSENT},
                     {"text": "fever", "cui": "", "polarity": PRESENT}],
           "expected_scorable": [{"cui": "C1", "weak": False}], "exclude": ["C2"],
           "expect_warnings": ["unrecognized"]}
    intents = [Intent("x", "C1", "x", PRESENT, "direct", columns=[col("C1")]),
               Intent("y", "C3", "y", ABSENT, "direct", columns=[col("C2")])]
    got = ke.layer2(row, intents, {"unrecognized", "weak_quality"})
    assert (got["cui_hits"], got["cui_total"]) == (1, 2)            # C2 was not the chosen CUI
    assert got["columns_exact"] and got["columns_f1"] == 1.0
    assert got["warnings_all"] and got["warnings_extra"] == ["weak_quality"]


def test_transforms_turn_one_expansion_off_equalise_suggests_and_mark_max():
    intents = [Intent("a", "A", "a", PRESENT, "wider", NOISY_OR, [col("C1"), col("C2")]),
               Intent("d", "D", "d", PRESENT, "disease", WEIGHTED,
                      [col("C1", 0.75), col("C3", 0.25)])]
    assert ke.transform(intents, "no_wider")[0].columns == []
    assert intents[0].columns                                       # the original is untouched
    assert [c.weight for c in ke.transform(intents, "suggests_equal")[1].columns] == [0.5, 0.5]
    assert ke.transform(intents, "combine_max")[0].combine == "max"


def test_max_combination_scores_by_the_largest_column():
    m = ScoreMatrix(["C1", "C2"], np.array([[3.0, -3.0, 0.0], [-3.0, 3.0, 0.0]]),
                    ["s1", "s2", "s3"], ["i1", "i2", "i3"], np.array(["AP", "AP", "PA"]))
    i = Intent("a", "A", "a", PRESENT, "wider", "max", [col("C1"), col("C2")])
    scores = ke.score_with_max(m, [i])
    assert scores[0] == pytest.approx(scores[1]) and scores[0] > scores[2]
    assert ke.rescore(m, [i], 2, ("PA",)) == ["s3"]
    assert ke.rescore(m, [Intent("n", "N", "n", PRESENT, "none")], 2, ()) == []


def test_the_pool_is_the_union_of_every_run_and_an_intent_survives_its_dict():
    runs = {"B1": {"q": {"studies": ["a", "b"]}}, "B2": {"q": {"studies": ["b", "c"]}}}
    assert kr.pool(runs, [{"id": "q"}], 10) == {"q": ["a", "b", "c"]}
    i = Intent("a", "A", "a", PRESENT, "wider", NOISY_OR, [col("C1")])
    back = kr.intent_from_dict(i.to_dict())
    assert back.columns == i.columns and back.combine == NOISY_OR


def test_the_corpus_top_honours_views_and_an_allow_list():
    c = ke.Corpus(ke.unit_rows(np.eye(3, dtype=np.float32)), ["i1", "i2", "i3"],
                  ["s1", "s2", "s3"], np.array(["AP", "PA", "PA"]))
    assert c.top(np.array([1, 0.5, 0.2]), 2) == [0, 1]
    assert c.top(np.array([1, 0.5, 0.2]), 2, ("PA",)) == [1, 2]
    assert c.top(np.array([1, 0.5, 0.2]), 2, (), np.array([False, False, True])) == [2]
