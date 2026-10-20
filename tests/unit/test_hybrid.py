"""Hybrid fusion: RRF over the vector and report legs, inside the label-allowed set."""

from __future__ import annotations

import pytest

from src.common.config import load_config
from src.retrieval.hybrid import FusionParams, _order_key, rrf, search_hybrid
from src.retrieval.image_search import ImageCandidate
from src.retrieval.report_search import ReportCandidate, ReportSearchError

P = FusionParams(rrf_k=60, leg_weights={"vector": 1.0, "report": 1.0},
                 vector_top_k=100, report_top_k=100)


def cands(study_ids):
    return [ImageCandidate(image_id=f"{s}.jpg", study_id=s, score=1.0 - i / 10, rank=i + 1)
            for i, s in enumerate(study_ids)]


def rcands(study_ids):
    return [ReportCandidate(study_id=s, score=5.0 - i, rank=i + 1)
            for i, s in enumerate(study_ids)]


def test_rrf_by_hand():
    s = rrf({"vector": ["a", "b"], "report": ["b", "c"]}, {"vector": 1.0, "report": 1.0}, 60)
    assert s == pytest.approx({"a": 1 / 61, "b": 1 / 62 + 1 / 61, "c": 1 / 62})


def test_rrf_weights_scale_each_leg():
    s = rrf({"vector": ["a"], "report": ["a"]}, {"vector": 2.0, "report": 0.5}, 60)
    assert s == pytest.approx({"a": 2.5 / 61})


def test_fusion_orders_ties_deterministically():
    out = search_hybrid(top_k=10, label_order=["a", "b", "c"],
                        vector_leg=lambda k: cands(["a", "b"]),
                        report_leg=lambda k: rcands(["b", "c"]), fusion=P)
    assert [r.study_id for r in out.results] == ["b", "a", "c"]
    assert out.results[0].vector_rank == 2 and out.results[0].report_rank == 1
    assert out.total_matches == 3 and out.legs == {"vector": 2, "report": 2}
    assert out.warnings == []


def test_single_leg_degenerates_to_its_order():
    out = search_hybrid(top_k=10, label_order=["a", "b", "c"],
                        vector_leg=lambda k: cands(["c", "a", "b"]), report_leg=None, fusion=P)
    assert [r.study_id for r in out.results] == ["c", "a", "b"]
    assert [r.report_rank for r in out.results] == [None] * 3
    assert out.warnings == [] and out.legs == {"vector": 3, "report": 0}


def test_top_k_truncates_and_ranks_are_contiguous():
    out = search_hybrid(top_k=2, label_order=None,
                        vector_leg=lambda k: cands(["a", "b", "c", "d"]), report_leg=None,
                        fusion=P)
    assert [r.rank for r in out.results] == [1, 2]
    assert [r.study_id for r in out.results] == ["a", "b"]
    assert out.total_matches is None


def test_vector_candidates_are_deduplicated_by_study():
    images = [ImageCandidate("a1", "a", 0.9, 1), ImageCandidate("b1", "b", 0.8, 2),
              ImageCandidate("a2", "a", 0.7, 3)]
    out = search_hybrid(top_k=10, label_order=["a", "b"], vector_leg=lambda k: images,
                        report_leg=None, fusion=P)
    assert [r.study_id for r in out.results] == ["a", "b"]
    assert out.results[0].vector_rank == 1 and out.results[1].vector_rank == 2


def test_no_signal_falls_back_to_label_order():
    out = search_hybrid(top_k=2, label_order=["x", "y", "z"], vector_leg=None, report_leg=None,
                        fusion=P)
    assert [r.study_id for r in out.results] == ["x", "y"] and out.results[0].score == 0.0
    assert out.results[0].vector_rank is None and out.results[0].report_rank is None
    assert [r.rank for r in out.results] == [1, 2] and out.total_matches == 3
    assert out.warnings == ["no ranking signal; results in label order"]


def test_both_legs_empty_also_falls_back_to_label_order():
    out = search_hybrid(top_k=5, label_order=["x"], vector_leg=lambda k: [],
                        report_leg=lambda k: [], fusion=P)
    assert [r.study_id for r in out.results] == ["x"]
    assert out.warnings == ["no ranking signal; results in label order"]


def test_no_signal_without_constraints_is_empty():
    assert search_hybrid(top_k=5, label_order=None, vector_leg=None, report_leg=None,
                         fusion=P).results == []


def test_report_leg_error_drops_the_leg_with_a_warning():
    def boom(k):
        raise ReportSearchError("locked")

    out = search_hybrid(top_k=5, label_order=["a"], vector_leg=lambda k: cands(["a"]),
                        report_leg=boom, fusion=P)
    assert [r.study_id for r in out.results] == ["a"]
    assert out.warnings == ["report leg failed: locked"] and out.legs["report"] == 0


def test_vector_leg_errors_are_not_caught():
    def boom(k):
        raise RuntimeError("qdrant down")

    with pytest.raises(RuntimeError):
        search_hybrid(top_k=5, label_order=None, vector_leg=boom, report_leg=None, fusion=P)


def test_results_never_leave_the_allowed_set():
    out = search_hybrid(top_k=10, label_order=["a", "b"],
                        vector_leg=lambda k: cands(["stray", "a", "b"]),
                        report_leg=lambda k: rcands(["b", "stray2"]), fusion=P)
    assert {r.study_id for r in out.results} == {"a", "b"}
    assert out.total_matches == 2 and out.legs == {"vector": 2, "report": 1}


def test_legs_that_only_return_strays_fall_back_to_label_order():
    out = search_hybrid(top_k=5, label_order=["a"], vector_leg=lambda k: cands(["stray"]),
                        report_leg=None, fusion=P)
    assert [r.study_id for r in out.results] == ["a"]
    assert out.warnings == ["no ranking signal; results in label order"]


def test_equal_scores_go_to_the_study_the_vector_leg_ranked():
    # "z" (vector only) and "b" (report only) tie at 1/61; the vector rule beats alphabetical.
    out = search_hybrid(top_k=5, label_order=None, vector_leg=lambda k: cands(["z"]),
                        report_leg=lambda k: rcands(["b"]), fusion=P)
    assert out.results[0].score == out.results[1].score
    assert [r.study_id for r in out.results] == ["z", "b"]


def test_order_key_tie_break_chain():
    # score first, then "has a vector rank", then the vector rank, then the study id
    keys = [_order_key("b", 0.5, None), _order_key("a", 0.5, None), _order_key("z", 0.5, 2),
            _order_key("y", 0.5, 1), _order_key("q", 0.9, None)]
    assert [k[-1] for k in sorted(keys)] == ["q", "y", "z", "a", "b"]


def test_legs_are_called_with_their_top_k():
    seen = {}
    fusion = FusionParams(60, {"vector": 1.0, "report": 1.0}, vector_top_k=7, report_top_k=9)

    def vector(k):
        seen["vector"] = k
        return cands(["a"])

    def report(k):
        seen["report"] = k
        return rcands(["a"])

    search_hybrid(top_k=3, label_order=["a"], vector_leg=vector, report_leg=report, fusion=fusion)
    assert seen == {"vector": 7, "report": 9}


def test_from_config_reads_retrieval_yaml():
    assert FusionParams.from_config(load_config("retrieval")) == FusionParams(
        60, {"vector": 1.0, "report": 1.0}, 100, 100)
