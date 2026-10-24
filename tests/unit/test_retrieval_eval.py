"""Known-answer tests for the retrieval evaluation (spec 11.3-11.5): relevance, grades,
metrics, the fusion grid and the verdict. A six-study toy world; `search` is a fake."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import load_config
from src.evaluation import retrieval_eval as rev
from src.evaluation.parser_benchmark import BenchmarkQuery, derive_groups
from src.query.parser import QueryParser
from src.query.rule_parser import RuleParser
from src.query.schema import ParseResult, SearchQuery
from src.retrieval import label_filter as lf
from src.retrieval.evidence import Evidence
from src.retrieval.hybrid import FusionParams
from src.retrieval.pipeline import Hit, SearchOutcome
from tests.unit.test_label_filter import labels_frame

PRIORITY = ("CHEXPERT_V1", "CHEXBERT_IMPRESSION", "CHEXBERT_FINDINGS", "CHEXBERT_REPORT")
STUDIES = ["s1", "s2", "s3", "s4", "s5", "s6"]
PROJECTION = {"s1": "AP", "s2": "PA", "s3": "AP", "s4": "AP", "s5": "PA", "s6": "AP"}
LABELS = [
    ("s1", "Edema", "PRESENT", "CHEXPERT_V1"),
    ("s1", "Edema", "PRESENT", "CHEXBERT_IMPRESSION"),
    ("s1", "Edema", "ABSENT", "CHEXBERT_REPORT"),             # a dissenting source
    ("s1", "Pleural Effusion", "ABSENT", "CHEXPERT_V1"),
    ("s2", "Edema", "PRESENT", "CHEXPERT_V1"),
    ("s2", "Edema", "PRESENT", "CHEXBERT_IMPRESSION"),
    ("s2", "Edema", "PRESENT", "CHEXBERT_REPORT"),
    ("s2", "Pleural Effusion", "ABSENT", "CHEXPERT_V1"),
    ("s2", "Pleural Effusion", "ABSENT", "CHEXBERT_IMPRESSION"),
    ("s3", "Edema", "PRESENT", "CHEXBERT_IMPRESSION"),
    ("s3", "Pleural Effusion", "PRESENT", "CHEXPERT_V1"),
    ("s4", "Edema", "UNCERTAIN", "CHEXPERT_V1"),
    ("s5", "Cardiomegaly", "PRESENT", "CHEXPERT_V1"),
    ("s6", "Edema", "PRESENT", "CHEXPERT_V1"),
    ("s6", "Edema", "NOT_MENTIONED", "CHEXBERT_IMPRESSION"),   # silence is not a vote
]
THRESHOLDS = {"simple_label_precision10_min": 0.70, "constraint_satisfaction_min": 0.95,
              "hybrid_ndcg10_relative_gain_min": 0.05}


def q(include=(), exclude=(), views=(), language="en") -> SearchQuery:
    return SearchQuery(language=language, views=list(views), exclude=list(exclude),
                       include=[{"concept": c, "status": s} for c, s in include])


def bq(id, query, gold, language="en", group="single") -> BenchmarkQuery:
    return BenchmarkQuery(id, "dev", language, group, derive_groups(gold), query, gold)


def ev(concept, polarity) -> Evidence:
    return Evidence(concept=concept, text="t", start=0, end=1, polarity=polarity)


def weights(vector, report, image=0.0) -> dict:
    return {"vector": vector, "report": report, "image": image}


@pytest.fixture(scope="module")
def matrix():
    return lf.build_label_matrix(labels_frame(LABELS), STUDIES, PRIORITY, seed=1)


@pytest.fixture(scope="module")
def agreement():
    # Reversed on purpose: rows must follow the ids given, not the frame or the label matrix.
    return rev.build_agreement_matrix(labels_frame(LABELS), list(reversed(STUDIES)))


@pytest.fixture(scope="module")
def parser():
    return QueryParser(RuleParser(load_config("concepts")), template="x {}")


@pytest.fixture
def relevance(matrix, agreement):
    def build(gold, views=(), absence="exclude_present", uncertainty="EXCLUDE"):
        return rev.relevance_for(gold, matrix, agreement, views=views, projection_of=PROJECTION,
                                 absence_policy=absence, uncertainty_policy=uncertainty)
    return build


@pytest.fixture
def rows():
    return [bq("en-001", "edema", q([("Edema", "PRESENT")])),
            bq("en-002", "edema without pleural effusion",
               q([("Edema", "PRESENT")], ["Pleural Effusion"]), group="negation"),
            bq("vi-001", "phù phổi tư thế PA", q([("Edema", "PRESENT")], views=["PA"],
                                                 language="vi"), language="vi", group="view"),
            bq("en-013", "is this study normal", q(), group="other")]


def fake(rankings):
    """A `search` with the pipeline's signature: `rankings(mode, fusion)` -> study ids; s1 is
    the only study quoted (Edema, affirmed). Every call is recorded on `.calls`."""
    calls = []

    def search(state, query, mode, *, top_k, views, absence_policy, uncertainty_policy,
               fusion=None):
        calls.append({"state": state, "query": query, "mode": mode, "top_k": top_k,
                      "views": views, "policy": (absence_policy, uncertainty_policy),
                      "fusion": fusion})
        hits = [Hit(rank, s, f"img-{s}", evidence=[ev("Edema", "affirmed")] if s == "s1" else [])
                for rank, s in enumerate(rankings(mode, fusion), start=1)]
        return SearchOutcome(hits, None, [])

    search.calls = calls
    return search


@pytest.fixture
def fake_search():
    fixed = {"label": ["s1", "s3"], "vector": ["s5", "s3", "s1"], "image": ["s2", "s6"],
             "hybrid": ["s2", "s1", "s6"]}
    return fake(lambda mode, fusion: fixed[mode])


def test_agreement_counts_sources_per_status():
    extra = [("zz", "Edema", "PRESENT", "CHEXPERT_V1")]       # not in the corpus: dropped
    a = rev.build_agreement_matrix(labels_frame(LABELS + extra), list(reversed(STUDIES)))
    assert a.counts.shape == (6, 14, 3) and a.counts.dtype == np.int8
    assert a.row_of["s6"] == 0 and a.row_of["s1"] == 5
    edema, pe = a.col_of["Edema"], a.col_of["Pleural Effusion"]
    # axis 2 is (PRESENT, UNCERTAIN, ABSENT)
    assert a.counts[a.row_of["s1"], edema].tolist() == [2, 0, 1]
    assert a.counts[a.row_of["s2"], edema].tolist() == [3, 0, 0]
    assert a.counts[a.row_of["s2"], pe].tolist() == [0, 0, 2]
    assert a.counts[a.row_of["s4"], edema].tolist() == [0, 1, 0]
    assert a.counts[a.row_of["s6"], edema].tolist() == [1, 0, 0]   # NOT_MENTIONED counts nowhere
    assert int(a.counts.sum()) == 14                               # 15 rows, one NOT_MENTIONED


def test_relevance_binary_matches_the_label_filter_and_applies_views(matrix, relevance):
    gold = q([("Edema", "PRESENT")])
    mask = lf.filter_studies_by_labels(matrix, gold.include, gold.exclude,
                                       absence_policy="exclude_present",
                                       uncertainty_policy="EXCLUDE")
    rel = relevance(gold)
    assert rel.relevant == frozenset(matrix.study_ids[mask]) == {"s1", "s2", "s3", "s6"}
    assert relevance(gold, views=["AP"]).relevant == {"s1", "s3", "s6"}
    assert relevance(gold, views=["PA"]).relevant == {"s2"}
    assert relevance(gold, views=["AP", "PA"]).relevant == rel.relevant
    # the policy is the label filter's: INCLUDE admits s4 (Edema UNCERTAIN), no source says PRESENT
    loose = relevance(gold, uncertainty="INCLUDE")
    assert loose.relevant == {"s1", "s2", "s3", "s4", "s6"} and loose.grade["s4"] == 1.0


def test_grade_is_one_plus_mean_agreement(relevance):
    gold = q([("Edema", "PRESENT")], ["Pleural Effusion"])
    rel = relevance(gold)
    # s1: 1 + (2 Edema PRESENT + 1 PE ABSENT) / 2; s2: 1 + (3 + 2) / 2; s6: 1 + (1 + 0) / 2.
    # s3 (PE PRESENT) is not relevant, so it is absent from the grades.
    assert rel.relevant == {"s1", "s2", "s6"}
    assert rel.grade == {"s1": 2.5, "s2": 3.5, "s6": 1.5}
    assert relevance(gold, absence="explicit_absence_required").grade == {"s1": 2.5, "s2": 3.5}
    assert relevance(q([("Edema", "UNCERTAIN")])).grade == {"s4": 2.0}    # 1 + 1 UNCERTAIN


def test_ndcg_by_hand():
    grade = {"a": 3.0, "b": 1.0, "c": 2.0}
    assert rev.ndcg_at(["b", "a"], grade, 10) == pytest.approx(
        (1 / 1 + 3 / np.log2(3)) / (3 / 1 + 2 / np.log2(3) + 1 / np.log2(4)))
    assert rev.ndcg_at(["b", "a", "c"], grade, 1) == pytest.approx(1 / 3)   # cut at k
    assert rev.ndcg_at([], grade, 10) == 0.0 and rev.ndcg_at(["z"], {}, 10) == 0.0


def test_query_metrics_divide_by_k_and_count_violations(matrix, relevance):
    gold = q([("Edema", "PRESENT")], ["Pleural Effusion"])
    rel = relevance(gold)                                # grades s1 2.5, s2 3.5, s6 1.5
    m = rev.query_metrics(["s1", "s3"], rel, gold, matrix)
    assert m["p@5"] == 0.2 and m["p@10"] == 0.1 and m["returned@10"] == 0.2
    assert m["mrr"] == 1.0 and m["exclude_violation@10"] == 0.5    # s3 has PE PRESENT
    assert m["ndcg@10"] == pytest.approx(2.5 / (3.5 + 2.5 / np.log2(3) + 1.5 / np.log2(4)))
    assert m["ndcg_binary@10"] == pytest.approx(1 / (1 + 1 / np.log2(3) + 1 / np.log2(4)))
    assert rev.query_metrics(["s3", "s6"], rel, gold, matrix)["mrr"] == 0.5
    assert rev.query_metrics([], rel, gold, matrix)["exclude_violation@10"] is None
    plain = q([("Edema", "PRESENT")])
    assert rev.query_metrics([], relevance(plain), plain, matrix) == {
        "p@5": 0.0, "p@10": 0.0, "ndcg@10": 0.0, "ndcg_binary@10": 0.0, "mrr": 0.0,
        "exclude_violation@10": None, "returned@10": 0.0, "hnrr@10": None}


def test_query_metrics_hnrr_at_10(matrix, relevance):
    """The share of what came back that has the include AND the excluded finding (s3)."""
    gold = q([("Edema", "PRESENT")], ["Pleural Effusion"])
    rel = relevance(gold)
    assert rev.query_metrics(["s1", "s3"], rel, gold, matrix)["hnrr@10"] == 0.5
    assert rev.query_metrics(["s1", "s2"], rel, gold, matrix)["hnrr@10"] == 0.0
    assert rev.query_metrics([], rel, gold, matrix)["hnrr@10"] is None
    include_only = q([("Edema", "PRESENT")])
    assert rev.query_metrics(["s3"], relevance(include_only), include_only,
                             matrix)["hnrr@10"] is None
    exclude_only = q(exclude=["Pleural Effusion"])
    assert rev.query_metrics(["s3"], relevance(exclude_only), exclude_only,
                             matrix)["hnrr@10"] is None


def test_mode_comparison_on_common_queries():
    def entry(rows):
        return {"per_query": {i: {"ndcg_binary@10": n, "returned@10": r, "hnrr@10": h}
                              for i, (n, r, h) in rows.items()}}

    report = {"parsers": {"gold": {"modes": {
        "image_joint": entry({"a": (0.8, 1.0, 0.1), "b": (0.5, 1.0, None), "c": (0.0, 0.0, None),
                              "d": (0.2, 1.0, 0.3)}),
        "image": entry({"a": (0.6, 1.0, 0.3), "b": (0.5, 1.0, None), "c": (0.4, 1.0, 0.0),
                        "d": (0.4, 1.0, 0.1)}),
        "vector": entry({"a": (0.1, 1.0, 0.5), "b": (0.2, 0.0, None), "c": (0.1, 1.0, None),
                         "d": (0.2, 1.0, 0.2)})}}}}
    out = rev.mode_comparison(report, "image_joint")
    image, vector = out["image"], out["vector"]
    assert image["n_common"] == 3                         # c: image_joint returned nothing
    assert image["ndcg_binary@10"] == pytest.approx({"image_joint": 0.5, "image": 0.5})
    assert (image["wins"], image["ties"], image["losses"]) == (1, 1, 1)
    assert image["hnrr@10"] == pytest.approx({"image_joint": 0.2, "image": 0.2})
    assert image["n_hnrr"] == 2                           # a and d have a rate on both sides
    assert vector["n_common"] == 2                        # b: vector returned nothing
    assert (vector["wins"], vector["ties"], vector["losses"]) == (1, 1, 0)
    assert vector["hnrr@10"] == pytest.approx({"image_joint": 0.2, "vector": 0.35})
    assert rev.mode_comparison(report, "image_decomposed") is None
    only = {"parsers": {"gold": {"modes": {"image_joint": entry({"a": (0.8, 1.0, 0.1)})}}}}
    assert rev.mode_comparison(only, "image_joint") == {}


def test_evidence_metrics_count_include_pairs_and_the_wanted_polarity():
    query = q([("Edema", "PRESENT"), ("Pneumonia", "UNCERTAIN")], ["Pleural Effusion"])
    hits = [Hit(1, "s1", "i1", evidence=[ev("Edema", "affirmed"), ev("Pneumonia", "affirmed"),
                                         ev("Pleural Effusion", "negated")]),
            Hit(2, "s2", "i2", evidence=[ev("Edema", "negated")]),
            Hit(3, "s3", "i3")]
    # 3 hits x 2 include concepts = 6 pairs, 3 quoted; only s1 Edema has the wanted polarity.
    # The exclude quote is not an include pair and is not counted.
    assert rev.evidence_metrics(hits, query) == pytest.approx(
        {"evidence_coverage": 0.5, "evidence_polarity_match": 1 / 3})
    assert rev.evidence_metrics(hits[2:], query) == {"evidence_coverage": 0.0,
                                                     "evidence_polarity_match": None}
    none = {"evidence_coverage": None, "evidence_polarity_match": None}
    assert rev.evidence_metrics(hits, q(exclude=["Pleural Effusion"])) == none
    assert rev.evidence_metrics([], query) == none


def test_aggregate_is_none_aware():
    rows = [{"p@10": 0.2, "exclude_violation@10": None},
            {"p@10": 0.4, "exclude_violation@10": 0.5}]
    assert rev.aggregate(rows) == {"p@10": pytest.approx(0.3), "exclude_violation@10": 0.5,
                                   "n": 2}
    assert rev.aggregate([{"x": None}]) == {"x": None, "n": 1}
    assert rev.aggregate([]) == {"n": 0}


def test_evaluate_skips_rows_without_gold_and_lays_out_the_report(rows, parser, matrix,
                                                                  agreement, fake_search):
    report = rev.evaluate_retrieval(
        rows, state=None, parser=parser, matrix=matrix, agreement=agreement,
        projection_of=PROJECTION, parsers=["gold", "rule"], modes=["label", "hybrid"],
        policies=[rev.DEFAULT_POLICY, rev.NEGATION_POLICY], fusion=rev.GRID[0],
        search=fake_search)
    assert report["skipped_no_gold"] == 1 and report["n"] == 3
    assert report["policies"] == ["exclude_present/EXCLUDE", "explicit_absence_required/EXCLUDE"]
    assert set(report["parsers"]) == {"gold", "rule"}
    entry = report["parsers"]["gold"]["modes"]["hybrid"]
    assert set(entry) >= {"metrics", "by_group", "by_language", "by_policy", "latency_ms",
                          "per_query"}

    # hybrid returns s2, s1, s6 for every query
    per = entry["per_query"]
    assert set(per) == {"en-001", "en-002", "vi-001"}
    # en-001: relevant s1 s2 s3 s6, grades 3 4 2 2
    assert per["en-001"]["p@10"] == pytest.approx(0.3) and per["en-001"]["p@5"] == 0.6
    assert per["en-001"]["ndcg@10"] == pytest.approx(
        (4 + 3 / np.log2(3) + 2 / np.log2(4))
        / (4 + 3 / np.log2(3) + 2 / np.log2(4) + 2 / np.log2(5)))
    assert per["en-001"]["evidence_coverage"] == pytest.approx(1 / 3)   # only s1 is quoted
    assert per["en-001"]["evidence_polarity_match"] == 1.0
    assert per["en-002"]["ndcg@10"] == pytest.approx(1.0)               # already ideal
    assert per["en-002"]["exclude_violation@10"] == 0.0
    assert per["vi-001"]["p@10"] == 0.1                                 # PA: only s2
    assert entry["metrics"]["p@10"] == pytest.approx((0.3 + 0.3 + 0.1) / 3)
    assert entry["metrics"]["exclude_violation@10"] == 0.0 and entry["metrics"]["n"] == 3

    assert set(entry["by_group"]) == {"single", "multi", "negation", "view"}
    assert entry["by_group"]["single"]["p@10"] == pytest.approx(0.2)    # en-001, vi-001
    assert entry["by_language"]["vi"] == per["vi-001"] | {"n": 1}
    # the negation policy runs on en-002 only: PE must be ABSENT, so s6 drops out
    by_policy = entry["by_policy"]
    assert by_policy["exclude_present/EXCLUDE"]["p@10"] == pytest.approx(0.3)
    assert by_policy["explicit_absence_required/EXCLUDE"]["p@10"] == pytest.approx(0.2)
    assert by_policy["explicit_absence_required/EXCLUDE"]["n"] == 1
    assert entry["latency_ms"]["n"] == 2                                # first row dropped

    # label returns s1, s3; the rule parser found no view, so relevance has none either
    assert report["parsers"]["rule"]["modes"]["label"]["per_query"]["en-001"]["p@10"] == 0.2
    calls = fake_search.calls
    assert len(calls) == 2 * 2 * (3 + 1)                 # the skipped row is never searched
    assert all(c["state"] is None and c["top_k"] == 10 and c["fusion"] == rev.GRID[0]
               for c in calls)
    assert all(c["views"] == c["query"].views for c in calls)
    negation_calls = [c for c in calls if c["policy"] == rev.NEGATION_POLICY]
    assert len(negation_calls) == 4
    assert all(c["query"].exclude == ["Pleural Effusion"] for c in negation_calls)


def test_fallback_rate_counts_rows_whose_parse_fell_back(rows, matrix, agreement, fake_search):
    class Flaky:
        def finalize(self, gold, text):
            return gold

        def parse(self, text, mode):
            return ParseResult(q([("Edema", "PRESENT")]), "rule",
                               fallback_used=text == "edema")

    report = rev.evaluate_retrieval(
        rows, state=None, parser=Flaky(), matrix=matrix, agreement=agreement,
        projection_of=PROJECTION, parsers=["gold", "auto"], modes=["label"],
        policies=[rev.DEFAULT_POLICY], search=fake_search)
    assert report["parsers"]["gold"]["fallback_rate"] is None
    assert report["parsers"]["auto"]["fallback_rate"] == pytest.approx(1 / 3)


def test_relevance_uses_the_gold_views_when_the_parse_drops_them(matrix, agreement):
    class DropsView:
        def finalize(self, gold, text):
            return gold

        def parse(self, text, mode):
            return ParseResult(q([("Edema", "PRESENT")]), "rule")   # the gold asks for PA

    search = fake(lambda mode, fusion: ["s1", "s2"])   # s1 AP, s2 PA, both Edema PRESENT
    row = bq("en-020", "edema on pa films", q([("Edema", "PRESENT")], views=["PA"]),
             group="view")
    report = rev.evaluate_retrieval(
        [row], state=None, parser=DropsView(), matrix=matrix, agreement=agreement,
        projection_of=PROJECTION, parsers=["rule"], modes=["hybrid"],
        policies=[rev.DEFAULT_POLICY], search=search)
    scored = report["parsers"]["rule"]["modes"]["hybrid"]["per_query"]["en-020"]
    # Relevance follows the gold (PA only: s2), the search still runs the parse's views.
    assert scored["p@10"] == pytest.approx(0.1) and scored["mrr"] == pytest.approx(0.5)
    assert [call["views"] for call in search.calls] == [[]]


def test_grid_has_36_configs_and_choose_uses_the_given_baseline():
    assert len(rev.GRID) == 36
    assert rev.GRID[0] == FusionParams(10, weights(0.0, 0.5, 0.0), 100, 100, 100)
    assert [(g.rrf_k, *(g.leg_weights[leg] for leg in ("vector", "report", "image")))
            for g in rev.GRID] == [
        (k, v, r, i) for k in (10, 60) for v in (0.0, 0.5, 1.0) for r in (0.5, 1.0)
        for i in (0.0, 0.5, 1.0)]
    assert all(g.vector_top_k == g.report_top_k == g.image_top_k == 100 for g in rev.GRID)

    baseline = FusionParams(10, weights(0.5, 1.0), 100, 100, 100)
    near = [{"rrf_k": 10, "leg_weights": weights(0.5, 1.0), "ndcg@10": 0.500},
            {"rrf_k": 60, "leg_weights": weights(1.0, 1.0, 0.5), "ndcg@10": 0.504}]
    assert rev.choose(near, baseline) == baseline                # 0.004 < 0.005
    far = [near[0], {"rrf_k": 60, "leg_weights": weights(1.0, 1.0, 0.5), "ndcg@10": 0.505}]
    assert rev.choose(far, baseline) == FusionParams(60, weights(1.0, 1.0, 0.5), 100, 100, 100)
    # top_k values come from the baseline, not from the grid entry
    wide = FusionParams(10, weights(0.5, 1.0), 50, 60, 70)
    assert rev.choose(far, wide) == FusionParams(60, weights(1.0, 1.0, 0.5), 50, 60, 70)
    # a config without an image key matches the baseline through _weights
    legacy = [{"rrf_k": 10, "leg_weights": {"vector": 0.5, "report": 1.0}, "ndcg@10": 0.5},
              {"rrf_k": 60, "leg_weights": weights(1.0, 1.0), "ndcg@10": 0.503}]
    assert rev.choose(legacy, baseline) == baseline
    with pytest.raises(ValueError, match="baseline"):
        rev.choose(legacy[1:], baseline)
    # two configs share the best ndcg: the earlier one in `configs` wins
    tie = [near[0], {"rrf_k": 60, "leg_weights": weights(1.0, 1.0, 0.5), "ndcg@10": 0.51},
           {"rrf_k": 10, "leg_weights": weights(0.0, 1.0, 1.0), "ndcg@10": 0.51}]
    assert rev.choose(tie, baseline) == FusionParams(60, weights(1.0, 1.0, 0.5), 100, 100, 100)
    assert rev.choose([*tie[:1], *tie[:0:-1]], baseline) == FusionParams(
        10, weights(0.0, 1.0, 1.0), 100, 100, 100)


def test_grid_search_scores_every_config_on_gold_hybrid(rows, parser, matrix, agreement):
    def ranking(mode, fusion):
        good = fusion.rrf_k == 60 and fusion.leg_weights["report"] == 0.5
        return ["s2", "s1", "s6"] if good else ["s3", "s5", "s1"]

    baseline = FusionParams(10, weights(0.5, 1.0), 100, 100, 100)
    search = fake(ranking)
    out = rev.grid_search(rows, state=None, parser=parser, matrix=matrix, agreement=agreement,
                          projection_of=PROJECTION, baseline=baseline, search=search)
    assert [(c["rrf_k"], c["leg_weights"]) for c in out["configs"]] == [
        (g.rrf_k, g.leg_weights) for g in rev.GRID]
    # s3 s5 s1: en-001 grades s3 2, s1 3 of (4, 3, 2, 2); en-002 s1 2.5 of (3.5, 2.5, 1.5);
    # vi-001 (PA, only s2 relevant) 0
    en001 = (2 + 3 / np.log2(4)) / (4 + 3 / np.log2(3) + 2 / np.log2(4) + 2 / np.log2(5))
    en002 = (2.5 / np.log2(4)) / (3.5 + 2.5 / np.log2(3) + 1.5 / np.log2(4))
    assert out["baseline_ndcg"] == pytest.approx((en001 + en002 + 0) / 3)
    assert "default_ndcg" not in out
    at = next(i for i, g in enumerate(rev.GRID) if g == baseline)
    assert out["configs"][at]["ndcg@10"] == out["baseline_ndcg"]
    assert out["configs"][at]["p@10"] == pytest.approx((0.2 + 0.1 + 0) / 3)
    assert out["chosen"]["rrf_k"] == 60 and out["chosen"]["leg_weights"] == weights(0.0, 0.5)
    assert len(search.calls) == 36 * 3
    assert all(c["mode"] == "hybrid" and c["policy"] == rev.DEFAULT_POLICY
               for c in search.calls)


def test_image_vs_vector_uses_only_queries_vector_answered():
    def entry(rows):
        return {"per_query": {i: {"ndcg@10": n, "returned@10": r, "exclude_violation@10": v}
                              for i, (n, r, v) in rows.items()}}

    report = {"parsers": {"gold": {"modes": {
        "vector": entry({"a": (0.4, 1.0, 0.5), "b": (0.6, 0.5, None), "c": (0.0, 0.0, None)}),
        "image": entry({"a": (0.6, 1.0, 0.2), "b": (0.6, 1.0, None), "c": (0.3, 1.0, 0.0)}),
        "label": entry({"a": (0.9, 1.0, 0.0), "b": (0.9, 1.0, None), "c": (0.1, 1.0, None)})}}}}
    out = rev.image_vs_vector(report)
    assert out["n_common"] == 2                       # c: vector returned nothing
    assert out["ndcg@10"] == pytest.approx({"image": 0.6, "vector": 0.5})
    assert (out["wins"], out["ties"], out["losses"]) == (1, 1, 0)
    # only a has a vector violation rate; c is out of scope although image has one
    assert out["exclude_violation@10"] == pytest.approx({"image": 0.2, "vector": 0.5})
    assert out["vector_unanswered"]["n"] == 1
    assert out["vector_unanswered"]["ndcg@10"] == pytest.approx({"image": 0.3, "label": 0.1})
    assert out["s2_passed"] is True

    worse = {"parsers": {"gold": {"modes": {
        "vector": entry({"a": (0.4, 1.0, 0.1)}), "image": entry({"a": (0.6, 1.0, 0.2)})}}}}
    out = rev.image_vs_vector(worse)
    assert out["s2_passed"] is False                  # better ndcg, but more violations
    assert out["vector_unanswered"] == {"n": 0, "ndcg@10": {"image": None, "label": None}}

    plain = {"parsers": {"gold": {"modes": {
        "vector": entry({"a": (0.4, 1.0, None)}), "image": entry({"a": (0.6, 1.0, None)})}}}}
    out = rev.image_vs_vector(plain)
    assert out["exclude_violation@10"] == {"image": None, "vector": None}
    assert out["s2_passed"] is True                   # no exclude query: that half holds

    # loss: image returned nothing (ndcg 0) where vector scored; tie: diff 5e-10 is within
    # tolerance; image has no violation rate for d (it returned nothing), so d is skipped on
    # both sides although vector has one
    mixed = {"parsers": {"gold": {"modes": {
        "vector": entry({"a": (0.5, 1.0, 0.4), "b": (0.5, 1.0, 0.2), "d": (0.7, 1.0, 0.9),
                         "e": (0.2, 1.0, 0.0)}),
        "image": entry({"a": (0.5 + 5e-10, 1.0, 0.1), "b": (0.9, 1.0, 0.0),
                        "d": (0.0, 0.0, None), "e": (0.1, 1.0, 0.0)})}}}}
    out = rev.image_vs_vector(mixed)
    assert out["n_common"] == 4
    assert (out["wins"], out["ties"], out["losses"]) == (1, 1, 2)    # b; a; d and e
    assert out["ndcg@10"] == pytest.approx({"image": (0.5 + 0.9 + 0.0 + 0.1) / 4,
                                            "vector": (0.5 + 0.5 + 0.7 + 0.2) / 4})
    assert out["exclude_violation@10"] == pytest.approx(
        {"image": (0.1 + 0.0 + 0.0) / 3, "vector": (0.4 + 0.2 + 0.0) / 3})
    assert out["s2_passed"] is False                  # ndcg 0.375 vs 0.475

    assert rev.image_vs_vector({"parsers": {"gold": {"modes": {"vector": entry({})}}}}) is None
    assert rev.image_vs_vector({"parsers": {"rule": {"modes": {}}}}) is None


def test_ablation_runs_every_configuration_through_search(rows, parser, matrix, agreement,
                                                           fake_search):
    fusion = FusionParams(60, weights(1.0, 1.0, 0.5), 80, 90, 70)
    out = rev.ablation(rows, state=None, parser=parser, matrix=matrix, agreement=agreement,
                       projection_of=PROJECTION, fusion=fusion, search=fake_search)
    names = ["label", "vector", "image", "filter+vector", "filter+report", "filter+image",
             "hybrid_main", "hybrid"]
    assert out["n"] == 3 and out["policy"] == "exclude_present/EXCLUDE"
    assert list(out["configs"]) == names and list(out["per_query"]) == names
    assert set(out["per_query"]["label"]) == {"en-001", "en-002", "vi-001"}
    assert out["configs"]["label"]["ndcg@10"] == pytest.approx(
        sum(out["per_query"]["label"].values()) / 3)

    calls = fake_search.calls
    assert len(calls) == 8 * 3                         # the row without gold is never searched
    by_name = {name: calls[i * 3:(i + 1) * 3] for i, name in enumerate(names)}
    assert {n: {c["mode"] for c in cs} for n, cs in by_name.items()} == {
        "label": {"label"}, "vector": {"vector"}, "image": {"image"},
        "filter+vector": {"hybrid"}, "filter+report": {"hybrid"}, "filter+image": {"hybrid"},
        "hybrid_main": {"hybrid"}, "hybrid": {"hybrid"}}
    assert all(c["policy"] == rev.DEFAULT_POLICY for c in calls)
    report_only = by_name["filter+report"][0]["fusion"]
    assert (report_only.weight("vector"), report_only.weight("report"),
            report_only.weight("image")) == (0.0, 1.0, 0.0)
    assert by_name["filter+image"][0]["fusion"].weight("image") == 1.0
    assert by_name["filter+vector"][0]["fusion"].rrf_k == 60
    assert by_name["hybrid_main"][0]["fusion"] == FusionParams(10, weights(0.5, 1.0), 80, 90, 70)
    assert by_name["hybrid"][0]["fusion"] == fusion
    assert (report_only.vector_top_k, report_only.report_top_k, report_only.image_top_k) == (
        80, 90, 70)


def test_verdict_uses_the_three_thresholds():
    def report(single, p10, hybrid_ndcg, vector_ndcg):
        return {"parsers": {
            "gold": {"modes": {
                "hybrid": {"metrics": {"p@10": p10, "ndcg@10": hybrid_ndcg},
                           "by_group": {"single": {"p@10": single}}},
                "vector": {"metrics": {"p@10": 0.4, "ndcg@10": vector_ndcg}, "by_group": {}}}},
            "rule": {"modes": {                       # never judged
                "hybrid": {"metrics": {"p@10": 0.0, "ndcg@10": 0.0},
                           "by_group": {"single": {"p@10": 0.0}}},
                "vector": {"metrics": {"p@10": 0.0, "ndcg@10": 1.0}, "by_group": {}}}}}}

    v = rev.verdict(report(0.8, 0.96, 0.6, 0.5), THRESHOLDS)
    assert v["hybrid_gain_over_vector"]["value"] == pytest.approx((0.6 - 0.5) / 0.5)
    assert v == {"single_p10": {"value": 0.8, "threshold": 0.70, "passed": True},
                 "constraint_satisfaction": {"value": 0.96, "threshold": 0.95, "passed": True},
                 "hybrid_gain_over_vector": {"value": pytest.approx(0.2), "threshold": 0.05,
                                             "passed": True},
                 "passed": True}
    low = rev.verdict(report(0.8, 0.94, 0.6, 0.5), THRESHOLDS)
    assert low["constraint_satisfaction"]["passed"] is False and low["passed"] is False
    small = rev.verdict(report(0.8, 0.96, 0.52, 0.5), THRESHOLDS)      # gain 0.04
    assert small["hybrid_gain_over_vector"]["passed"] is False and small["passed"] is False
    undefined = rev.verdict(report(0.8, 0.96, 0.6, 0.0), THRESHOLDS)
    assert undefined["hybrid_gain_over_vector"] == {"value": None, "threshold": 0.05,
                                                    "passed": None}
    assert undefined["passed"] is False
