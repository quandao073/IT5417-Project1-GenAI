"""Known-answer tests for the retrieval evaluation (spec 11.3-11.4): relevance, grades and
metrics of the one search path. A six-study toy world; `search` is a fake."""

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


def q(include=(), exclude=(), views=(), language="en") -> SearchQuery:
    return SearchQuery(language=language, views=list(views), exclude=list(exclude),
                       include=[{"concept": c, "status": s} for c, s in include])


def bq(id, query, gold, language="en", group="single") -> BenchmarkQuery:
    return BenchmarkQuery(id, "dev", language, group, derive_groups(gold), query, gold)


def ev(concept, polarity) -> Evidence:
    return Evidence(concept=concept, text="t", start=0, end=1, polarity=polarity)


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


def fake(ranking):
    """A `search` with the pipeline's signature returning `ranking` (study ids) for every
    query; s1 is the only study quoted (Edema, affirmed). Every call is recorded on `.calls`."""
    calls = []

    def search(state, query, *, top_k, views):
        calls.append({"state": state, "query": query, "top_k": top_k, "views": views})
        hits = [Hit(rank, s, f"img-{s}", 0.5,
                    evidence=[ev("Edema", "affirmed")] if s == "s1" else [])
                for rank, s in enumerate(ranking, start=1)]
        return SearchOutcome(hits, "template", [])

    search.calls = calls
    return search


@pytest.fixture
def fake_search():
    return fake(["s2", "s1", "s6"])


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
    assert m["ndcg_binary@10"] == pytest.approx(1 / (1 + 1 / np.log2(3) + 1 / np.log2(4)))
    assert m["ndcg_graded@10"] == pytest.approx(
        2.5 / (3.5 + 2.5 / np.log2(3) + 1.5 / np.log2(4)))
    assert rev.query_metrics(["s3", "s6"], rel, gold, matrix)["mrr"] == 0.5
    assert rev.query_metrics([], rel, gold, matrix)["exclude_violation@10"] is None
    plain = q([("Edema", "PRESENT")])
    empty = rev.query_metrics([], relevance(plain), plain, matrix)
    assert empty == {"ndcg_binary@10": 0.0, "ndcg_graded@10": 0.0, "p@5": 0.0, "p@10": 0.0,
                     "mrr": 0.0, "exclude_violation@10": None, "returned@10": 0.0}
    assert next(iter(empty)) == "ndcg_binary@10"          # the headline metric comes first


def test_evidence_metrics_count_include_pairs_and_the_wanted_polarity():
    query = q([("Edema", "PRESENT"), ("Pneumonia", "UNCERTAIN")], ["Pleural Effusion"])
    hits = [Hit(1, "s1", "i1", 0.5, evidence=[ev("Edema", "affirmed"),
                                              ev("Pneumonia", "affirmed"),
                                              ev("Pleural Effusion", "negated")]),
            Hit(2, "s2", "i2", 0.4, evidence=[ev("Edema", "negated")]),
            Hit(3, "s3", "i3", 0.3)]
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
        projection_of=PROJECTION, parsers=["gold", "rule"],
        policies=[rev.DEFAULT_POLICY, rev.NEGATION_POLICY], search=fake_search)
    assert report["skipped_no_gold"] == 1 and report["n"] == 3
    assert report["policies"] == ["exclude_present/EXCLUDE", "explicit_absence_required/EXCLUDE"]
    assert set(report["parsers"]) == {"gold", "rule"}
    entry = report["parsers"]["gold"]
    assert set(entry) == {"fallback_rate", "metrics", "by_group", "by_language", "by_policy",
                          "latency_ms", "per_query"}

    # the search returns s2, s1, s6 for every query
    per = entry["per_query"]
    assert set(per) == {"en-001", "en-002", "vi-001"}
    # en-001: relevant s1 s2 s3 s6, grades 3 4 2 2
    assert per["en-001"]["p@10"] == pytest.approx(0.3) and per["en-001"]["p@5"] == 0.6
    assert per["en-001"]["ndcg_binary@10"] == pytest.approx(
        (1 + 1 / np.log2(3) + 1 / np.log2(4))
        / (1 + 1 / np.log2(3) + 1 / np.log2(4) + 1 / np.log2(5)))
    assert per["en-001"]["ndcg_graded@10"] == pytest.approx(
        (4 + 3 / np.log2(3) + 2 / np.log2(4))
        / (4 + 3 / np.log2(3) + 2 / np.log2(4) + 2 / np.log2(5)))
    assert per["en-001"]["evidence_coverage"] == pytest.approx(1 / 3)   # only s1 is quoted
    assert per["en-001"]["evidence_polarity_match"] == 1.0
    assert per["en-002"]["ndcg_graded@10"] == pytest.approx(1.0)        # already ideal
    assert per["en-002"]["exclude_violation@10"] == 0.0
    assert per["vi-001"]["p@10"] == 0.1                                 # PA: only s2
    assert entry["metrics"]["p@10"] == pytest.approx((0.3 + 0.3 + 0.1) / 3)
    assert entry["metrics"]["exclude_violation@10"] == 0.0 and entry["metrics"]["n"] == 3

    assert set(entry["by_group"]) == {"single", "multi", "negation", "view"}
    assert entry["by_group"]["single"]["p@10"] == pytest.approx(0.2)    # en-001, vi-001
    assert entry["by_language"]["vi"] == per["vi-001"] | {"n": 1}
    # the negation policy scores en-002 only: PE must be ABSENT, so s6 drops out
    by_policy = entry["by_policy"]
    assert by_policy["exclude_present/EXCLUDE"]["p@10"] == pytest.approx(0.3)
    assert by_policy["explicit_absence_required/EXCLUDE"]["p@10"] == pytest.approx(0.2)
    assert by_policy["explicit_absence_required/EXCLUDE"]["n"] == 1
    assert entry["latency_ms"]["n"] == 2                                # first row dropped

    # the rule parser found no view, so en-001 scores the same; the skipped row is never
    # searched, and a policy only changes relevance, so each row is searched once per parser
    assert report["parsers"]["rule"]["per_query"]["en-001"]["p@10"] == pytest.approx(0.3)
    calls = fake_search.calls
    assert len(calls) == 2 * 3
    assert all(c["state"] is None and c["top_k"] == 10 for c in calls)
    assert all(c["views"] == c["query"].views for c in calls)


def test_fallback_rate_counts_rows_whose_parse_fell_back(rows, matrix, agreement, fake_search):
    class Flaky:
        def finalize(self, gold, text):
            return gold

        def parse(self, text, mode):
            return ParseResult(q([("Edema", "PRESENT")]), "rule",
                               fallback_used=text == "edema")

    report = rev.evaluate_retrieval(
        rows, state=None, parser=Flaky(), matrix=matrix, agreement=agreement,
        projection_of=PROJECTION, parsers=["gold", "auto"],
        policies=[rev.DEFAULT_POLICY], search=fake_search)
    assert report["parsers"]["gold"]["fallback_rate"] is None
    assert report["parsers"]["auto"]["fallback_rate"] == pytest.approx(1 / 3)


def test_relevance_uses_the_gold_views_when_the_parse_drops_them(matrix, agreement):
    class DropsView:
        def finalize(self, gold, text):
            return gold

        def parse(self, text, mode):
            return ParseResult(q([("Edema", "PRESENT")]), "rule")   # the gold asks for PA

    search = fake(["s1", "s2"])                        # s1 AP, s2 PA, both Edema PRESENT
    row = bq("en-020", "edema on pa films", q([("Edema", "PRESENT")], views=["PA"]),
             group="view")
    report = rev.evaluate_retrieval(
        [row], state=None, parser=DropsView(), matrix=matrix, agreement=agreement,
        projection_of=PROJECTION, parsers=["rule"],
        policies=[rev.DEFAULT_POLICY], search=search)
    scored = report["parsers"]["rule"]["per_query"]["en-020"]
    # Relevance follows the gold (PA only: s2), the search still runs the parse's views.
    assert scored["p@10"] == pytest.approx(0.1) and scored["mrr"] == pytest.approx(0.5)
    assert [call["views"] for call in search.calls] == [[]]
