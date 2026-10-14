"""Known-answer tests for the parser metrics (spec 8.1) and the evaluation driver."""

from __future__ import annotations

import pytest

from src.common.config import load_config
from src.evaluation import parser_eval as pe
from src.evaluation.parser_benchmark import BenchmarkQuery, derive_groups
from src.query.parser import QueryParser
from src.query.rule_parser import RuleParser
from src.query.schema import SearchQuery
from src.retrieval import label_filter as lf
from tests.unit.fakes import FakeOllama
from tests.unit.test_label_filter import labels_frame


def q(include=(), exclude=(), views=(), language="en") -> SearchQuery:
    return SearchQuery(language=language, views=list(views), exclude=list(exclude),
                       include=[{"concept": c, "status": s} for c, s in include])


class TestItems:
    def test_items_are_sets_over_the_three_axes(self):
        query = q([("Edema", "PRESENT"), ("Pneumonia", "UNCERTAIN")], ["Fracture"], ["PA"])
        assert pe.concept_items(query) == {"Edema", "Pneumonia", "Fracture"}
        assert pe.status_items(query) == {("Edema", "PRESENT"), ("Pneumonia", "UNCERTAIN"),
                                          ("Fracture", "EXCLUDED")}
        assert pe.view_items(query) == {"PA"}


class TestMacroF1:
    def test_hand_computed_two_classes(self):
        # class A: tp 1, fn 1 -> P 1, R .5, F1 .667 ; class B: fp 1 -> F1 0 ; macro .333
        preds = [{"A"}, {"B"}]
        golds = [{"A"}, {"A"}]
        out = pe.macro_f1(preds, golds, lambda item: item)
        assert out["per_class"]["A"]["f1"] == pytest.approx(2 / 3)
        assert out["per_class"]["B"] == {"tp": 0, "fp": 1, "fn": 0, "precision": 0.0,
                                         "recall": 0.0, "f1": 0.0, "support": 0}
        assert out["macro_f1"] == pytest.approx(1 / 3)

    def test_perfect_and_empty(self):
        assert pe.macro_f1([{"A"}], [{"A"}], lambda i: i)["macro_f1"] == 1.0
        assert pe.macro_f1([set()], [set()], lambda i: i)["macro_f1"] is None

    def test_class_of_groups_tuples(self):
        preds = [{("Edema", "PRESENT")}]
        golds = [{("Edema", "EXCLUDED")}]
        out = pe.macro_f1(preds, golds, lambda item: item[1])
        assert set(out["per_class"]) == {"PRESENT", "EXCLUDED"} and out["macro_f1"] == 0.0


class TestEvaluate:
    def test_three_queries_by_hand(self):
        golds = [q([("Edema", "PRESENT")], ["Pleural Effusion"]),
                 q([("Pneumonia", "UNCERTAIN")], views=["PA"]),
                 q([("Cardiomegaly", "PRESENT")])]
        preds = [q([("Edema", "PRESENT")], ["Pleural Effusion"]),      # exact
                 q([("Pneumonia", "PRESENT")], views=["PA"]),          # status wrong
                 None]                                                  # invalid parse
        out = pe.evaluate(golds, preds)
        assert out["n"] == 3 and out["schema_valid_rate"] == pytest.approx(2 / 3)
        assert out["exact_match_rate"] == pytest.approx(1 / 3)
        assert out["view_macro_f1"] == 1.0
        # concepts: Edema, Pleural Effusion, Pneumonia perfect; Cardiomegaly fn -> F1 0
        assert out["concept_macro_f1"] == pytest.approx(3 / 4)
        # statuses: PRESENT tp1 fp1 fn1 -> .5 ; EXCLUDED 1 ; UNCERTAIN fn -> 0 ; macro .5
        assert out["status_macro_f1"] == pytest.approx(0.5)


def bq(id, query, gold, split="heldout", language="en", group="single"):
    return BenchmarkQuery(id, split, language, group, derive_groups(gold), query, gold)


@pytest.fixture
def rows():
    return [bq("en-001", "cardiomegaly", q([("Cardiomegaly", "PRESENT")])),
            bq("en-002", "edema with no pleural effusion",
               q([("Edema", "PRESENT")], ["Pleural Effusion"]), group="negation"),
            bq("vi-001", "tim to", q([("Cardiomegaly", "PRESENT")], language="vi"),
               language="vi")]


@pytest.fixture
def rule_parser():
    return QueryParser(RuleParser(load_config("concepts")), template="x {}")


class TestRunParser:
    def test_rule_mode_predicts_every_row(self, rows, rule_parser):
        preds = pe.run_parser(rule_parser, rows, "rule")
        assert [p.query is not None for p in preds] == [True, True, True]
        assert all(p.parser == "rule" and not p.fallback_used for p in preds)
        assert all(p.ms >= 0 for p in preds)

    def test_llm_mode_counts_failures_as_invalid(self, rows):
        fake = FakeOllama("ok", ["{bad", "{bad",
                                 '{"findings": "", "include": [], "exclude": [], "views": [], '
                                 '"language": "en"}', "{bad", "{bad"])
        parser = QueryParser(RuleParser(load_config("concepts")), template="x {}",
                             llm_client=fake, system_prompt="s")
        preds = pe.run_parser(parser, rows, "llm")
        assert [p.query is not None for p in preds] == [False, True, False]
        assert preds[0].error.startswith("llm output invalid")


class TestBreakdownsAndVerdict:
    def test_breakdowns_by_language_and_group(self, rows, rule_parser):
        preds = pe.run_parser(rule_parser, rows, "rule")
        out = pe.breakdowns(rows, preds)
        assert set(out["by_language"]) == {"en", "vi"} and out["by_language"]["vi"]["n"] == 1
        assert out["by_group"]["negation"]["n"] == 1 and out["by_group"]["single"]["n"] == 2
        assert "view" not in out["by_group"]

    def test_verdict_applies_every_threshold(self):
        thresholds = {"parser_schema_valid_rate_min": 0.99, "parser_macro_f1_min": 0.9}
        good = {"schema_valid_rate": 1.0, "concept_macro_f1": 0.95, "status_macro_f1": 0.91,
                "view_macro_f1": 1.0}
        assert pe.verdict(good, thresholds) == {"schema_valid": True, "concept_macro_f1": True,
                                                "status_macro_f1": True, "view_macro_f1": True,
                                                "passed": True}
        bad = {**good, "status_macro_f1": 0.89}
        assert pe.verdict(bad, thresholds)["passed"] is False
        assert pe.verdict({**good, "view_macro_f1": None}, thresholds)["view_macro_f1"] is None


class TestLabelFilterCounts:
    def test_counts_every_policy_combination(self, rows):
        frame = labels_frame([
            ("a", "Edema", "PRESENT", "CHEXPERT_V1"),
            ("a", "Pleural Effusion", "ABSENT", "CHEXPERT_V1"),
            ("b", "Edema", "PRESENT", "CHEXPERT_V1"),
            ("b", "Pleural Effusion", "UNCERTAIN", "CHEXPERT_V1"),
            ("c", "Edema", "PRESENT", "CHEXPERT_V1")])
        matrix = lf.build_label_matrix(frame, ["a", "b", "c"], ("CHEXPERT_V1",), seed=1)
        out = pe.label_filter_counts(rows, matrix)
        row = out["per_query"]["en-002"]
        assert row == {"exclude_present/EXCLUDE": 2, "exclude_present/INCLUDE": 3,
                       "explicit_absence_required/EXCLUDE": 1,
                       "explicit_absence_required/INCLUDE": 1}
        assert out["summary"]["negation"]["exclude_present/EXCLUDE"]["median"] == 2
        assert out["summary"]["negation"]["explicit_absence_required/EXCLUDE"]["zero"] == 0


class TestEvaluateBenchmark:
    def test_end_to_end_structure(self, rows, rule_parser):
        thresholds = {"parser_schema_valid_rate_min": 0.99, "parser_macro_f1_min": 0.9}
        out = pe.evaluate_benchmark(rows, rule_parser, ["rule", "auto"], thresholds)
        assert set(out["parsers"]) == {"rule", "auto"}
        assert out["parsers"]["rule"]["metrics"]["n"] == 3
        assert "by_language" in out["parsers"]["rule"] and "latency_ms" in out["parsers"]["rule"]
        assert out["verdict"]["passed"] in (True, False)
        assert out["n"] == 3 and out["thresholds"] == thresholds
