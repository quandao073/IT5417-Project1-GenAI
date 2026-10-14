"""SearchQuery is the one contract between the parsers and the retrieval legs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.query.schema import Constraint, ParseResult, SearchQuery, canonical_concept


class TestCanonicalConcept:
    @pytest.mark.parametrize("raw, expected", [
        ("Cardiomegaly", "Cardiomegaly"), ("cardiomegaly", "Cardiomegaly"),
        ("PLEURAL  effusion", "Pleural Effusion"), (" no finding ", "No Finding"),
    ])
    def test_normalises_case_and_spacing(self, raw, expected):
        assert canonical_concept(raw) == expected

    @pytest.mark.parametrize("raw", ["Pleural Effusions", "edema, mild", "", "Lung"])
    def test_rejects_anything_outside_the_fourteen_labels(self, raw):
        with pytest.raises(ValueError, match="unknown concept"):
            canonical_concept(raw)


class TestSearchQuery:
    def test_minimal_query_has_empty_lists(self):
        q = SearchQuery(language="en")
        assert (q.semantic_query, q.include, q.exclude, q.views) == ("", [], [], [])

    def test_include_is_canonicalised_and_deduped_keeping_the_first(self):
        q = SearchQuery(language="en", include=[
            {"concept": "edema", "status": "PRESENT"},
            {"concept": "Edema", "status": "UNCERTAIN"},
            {"concept": "cardiomegaly", "status": "PRESENT"}])
        assert q.include == [Constraint(concept="Edema", status="PRESENT"),
                             Constraint(concept="Cardiomegaly", status="PRESENT")]

    def test_exclude_is_canonicalised_and_deduped(self):
        q = SearchQuery(language="en",
                        exclude=["pleural effusion", "Pleural Effusion", "fracture"])
        assert q.exclude == ["Pleural Effusion", "Fracture"]

    def test_a_concept_cannot_be_both_included_and_excluded(self):
        with pytest.raises(ValidationError, match="both include and exclude"):
            SearchQuery(language="en", include=[{"concept": "Edema", "status": "PRESENT"}],
                        exclude=["edema"])

    def test_views_are_deduped_in_order_and_restricted_to_ap_pa(self):
        assert SearchQuery(language="en", views=["PA", "AP", "PA"]).views == ["PA", "AP"]
        with pytest.raises(ValidationError):
            SearchQuery(language="en", views=["LATERAL"])

    @pytest.mark.parametrize("status", ["ABSENT", "NOT_MENTIONED", "present"])
    def test_include_status_is_present_or_uncertain_only(self, status):
        with pytest.raises(ValidationError):
            SearchQuery(language="en", include=[{"concept": "Edema", "status": status}])

    def test_semantic_query_whitespace_is_collapsed_and_length_capped(self):
        assert SearchQuery(language="en", semantic_query="  a   b ").semantic_query == "a b"
        with pytest.raises(ValidationError):
            SearchQuery(language="en", semantic_query="x" * 301)

    def test_language_is_required(self):
        with pytest.raises(ValidationError):
            SearchQuery()

    def test_constrained_concepts_lists_include_then_exclude_once_each(self):
        q = SearchQuery(language="vi",
                        include=[{"concept": "Edema", "status": "PRESENT"},
                                 {"concept": "Cardiomegaly", "status": "UNCERTAIN"}],
                        exclude=["Pleural Effusion"])
        assert q.constrained_concepts() == ["Edema", "Cardiomegaly", "Pleural Effusion"]


def test_parse_result_defaults():
    r = ParseResult(SearchQuery(language="en"), "rule")
    assert (r.fallback_used, r.warnings, r.llm_ms, r.cached) == (False, (), None, False)
