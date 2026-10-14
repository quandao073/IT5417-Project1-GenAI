"""Label mode: the truth table of spec 5.2 (WORKLOG T7, D9) and the fixed order (D18)."""

from __future__ import annotations

from itertools import product

import numpy as np
import polars as pl
import pytest

from src.query.schema import Constraint
from src.retrieval import label_filter as lf

PRIORITY = ("CHEXPERT_V1", "CHEXBERT_IMPRESSION", "CHEXBERT_FINDINGS", "CHEXBERT_REPORT")
STATUSES = ("PRESENT", "UNCERTAIN", "ABSENT", "NOT_MENTIONED")


def labels_frame(rows) -> pl.DataFrame:
    """rows of (study_id, concept, status, source); source_version is constant."""
    return pl.DataFrame(
        [(*r, "t") for r in rows],
        schema={"study_id": pl.Utf8, "concept": pl.Utf8, "status": pl.Utf8,
                "source": pl.Utf8, "source_version": pl.Utf8},
        orient="row",
    )


@pytest.fixture
def one_concept_per_status():
    """Four studies; study s_<STATUS> has Edema in that status, nothing else."""
    rows = [(f"s_{s}", "Edema", s, "CHEXBERT_IMPRESSION") for s in STATUSES
            if s != "NOT_MENTIONED"]
    ids = [f"s_{s}" for s in STATUSES]
    return lf.build_label_matrix(labels_frame(rows), ids, PRIORITY, seed=1)


def matches(matrix, include=(), exclude=(), absence="exclude_present", uncertainty="EXCLUDE"):
    mask = lf.filter_studies_by_labels(
        matrix, [Constraint(concept=c, status=s) for c, s in include], list(exclude),
        absence_policy=absence, uncertainty_policy=uncertainty)
    return set(matrix.study_ids[mask])


class TestBuild:
    def test_codes_sources_and_shape(self, one_concept_per_status):
        m = one_concept_per_status
        assert m.status.shape == (4, 14) and m.status.dtype == np.int8
        j = m.col_of["Edema"]
        assert [lf.CODE_STATUS[int(m.status[m.row_of[f"s_{s}"], j])] for s in STATUSES] == list(
            STATUSES)
        assert m.source[m.row_of["s_PRESENT"], j] == "CHEXBERT_IMPRESSION"
        assert m.source[m.row_of["s_NOT_MENTIONED"], j] is None
        # every other concept of every study is NOT_MENTIONED with no source
        other = [k for k in range(14) if k != j]
        assert (m.status[:, other] == lf.STATUS_CODE["NOT_MENTIONED"]).all()
        assert all(v is None for v in m.source[:, other].ravel())

    def test_priority_picks_the_first_source_that_speaks(self):
        rows = [("a", "Edema", "NOT_MENTIONED", "CHEXPERT_V1"),
                ("a", "Edema", "ABSENT", "CHEXBERT_IMPRESSION"),
                ("a", "Edema", "PRESENT", "CHEXBERT_REPORT")]
        m = lf.build_label_matrix(labels_frame(rows), ["a"], PRIORITY, seed=1)
        j = m.col_of["Edema"]
        assert lf.CODE_STATUS[int(m.status[0, j])] == "ABSENT"
        assert m.source[0, j] == "CHEXBERT_IMPRESSION"

    def test_a_study_without_any_row_is_all_not_mentioned(self):
        m = lf.build_label_matrix(labels_frame([]), ["lonely"], PRIORITY, seed=1)
        assert (m.status == lf.STATUS_CODE["NOT_MENTIONED"]).all()

    def test_rows_follow_the_given_study_order(self):
        rows = [("b", "Edema", "PRESENT", "CHEXPERT_V1"), ("a", "Edema", "ABSENT", "CHEXPERT_V1")]
        m = lf.build_label_matrix(labels_frame(rows), ["b", "a"], PRIORITY, seed=1)
        assert list(m.study_ids) == ["b", "a"] and m.row_of == {"b": 0, "a": 1}

    def test_order_is_a_permutation_fixed_by_the_seed(self):
        ids = [f"s{i}" for i in range(50)]
        a = lf.build_label_matrix(labels_frame([]), ids, PRIORITY, seed=20260101)
        b = lf.build_label_matrix(labels_frame([]), ids, PRIORITY, seed=20260101)
        c = lf.build_label_matrix(labels_frame([]), ids, PRIORITY, seed=7)
        assert sorted(a.order) == list(range(50))
        assert (a.order == b.order).all() and not (a.order == c.order).all()
        assert list(a.order) != list(range(50))


EXPECTED = {
    # (constraint, absence, uncertainty) -> statuses that satisfy it (spec 5.2)
    ("include_present", "exclude_present", "EXCLUDE"): {"PRESENT"},
    ("include_present", "exclude_present", "INCLUDE"): {"PRESENT", "UNCERTAIN"},
    ("include_present", "explicit_absence_required", "EXCLUDE"): {"PRESENT"},
    ("include_present", "explicit_absence_required", "INCLUDE"): {"PRESENT", "UNCERTAIN"},
    ("include_uncertain", "exclude_present", "EXCLUDE"): {"UNCERTAIN"},
    ("include_uncertain", "exclude_present", "INCLUDE"): {"UNCERTAIN"},
    ("include_uncertain", "explicit_absence_required", "EXCLUDE"): {"UNCERTAIN"},
    ("include_uncertain", "explicit_absence_required", "INCLUDE"): {"UNCERTAIN"},
    ("exclude", "exclude_present", "EXCLUDE"): {"ABSENT", "NOT_MENTIONED"},
    ("exclude", "exclude_present", "INCLUDE"): {"ABSENT", "NOT_MENTIONED", "UNCERTAIN"},
    ("exclude", "explicit_absence_required", "EXCLUDE"): {"ABSENT"},
    ("exclude", "explicit_absence_required", "INCLUDE"): {"ABSENT"},
}


class TestTruthTable:
    @pytest.mark.parametrize("kind, absence, uncertainty, status", [
        (k, a, u, s) for (k, a, u) in EXPECTED for s in STATUSES])
    def test_each_cell(self, one_concept_per_status, kind, absence, uncertainty, status):
        if kind == "exclude":
            got = matches(one_concept_per_status, exclude=["Edema"], absence=absence,
                          uncertainty=uncertainty)
        else:
            want = "PRESENT" if kind == "include_present" else "UNCERTAIN"
            got = matches(one_concept_per_status, include=[("Edema", want)], absence=absence,
                          uncertainty=uncertainty)
        assert (f"s_{status}" in got) == (status in EXPECTED[(kind, absence, uncertainty)])

    def test_not_mentioned_is_never_absent(self, one_concept_per_status):
        for absence, uncertainty in product(("exclude_present", "explicit_absence_required"),
                                            ("EXCLUDE", "INCLUDE")):
            got = matches(one_concept_per_status, exclude=["Edema"], absence=absence,
                          uncertainty=uncertainty)
            assert ("s_NOT_MENTIONED" in got) == (absence == "exclude_present")


class TestCombinations:
    def test_constraints_are_anded(self):
        rows = [("a", "Edema", "PRESENT", "CHEXPERT_V1"),
                ("a", "Pleural Effusion", "ABSENT", "CHEXPERT_V1"),
                ("b", "Edema", "PRESENT", "CHEXPERT_V1"),
                ("b", "Pleural Effusion", "PRESENT", "CHEXPERT_V1"),
                ("c", "Edema", "ABSENT", "CHEXPERT_V1"),
                ("c", "Pleural Effusion", "ABSENT", "CHEXPERT_V1")]
        m = lf.build_label_matrix(labels_frame(rows), ["a", "b", "c"], PRIORITY, seed=1)
        assert matches(m, include=[("Edema", "PRESENT")], exclude=["Pleural Effusion"]) == {"a"}
        assert matches(m, include=[("Edema", "PRESENT")]) == {"a", "b"}
        assert matches(m, exclude=["Pleural Effusion"]) == {"a", "c"}

    def test_no_constraint_is_an_error_not_the_whole_corpus(self, one_concept_per_status):
        with pytest.raises(ValueError, match="at least one constraint"):
            lf.filter_studies_by_labels(one_concept_per_status, [], [],
                                        absence_policy="exclude_present",
                                        uncertainty_policy="EXCLUDE")

    def test_empty_mask_gives_no_matches(self, one_concept_per_status):
        mask = lf.filter_studies_by_labels(
            one_concept_per_status, [Constraint(concept="Fracture", status="PRESENT")], [],
            absence_policy="exclude_present", uncertainty_policy="EXCLUDE")
        assert not mask.any() and lf.ordered_matches(one_concept_per_status, mask) == []


class TestOrderAndProvenance:
    def test_ordered_matches_follow_the_permutation(self):
        ids = [f"s{i}" for i in range(20)]
        rows = [(i, "Edema", "PRESENT", "CHEXPERT_V1") for i in ids]
        m = lf.build_label_matrix(labels_frame(rows), ids, PRIORITY, seed=3)
        mask = np.ones(20, dtype=bool)
        got = lf.ordered_matches(m, mask)
        assert got == [ids[k] for k in m.order] and sorted(got) == sorted(ids)
        mask[m.order[0]] = False
        assert lf.ordered_matches(m, mask)[0] == ids[m.order[1]]

    def test_resolved_labels_carry_status_and_source(self, one_concept_per_status):
        got = lf.resolved_labels(one_concept_per_status, "s_PRESENT", ["Edema", "Fracture"])
        assert got == [{"concept": "Edema", "status": "PRESENT", "source": "CHEXBERT_IMPRESSION"},
                       {"concept": "Fracture", "status": "NOT_MENTIONED", "source": None}]
