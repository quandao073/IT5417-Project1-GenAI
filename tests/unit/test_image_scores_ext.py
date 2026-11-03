"""Extra score-matrix columns: candidates, RadGraph silver labels, lenient AUROC (D55, D62)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.retrieval import image_scores_ext as iext

CFG = {"max_candidates": 3, "captions": 2, "forced": ["C4"], "generic": {"C5": "too general"}}


def test_descendants_follow_is_a_downward():
    edges = pl.DataFrame({"src": ["C2", "C3", "C9"], "rel": ["IS_A", "IS_A", "PART_OF"],
                          "dst": ["C1", "C2", "C1"]})
    assert iext.descendants(edges, ["C1", "C3"]) == {"C1": {"C1", "C2", "C3"}, "C3": {"C3"}}


def test_candidates_exclude_by_reason_and_keep_the_forced_ones():
    nodes = pl.DataFrame({
        "cui": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        "name": ["Emphysema", "Normal heart size", "Degenerative abnormality", "Granuloma",
                 "Lesion", "Lung", "Old scar"],
        "labels": ["Finding", "Finding", "Finding", "Finding", "Finding", "Anatomy", "Finding"],
        "n_pool": [900, 800, 700, 10, 600, 5000, 650],
        "in_core": [True] * 7, "chexpert_label": [None] * 7},
        schema_overrides={"chexpert_label": pl.String})
    strings = pl.DataFrame({"cui": ["C1", "C2", "C3", "C3", "C3", "C4", "C5", "C7"],
                            "string": ["emphysema", "normal heart size", "degenerative",
                                       "degenerative change", "spondylosis", "granuloma",
                                       "lesion", "scar"],
                            "mentions": [9, 8, 7, 6, 1, 1, 5, 4]})
    table = iext.candidates(nodes, strings, {}, CFG)
    got = {c: (s, r) for c, s, r in table.select("cui", "status", "reason").iter_rows()}
    assert got["C2"] == ("excluded", "normal statement")
    assert got["C3"][0] == "candidate"                     # "abnormality" is no normal word
    assert got["C4"] == ("candidate", "")                  # forced despite its rank
    assert got["C5"] == ("excluded", "generic: too general")
    assert got["C6"] == ("excluded", "anatomy or device")
    assert got["C1"][0] == "candidate"
    assert got["C7"] == ("excluded", "beyond the first 3")  # C1, C3 and the forced C4 fill it
    caps = dict(table.select("cui", "captions").iter_rows())
    assert caps["C3"] == "degenerative | degenerative change"   # the most frequent two


def test_silver_statuses_prefer_present_then_uncertain_then_absent():
    mcuis = pl.DataFrame({
        "study_id": ["s1", "s1", "s2", "s3", "s3", "s4"],
        "kind": ["Observation"] * 6,
        "polarity": ["absent", "present", "uncertain", "absent", "absent", "present"],
        "cui": ["C1", "C2", "C1", "C1", "C2", "C9"]})
    got = iext.silver_statuses(mcuis, ["s1", "s2", "s3", "s4", "s5"], {"C1": {"C1", "C2"}})
    assert got[:, 0].tolist() == ["PRESENT", "UNCERTAIN", "ABSENT", "NOT_MENTIONED",
                                  "NOT_MENTIONED"]


def test_lenient_auroc_drops_uncertain_and_counts_unmentioned_as_negative():
    statuses = np.array(["PRESENT", "PRESENT", "UNCERTAIN", "ABSENT", "NOT_MENTIONED"],
                        dtype=object)
    logits = np.array([0.9, 0.2, 5.0, 0.1, 0.5])
    res = iext.lenient_auroc(logits, statuses, n_boot=50, seed=1)
    assert res["auroc_lenient"] == pytest.approx(0.75)
    assert (res["n_present"], res["n_negative"]) == (2, 2)
    assert 0.0 <= res["ci95"][0] <= res["ci95"][1] <= 1.0
