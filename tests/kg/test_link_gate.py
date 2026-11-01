"""The V2-3 gate and the CheXpert phrase check (D56), on invented links."""

from __future__ import annotations

import json

import httpx
import polars as pl
import pytest

from src.common.config import load_config
from src.kg import chexpert_phrases as cp
from src.kg import link_gate as lg

GATE = {"precision": 0.85, "coverage_by_mentions": 0.90, "sample": 3, "seed": 20260101}


def links(n=6):
    return pl.DataFrame({
        "kind": ["Observation"] * n, "string": [f"s{i}" for i in range(n)],
        "mentions": list(range(1, n + 1)),
        "method": ["llm"] * (n - 2) + ["adjudicated", "override"],
        "decision": ["cui"] * n, "cui": [f"C000000{i}" for i in range(n)],
        "argmax_cui": ["C0000000", "C0000009", *[f"C000000{i}" for i in range(2, n)]]})


INFO = pl.DataFrame({"cui": [f"C000000{i}" for i in range(10)],
                     "name": [f"name {i}" for i in range(10)], "semantic_types": ["T033"] * 10})


def test_the_sample_is_seeded_and_takes_llm_links_only():
    first = lg.sample(links(), INFO, {("Observation", "s1"): [["lung", 2]]}, GATE)
    again = lg.sample(links(), INFO, {}, GATE)
    assert first["string"].to_list() == again["string"].to_list()
    assert first.height == 3 and set(first["string"]) <= {"s0", "s1", "s2", "s3"}
    row = {r["string"]: r for r in first.iter_rows(named=True)}
    if "s1" in row:
        assert row["s1"]["argmax_grade"] == "" and row["s1"]["located"] == "lung (2)"
    if "s0" in row:
        assert row["s0"]["argmax_grade"] == lg.SAME


def test_precision_wilson_and_the_gate():
    graded = pl.DataFrame({"link_grade": ["correct"] * 17 + ["incorrect"] * 3 + ["unsure"],
                           "argmax_grade": [lg.SAME] * 10 + ["incorrect"] * 11})
    link = lg.precision(graded, "link_grade")
    assert (link["correct"], link["graded"], link["unsure"], link["precision"]) == (17, 20, 1,
                                                                                    0.85)
    assert link["wilson95"] == pytest.approx([0.6396, 0.9476], abs=1e-4)
    assert lg.precision(graded, "argmax_grade")["precision"] == 10 / 21
    result = lg.gate(graded, {"all": {"decided": 0.91}}, GATE)
    assert result["precision_passed"] and result["coverage_passed"]


def test_judge_items_and_grades_round_trip():
    sample = lg.sample(links(), INFO, {}, GATE)
    items = lg.judge_items(sample)
    assert len(items) == sample.height + (sample["argmax_grade"] != lg.SAME).sum()
    grades = {(i["kind"], i["string"]): {"grade": "correct", "reason": "r"} for i in items}
    judged = lg.apply_grades(sample, grades, "llm:judge (silver)")
    assert set(judged["link_grade"]) == {"correct"} and set(judged["grader"]) == {
        "llm:judge (silver)"}
    assert set(judged["argmax_grade"]) <= {"correct", lg.SAME}


@pytest.fixture
def phrase_dir(tmp_path):
    texts = {"pleural_effusion": "pleural effusion\neffusion\n", "edema": "edema\n"}
    for stem in cp.LABEL_FILES:
        (tmp_path / f"{stem}.txt").write_text(texts.get(stem, ""), encoding="utf-8")
    return tmp_path


def test_phrases_load_by_label_and_mismatches_are_found(phrase_dir, tmp_path):
    phrases = cp.load(phrase_dir)
    assert "No Finding" not in phrases and phrases["Pleural Effusion"] == {"pleural effusion",
                                                                           "effusion"}
    umls = tmp_path / "umls"
    umls.mkdir()
    pl.DataFrame({"CUI1": ["C0032227"], "REL": ["CHD"], "CUI2": ["C0000099"]}).write_parquet(
        umls / "mrrel.parquet")
    mapping = {"Pleural Effusion": {"id": "C0032227"}, "Edema": {"id": "C0034063"},
               "Support Devices": {"id": "PRJ:support_devices", "is_a": "C0025080",
                                   "children": ["C0008034"]}}
    expected = cp.expected_cuis(mapping, umls, {"C0000050"}, "PRJ:support_devices")
    assert expected["Pleural Effusion"] == {"C0032227", "C0000099"}
    # children and any device of the domain, never the is_a parent
    assert expected["Support Devices"] == {"C0008034", "C0000050"}
    found = pl.DataFrame({"kind": ["Observation"] * 3, "string": ["effusion", "edema", "mass"],
                          "cui": ["C0000099", "C0013687", "C0000001"],
                          "argmax_cui": ["C0032227", "C0034063", None]})
    stats, bad = cp.mismatches(found, phrases, expected)
    assert bad == {("Observation", "edema")}
    assert cp.rate(stats) == {"strings": 2, "linked": 2, "ok": 1, "rate": 0.5}
    assert cp.rate(cp.mismatches(found, phrases, expected, "argmax_cui")[0])["rate"] == 1.0


def test_fetch_pins_the_commit_and_records_the_license(tmp_path):
    asked = []

    def handler(request):
        asked.append(str(request.url))
        return httpx.Response(200, text="phrase\n")

    cfg = load_config("kg")["linker"]["chexpert_phrases"]
    target = cp.fetch(cfg, tmp_path, transport=httpx.MockTransport(handler))
    assert len(asked) == len(cp.LABEL_FILES) and all(cfg["commit"] in url for url in asked)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["license"] == "MIT" and manifest["commit"] == cfg["commit"]
