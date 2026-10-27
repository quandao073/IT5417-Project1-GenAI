"""`cxr kg-chexpert-candidates` (TIP V2-2 step 5) on invented MRCONSO rows; no real UMLS content."""

from __future__ import annotations

import polars as pl
import pytest
import yaml

from src.cli.main import main
from src.data.labels import CHEXPERT_LABELS
from src.kg import chexpert_candidates as cc

CONCEPTS = {"concepts": {
    "Pleural Effusion": {"en": ["pleural effusion", "effusion"]},
    "Support Devices": {"en": ["support devices", "lines and tubes"]},
    "No Finding": {"en": ["no finding"]},
}}


def conso(cui, text, sab, preferred=False):
    return {"CUI": cui, "STR": text, "SAB": sab, "TS": "P" if preferred else "S",
            "ISPREF": "Y" if preferred else "N", "STT": "PF" if preferred else "VO"}


ROWS = [
    conso("C9000001", "Pleural effusion", "MSH", preferred=True),
    conso("C9000001", "Effusion", "MSH"),
    conso("C9000002", "Effusion", "NCI", preferred=True),
    conso("C9000003", "pleural effusion", "HPO"),
    conso("C9000005", "effusion", "FMA"),
    conso("C9000010", "Device", "NCI", preferred=True),
    conso("C9000011", "Tube", "MSH", preferred=True),
]
STY = [{"CUI": c, "TUI": t, "STY": s} for c, t, s in [
    ("C9000001", "T046", "Pathologic Function"), ("C9000002", "T046", "Pathologic Function"),
    ("C9000003", "T184", "Sign or Symptom"), ("C9000005", "T033", "Finding"),
    ("C9000010", "T074", "Medical Device"), ("C9000011", "T074", "Medical Device")]]


@pytest.fixture
def umls(tmp_path):
    pl.DataFrame(ROWS).write_parquet(tmp_path / "mrconso.parquet")
    pl.DataFrame(STY).write_parquet(tmp_path / "mrsty.parquet")
    return tmp_path


def test_exact_matches_are_ranked_and_cut_at_three(umls):
    found = cc.candidates(CONCEPTS, umls / "mrconso.parquet", umls / "mrsty.parquet")
    effusion = found.filter(pl.col("label") == "Pleural Effusion").sort("rank")
    # 2 forms > preferred > number of sources > CUI; C9000005 is the fourth and is cut
    assert effusion["cui"].to_list() == ["C9000001", "C9000002", "C9000003"]
    assert effusion["kind"].unique().to_list() == ["cui"]
    first = effusion.row(0, named=True)
    assert first["name"] == "Pleural effusion" and "T046" in first["semantic_types"]


def test_a_label_without_an_exact_match_gets_a_project_id_and_is_a_candidates(umls):
    found = cc.candidates(CONCEPTS, umls / "mrconso.parquet", umls / "mrsty.parquet")
    devices = found.filter(pl.col("label") == "Support Devices").sort("rank")
    assert devices["kind"].unique().to_list() == ["is_a"]
    assert devices["proposed_id"].unique().to_list() == ["PRJ:support_devices"]
    assert devices["cui"].to_list() == ["C9000010", "C9000011"]   # from "device", "tube"


def test_no_finding_is_the_special_project_id_without_candidates(umls):
    found = cc.candidates(CONCEPTS, umls / "mrconso.parquet", umls / "mrsty.parquet")
    row = found.filter(pl.col("label") == "No Finding")
    assert row.height == 1 and row["proposed_id"][0] == "PRJ:no_finding"
    assert row["cui"][0] is None


def test_a_label_with_neither_kind_of_candidate_stops(umls):
    concepts = {"concepts": {"Pleural Other": {"en": ["pleural other"]}}}
    with pytest.raises(cc.CandidateError, match="Pleural Other"):
        cc.candidates(concepts, umls / "mrconso.parquet", umls / "mrsty.parquet")


def valid_mapping():
    labels = {label: {"id": f"C90000{i:02d}"} for i, label in enumerate(CHEXPERT_LABELS)}
    labels["Support Devices"] = {"id": "PRJ:support_devices", "is_a": "C9000010"}
    labels["No Finding"] = {"id": "PRJ:no_finding"}
    return {"labels": labels}


def test_check_mapping_accepts_codes_only():
    assert cc.check_mapping(valid_mapping()) == []


@pytest.mark.parametrize("label, entry, problem", [
    ("Edema", None, "missing"),
    ("Edema", {"id": "C9000099", "is_a": "C9000010"}, "is_a"),
    ("Support Devices", {"id": "PRJ:support_devices"}, "is_a"),
    ("No Finding", {"id": "PRJ:no_finding", "is_a": "C9000010"}, "No Finding"),
    ("Edema", {"id": "pulmonary edema"}, "neither"),
])
def test_check_mapping_names_each_problem(label, entry, problem):
    mapping = valid_mapping()
    if entry is None:
        del mapping["labels"][label]
    else:
        mapping["labels"][label] = entry
    assert any(problem in p and label in p for p in cc.check_mapping(mapping))


def test_cli_writes_the_candidate_file_without_printing_names(umls, tmp_path, monkeypatch,
                                                              capsys):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "concepts.yaml").write_text(yaml.safe_dump(CONCEPTS), encoding="utf-8")
    monkeypatch.setenv("CONFIG_DIR", str(configs))
    out = tmp_path / "vocab" / "chexpert_candidates.csv"
    assert main(["kg-chexpert-candidates", "--umls-dir", str(umls), "--out", str(out)]) == 0
    assert pl.read_csv(out).height == 6   # 3 + 2 + No Finding
    printed = capsys.readouterr().out
    assert "Pleural Effusion" in printed and "Pleural effusion" not in printed
