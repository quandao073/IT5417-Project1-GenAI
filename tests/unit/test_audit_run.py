"""Audit report: join measurement and corpus decision combined into one artifact."""

import json

from cxr_retrieval.data.audit import run_audit


def _make_small_tree(root, patients):
    for patient in patients:
        study = root / "train" / patient / "study1"
        study.mkdir(parents=True)
        (study / "view1_frontal.jpg").write_bytes(b"")


def _make_plus_csv(path, patients):
    rows = "\n".join(f"PNG/train/{p}/study1/view1_frontal.png" for p in patients)
    path.write_text(f"path_to_image\n{rows}\n", encoding="utf-8")


def test_full_match_reports_the_winning_strategy_and_unlocks_full_corpus(tmp_path):
    small = tmp_path / "small"
    _make_small_tree(small, ["patient00001", "patient00002"])
    csv = tmp_path / "plus.csv"
    _make_plus_csv(csv, ["patient00001", "patient00002"])

    report = run_audit(csv, small)

    assert report["best_strategy"] == "image_key"
    assert report["join_rate"] == 1.0
    assert report["decision"] == "full_frontal"
    assert report["plus_total"] == 2
    assert report["small_total"] == 2


def test_poor_match_records_the_abort_decision(tmp_path):
    small = tmp_path / "small"
    _make_small_tree(small, ["patient00001"])
    csv = tmp_path / "plus.csv"
    _make_plus_csv(csv, ["patient00001"] + [f"patient9000{i}" for i in range(4)])

    report = run_audit(csv, small)

    assert report["join_rate"] == 0.2
    assert report["decision"] == "abort"


def test_every_strategy_is_reported_so_the_choice_is_auditable(tmp_path):
    small = tmp_path / "small"
    _make_small_tree(small, ["patient00001"])
    csv = tmp_path / "plus.csv"
    _make_plus_csv(csv, ["patient00001"])

    report = run_audit(csv, small)

    assert {s["strategy"] for s in report["strategies"]} == {
        "exact",
        "posix_lower",
        "image_key",
    }


def test_report_is_json_serializable(tmp_path):
    small = tmp_path / "small"
    _make_small_tree(small, ["patient00001"])
    csv = tmp_path / "plus.csv"
    _make_plus_csv(csv, ["patient00001"])

    assert json.loads(json.dumps(run_audit(csv, small)))["join_rate"] == 1.0
