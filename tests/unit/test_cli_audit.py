"""CLI `cxr audit` - cong chan phai dung pipeline khi khong dat."""

import json

from cxr_retrieval.cli.main import main


def _fixture(tmp_path, plus_patients, small_patients):
    small = tmp_path / "small"
    for patient in small_patients:
        study = small / "train" / patient / "study1"
        study.mkdir(parents=True)
        (study / "view1_frontal.jpg").write_bytes(b"")
    csv = tmp_path / "plus.csv"
    rows = "\n".join(f"PNG/train/{p}/study1/view1_frontal.png" for p in plus_patients)
    csv.write_text(f"path_to_image\n{rows}\n", encoding="utf-8")
    return csv, small


def test_writes_the_audit_artifact(tmp_path):
    csv, small = _fixture(tmp_path, ["patient00001"], ["patient00001"])
    out = tmp_path / "data_audit.json"

    main(["audit", "--plus-csv", str(csv), "--small-root", str(small), "--out", str(out)])

    assert json.loads(out.read_text(encoding="utf-8"))["decision"] == "full_frontal"


def test_exits_zero_when_the_gate_passes(tmp_path):
    csv, small = _fixture(tmp_path, ["patient00001"], ["patient00001"])
    out = tmp_path / "data_audit.json"

    code = main(
        ["audit", "--plus-csv", str(csv), "--small-root", str(small), "--out", str(out)]
    )

    assert code == 0


def test_exits_nonzero_when_the_gate_aborts(tmp_path):
    csv, small = _fixture(
        tmp_path, ["patient00001"] + [f"patient9000{i}" for i in range(4)], ["patient00001"]
    )
    out = tmp_path / "data_audit.json"

    code = main(
        ["audit", "--plus-csv", str(csv), "--small-root", str(small), "--out", str(out)]
    )

    assert code != 0


def test_artifact_is_still_written_when_the_gate_aborts(tmp_path):
    csv, small = _fixture(
        tmp_path, ["patient00001"] + [f"patient9000{i}" for i in range(4)], ["patient00001"]
    )
    out = tmp_path / "nested" / "data_audit.json"

    main(["audit", "--plus-csv", str(csv), "--small-root", str(small), "--out", str(out)])

    assert json.loads(out.read_text(encoding="utf-8"))["decision"] == "abort"
