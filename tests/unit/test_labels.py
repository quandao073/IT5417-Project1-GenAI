"""Four-state label mapping and multi-source resolution.

The whole project's argument rests on NOT_MENTIONED never being treated as
ABSENT, so that rule is tested explicitly rather than left implicit.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from src.data import labels

CHEXPERT_HEADER = (
    "Path,Sex,Age,Frontal/Lateral,AP/PA,No Finding,Enlarged Cardiomediastinum,"
    "Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,"
    "Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices"
)


def write_chexpert_csv(path, rows):
    path.write_text("\n".join([CHEXPERT_HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def write_chexbert_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


class TestToStatus:
    @pytest.mark.parametrize("value", [1.0, 1, "1.0", "1"])
    def test_one_is_present(self, value):
        assert labels.to_status(value) == "PRESENT"

    @pytest.mark.parametrize("value", [0.0, 0, "0.0", "0"])
    def test_zero_is_absent(self, value):
        assert labels.to_status(value) == "ABSENT"

    @pytest.mark.parametrize("value", [-1.0, -1, "-1.0", "-1"])
    def test_minus_one_is_uncertain(self, value):
        assert labels.to_status(value) == "UNCERTAIN"

    @pytest.mark.parametrize("value", [None, "", "   ", float("nan")])
    def test_missing_is_not_mentioned(self, value):
        assert labels.to_status(value) == "NOT_MENTIONED"

    @pytest.mark.parametrize("value", [None, "", float("nan")])
    def test_missing_is_never_absent(self, value):
        """The single rule the whole retrieval argument depends on."""
        assert labels.to_status(value) != "ABSENT"

    @pytest.mark.parametrize("value", [2.0, -2, "yes", "PRESENT"])
    def test_an_unexpected_value_is_an_error(self, value):
        """Fail loudly: silently folding an unknown code into a state hides data bugs."""
        with pytest.raises(ValueError):
            labels.to_status(value)


class TestReadChexpertCsv:
    def test_emits_one_row_per_image_and_concept(self, tmp_path):
        csv = write_chexpert_csv(
            tmp_path / "train.csv",
            [
                "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg,"
                "Female,68,Frontal,AP,1.0,,,,,,,,,0.0,,,,1.0"
            ],
        )
        df = labels.read_chexpert_csv(csv)
        assert df.height == len(labels.CHEXPERT_LABELS) == 14
        assert set(df.columns) == {"path_to_image", "concept", "status", "source", "source_version"}

    def test_normalizes_the_path(self, tmp_path):
        csv = write_chexpert_csv(
            tmp_path / "train.csv",
            [
                "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg,"
                "Female,68,Frontal,AP,1.0,,,,,,,,,0.0,,,,1.0"
            ],
        )
        paths = labels.read_chexpert_csv(csv).get_column("path_to_image").unique().to_list()
        assert paths == ["train/patient00001/study1/view1_frontal.jpg"]

    def test_maps_the_four_states(self, tmp_path):
        csv = write_chexpert_csv(
            tmp_path / "train.csv",
            [
                "CheXpert-v1.0-small/train/patient00002/study1/view1_frontal.jpg,"
                "Male,40,Frontal,PA,,,-1.0,1.0,,,,,,,0.0,,,"
            ],
        )
        got = dict(labels.read_chexpert_csv(csv).select("concept", "status").iter_rows())
        assert got["Cardiomegaly"] == "UNCERTAIN"
        assert got["Lung Opacity"] == "PRESENT"
        assert got["Pleural Effusion"] == "ABSENT"
        assert got["No Finding"] == "NOT_MENTIONED"

    def test_records_its_own_provenance(self, tmp_path):
        csv = write_chexpert_csv(
            tmp_path / "valid.csv",
            [
                "CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg,"
                "Male,40,Frontal,PA,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"
            ],
        )
        df = labels.read_chexpert_csv(csv)
        assert df.get_column("source").unique().to_list() == ["CHEXPERT_V1"]
        assert df.get_column("source_version").unique().to_list() == ["valid.csv"]


class TestReadChexbertJsonl:
    def test_emits_one_row_per_image_and_concept(self, tmp_path):
        jsonl = write_chexbert_jsonl(
            tmp_path / "impression_fixed.json",
            [
                {
                    "path_to_image": "train/patient42142/study5/view1_frontal.jpg",
                    **{name: None for name in labels.CHEXPERT_LABELS},
                }
            ],
        )
        df = labels.read_chexbert_jsonl(jsonl, "CHEXBERT_IMPRESSION")
        assert df.height == 14
        assert df.get_column("status").unique().to_list() == ["NOT_MENTIONED"]
        assert df.get_column("source").unique().to_list() == ["CHEXBERT_IMPRESSION"]

    def test_maps_values_and_keeps_every_concept(self, tmp_path):
        record = {name: None for name in labels.CHEXPERT_LABELS}
        record.update({"Lung Opacity": 0.0, "Support Devices": 1.0, "Edema": -1.0})
        jsonl = write_chexbert_jsonl(
            tmp_path / "impression_fixed.json",
            [{"path_to_image": "train/patient00001/study1/view1_frontal.jpg", **record}],
        )
        got = dict(
            labels.read_chexbert_jsonl(jsonl, "CHEXBERT_IMPRESSION")
            .select("concept", "status")
            .iter_rows()
        )
        assert got["Lung Opacity"] == "ABSENT"
        assert got["Support Devices"] == "PRESENT"
        assert got["Edema"] == "UNCERTAIN"
        assert got["Pneumonia"] == "NOT_MENTIONED"

    def test_rejects_a_source_outside_the_contract(self, tmp_path):
        jsonl = write_chexbert_jsonl(tmp_path / "x.json", [])
        with pytest.raises(ValueError):
            labels.read_chexbert_jsonl(jsonl, "MADE_UP_SOURCE")


def long_frame(rows):
    return pl.DataFrame(
        rows,
        schema={"study_id": pl.Utf8, "concept": pl.Utf8, "status": pl.Utf8, "source": pl.Utf8},
        orient="row",
    )


class TestResolve:
    PRIORITY = ("CHEXPERT_V1", "CHEXBERT_IMPRESSION", "CHEXBERT_FINDINGS", "CHEXBERT_REPORT")

    def test_returns_exactly_one_status_per_study_and_concept(self):
        df = long_frame(
            [
                ("p1/s1", "Edema", "PRESENT", "CHEXPERT_V1"),
                ("p1/s1", "Edema", "ABSENT", "CHEXBERT_IMPRESSION"),
                ("p1/s1", "Edema", "UNCERTAIN", "CHEXBERT_REPORT"),
            ]
        )
        out = labels.resolve(df, self.PRIORITY)
        assert out.height == 1
        assert set(out.columns) == {"study_id", "concept", "status"}

    def test_the_highest_priority_source_wins(self):
        df = long_frame(
            [
                ("p1/s1", "Edema", "PRESENT", "CHEXPERT_V1"),
                ("p1/s1", "Edema", "ABSENT", "CHEXBERT_IMPRESSION"),
            ]
        )
        assert labels.resolve(df, self.PRIORITY).row(0)[2] == "PRESENT"

    def test_skips_a_higher_source_that_says_nothing(self):
        df = long_frame(
            [
                ("p1/s1", "Edema", "NOT_MENTIONED", "CHEXPERT_V1"),
                ("p1/s1", "Edema", "PRESENT", "CHEXBERT_IMPRESSION"),
            ]
        )
        assert labels.resolve(df, self.PRIORITY).row(0)[2] == "PRESENT"

    def test_all_silent_stays_not_mentioned(self):
        df = long_frame(
            [
                ("p1/s1", "Edema", "NOT_MENTIONED", "CHEXPERT_V1"),
                ("p1/s1", "Edema", "NOT_MENTIONED", "CHEXBERT_IMPRESSION"),
            ]
        )
        assert labels.resolve(df, self.PRIORITY).row(0)[2] == "NOT_MENTIONED"

    def test_silence_never_resolves_to_absent(self):
        df = long_frame([("p1/s1", "Edema", "NOT_MENTIONED", "CHEXPERT_V1")])
        assert labels.resolve(df, self.PRIORITY).row(0)[2] != "ABSENT"

    def test_keeps_studies_and_concepts_separate(self):
        df = long_frame(
            [
                ("p1/s1", "Edema", "PRESENT", "CHEXPERT_V1"),
                ("p1/s1", "Cardiomegaly", "ABSENT", "CHEXPERT_V1"),
                ("p2/s1", "Edema", "UNCERTAIN", "CHEXPERT_V1"),
            ]
        )
        out = labels.resolve(df, self.PRIORITY).sort("study_id", "concept")
        assert out.rows() == [
            ("p1/s1", "Cardiomegaly", "ABSENT"),
            ("p1/s1", "Edema", "PRESENT"),
            ("p2/s1", "Edema", "UNCERTAIN"),
        ]


def test_the_fourteen_labels_match_the_data_config():
    """labels.py must not drift from configs/data.yaml:labels.pathologies."""
    import yaml

    with open("configs/data.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert set(config["labels"]["pathologies"]) == set(labels.CHEXPERT_LABELS)
