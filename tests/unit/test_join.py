"""Building the eligible study pool from CheXpert Plus.

Eligibility is three filters: frontal AP/PA, a non-empty impression, and an
image that is actually on disk. Findings is never a filter - it exists for only
26.6% of rows.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.data import join

PLUS_HEADER = (
    "path_to_image,frontal_lateral,ap_pa,deid_patient_id,section_findings,section_impression"
)


def write_plus_csv(path, rows):
    path.write_text("\n".join([PLUS_HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def rows(records):
    """Each record is (path, study_id, patient_id, projection, findings, impression).

    The CSV row index is appended automatically so the test bodies stay readable.
    """
    return pl.DataFrame(
        [record + (index,) for index, record in enumerate(records)],
        schema={
            "path_to_image": pl.Utf8,
            "study_id": pl.Utf8,
            "patient_id": pl.Utf8,
            "projection": pl.Utf8,
            "findings": pl.Utf8,
            "impression": pl.Utf8,
            "source_row_index": pl.Int64,
        },
        orient="row",
    )


FRONTAL_AP = ("train/patient1/study1/view1_frontal.jpg", "patient1/study1", "patient1", "AP")


class TestReadPlusRows:
    def test_derives_study_and_patient_from_the_path(self, tmp_path):
        csv = write_plus_csv(
            tmp_path / "plus.csv",
            ["train/patient00003/study1/view1_frontal.jpg,Frontal,AP,pat3,,Mild edema."],
        )
        row = join.read_plus_rows(csv).row(0, named=True)
        assert row["study_id"] == "patient00003/study1"
        assert row["patient_id"] == "patient00003"
        assert row["projection"] == "AP"
        assert row["impression"] == "Mild edema."

    def test_strips_the_leading_newline_impressions_carry(self, tmp_path):
        csv = write_plus_csv(
            tmp_path / "plus.csv",
            ['train/patient1/study1/view1_frontal.jpg,Frontal,PA,p1,,"\n1. NO PNEUMOTHORAX.\n"'],
        )
        assert join.read_plus_rows(csv).row(0, named=True)["impression"] == "1. NO PNEUMOTHORAX."

    def test_records_the_csv_row_each_report_came_from(self, tmp_path):
        """reports.parquet carries source_row_index so a row can be traced back."""
        csv = write_plus_csv(
            tmp_path / "plus.csv",
            [
                "train/patient1/study1/view1_frontal.jpg,Frontal,PA,p1,,First.",
                "train/patient2/study1/view1_frontal.jpg,Frontal,AP,p2,,Second.",
            ],
        )
        assert join.read_plus_rows(csv).get_column("source_row_index").to_list() == [0, 1]

    def test_keeps_an_absent_findings_section_as_null(self, tmp_path):
        csv = write_plus_csv(
            tmp_path / "plus.csv",
            ["train/patient1/study1/view1_frontal.jpg,Frontal,PA,p1,,Impression text."],
        )
        assert join.read_plus_rows(csv).row(0, named=True)["findings"] is None


class TestFilterEligible:
    AVAILABLE = {
        "train/patient1/study1/view1_frontal.jpg",
        "train/patient2/study1/view1_frontal.jpg",
        "train/patient3/study1/view1_lateral.jpg",
    }

    def test_keeps_a_frontal_ap_pa_row_with_an_impression_and_a_file(self):
        df = rows([(*FRONTAL_AP, None, "Mild edema.")])
        assert join.filter_eligible(df, self.AVAILABLE).height == 1

    @pytest.mark.parametrize("projection", ["LATERAL", "", None])
    def test_drops_a_projection_outside_ap_and_pa(self, projection):
        df = rows([(*FRONTAL_AP[:3], projection, None, "Mild edema.")])
        assert join.filter_eligible(df, self.AVAILABLE).height == 0

    @pytest.mark.parametrize("impression", [None, "", "   ", "\n"])
    def test_drops_a_row_without_an_impression(self, impression):
        df = rows([(*FRONTAL_AP, "Findings text.", impression)])
        assert join.filter_eligible(df, self.AVAILABLE).height == 0

    def test_drops_a_row_whose_image_is_not_on_disk(self):
        df = rows(
            [
                (
                    "train/patient9/study1/view1_frontal.jpg",
                    "patient9/study1",
                    "patient9",
                    "AP",
                    None,
                    "Mild edema.",
                )
            ]
        )
        assert join.filter_eligible(df, self.AVAILABLE).height == 0

    def test_never_requires_findings(self):
        """Findings exists for only 26.6% of rows; requiring it would gut the corpus."""
        df = rows([(*FRONTAL_AP, None, "Mild edema.")])
        assert join.filter_eligible(df, self.AVAILABLE).height == 1


class TestSelectOneImagePerStudy:
    def test_picks_the_lowest_view_index(self):
        df = rows(
            [
                ("train/p1/study1/view3_frontal.jpg", "p1/study1", "p1", "AP", None, "x"),
                ("train/p1/study1/view1_frontal.jpg", "p1/study1", "p1", "AP", None, "x"),
                ("train/p1/study1/view2_frontal.jpg", "p1/study1", "p1", "AP", None, "x"),
            ]
        )
        out = join.select_one_image_per_study(df)
        assert out.height == 1
        assert out.row(0, named=True)["path_to_image"] == "train/p1/study1/view1_frontal.jpg"

    def test_breaks_ties_on_the_path_so_the_choice_is_deterministic(self):
        df = rows(
            [
                ("train/p1/study1/view1_lateral.jpg", "p1/study1", "p1", "AP", None, "x"),
                ("train/p1/study1/view1_frontal.jpg", "p1/study1", "p1", "AP", None, "x"),
            ]
        )
        first = join.select_one_image_per_study(df).row(0, named=True)["path_to_image"]
        second = join.select_one_image_per_study(df.reverse()).row(0, named=True)["path_to_image"]
        assert first == second == "train/p1/study1/view1_frontal.jpg"

    def test_keeps_every_study(self):
        df = rows(
            [
                ("train/p1/study1/view1_frontal.jpg", "p1/study1", "p1", "AP", None, "x"),
                ("train/p1/study2/view1_frontal.jpg", "p1/study2", "p1", "PA", None, "y"),
                ("train/p2/study1/view1_frontal.jpg", "p2/study1", "p2", "AP", None, "z"),
            ]
        )
        out = join.select_one_image_per_study(df)
        assert out.height == 3
        assert out.get_column("study_id").n_unique() == 3
