"""The canonical corpus build, end to end on a miniature dataset.

The fixture mirrors the real layout: a CheXpert-small tree with real JPEGs and
train/valid CSVs, a CheXpert Plus CSV, and the three CheXbert JSONL files. What
is asserted here is the data contract the rest of the project reads.
"""

from __future__ import annotations

import json

import polars as pl
import pytest
from PIL import Image

from src.data import canonical, contracts, labels

CHEXPERT_HEADER = (
    "Path,Sex,Age,Frontal/Lateral,AP/PA,No Finding,Enlarged Cardiomediastinum,"
    "Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,"
    "Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices"
)
PLUS_HEADER = (
    "path_to_image,frontal_lateral,ap_pa,deid_patient_id,section_findings,section_impression"
)


class Spec:
    """One image in the miniature dataset."""

    def __init__(self, split, patient, study, view, projection, present, findings=None):
        self.split, self.patient, self.study, self.view = split, patient, study, view
        self.projection, self.present, self.findings = projection, present, findings

    @property
    def path(self):
        return f"{self.split}/{self.patient}/{self.study}/view{self.view}_frontal.jpg"

    @property
    def study_id(self):
        return f"{self.patient}/{self.study}"

    @property
    def impression(self):
        return f"\n1. {', '.join(self.present) or 'NO ACUTE ABNORMALITY'}.\n"


def label_cells(present):
    return [("1.0" if name in present else "") for name in labels.CHEXPERT_LABELS]


@pytest.fixture
def dataset(tmp_path):
    """Build a miniature CheXpert-small + CheXpert Plus on disk."""
    specs = []
    # Five train patients, two studies each, the first study having two views.
    for p in range(1, 6):
        patient = f"patient{p:05d}"
        present = [labels.CHEXPERT_LABELS[p % len(labels.CHEXPERT_LABELS)]]
        specs.append(Spec("train", patient, "study1", 1, "AP", present, "Findings text."))
        specs.append(Spec("train", patient, "study1", 2, "AP", present))
        specs.append(Spec("train", patient, "study2", 1, "PA", ["Cardiomegaly"]))
    # One radiologist-labelled valid patient: must survive any --limit.
    specs.append(Spec("valid", "patient64541", "study1", 1, "PA", ["Edema"]))
    # A lateral row and a row with no impression: both must be filtered out.
    excluded = [
        Spec("train", "patient00009", "study1", 1, "LATERAL", ["Edema"]),
        Spec("train", "patient00010", "study1", 1, "AP", []),
    ]

    small_root = tmp_path / "CheXpert-v1.0-small"
    for spec in specs + excluded:
        target = small_root / spec.path
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (320 + spec.view, 390), color=7 * spec.view).save(target)

    for split in ("train", "valid"):
        rows = [
            f"CheXpert-v1.0-small/{s.path},Female,60,Frontal,{s.projection},"
            + ",".join(label_cells(s.present))
            for s in specs + excluded
            if s.split == split
        ]
        (small_root / f"{split}.csv").write_text(
            "\n".join([CHEXPERT_HEADER, *rows]) + "\n", encoding="utf-8"
        )

    plus_rows = []
    for s in specs + excluded:
        blank = not s.present and s.patient.endswith("10")
        impression = "" if blank else s.impression
        plus_rows.append(
            f'{s.path},{"Frontal" if s.projection != "LATERAL" else "Lateral"},'
            f'{s.projection},{s.patient},"{s.findings or ""}","{impression}"'
        )
    plus_csv = tmp_path / "df_chexpert_plus.csv"
    plus_csv.write_text("\n".join([PLUS_HEADER, *plus_rows]) + "\n", encoding="utf-8")

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for section in ("findings", "impression", "report"):
        records = [
            {
                "path_to_image": s.path,
                **{n: (1.0 if n in s.present else None) for n in labels.CHEXPERT_LABELS},
            }
            for s in specs + excluded
        ]
        (labels_dir / f"{section}_fixed.json").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )

    return {
        "plus_csv": plus_csv,
        "small_root": small_root,
        "chexbert": {
            "CHEXBERT_FINDINGS": labels_dir / "findings_fixed.json",
            "CHEXBERT_IMPRESSION": labels_dir / "impression_fixed.json",
            "CHEXBERT_REPORT": labels_dir / "report_fixed.json",
        },
        "specs": specs,
    }


@pytest.fixture
def build(dataset):
    return canonical.build_corpus(
        plus_csv=dataset["plus_csv"],
        small_root=dataset["small_root"],
        chexbert=dataset["chexbert"],
        limit=10,
        seed=20260101,
    )


class TestContracts:
    @pytest.mark.parametrize("table", sorted(contracts.TABLES))
    def test_every_table_satisfies_its_contract(self, build, table):
        contracts.validate(getattr(build, table), table)

    def test_the_corpus_holds_exactly_the_requested_number_of_studies(self, build):
        assert build.studies.height == 10
        assert build.images.height == 10
        assert build.reports.height == 10

    def test_one_image_per_study(self, build):
        assert build.images.get_column("study_id").n_unique() == build.images.height

    def test_every_selected_image_exists_and_is_readable(self, build, dataset):
        for path in build.images.get_column("local_image_path"):
            with Image.open(path) as image:
                image.verify()

    def test_recorded_dimensions_match_the_files(self, build):
        for row in build.images.iter_rows(named=True):
            with Image.open(row["local_image_path"]) as image:
                assert image.size == (row["width"], row["height"])

    def test_every_projection_is_ap_or_pa(self, build):
        assert set(build.images.get_column("projection")) <= {"AP", "PA"}

    def test_the_lateral_and_impression_less_rows_are_excluded(self, build):
        studies = set(build.studies.get_column("study_id"))
        assert "patient00009/study1" not in studies
        assert "patient00010/study1" not in studies

    def test_the_radiologist_labelled_study_is_always_included(self, build):
        assert "patient64541/study1" in set(build.studies.get_column("study_id"))

    def test_image_and_study_ids_are_consistent_across_tables(self, build):
        studies = set(build.studies.get_column("study_id"))
        assert set(build.images.get_column("study_id")) == studies
        assert set(build.reports.get_column("study_id")) == studies
        assert set(build.sample_manifest.get_column("study_id")) >= studies
        assert set(build.labels.get_column("study_id")) == studies
        selected = set(build.studies.get_column("selected_image_id"))
        assert selected == set(build.images.get_column("image_id"))


class TestLabels:
    def test_every_source_is_kept_with_its_provenance(self, build):
        assert set(build.labels.get_column("source")) == contracts.LABEL_SOURCES

    def test_no_assertion_overwrites_another(self, build):
        """One row per (study, concept, source): conflicts are data, not errors."""
        key = build.labels.select("study_id", "concept", "source")
        assert not key.is_duplicated().any()

    def test_every_study_carries_all_fourteen_concepts_per_covering_source(self, build):
        counts = build.labels.group_by("study_id", "source").len().get_column("len")
        assert set(counts) == {14}

    def test_the_findings_labeller_is_silenced_where_there_is_no_findings_section(self, build):
        """CheXbert run on an empty findings section emits "No Finding = present"
        for 100% of such studies - measured 18,598/18,598 on the real corpus.

        Those assertions are an artefact of empty input, not a reading of the
        image, and they outrank the honest sources in `label_resolution`. A
        source with no input has no assertion, so it contributes no rows.
        """
        without = build.studies.filter(~pl.col("has_findings")).get_column("study_id")
        assert len(without) > 0, "fixture must contain a study with no findings section"
        leaked = build.labels.filter(
            (pl.col("source") == "CHEXBERT_FINDINGS") & pl.col("study_id").is_in(without.to_list())
        )
        assert leaked.height == 0

    def test_the_findings_labeller_is_kept_where_a_findings_section_exists(self, build):
        with_findings = build.studies.filter(pl.col("has_findings")).get_column("study_id")
        assert len(with_findings) > 0
        kept = build.labels.filter(
            (pl.col("source") == "CHEXBERT_FINDINGS")
            & pl.col("study_id").is_in(with_findings.to_list())
        )
        assert kept.height == 14 * len(with_findings)


class TestReports:
    def test_search_text_is_impression_first(self, build):
        for row in build.reports.iter_rows(named=True):
            assert row["search_text"].startswith(row["impression"])

    def test_findings_are_appended_when_present(self, build):
        with_findings = build.reports.filter(pl.col("findings").is_not_null())
        assert with_findings.height > 0
        for row in with_findings.iter_rows(named=True):
            assert row["findings"] in row["search_text"]

    def test_impression_is_never_blank(self, build):
        assert all(text.strip() for text in build.reports.get_column("impression"))

    def test_one_report_per_study(self, build):
        assert build.reports.get_column("study_id").n_unique() == build.reports.height


class TestReproducibility:
    def test_the_same_seed_rebuilds_the_same_corpus(self, dataset):
        kwargs = dict(
            plus_csv=dataset["plus_csv"],
            small_root=dataset["small_root"],
            chexbert=dataset["chexbert"],
            limit=10,
            seed=20260101,
        )
        first = canonical.build_corpus(**kwargs)
        second = canonical.build_corpus(**kwargs)
        for table in sorted(contracts.TABLES):
            assert getattr(first, table).equals(getattr(second, table)), table

    def test_a_prefix_of_the_manifest_is_what_a_smaller_limit_selects(self, dataset):
        kwargs = dict(
            plus_csv=dataset["plus_csv"],
            small_root=dataset["small_root"],
            chexbert=dataset["chexbert"],
            seed=20260101,
        )
        big = canonical.build_corpus(limit=10, **kwargs)
        small = canonical.build_corpus(limit=4, **kwargs)
        assert set(small.studies.get_column("study_id")) <= set(
            big.studies.get_column("study_id")
        )


class TestCorpusReport:
    def test_records_how_the_pool_was_narrowed(self, build):
        counts = build.report["counts"]
        assert counts["plus_rows"] > counts["eligible_rows"] >= counts["eligible_studies"]
        assert counts["selected_studies"] == 10

    def test_records_the_stratum_priority_so_the_sample_can_be_reproduced(self, build):
        assert build.report["stratum_priority"]

    def test_records_absent_prevalence_per_concept(self, build):
        """Phase 1 must supply the number that settles the absence_policy default."""
        prevalence = build.report["status_prevalence"]
        assert set(prevalence) == set(labels.CHEXPERT_LABELS)
        assert all("ABSENT" in counts for counts in prevalence.values())

    def test_records_the_projection_mix(self, build):
        assert set(build.report["projection_mix"]) <= {"AP", "PA"}


class TestWriteAndCli:
    def test_write_corpus_emits_every_table(self, build, tmp_path):
        written = canonical.write_corpus(build, tmp_path / "canonical")
        assert {p.name for p in written} == {f"{t}.parquet" for t in contracts.TABLES}
        for path in written:
            assert pl.read_parquet(path).height > 0

    def test_the_cli_writes_the_tables_and_the_audit_report(self, dataset, tmp_path):
        import json as _json

        from src.cli.main import main

        out = tmp_path / "canonical"
        report = tmp_path / "corpus_report.json"
        code = main(
            [
                "build-corpus",
                "--plus-csv", str(dataset["plus_csv"]),
                "--small-root", str(dataset["small_root"]),
                "--chexbert-dir", str(dataset["chexbert"]["CHEXBERT_IMPRESSION"].parent),
                "--limit", "8",
                "--seed", "20260101",
                "--out", str(out),
                "--report", str(report),
            ]
        )
        assert code == 0
        assert (out / "images.parquet").is_file()
        assert pl.read_parquet(out / "studies.parquet").height == 8
        assert _json.loads(report.read_text(encoding="utf-8"))["counts"]["selected_studies"] == 8

    def test_the_cli_refuses_a_limit_larger_than_the_pool(self, dataset, tmp_path):
        from src.cli.main import main

        code = main(
            [
                "build-corpus",
                "--plus-csv", str(dataset["plus_csv"]),
                "--small-root", str(dataset["small_root"]),
                "--chexbert-dir", str(dataset["chexbert"]["CHEXBERT_IMPRESSION"].parent),
                "--limit", "9999",
                "--out", str(tmp_path / "canonical"),
                "--report", str(tmp_path / "corpus_report.json"),
            ]
        )
        assert code == 1
