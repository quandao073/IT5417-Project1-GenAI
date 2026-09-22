"""Schemas for the five canonical Parquet tables.

Everything downstream reads these tables, so a schema drift has to fail at the
writer rather than three phases later in the API.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.data import contracts

VALID_ROWS: dict[str, dict] = {
    "images": {
        "image_id": ["a" * 64],
        "study_id": ["patient00001/study1"],
        "patient_id": ["patient00001"],
        "path_to_image": ["train/patient00001/study1/view1_frontal.jpg"],
        "local_image_path": ["data/raw/CheXpert-v1.0-small/train/p/s/v.jpg"],
        "view": ["FRONTAL"],
        "projection": ["AP"],
        "dataset_split": ["train"],
        "image_checksum": ["b" * 64],
        "width": [320],
        "height": [390],
    },
    "studies": {
        "study_id": ["patient00001/study1"],
        "patient_id": ["patient00001"],
        "selected_image_id": ["a" * 64],
        "has_report": [True],
        "has_impression": [True],
        "has_findings": [False],
        "ingestion_status": ["OK"],
    },
    "labels": {
        "study_id": ["patient00001/study1"],
        "concept": ["Cardiomegaly"],
        "status": ["PRESENT"],
        "source": ["CHEXPERT_V1"],
        "source_version": ["train.csv"],
    },
    "reports": {
        "study_id": ["patient00001/study1"],
        "patient_id": ["patient00001"],
        "findings": [None],
        "impression": ["Mild cardiomegaly."],
        "search_text": ["Mild cardiomegaly."],
        "source_row_index": [17],
        "report_checksum": ["c" * 64],
        "dataset_version": ["df_chexpert_plus_240401"],
    },
    "sample_manifest": {
        "study_id": ["patient00001/study1"],
        "stratum": ["Cardiomegaly"],
        "rank_in_manifest": [0],
        "selection_seed": [20260101],
    },
}


def frame(table: str, **overrides) -> pl.DataFrame:
    """A minimal valid frame for `table`, with optional column overrides."""
    data = dict(VALID_ROWS[table]) | overrides
    schema = {name: contracts.TABLES[table].columns[name] for name in data}
    return pl.DataFrame(data, schema=schema)


class TestSpecs:
    def test_the_five_canonical_tables_are_defined(self):
        assert set(contracts.TABLES) == {
            "images",
            "studies",
            "labels",
            "reports",
            "sample_manifest",
        }

    @pytest.mark.parametrize("table", sorted(VALID_ROWS))
    def test_every_primary_key_column_exists_in_the_schema(self, table):
        spec = contracts.TABLES[table]
        assert spec.primary_key
        assert set(spec.primary_key) <= set(spec.columns)

    @pytest.mark.parametrize("table", sorted(VALID_ROWS))
    def test_every_constrained_column_exists_in_the_schema(self, table):
        spec = contracts.TABLES[table]
        assert set(spec.allowed_values) <= set(spec.columns)

    def test_no_table_carries_a_corpus_split_column(self):
        """The corpus is never split; see docs/adr/0002 and configs/evaluation.yaml."""
        for spec in contracts.TABLES.values():
            assert "project_split" not in spec.columns
            assert "corpus_tier" not in spec.columns


class TestValidate:
    @pytest.mark.parametrize("table", sorted(VALID_ROWS))
    def test_accepts_a_well_formed_frame(self, table):
        contracts.validate(frame(table), table)

    def test_rejects_an_unknown_table_name(self):
        with pytest.raises(KeyError):
            contracts.validate(frame("images"), "not_a_table")

    def test_rejects_a_missing_column(self):
        df = frame("images").drop("width")
        with pytest.raises(contracts.ContractError, match="width"):
            contracts.validate(df, "images")

    def test_rejects_an_unexpected_column(self):
        df = frame("images").with_columns(pl.lit("x").alias("corpus_tier"))
        with pytest.raises(contracts.ContractError, match="corpus_tier"):
            contracts.validate(df, "images")

    def test_rejects_a_wrong_dtype(self):
        df = frame("images").with_columns(pl.col("width").cast(pl.Utf8))
        with pytest.raises(contracts.ContractError, match="width"):
            contracts.validate(df, "images")

    def test_rejects_a_duplicate_primary_key(self):
        one = frame("images")
        with pytest.raises(contracts.ContractError, match="duplicate"):
            contracts.validate(pl.concat([one, one]), "images")

    def test_rejects_a_null_primary_key(self):
        df = frame("images", image_id=[None])
        with pytest.raises(contracts.ContractError, match="null"):
            contracts.validate(df, "images")

    def test_accepts_a_composite_primary_key_that_differs_in_one_column(self):
        rows = pl.concat([frame("labels"), frame("labels", source=["CHEXBERT_IMPRESSION"])])
        contracts.validate(rows, "labels")

    def test_rejects_a_projection_outside_ap_and_pa(self):
        df = frame("images", projection=["LATERAL"])
        with pytest.raises(contracts.ContractError, match="projection"):
            contracts.validate(df, "images")

    def test_rejects_a_label_status_outside_the_four_states(self):
        df = frame("labels", status=["MAYBE"])
        with pytest.raises(contracts.ContractError, match="status"):
            contracts.validate(df, "labels")

    @pytest.mark.parametrize("status", ["PRESENT", "ABSENT", "UNCERTAIN", "NOT_MENTIONED"])
    def test_accepts_each_of_the_four_label_states(self, status):
        contracts.validate(frame("labels", status=[status]), "labels")


def test_label_states_match_the_data_config():
    """contracts.py must not drift from configs/data.yaml:labels.mapping."""
    import yaml

    with open("configs/data.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert frozenset(config["labels"]["mapping"].values()) == contracts.LABEL_STATUSES
