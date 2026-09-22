"""Schemas for the five canonical Parquet tables. See plan section 8.

A writer validates against these before it writes, so a schema drift fails at
the point it is introduced rather than three phases later in the API.

Two deliberate omissions, both recorded in docs/adr/0002:
  - no `project_split`: the corpus is never split, only the query set is
  - no `corpus_tier`:   impression covers 99.93% of eligible rows, so the
                        Tier A / Tier B split of plan 7.4 is empty by
                        construction
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import polars as pl

LABEL_STATUSES = frozenset({"PRESENT", "ABSENT", "UNCERTAIN", "NOT_MENTIONED"})
LABEL_SOURCES = frozenset(
    {"CHEXPERT_V1", "CHEXBERT_FINDINGS", "CHEXBERT_IMPRESSION", "CHEXBERT_REPORT"}
)
PROJECTIONS = frozenset({"AP", "PA"})


class ContractError(ValueError):
    """A frame does not satisfy its table's contract."""


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: Mapping[str, pl.DataType]
    primary_key: tuple[str, ...]
    allowed_values: Mapping[str, frozenset[str]] = field(default_factory=dict)


TABLES: dict[str, TableSpec] = {
    "images": TableSpec(
        name="images",
        columns={
            "image_id": pl.Utf8,
            "study_id": pl.Utf8,
            "patient_id": pl.Utf8,
            "path_to_image": pl.Utf8,
            "local_image_path": pl.Utf8,
            "view": pl.Utf8,
            "projection": pl.Utf8,
            "dataset_split": pl.Utf8,
            "image_checksum": pl.Utf8,
            "width": pl.Int32,
            "height": pl.Int32,
        },
        primary_key=("image_id",),
        allowed_values={
            "view": frozenset({"FRONTAL"}),
            "projection": PROJECTIONS,
            "dataset_split": frozenset({"train", "valid"}),
        },
    ),
    "studies": TableSpec(
        name="studies",
        columns={
            "study_id": pl.Utf8,
            "patient_id": pl.Utf8,
            "selected_image_id": pl.Utf8,
            "has_report": pl.Boolean,
            "has_impression": pl.Boolean,
            "has_findings": pl.Boolean,
            "ingestion_status": pl.Utf8,
        },
        primary_key=("study_id",),
    ),
    "labels": TableSpec(
        name="labels",
        columns={
            "study_id": pl.Utf8,
            "concept": pl.Utf8,
            "status": pl.Utf8,
            "source": pl.Utf8,
            "source_version": pl.Utf8,
        },
        # One assertion per source is kept, never overwritten, so the key has to
        # carry the source. Conflicts between sources are data, not errors.
        primary_key=("study_id", "concept", "source"),
        allowed_values={"status": LABEL_STATUSES, "source": LABEL_SOURCES},
    ),
    "reports": TableSpec(
        name="reports",
        columns={
            "study_id": pl.Utf8,
            "patient_id": pl.Utf8,
            "findings": pl.Utf8,
            "impression": pl.Utf8,
            "search_text": pl.Utf8,
            "source_row_index": pl.Int64,
            "report_checksum": pl.Utf8,
            "dataset_version": pl.Utf8,
        },
        primary_key=("study_id",),
    ),
    "sample_manifest": TableSpec(
        name="sample_manifest",
        columns={
            "study_id": pl.Utf8,
            "stratum": pl.Utf8,
            "rank_in_manifest": pl.Int64,
            "selection_seed": pl.Int64,
        },
        primary_key=("study_id",),
    ),
}


def validate(df: pl.DataFrame, table: str) -> None:
    """Raise ContractError unless `df` satisfies `table`'s contract."""
    spec = TABLES[table]

    present = set(df.columns)
    expected = set(spec.columns)
    if missing := sorted(expected - present):
        raise ContractError(f"{table}: missing columns {missing}")
    if unexpected := sorted(present - expected):
        raise ContractError(f"{table}: unexpected columns {unexpected}")

    for column, dtype in spec.columns.items():
        if df.schema[column] != dtype:
            raise ContractError(
                f"{table}: column {column!r} is {df.schema[column]}, expected {dtype}"
            )

    key = list(spec.primary_key)
    if df.select(key).null_count().sum_horizontal().item() > 0:
        raise ContractError(f"{table}: primary key {key} contains null values")
    if df.select(key).is_duplicated().any():
        raise ContractError(f"{table}: primary key {key} contains duplicate values")

    for column, allowed in spec.allowed_values.items():
        seen = set(df.get_column(column).drop_nulls().unique().to_list())
        if illegal := sorted(seen - allowed):
            raise ContractError(
                f"{table}: column {column!r} holds values outside "
                f"{sorted(allowed)}: {illegal}"
            )
