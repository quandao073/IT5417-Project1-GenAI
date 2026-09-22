"""Building the eligible study pool from CheXpert Plus.

Eligibility is three filters, in this order:
  1. frontal AP/PA           - 191,054 of 223,462 rows
  2. a non-empty impression  - 190,913 of those (99.93%)
  3. the image is on disk

Findings is never a filter: it exists for only 26.6% of rows.

The CSV holds newlines inside `report` and `section_*`, so it must go through a
real CSV parser - 223,462 records span 10.7M physical lines. Polars handles it.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from pathlib import Path

import polars as pl

from src.common.ids import normalize_image_path

FRONTAL = "Frontal"
PROJECTIONS = ("AP", "PA")

PLUS_COLUMNS = [
    "path_to_image",
    "frontal_lateral",
    "ap_pa",
    "deid_patient_id",
    "section_findings",
    "section_impression",
]

# viewN_*.jpg. A path without one sorts last but still deterministically.
_VIEW_INDEX = re.compile(r"/view(\d+)_")
_NO_VIEW_INDEX = 1_000_000


def read_plus_rows(csv_path: Path) -> pl.DataFrame:
    """Read the columns the corpus build needs, one row per image."""
    frame = pl.read_csv(csv_path, columns=PLUS_COLUMNS).with_row_index("source_row_index")
    return frame.select(
        pl.col("path_to_image")
        .map_elements(normalize_image_path, return_dtype=pl.Utf8)
        .alias("path_to_image"),
        pl.col("path_to_image").str.extract(r"(patient\d+/study\d+)").alias("study_id"),
        pl.col("path_to_image").str.extract(r"(patient\d+)").alias("patient_id"),
        pl.col("frontal_lateral").alias("view"),
        pl.col("ap_pa").alias("projection"),
        # Impressions routinely open with a newline and carry "1. 2. 3."
        # numbering; strip before anything decides whether they are empty.
        _blank_to_null("section_findings").alias("findings"),
        _blank_to_null("section_impression").alias("impression"),
        # Lets a report row be traced back to the exact CSV record it came from.
        pl.col("source_row_index").cast(pl.Int64),
    ).drop("view")


def _blank_to_null(column: str) -> pl.Expr:
    stripped = pl.col(column).str.strip_chars()
    return pl.when(stripped.str.len_chars() > 0).then(stripped).otherwise(None)


def filter_eligible(rows: pl.DataFrame, available: Collection[str]) -> pl.DataFrame:
    """Keep rows that are frontal AP/PA, carry an impression, and have their image."""
    available = set(available)
    return rows.filter(
        pl.col("projection").is_in(PROJECTIONS)
        # Blank-after-strip counts as missing, checked here rather than trusting
        # the caller to have normalized: this is the eligibility rule, and a
        # whitespace-only impression would otherwise become an empty report.
        & (pl.col("impression").str.strip_chars().str.len_chars() > 0)
        & pl.col("path_to_image").is_in(available)
    )


def select_one_image_per_study(rows: pl.DataFrame) -> pl.DataFrame:
    """Reduce to one image per study: the lowest view index, path breaking ties.

    Unbiased and deterministic. Preferring PA would give cleaner frontals but
    skew the corpus away from the dataset's real 161,622 AP / 29,432 PA mix.
    """
    return (
        rows.with_columns(
            pl.col("path_to_image")
            .str.extract(_VIEW_INDEX.pattern, 1)
            .cast(pl.Int64)
            .fill_null(_NO_VIEW_INDEX)
            .alias("_view_index")
        )
        .sort("_view_index", "path_to_image")
        .group_by("study_id")
        .first()
        .drop("_view_index")
        .sort("study_id")
    )
