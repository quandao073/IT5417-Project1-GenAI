"""The 14 CheXpert labels: four-state mapping, source readers, and resolution.

Two sources spell the same assertion differently:
  - CheXpert-small's train.csv/valid.csv, one row per image, 14 label columns
  - CheXpert Plus's labels/*_fixed.json, JSONL despite the extension, one object
    per report section, keyed by path_to_image

Both use 1 / 0 / -1 / missing. `NOT_MENTIONED` is never `ABSENT`: the report
simply did not say. Collapsing the two is the one mistake that would
invalidate every negation result in the evaluation.

labels.parquet keeps every source with its provenance and never overwrites, so
`resolve` is what turns that into the single status the label filter needs.
See configs/retrieval.yaml:label_resolution.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from src.common.ids import normalize_image_path
from src.data.contracts import LABEL_SOURCES

# CSV column order, which is also the order CheXbert's JSONL uses.
CHEXPERT_LABELS: tuple[str, ...] = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)

_CODE_TO_STATUS = {1: "PRESENT", 0: "ABSENT", -1: "UNCERTAIN"}

_LONG_SCHEMA = {
    "path_to_image": pl.Utf8,
    "concept": pl.Utf8,
    "status": pl.Utf8,
    "source": pl.Utf8,
    "source_version": pl.Utf8,
}


def to_status(value: object) -> str:
    """Map a raw label cell to one of the four states.

    Missing means NOT_MENTIONED, never ABSENT. An unrecognised code raises
    rather than being folded into a state, so a data change cannot pass silently.
    """
    if value is None:
        return "NOT_MENTIONED"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "NOT_MENTIONED"
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"Unrecognised label value: {value!r}") from exc
    if isinstance(value, float) and math.isnan(value):
        return "NOT_MENTIONED"
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Unrecognised label value: {value!r}")
    code = int(value)
    if code != value or code not in _CODE_TO_STATUS:
        raise ValueError(f"Unrecognised label value: {value!r}")
    return _CODE_TO_STATUS[code]


def read_chexpert_csv(path: Path) -> pl.DataFrame:
    """Read CheXpert-small's train.csv or valid.csv into long-format assertions."""
    path = Path(path)
    frame = pl.read_csv(
        path,
        columns=["Path", *CHEXPERT_LABELS],
        schema_overrides={name: pl.Float64 for name in CHEXPERT_LABELS},
    )
    return _to_long(
        frame.rename({"Path": "path_to_image"}),
        source="CHEXPERT_V1",
        source_version=path.name,
    )


def read_chexbert_jsonl(path: Path, source: str) -> pl.DataFrame:
    """Read one CheXpert Plus labels/*_fixed.json file into long-format assertions.

    The extension says JSON but the file is JSONL: one object per line.
    """
    if source not in LABEL_SOURCES:
        raise ValueError(
            f"Unknown label source {source!r}, expected one of {sorted(LABEL_SOURCES)}"
        )
    path = Path(path)

    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not (line := line.strip()):
                continue
            record = json.loads(line)
            image = normalize_image_path(record["path_to_image"])
            for concept in CHEXPERT_LABELS:
                rows.append((image, concept, to_status(record.get(concept)), source, path.name))
    return pl.DataFrame(rows, schema=_LONG_SCHEMA, orient="row")


def _to_long(frame: pl.DataFrame, source: str, source_version: str) -> pl.DataFrame:
    """Melt a wide label frame into one row per (image, concept)."""
    rows = []
    for record in frame.iter_rows(named=True):
        image = normalize_image_path(record["path_to_image"])
        for concept in CHEXPERT_LABELS:
            rows.append((image, concept, to_status(record[concept]), source, source_version))
    return pl.DataFrame(rows, schema=_LONG_SCHEMA, orient="row")


def resolve(long: pl.DataFrame, priority: Sequence[str]) -> pl.DataFrame:
    """Reduce multi-source assertions to one status per (study_id, concept).

    Takes the first source in `priority` that actually says something; a source
    that is silent is skipped rather than allowed to mask a later one. When every
    source is silent the result stays NOT_MENTIONED.
    """
    rank = {source: index for index, source in enumerate(priority)}
    return (
        long.with_columns(
            pl.col("source").replace_strict(rank, return_dtype=pl.Int32).alias("_rank"),
            (pl.col("status") == "NOT_MENTIONED").alias("_silent"),
        )
        # Silent assertions sort last, so the first row of each group is the
        # highest-priority source that said something - or, if none did, a
        # silent one, which correctly yields NOT_MENTIONED.
        .sort("_silent", "_rank")
        .group_by("study_id", "concept")
        .first()
        .select("study_id", "concept", "status")
    )
