"""Build the canonical corpus: five Parquet tables plus an audit report.

The order matters. Stratified sampling has to run over the *whole* eligible
pool, otherwise the corpus over-represents whatever happened to sort first, so
labels are resolved before anything is selected:

    available images on disk
        -> eligible rows      (frontal AP/PA, has an impression, file exists)
        -> one image / study
        -> labels from all four sources, resolved to one status per concept
        -> stratum per study  (its rarest PRESENT label)
        -> interleaved manifest, radiologist-labelled studies pinned to the head
        -> take `limit`

See plan sections 7 and 8, and docs/adr/0002 for what this deliberately omits:
no `project_split` (the corpus is never split) and no `corpus_tier` (impression
covers 99.93% of eligible rows, so every selected study has a report).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from PIL import Image

from src.common.ids import image_id
from src.data import contracts, join, labels, sample
from src.data.audit import list_small_image_paths

DEFAULT_PRIORITY: tuple[str, ...] = (
    "CHEXPERT_V1",
    "CHEXBERT_IMPRESSION",
    "CHEXBERT_FINDINGS",
    "CHEXBERT_REPORT",
)
DATASET_VERSION = "df_chexpert_plus_240401"
PINNED_SPLIT = "valid/"


@dataclass(frozen=True)
class CorpusBuild:
    images: pl.DataFrame
    studies: pl.DataFrame
    labels: pl.DataFrame
    reports: pl.DataFrame
    sample_manifest: pl.DataFrame
    report: dict


def build_corpus(
    plus_csv: Path,
    small_root: Path,
    chexbert: Mapping[str, Path],
    limit: int,
    seed: int,
    priority: Sequence[str] = DEFAULT_PRIORITY,
) -> CorpusBuild:
    """Build the canonical tables for `limit` studies, reproducibly from `seed`."""
    small_root = Path(small_root)

    available = set(list_small_image_paths(small_root))
    plus_rows = join.read_plus_rows(plus_csv)
    eligible = join.filter_eligible(plus_rows, available)
    per_study = join.select_one_image_per_study(eligible)

    assertions = _read_all_labels(small_root, chexbert, per_study)
    resolved = labels.resolve(assertions, priority)

    stratum_priority = sample.stratum_priority(resolved)
    strata = sample.assign_strata(resolved, stratum_priority)
    pinned = [
        study
        for study, path in per_study.select("study_id", "path_to_image").iter_rows()
        if path.startswith(PINNED_SPLIT)
    ]
    manifest = sample.build_manifest(strata, seed=seed, pinned=pinned)

    chosen = manifest.sort("rank_in_manifest").head(limit)
    chosen_ids = chosen.get_column("study_id").to_list()
    selected = per_study.filter(pl.col("study_id").is_in(chosen_ids)).sort("study_id")

    images = _build_images(selected, small_root)
    reports = _build_reports(selected)
    study_labels = (
        assertions.filter(pl.col("study_id").is_in(chosen_ids))
        .select("study_id", "concept", "status", "source", "source_version")
        .sort("study_id", "concept", "source")
    )
    studies = _build_studies(selected, images)

    report = _corpus_report(
        plus_rows=plus_rows,
        eligible=eligible,
        per_study=per_study,
        images=images,
        resolved=resolved.filter(pl.col("study_id").is_in(chosen_ids)),
        stratum_priority=stratum_priority,
        chosen=chosen,
        seed=seed,
        limit=limit,
    )

    return CorpusBuild(
        images=images,
        studies=studies,
        labels=study_labels,
        reports=reports,
        sample_manifest=manifest,
        report=report,
    )


def _read_all_labels(
    small_root: Path,
    chexbert: Mapping[str, Path],
    per_study: pl.DataFrame,
) -> pl.DataFrame:
    """Every source's assertions for the selected images, keyed by study.

    Nothing is overwritten: a disagreement between sources is data, kept with
    its provenance. `labels.resolve` is what picks one status when a filter
    needs exactly one.
    """
    frames = [
        labels.read_chexpert_csv(small_root / name)
        for name in ("train.csv", "valid.csv")
        if (small_root / name).is_file()
    ]
    frames += [labels.read_chexbert_jsonl(path, source) for source, path in chexbert.items()]

    keys = per_study.select("path_to_image", "study_id")
    assertions = pl.concat(frames).join(keys, on="path_to_image", how="inner")

    # CheXbert run on an *empty* findings section emits "No Finding = present"
    # every time: measured 18,598 of 18,598 such studies, against 6.3% where a
    # findings section actually exists. Findings is absent for 74.4% of studies,
    # and because the honest sources are usually silent on "No Finding", the
    # artefact wins in `resolve` and marks three quarters of the corpus normal.
    # A source with no input has no assertion, so it contributes no rows.
    with_findings = (
        per_study.filter(pl.col("findings").is_not_null()).get_column("path_to_image").to_list()
    )
    assertions = assertions.filter(
        (pl.col("source") != "CHEXBERT_FINDINGS")
        | pl.col("path_to_image").is_in(with_findings)
    )
    return assertions.drop("path_to_image")


def _build_images(selected: pl.DataFrame, small_root: Path) -> pl.DataFrame:
    """One row per selected image, with its real dimensions and checksum.

    Dimensions and checksum come from a single open of each file: the cost is
    the first read off disk (~7 ms/image cold), not the hashing (~0.08 ms), so
    a second pass would double the only part that is expensive.
    """
    rows = []
    for record in selected.iter_rows(named=True):
        path = record["path_to_image"]
        local = small_root / path
        payload = local.read_bytes()
        with Image.open(local) as image:
            width, height = image.size
        rows.append(
            {
                "image_id": image_id(path),
                "study_id": record["study_id"],
                "patient_id": record["patient_id"],
                "path_to_image": path,
                "local_image_path": local.as_posix(),
                "view": "FRONTAL",
                "projection": record["projection"],
                "dataset_split": path.split("/", 1)[0],
                "image_checksum": hashlib.sha256(payload).hexdigest(),
                "width": width,
                "height": height,
            }
        )
    return pl.DataFrame(rows, schema=dict(contracts.TABLES["images"].columns)).sort("image_id")


def _build_reports(selected: pl.DataFrame) -> pl.DataFrame:
    """One deduplicated report per study, impression first.

    Findings exists for only 26.6% of rows, so it is appended when present and
    never required. Deduplication happens here rather than at FTS build time,
    because study_id is the primary key of this table.
    """
    search_text = pl.when(pl.col("findings").is_not_null()).then(
        pl.col("impression") + pl.lit("\n\n") + pl.col("findings")
    ).otherwise(pl.col("impression"))

    return (
        selected.select(
            "study_id",
            "patient_id",
            "findings",
            "impression",
            search_text.alias("search_text"),
            pl.col("source_row_index").cast(pl.Int64),
            pl.lit(DATASET_VERSION).alias("dataset_version"),
        )
        .with_columns(
            pl.col("search_text")
            .map_elements(
                lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest(),
                return_dtype=pl.Utf8,
            )
            .alias("report_checksum")
        )
        .unique(subset=["study_id"], keep="first")
        .select(list(contracts.TABLES["reports"].columns))
        .sort("study_id")
    )


def _build_studies(selected: pl.DataFrame, images: pl.DataFrame) -> pl.DataFrame:
    return (
        selected.join(images.select("study_id", "image_id"), on="study_id")
        .select(
            "study_id",
            "patient_id",
            pl.col("image_id").alias("selected_image_id"),
            pl.lit(True).alias("has_report"),
            pl.col("impression").is_not_null().alias("has_impression"),
            pl.col("findings").is_not_null().alias("has_findings"),
            pl.lit("OK").alias("ingestion_status"),
        )
        .sort("study_id")
    )


def _corpus_report(**kw) -> dict:
    """The numbers a reviewer needs to trust the corpus, written beside it."""
    resolved, images = kw["resolved"], kw["images"]
    prevalence = {
        concept: {"PRESENT": 0, "ABSENT": 0, "UNCERTAIN": 0, "NOT_MENTIONED": 0}
        for concept in labels.CHEXPERT_LABELS
    }
    for concept, status, count in (
        resolved.group_by("concept", "status").len().iter_rows()
    ):
        prevalence[concept][status] = count

    return {
        "seed": kw["seed"],
        "limit": kw["limit"],
        "counts": {
            "plus_rows": kw["plus_rows"].height,
            "eligible_rows": kw["eligible"].height,
            "eligible_studies": kw["per_study"].height,
            "eligible_patients": kw["per_study"].get_column("patient_id").n_unique(),
            "selected_studies": kw["chosen"].height,
            "selected_patients": images.get_column("patient_id").n_unique(),
        },
        "stratum_priority": list(kw["stratum_priority"]),
        "stratum_mix": dict(kw["chosen"].get_column("stratum").value_counts().iter_rows()),
        "projection_mix": dict(images.get_column("projection").value_counts().iter_rows()),
        "dataset_split_mix": dict(
            images.get_column("dataset_split").value_counts().iter_rows()
        ),
        # Feeds the absence_policy decision (WORKLOG P2): explicit_absence_required
        # is only viable if ABSENT is common enough to retrieve on.
        "status_prevalence": prevalence,
    }


def write_corpus(build: CorpusBuild, out_dir: Path) -> list[Path]:
    """Write the five tables, validating each against its contract first."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for table in contracts.TABLES:
        frame = getattr(build, table)
        contracts.validate(frame, table)
        path = out_dir / f"{table}.parquet"
        frame.write_parquet(path)
        written.append(path)
    return written
