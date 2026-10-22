"""The out-of-corpus pool for the image-concept probes.

Probes are trained on BiomedCLIP image embeddings of studies whose patients are
NOT among the 25,000 corpus studies, so a probe is never scored on a patient it
was fitted on. The pool is every eligible train/ study outside the corpus:
`valid/` holds the radiologist-labelled test studies and is never touched.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from src.common.ids import image_id
from src.data import join, labels
from src.data.audit import list_small_image_paths
from src.data.canonical import read_all_labels
from src.data.labels import CHEXPERT_LABELS

POOL_IMAGES = "pool_images.parquet"
POOL_LABELS = "pool_labels.parquet"
POOL_META = "pool.json"

_TRAIN_PREFIX = "train/"
_VALID_PREFIX = "valid/"


class PoolError(ValueError):
    """The pool would leak a corpus patient or a valid/ study, or is empty."""


def select_pool(per_study: pl.DataFrame, corpus_patients: Collection[str]) -> pl.DataFrame:
    """Studies of patients outside the corpus whose image lives under train/."""
    pool = per_study.filter(
        ~pl.col("patient_id").is_in(list(corpus_patients))
        & pl.col("path_to_image").str.starts_with(_TRAIN_PREFIX)
    )
    if pool.is_empty():
        raise PoolError("No eligible train/ study is left outside the corpus patients")
    return pool


def pool_images(pool: pl.DataFrame) -> pl.DataFrame:
    """One row per pool image, in the shape `cxr embed --images` reads."""
    return (
        pool.select(
            pl.col("path_to_image")
            .map_elements(image_id, return_dtype=pl.Utf8)
            .alias("image_id"),
            "study_id",
            "patient_id",
            "path_to_image",
            "projection",
        )
        .sort("image_id")
    )


def pool_sha256(images: pl.DataFrame) -> str:
    """Fingerprint of which image stands for which study, independent of row order."""
    digest = hashlib.sha256()
    for study, path in images.sort("study_id").select("study_id", "path_to_image").iter_rows():
        digest.update(f"{study}\t{path}\n".encode())
    return digest.hexdigest()


def resolved_status_matrix(
    resolved: pl.DataFrame,
    study_ids: Sequence[str],
    concepts: Sequence[str] = CHEXPERT_LABELS,
) -> np.ndarray:
    """Statuses from an already-resolved long frame, rows in `study_ids` order.

    A (study, concept) the frame does not mention is NOT_MENTIONED - never ABSENT.
    """
    wide = resolved.pivot(on="concept", index="study_id", values="status")
    for concept in concepts:
        if concept not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Utf8).alias(concept))
    ordered = pl.DataFrame({"study_id": list(study_ids)}).join(wide, on="study_id", how="left")
    return (
        ordered.select([pl.col(c).fill_null("NOT_MENTIONED") for c in concepts])
        .to_numpy()
        .astype(object)
    )


def build_pool(
    plus_csv: Path,
    small_root: Path,
    chexbert: Mapping[str, Path],
    corpus_images: pl.DataFrame,
    priority: Sequence[str],
    *,
    corpus_images_path: Path | None = None,
    git_commit: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """The pool's images, its resolved labels and the metadata that pins them.

    `corpus_images_path` is the parquet `corpus_images` was read from; its file
    hash goes into the metadata so a pool can be matched to the corpus it avoids.
    """
    small_root = Path(small_root)
    available = set(list_small_image_paths(small_root))
    eligible = join.filter_eligible(join.read_plus_rows(plus_csv), available)
    per_study = join.select_one_image_per_study(eligible)

    corpus_patients = set(corpus_images.get_column("patient_id"))
    pool = select_pool(per_study, corpus_patients)
    images = pool_images(pool)

    # select_pool already guarantees both; a post-condition keeps a later edit
    # to it from silently leaking a corpus patient or a test study.
    if overlap := set(images.get_column("patient_id")) & corpus_patients:
        raise PoolError(f"{len(overlap)} pool patients are also corpus patients")
    if images.filter(pl.col("path_to_image").str.starts_with(_VALID_PREFIX)).height:
        raise PoolError("The pool contains valid/ images")

    resolved = labels.resolve(read_all_labels(small_root, chexbert, pool), priority)

    meta = {
        "studies": images.height,
        "patients": images.get_column("patient_id").n_unique(),
        "projection_mix": dict(images.get_column("projection").value_counts().iter_rows()),
        "corpus_images_sha256": (
            hashlib.sha256(Path(corpus_images_path).read_bytes()).hexdigest()
            if corpus_images_path is not None
            else None
        ),
        "pool_sha256": pool_sha256(images),
        "git_commit": git_commit,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return images, resolved, meta


def write_pool(
    images: pl.DataFrame, labels_resolved: pl.DataFrame, meta: dict, out_dir: Path
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / POOL_IMAGES, out_dir / POOL_LABELS, out_dir / POOL_META]
    images.write_parquet(paths[0])
    labels_resolved.write_parquet(paths[1])
    paths[2].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths
