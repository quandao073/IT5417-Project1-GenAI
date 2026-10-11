"""Upsert embedding shards into a Qdrant collection and write the build manifest.

The collection is named by build_id, so a stale index can never be served under
a current build (plan section 16). The manifest is what the API checks against
at startup and what `cxr validate-indexes` cross-checks with the shards.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from src.common.ids import qdrant_point_id
from src.embeddings import embed

MANIFEST_FILE = "build_manifest.json"
PAYLOAD_FIELDS = ("image_id", "study_id", "patient_id", "projection", "path_to_image")
UPSERT_BATCH = 500


def collection_name(prefix: str, build_id: str) -> str:
    return f"{prefix}_{build_id}"


def shards_digest(manifest: dict) -> str:
    """Stable sha256 of the shard list, so the build manifest pins the exact shards."""
    payload = json.dumps(manifest["shards"], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_shards(shards_dir: Path, manifest: dict) -> None:
    for entry in manifest["shards"]:
        path = shards_dir / entry["file"]
        if not path.is_file():
            raise ValueError(f"shard missing: {path}")
        vectors, _ = embed.read_shard(path)
        if embed.vectors_sha256(vectors) != entry["sha256"]:
            raise ValueError(f"{path.name}: sha256 differs from shards.json; re-run cxr embed")


def images_by_id(images: pl.DataFrame) -> dict[str, dict]:
    return {row["image_id"]: row for row in images.select(PAYLOAD_FIELDS).iter_rows(named=True)}


def build_points(vectors: np.ndarray, image_ids: Sequence[str], by_id: dict[str, dict]) -> list:
    from qdrant_client import models

    points = []
    for vector, image_id in zip(vectors, image_ids, strict=True):
        row = by_id.get(image_id)
        if row is None:
            raise ValueError(f"image_id {image_id} is not in images.parquet")
        points.append(models.PointStruct(
            id=str(qdrant_point_id(row["path_to_image"])),
            vector=vector.tolist(),
            payload={k: row[k] for k in PAYLOAD_FIELDS},
        ))
    return points


def _check_ids(shards_dir: Path, manifest: dict, by_id: dict[str, dict]) -> None:
    """Refuse duplicate or unknown image ids before anything is written to Qdrant."""
    seen: set[str] = set()
    dupes: set[str] = set()
    unknown: list[str] = []
    for entry in manifest["shards"]:
        _, ids = embed.read_shard(shards_dir / entry["file"])
        dupes.update(seen.intersection(ids))
        seen.update(ids)
        unknown.extend(i for i in ids if i not in by_id)
    if dupes:
        raise ValueError(f"duplicate image_id across shards: {sorted(dupes)[:3]}")
    if unknown:
        raise ValueError(
            f"{len(unknown)} image_ids are not in images.parquet, e.g. {unknown[0]}"
        )


def run_index(
    client,
    shards_dir: Path,
    images: pl.DataFrame,
    prefix: str,
    *,
    recreate: bool,
    config_hash: str,
    git_commit: str | None,
    dataset: dict,
    library_versions: dict,
) -> dict:
    from qdrant_client import models

    manifest = embed.load_shards_manifest(shards_dir)
    verify_shards(shards_dir, manifest)
    name = collection_name(prefix, manifest["build_id"])
    by_id = images_by_id(images)
    _check_ids(shards_dir, manifest, by_id)

    if client.collection_exists(name):
        if not recreate:
            raise ValueError(f"collection {name} already exists; pass --recreate to rebuild it")
        client.delete_collection(name)
    client.create_collection(
        name, vectors_config=models.VectorParams(size=manifest["dim"],
                                                 distance=models.Distance.COSINE),
    )
    client.create_payload_index(name, field_name="projection",
                                field_schema=models.PayloadSchemaType.KEYWORD)

    total = 0
    for entry in manifest["shards"]:
        vectors, ids = embed.read_shard(shards_dir / entry["file"])
        points = build_points(vectors, ids, by_id)
        for b in range(0, len(points), UPSERT_BATCH):
            client.upsert(name, points=points[b : b + UPSERT_BATCH], wait=True)
        total += len(points)

    return {
        "build_id": manifest["build_id"],
        "collection": name,
        "git_commit": git_commit,
        "config_hash": config_hash,
        "dataset": dataset,
        "sampling_seed": 20260101,
        "embedding_model": {
            "model_id": manifest["model_id"],
            "revision": manifest["revision"],
            "dimension": manifest["dim"],
            "preprocessing": manifest["preprocessing"],
        },
        "counts": {
            "studies": images["study_id"].n_unique(),
            "images": images.height,
            "reports": 0,
            "vectors": total,
        },
        "shards_sha256": shards_digest(manifest),
        "library_versions": library_versions,
        "indexed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    embed.write_json_atomic(path, manifest, ensure_ascii=False)
