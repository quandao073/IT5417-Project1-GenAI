"""Cross-check the build manifest, the shards and the Qdrant collection.

Six checks from the Phase 3 spec. Each returns a Check rather than raising, so
the CLI can print every PASS/FAIL at once; the exit code is 1 if any failed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src.common.ids import qdrant_point_id
from src.embeddings import embed, index


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


# Manifests written before `config_scope` existed hashed these four files.
_LEGACY_CONFIG_SCOPE = ("data", "models", "retrieval", "concepts")


def expected_config_hash(manifest: dict, config_dir: Path | None = None) -> str:
    """Recompute the hash over the scope the manifest itself recorded, so an
    index stays checkable when the default scope changes."""
    from src.common.config import config_hash

    return config_hash(manifest.get("config_scope", _LEGACY_CONFIG_SCOPE), config_dir)


def all_point_ids(client, collection: str, page: int = 5000) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(collection, limit=page, offset=offset,
                                       with_payload=False, with_vectors=False)
        ids.update(str(p.id) for p in points)
        if offset is None:
            return ids


def _guarded(name: str, check: Callable[[], Check]) -> Check:
    """An exception inside a check is a FAIL line, not a traceback that hides the other five."""
    try:
        return check()
    except Exception as exc:
        return Check(name, False, f"{type(exc).__name__}: {exc}")


def validate_indexes(
    client,
    manifest: dict,
    shards_dir: Path,
    images: pl.DataFrame,
    *,
    expected_config_hash: str,
    expected_revision: str,
    sample_size: int = 20,
    seed: int = 20260101,
) -> list[Check]:
    collection = manifest["collection"]
    shards = embed.load_shards_manifest(shards_dir)

    def config_hash() -> Check:
        got_hash = manifest["config_hash"]
        return Check("config_hash", got_hash == expected_config_hash,
                     f"manifest {got_hash[:12]} vs configs {expected_config_hash[:12]}")

    def counts() -> Check:
        n_qdrant = client.count(collection).count
        n_shards = sum(s["n"] for s in shards["shards"])
        found = (n_qdrant, images.height, n_shards, manifest["counts"]["vectors"])
        return Check("counts", len(set(found)) == 1,
                     f"qdrant={n_qdrant} images={images.height} shards={n_shards} "
                     f"manifest={manifest['counts']['vectors']}")

    def orphans() -> Check:
        # Both directions.
        qdrant_ids = all_point_ids(client, collection)
        table_ids = {str(qdrant_point_id(p)) for p in images["path_to_image"].to_list()}
        extra, missing = qdrant_ids - table_ids, table_ids - qdrant_ids
        return Check("orphans", not extra and not missing,
                     f"{len(extra)} points not in images.parquet, "
                     f"{len(missing)} images not indexed")

    def dimension() -> Check:
        # D6: collection == manifest == first shard
        dim_q = client.get_collection(collection).config.params.vectors.size
        first_vectors, _ = embed.read_shard(shards_dir / shards["shards"][0]["file"])
        dims = (dim_q, manifest["embedding_model"]["dimension"], int(first_vectors.shape[1]))
        return Check("dimension", len(set(dims)) == 1,
                     f"qdrant={dims[0]} manifest={dims[1]} shard={dims[2]}")

    def sample() -> Check:
        # Vectors and payload in Qdrant equal the shards and the table.
        by_id = index.images_by_id(images)
        located: dict[str, tuple[str, int]] = {}
        for entry in shards["shards"]:
            _, ids = embed.read_shard(shards_dir / entry["file"])
            located.update({image_id: (entry["file"], row) for row, image_id in enumerate(ids)})
        rng = np.random.default_rng(seed)
        picked = rng.choice(images["image_id"].to_list(), size=min(sample_size, images.height),
                            replace=False)
        bad = []
        for image_id in picked:
            row = by_id[image_id]
            found = client.retrieve(collection, ids=[str(qdrant_point_id(row["path_to_image"]))],
                                    with_vectors=True, with_payload=True)
            if not found:
                bad.append(f"{image_id[:8]}:missing")
                continue
            point = found[0]
            file, pos = located[image_id]
            vectors, _ = embed.read_shard(shards_dir / file)
            if not np.allclose(point.vector, vectors[pos], atol=1e-6):
                bad.append(f"{image_id[:8]}:vector")
            if point.payload != {k: row[k] for k in index.PAYLOAD_FIELDS}:
                bad.append(f"{image_id[:8]}:payload")
        return Check("sample", not bad, f"{len(picked)} sampled, mismatches: {bad[:3]}")

    def revision() -> Check:
        rev = manifest["embedding_model"]["revision"]
        return Check("revision", rev == expected_revision,
                     f"manifest {rev} vs models.yaml {expected_revision}")

    return [
        _guarded("config_hash", config_hash),
        _guarded("counts", counts),
        _guarded("orphans", orphans),
        _guarded("dimension", dimension),
        _guarded("sample", sample),
        _guarded("revision", revision),
    ]
