"""Embed the corpus into shard files (Phase 3, step one of two).

The expensive step - about 40 minutes of CPU for 25,000 images - is kept apart
from the cheap, changeable one (indexing into Qdrant): shards are plain .npz
files that `cxr index` reads and that the offline corpus scoring reuses. Shards
are fixed by row index, so a resume needs no state beyond which files exist
and are intact.
"""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

SHARDS_FILE = "shards.json"


def make_build_id(selected: str, day: date) -> str:
    return f"{day:%Y%m%d}_chexpert25k_{selected}_v1"


def shard_bounds(n_images: int, shard_size: int) -> list[tuple[int, int]]:
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    return [(s, min(s + shard_size, n_images)) for s in range(0, n_images, shard_size)]


def shard_path(out_dir: Path, k: int) -> Path:
    return out_dir / f"shard_{k:04d}.npz"


def vectors_sha256(vectors: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(vectors, dtype=np.float32).tobytes()).hexdigest()


def write_shard(path: Path, vectors: np.ndarray, image_ids: Sequence[str]) -> str:
    """Write to a .part file through an open handle (numpy would otherwise append
    .npz to the temp name), then rename, so a killed run never leaves a file that
    looks finished."""
    vectors = np.asarray(vectors, dtype=np.float32)
    part = path.with_suffix(".npz.part")
    with open(part, "wb") as handle:
        np.savez(handle, vectors=vectors, image_ids=np.asarray(list(image_ids)))
    part.replace(path)
    return vectors_sha256(vectors)


def read_shard(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path) as archive:
        return archive["vectors"].astype(np.float32, copy=False), archive["image_ids"].tolist()


def shard_is_complete(path: Path, expected_ids: Sequence[str]) -> bool:
    if not path.is_file():
        return False
    try:
        vectors, ids = read_shard(path)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False
    return ids == list(expected_ids) and vectors.shape[0] == len(ids)


def load_shards_manifest(out_dir: Path) -> dict:
    path = out_dir / SHARDS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `cxr embed` to completion first")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict, **dumps_kwargs) -> None:
    """Write to <name>.part, then rename, so a reader never sees a half-written file."""
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(payload, indent=2, **dumps_kwargs), encoding="utf-8")
    part.replace(path)


def _finished_manifest(out_dir: Path, build_id: str, ids_all: list[str],
                       shard_size: int) -> dict | None:
    """The existing manifest if it already describes a finished build, else None.

    A re-run that has nothing to embed must not rewrite the manifest: it would
    replace the real build's device, versions and commit with the re-run's.
    """
    try:
        manifest = load_shards_manifest(out_dir)
    except (FileNotFoundError, ValueError):
        return None
    bounds = shard_bounds(len(ids_all), shard_size)
    entries = manifest.get("shards")
    if (manifest.get("build_id") != build_id or manifest.get("n_images") != len(ids_all)
            or not isinstance(entries, list) or len(entries) != len(bounds)):
        return None
    for k, ((start, end), entry) in enumerate(zip(bounds, entries, strict=True)):
        path = shard_path(out_dir, k)
        if entry.get("file") != path.name or entry.get("n") != end - start:
            return None
        if not shard_is_complete(path, ids_all[start:end]):
            return None
        if vectors_sha256(read_shard(path)[0]) != entry.get("sha256"):
            return None
    return manifest


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _batch_failure(encoder, k: int, batch: Sequence[Path], exc: Exception) -> RuntimeError:
    """Name the first image that fails on its own; a batch error does not say which one."""
    for path in batch:
        try:
            encoder.encode_images([path])
        except Exception as single:
            return RuntimeError(f"shard {k}: failed to embed {path}: {single!r}")
    return RuntimeError(
        f"shard {k}: batch starting at {batch[0]} failed ({exc!r}) but every image "
        "re-embedded alone; the failure looks transient"
    )


def run_embed(
    images: pl.DataFrame,
    encoder,
    image_root: Path,
    out_dir: Path,
    build_id: str,
    *,
    shard_size: int,
    batch_size: int,
    library_versions: dict,
    git_commit: str | None,
) -> dict:
    """Embed every image in `images` (sorted by image_id) into shards under out_dir.

    Existing intact shards are skipped. Any image that fails to embed stops the
    run: a corpus with silently missing vectors would fail validate-indexes
    later, with far less context about what went wrong.
    """
    images = images.sort("image_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_all = images["image_id"].to_list()
    finished = _finished_manifest(out_dir, build_id, ids_all, shard_size)
    if finished is not None:
        return finished
    # A manifest left by an earlier run would describe shards this run may replace.
    (out_dir / SHARDS_FILE).unlink(missing_ok=True)
    started = _now()
    paths_all = [image_root / p for p in images["path_to_image"].to_list()]

    seconds, embedded, skipped = 0.0, 0, 0
    for k, (start, end) in enumerate(shard_bounds(len(ids_all), shard_size)):
        path = shard_path(out_dir, k)
        ids = ids_all[start:end]
        if shard_is_complete(path, ids):
            skipped += len(ids)
            continue
        chunks = []
        for b in range(start, end, batch_size):
            batch = paths_all[b : min(b + batch_size, end)]
            tick = time.perf_counter()
            try:
                chunks.append(encoder.encode_images(batch))
            except Exception as exc:
                raise _batch_failure(encoder, k, batch, exc) from exc
            seconds += time.perf_counter() - tick
        vectors = np.concatenate(chunks)
        if vectors.shape[0] != len(ids):
            raise RuntimeError(
                f"shard {k}: encoder returned {vectors.shape[0]} rows for {len(ids)} images"
            )
        write_shard(path, vectors, ids)
        embedded += len(ids)

    shards = []
    dim = None
    for k, (start, end) in enumerate(shard_bounds(len(ids_all), shard_size)):
        vectors, _ = read_shard(shard_path(out_dir, k))
        dim = dim or int(vectors.shape[1])
        shards.append({"file": shard_path(out_dir, k).name, "n": end - start,
                       "sha256": vectors_sha256(vectors)})

    manifest = {
        "build_id": build_id,
        "model_id": encoder.model_id,
        "revision": encoder.revision,
        "preprocessing": encoder.preprocessing,
        "dim": dim,
        "shard_size": shard_size,
        "n_images": len(ids_all),
        "shards": shards,
        "library_versions": library_versions,
        "git_commit": git_commit,
        "started_at": started,
        "finished_at": _now(),
        "device": encoder.device,
        "s_per_image": seconds / embedded if embedded else None,
        "embedded": embedded,
        "skipped": skipped,
    }
    write_json_atomic(out_dir / SHARDS_FILE, manifest)
    return manifest
