"""Cache the pool images at the training geometry, with the probe's patient split (H5).

One pass over `pool_images.parquet` decodes every JPEG once (the cost is disk I/O, WORKLOG
Phase 1) and writes uint8 [N, size, size] through a memmap, so an epoch later reads 50 KB per
image instead of decoding it. The impression of each study is joined from the CheXpert Plus CSV
through `join.read_plus_rows` (polars: the CSV has newlines inside its text columns, T2) and
every row must have one, as the pool's eligibility rule promises.

    uv run --project training python training/cache.py --pool-dir data/probe_pool \
        --image-root data/raw/CheXpert-v1.0-small \
        --plus-csv data/raw/chexpert-plus/df_chexpert_plus_240401.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from src.common.config import load_config
from src.data import join, probe_pool
from src.embeddings.embed import write_json_atomic
from src.embeddings.probe import patient_split

try:
    from training.data import CACHE_META, IMAGES_FILE, INDEX_FILE, preprocess_uint8
except ModuleNotFoundError:  # `python training/cache.py` puts training/ on sys.path, not the root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.data import CACHE_META, IMAGES_FILE, INDEX_FILE, preprocess_uint8

_PROGRESS_EVERY = 5000


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() or None


def build_cache(pool_dir: Path, image_root: Path, plus_csv: Path, out_dir: Path, *,
                size: int = 224, val_fraction: float, seed: int) -> dict:
    """Write images.npy, index.parquet and cache.json under out_dir; returns cache.json."""
    tick = time.perf_counter()
    pool_dir, image_root, out_dir = Path(pool_dir), Path(image_root), Path(out_dir)
    images = pl.read_parquet(pool_dir / probe_pool.POOL_IMAGES).sort("image_id")
    plus = join.read_plus_rows(Path(plus_csv)).select("path_to_image", "impression")
    # Sorted again after the join: polars does not promise to keep the left order.
    joined = images.join(plus, on="path_to_image", how="left").sort("image_id")
    missing = joined.filter(pl.col("impression").is_null())["study_id"].to_list()
    if missing:
        raise ValueError(
            f"{len(missing)} pool studies have no impression in the Plus CSV "
            f"(blank or absent), e.g. {missing[:5]}"
        )
    patients = joined["patient_id"].to_list()
    is_val = patient_split(patients, val_fraction=val_fraction, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    n = joined.height
    part = out_dir / (IMAGES_FILE + ".part")
    array = np.lib.format.open_memmap(part, mode="w+", dtype=np.uint8, shape=(n, size, size))
    try:
        for k, rel in enumerate(joined["path_to_image"].to_list()):
            with Image.open(image_root / rel) as img:
                array[k] = preprocess_uint8(img, size)
            if (k + 1) % _PROGRESS_EVERY == 0:
                print(f"cached {k + 1}/{n} images", flush=True)
        array.flush()
    finally:
        del array
    part.replace(out_dir / IMAGES_FILE)

    index = pl.DataFrame({
        "row": np.arange(n, dtype=np.int64),
        "image_id": joined["image_id"],
        "study_id": joined["study_id"],
        "patient_id": joined["patient_id"],
        "is_val": is_val,
        "impression": joined["impression"],
    })
    part = out_dir / (INDEX_FILE + ".part")
    index.write_parquet(part)
    part.replace(out_dir / INDEX_FILE)

    meta = {
        "pool_sha256": probe_pool.pool_sha256(images),
        "n": n,
        "n_val": int(is_val.sum()),
        "size": size,
        "val_fraction": val_fraction,
        "seed": seed,
        "seconds": round(time.perf_counter() - tick, 1),
        "git_commit": _git_commit(),
    }
    write_json_atomic(out_dir / CACHE_META, meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    cfg = load_config("training")
    parser = argparse.ArgumentParser(description="Cache the pool images for training.")
    parser.add_argument("--pool-dir", default="data/probe_pool")
    parser.add_argument("--image-root", default="data/raw/CheXpert-v1.0-small")
    parser.add_argument("--plus-csv", default="data/raw/chexpert-plus/df_chexpert_plus_240401.csv")
    parser.add_argument("--out", default=cfg["image"]["cache_dir"])
    parser.add_argument("--size", type=int, default=cfg["image"]["size"])
    parser.add_argument("--val-fraction", type=float, default=cfg["split"]["val_fraction"])
    parser.add_argument("--seed", type=int, default=cfg["split"]["seed"])
    args = parser.parse_args(argv)
    meta = build_cache(args.pool_dir, args.image_root, args.plus_csv, args.out,
                       size=args.size, val_fraction=args.val_fraction, seed=args.seed)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
