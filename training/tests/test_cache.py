"""The pool image cache (plan Task 9): uint8 images at the CXR-CLIP test-time geometry, the
patient split of the probe, and one non-empty impression per row."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from PIL import Image

from src.embeddings.probe import patient_split
from training.cache import CACHE_META, IMAGES_FILE, INDEX_FILE, build_cache
from training.data import preprocess_uint8
from training.tests.conftest import write_pool


def test_cache_shapes_and_split(pool, tmp_path):
    pool_dir, image_root, plus_csv = pool
    out = tmp_path / "cache"
    meta = build_cache(pool_dir, image_root, plus_csv, out, size=32, val_fraction=0.5, seed=3)

    images = np.load(out / IMAGES_FILE, mmap_mode="r")
    index = pl.read_parquet(out / INDEX_FILE)
    assert images.dtype == np.uint8 and images.shape == (6, 32, 32)
    assert index.columns == ["row", "image_id", "study_id", "patient_id", "is_val", "impression"]
    assert index["row"].to_list() == list(range(6))

    pool_images = pl.read_parquet(pool_dir / "pool_images.parquet")
    assert index["image_id"].to_list() == pool_images["image_id"].to_list()
    expected_val = patient_split(index["patient_id"].to_list(), val_fraction=0.5, seed=3)
    assert index["is_val"].to_list() == expected_val.tolist()
    assert 0 < int(expected_val.sum()) < 6

    for k, path in enumerate(pool_images["path_to_image"].to_list()):
        with Image.open(image_root / path) as img:
            assert np.array_equal(images[k], preprocess_uint8(img, 32))

    # Impressions are stripped: the first row's starts with a newline in the CSV.
    assert index["impression"][0].startswith("1. ") or not index["impression"][0].startswith("\n")
    assert all(text == text.strip() and text for text in index["impression"].to_list())

    written = json.loads((out / CACHE_META).read_text(encoding="utf-8"))
    assert written == meta
    assert meta["n"] == 6 and meta["n_val"] == int(expected_val.sum()) and meta["size"] == 32
    assert len(meta["pool_sha256"]) == 64 and meta["seconds"] >= 0
    assert "git_commit" in meta
    assert not (out / (IMAGES_FILE + ".part")).exists()


def test_cache_impression_is_nonempty_for_every_row(tmp_path):
    pool_dir, image_root, plus_csv = write_pool(tmp_path / "blank",
                                                blank_impression_for="patient00003/study2")
    with pytest.raises(ValueError, match="patient00003/study2"):
        build_cache(pool_dir, image_root, plus_csv, tmp_path / "c1", size=32,
                    val_fraction=0.1, seed=1)

    pool_dir, image_root, plus_csv = write_pool(tmp_path / "missing",
                                                drop_from_csv="patient00004/study1")
    with pytest.raises(ValueError, match="patient00004/study1"):
        build_cache(pool_dir, image_root, plus_csv, tmp_path / "c2", size=32,
                    val_fraction=0.1, seed=1)
