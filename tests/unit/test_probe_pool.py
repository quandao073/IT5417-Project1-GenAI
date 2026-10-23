"""The out-of-corpus pool: patient-disjoint from the corpus, train/ only."""

from __future__ import annotations

import hashlib
import json

import polars as pl
import pytest

from src.common.ids import image_id
from src.data import join, probe_pool
from src.data.labels import CHEXPERT_LABELS
from src.data.probe_pool import (
    PoolError,
    pool_images,
    pool_sha256,
    resolved_status_matrix,
    select_pool,
    write_pool,
)


def per_study_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "study_id": ["patient00001/study1", "patient00002/study1", "patient64600/study1"],
        "patient_id": ["patient00001", "patient00002", "patient64600"],
        "path_to_image": ["train/patient00001/study1/view1_frontal.jpg",
                          "train/patient00002/study1/view1_frontal.jpg",
                          "valid/patient64600/study1/view1_frontal.jpg"],
        "projection": ["AP", "PA", "AP"],
        "findings": [None, "x", None],
        "impression": ["a", "b", "c"],
    })


def test_select_pool_drops_corpus_patients_and_valid_paths():
    out = select_pool(per_study_frame(), {"patient00001"})
    assert out["study_id"].to_list() == ["patient00002/study1"]
    assert out.columns == per_study_frame().columns


def test_select_pool_raises_when_nothing_is_left():
    with pytest.raises(PoolError):
        select_pool(per_study_frame(), {"patient00001", "patient00002"})


def test_pool_images_columns_and_order():
    pool = per_study_frame().filter(pl.col("path_to_image").str.starts_with("train/"))
    out = pool_images(pool)
    assert out.columns == ["image_id", "study_id", "patient_id", "path_to_image", "projection"]
    assert out["image_id"].to_list() == sorted(out["image_id"].to_list())
    for row in out.iter_rows(named=True):
        assert row["image_id"] == image_id(row["path_to_image"])
    assert out.height == 2


def test_pool_sha256_is_order_independent_and_changes_with_content():
    images = pool_images(per_study_frame())
    shuffled = images.sort("path_to_image", descending=True)
    assert pool_sha256(images) == pool_sha256(shuffled)

    expected = hashlib.sha256()
    for study, path in images.sort("study_id").select("study_id", "path_to_image").iter_rows():
        expected.update(f"{study}\t{path}\n".encode())
    assert pool_sha256(images) == expected.hexdigest()

    changed = images.with_columns(
        pl.when(pl.col("study_id") == "patient00001/study1")
        .then(pl.lit("train/patient00001/study1/view2_frontal.jpg"))
        .otherwise(pl.col("path_to_image"))
        .alias("path_to_image")
    )
    assert pool_sha256(changed) != pool_sha256(images)


def test_resolved_status_matrix_defaults_to_not_mentioned_never_absent():
    resolved = pl.DataFrame(
        {"study_id": ["s1", "s1", "s2"],
         "concept": ["Edema", "Pneumonia", "Edema"],
         "status": ["PRESENT", "NOT_MENTIONED", "ABSENT"],
         "source": ["CHEXPERT_V1", None, "CHEXPERT_V1"]},
        schema={"study_id": pl.Utf8, "concept": pl.Utf8, "status": pl.Utf8, "source": pl.Utf8},
    )
    matrix = resolved_status_matrix(resolved, ["s2", "s1", "s3"])
    assert matrix.shape == (3, len(CHEXPERT_LABELS))
    edema = CHEXPERT_LABELS.index("Edema")
    assert matrix[:, edema].tolist() == ["ABSENT", "PRESENT", "NOT_MENTIONED"]
    others = [i for i in range(len(CHEXPERT_LABELS)) if i != edema]
    assert (matrix[:, others][2] == "NOT_MENTIONED").all()
    assert matrix[1, CHEXPERT_LABELS.index("Pneumonia")] == "NOT_MENTIONED"


def _patch_sources(monkeypatch, tmp_path, selected: pl.DataFrame):
    paths = selected["path_to_image"].to_list()
    monkeypatch.setattr(join, "read_plus_rows", lambda csv: selected.with_columns(
        pl.lit(0, dtype=pl.Int64).alias("source_row_index")))
    monkeypatch.setattr(probe_pool, "list_small_image_paths", lambda root: paths)
    monkeypatch.setattr(
        probe_pool, "read_all_labels",
        lambda root, chexbert, pool: pl.DataFrame(
            {"study_id": pool["study_id"].to_list(), "concept": ["Edema"] * pool.height,
             "status": ["PRESENT"] * pool.height, "source": ["CHEXPERT_V1"] * pool.height,
             "source_version": ["t"] * pool.height}))


def _corpus(tmp_path, patients):
    corpus = pl.DataFrame({"patient_id": patients})
    path = tmp_path / "images.parquet"
    corpus.write_parquet(path)
    return corpus, path


def test_build_pool_raises_on_patient_overlap(tmp_path, monkeypatch):
    selected = per_study_frame().filter(pl.col("path_to_image").str.starts_with("train/"))
    _patch_sources(monkeypatch, tmp_path, selected)
    # A broken select_pool that lets the corpus patient through.
    monkeypatch.setattr(probe_pool, "select_pool", lambda per_study, patients: per_study)
    corpus, _ = _corpus(tmp_path, ["patient00001"])
    with pytest.raises(PoolError, match="patient"):
        probe_pool.build_pool(tmp_path / "x.csv", tmp_path, {}, corpus, ["CHEXPERT_V1"])


def test_build_pool_raises_on_valid_paths(tmp_path, monkeypatch):
    selected = per_study_frame()
    _patch_sources(monkeypatch, tmp_path, selected)
    monkeypatch.setattr(probe_pool, "select_pool", lambda per_study, patients: per_study)
    corpus, _ = _corpus(tmp_path, ["patient09999"])
    with pytest.raises(PoolError, match="valid/"):
        probe_pool.build_pool(tmp_path / "x.csv", tmp_path, {}, corpus, ["CHEXPERT_V1"])


def test_build_pool_and_write_pool_round_trip(tmp_path, monkeypatch):
    _patch_sources(monkeypatch, tmp_path, per_study_frame())
    corpus, corpus_path = _corpus(tmp_path, ["patient00001"])
    images, labels, meta = probe_pool.build_pool(
        tmp_path / "x.csv", tmp_path, {}, corpus, ["CHEXPERT_V1"],
        corpus_images_path=corpus_path)
    assert images["study_id"].to_list() == ["patient00002/study1"]
    assert labels.columns == ["study_id", "concept", "status", "source"]
    assert meta["studies"] == 1 and meta["patients"] == 1
    assert meta["projection_mix"] == {"PA": 1}
    assert meta["pool_sha256"] == pool_sha256(images)
    assert meta["corpus_images_sha256"] == hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    written = write_pool(images, labels, meta, tmp_path / "out")
    assert sorted(p.name for p in written) == sorted(
        [probe_pool.POOL_IMAGES, probe_pool.POOL_LABELS, probe_pool.POOL_META])
    assert pl.read_parquet(tmp_path / "out" / probe_pool.POOL_IMAGES).equals(images)
    assert json.loads((tmp_path / "out" / probe_pool.POOL_META).read_text())["studies"] == 1
