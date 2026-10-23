"""Synthetic pool for the training-package tests: six grayscale JPEGs of unequal sizes, the
pool parquet that names them, and a CheXpert Plus CSV with newlines inside the impressions (T2).
No real data is ever present in the worktree."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image

from src.common.ids import image_id
from src.data.labels import CHEXPERT_LABELS

POOL_ROWS = [
    # (patient, study, projection, impression, (height, width))
    ("patient00001", "study1", "AP", "\n1. Edema.\n2. No effusion.\n", (40, 56)),
    ("patient00002", "study1", "PA", "Cardiomegaly.", (56, 40)),
    ("patient00003", "study2", "AP", "No acute\ncardiopulmonary process.", (48, 48)),
    ("patient00004", "study1", "AP", "Pneumothorax, left.", (64, 36)),
    ("patient00005", "study3", "PA", "Fracture of the\n\nright clavicle.", (36, 64)),
    ("patient00006", "study1", "AP", "Support devices.", (44, 52)),
]
COL = {name: i for i, name in enumerate(CHEXPERT_LABELS)}


def status_row(statuses: dict[str, str]) -> np.ndarray:
    out = np.array(["NOT_MENTIONED"] * len(CHEXPERT_LABELS), dtype=object)
    for concept, status in statuses.items():
        out[COL[concept]] = status
    return out


def synthetic_image(seed: int, shape: tuple[int, int]) -> Image.Image:
    """Random grey noise with a bright block on the LEFT, so a flip would be visible."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 120, size=shape, dtype=np.uint8)
    pixels[:, : shape[1] // 3] = 230
    return Image.fromarray(pixels, mode="L")


def write_pool(root: Path, rows=POOL_ROWS, *, blank_impression_for: str | None = None,
               drop_from_csv: str | None = None) -> tuple[Path, Path, Path]:
    """Returns (pool_dir, image_root, plus_csv)."""
    image_root = root / "images"
    pool_dir = root / "pool"
    pool_dir.mkdir(parents=True)
    records, csv_rows = [], []
    for k, (patient, study, projection, impression, shape) in enumerate(rows):
        rel = f"train/{patient}/{study}/view1_frontal.jpg"
        path = image_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_image(k, shape).save(path, format="JPEG", quality=95)
        study_id = f"{patient}/{study}"
        records.append({"image_id": image_id(rel), "study_id": study_id, "patient_id": patient,
                        "path_to_image": rel, "projection": projection})
        if study_id == drop_from_csv:
            continue
        text = "  \n " if study_id == blank_impression_for else impression
        csv_rows.append([rel, "Frontal", projection, patient, "", text])
    pl.DataFrame(records).sort("image_id").write_parquet(pool_dir / "pool_images.parquet")
    plus_csv = root / "plus.csv"
    with open(plus_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_to_image", "frontal_lateral", "ap_pa", "deid_patient_id",
                         "section_findings", "section_impression"])
        writer.writerows(csv_rows)
    return pool_dir, image_root, plus_csv


@pytest.fixture
def pool(tmp_path):
    return write_pool(tmp_path)
