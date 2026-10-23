"""Preprocessing and the training dataset (plan Task 9): the cache plus `eval_transform` is
exactly the CXR-CLIP test transform; augmentation never flips."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchvision import transforms as T

from src.common.config import load_config
from src.retrieval.templates import training_concepts
from training.cache import build_cache
from training.captions import CaptionSampler
from training.data import (
    PoolDataset,
    eval_transform,
    load_cache,
    preprocess_uint8,
    train_transform,
)
from training.tests.conftest import status_row, synthetic_image

CFG = load_config("concepts")
CONCEPTS = training_concepts(load_config("training"))
MIX = {"single": 0.5, "pair": 0.25, "negation": 0.25}
IDENTITY_AUG = {"random_resized_crop_scale": [1.0, 1.0], "rotation_degrees": 0,
                "brightness": 0.0, "contrast": 0.0, "horizontal_flip": False}
REAL_AUG = load_config("training")["augmentation"]


@pytest.mark.parametrize("shape", [(40, 56), (56, 40), (33, 33)])
def test_preprocess_matches_torchvision(shape):
    img = synthetic_image(1, shape)
    cached = preprocess_uint8(img, 32)
    assert cached.dtype == np.uint8 and cached.shape == (32, 32)
    reference = T.Compose([T.Resize(32), T.CenterCrop(32), T.ToTensor(),
                           T.Normalize([0.5] * 3, [0.5] * 3)])(img.convert("RGB"))
    ours = eval_transform()(cached)
    assert ours.dtype == torch.float32 and ours.shape == (3, 32, 32)
    assert torch.allclose(ours, reference, atol=1e-6)


def _statuses():
    return np.stack([
        status_row({"Edema": "PRESENT", "Pleural Effusion": "ABSENT"}),
        status_row({"Cardiomegaly": "PRESENT"}),
        status_row({"No Finding": "PRESENT"}),
        status_row({"Pneumothorax": "PRESENT", "Edema": "ABSENT"}),
        status_row({"Fracture": "PRESENT"}),
        status_row({"Support Devices": "UNCERTAIN"}),   # not eligible
    ])


def test_dataset_item_shapes_and_no_flip(pool, tmp_path):
    pool_dir, image_root, plus_csv = pool
    out = tmp_path / "cache"
    build_cache(pool_dir, image_root, plus_csv, out, size=32, val_fraction=0.5, seed=3)
    images, index, _ = load_cache(out)
    statuses = _statuses()
    sampler = CaptionSampler(statuses, CONCEPTS, CFG, mix=MIX, seed=1)

    train = PoolDataset(out, sampler, train=True, augment_cfg=IDENTITY_AUG)
    val = PoolDataset(out, sampler, train=False, augment_cfg=None)
    is_val = index["is_val"].to_numpy()
    eligible = sampler.eligible()
    assert len(train) == int((~is_val & eligible).sum())
    assert len(val) == int((is_val & eligible).sum())
    assert len(train) + len(val) == 5

    pixels, text, y = train[0]
    assert pixels.dtype == torch.float32 and pixels.shape == (3, 32, 32)
    assert isinstance(text, str) and text
    assert y.dtype == torch.int8 and tuple(y.shape) == (len(CONCEPTS),)
    # Identity augmentation reproduces the eval transform of the same cached image.
    assert torch.allclose(pixels, eval_transform()(images[train.rows[0]]), atol=1e-5)

    # Real augmentation: the bright block stays on the left in every draw.
    torch.manual_seed(0)
    aug = train_transform(REAL_AUG, 32)
    marked = preprocess_uint8(synthetic_image(2, (48, 48)), 32)
    for _ in range(50):
        x = aug(marked)
        assert x.shape == (3, 32, 32)
        assert x[0, :, :10].mean() > x[0, :, -10:].mean()

    with pytest.raises(ValueError, match="flip"):
        train_transform({**REAL_AUG, "horizontal_flip": True}, 32)
