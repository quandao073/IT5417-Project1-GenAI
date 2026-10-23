"""Pool images for training: preprocessing, the cached-array reader and the Dataset (H5).

The cache holds every pool image as uint8 [size, size] after the CXR-CLIP test-time geometry
(Resize(size) on the short side, bilinear, then CenterCrop(size)) on the grayscale image. CXR
JPEGs are single-channel, so one channel is stored and expanded to three at load time;
`eval_transform` then equals the adapter's ToTensor + Normalize(0.5, 0.5) exactly
(tests/test_data.py pins that). Augmentation is the configured RandomResizedCrop, rotation and
brightness/contrast jitter and never a horizontal flip: a flip teaches the wrong side.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms import functional as F

from training.captions import CaptionSampler

IMAGES_FILE = "images.npy"
INDEX_FILE = "index.parquet"
CACHE_META = "cache.json"
MEAN = STD = 0.5

Transform = Callable[[np.ndarray], torch.Tensor]


def preprocess_uint8(img: Image.Image, size: int = 224) -> np.ndarray:
    """Grayscale, Resize(size) short side bilinear, CenterCrop(size) -> uint8 [size, size]."""
    grey = F.center_crop(F.resize(img.convert("L"), size, InterpolationMode.BILINEAR), size)
    return np.asarray(grey, dtype=np.uint8)


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    # Copy: the array is usually a read-only memmap slice, which torch refuses to wrap.
    return torch.from_numpy(np.array(arr, dtype=np.uint8, copy=True)).unsqueeze(0)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32).div_(255.0).sub_(MEAN).div_(STD)
    return x.expand(3, -1, -1).contiguous()


def eval_transform() -> Transform:
    """uint8 [H, W] -> float32 [3, H, W]: ToTensor and Normalize(0.5, 0.5) on three channels."""
    return lambda arr: _normalize(_to_tensor(arr))


def train_transform(cfg: dict, size: int = 224) -> Transform:
    """training.yaml:augmentation on a uint8 [H, W] array, then the eval normalisation."""
    if cfg.get("horizontal_flip", False):
        raise ValueError("horizontal flip is never allowed: it swaps left and right")
    lo, hi = cfg["random_resized_crop_scale"]
    aug = v2.Compose([
        # ratio fixed at 1: only the scale is jittered, as the config says.
        v2.RandomResizedCrop(size, scale=(float(lo), float(hi)), ratio=(1.0, 1.0),
                             interpolation=InterpolationMode.BILINEAR, antialias=True),
        v2.RandomRotation(float(cfg["rotation_degrees"]),
                          interpolation=InterpolationMode.BILINEAR),
        v2.ColorJitter(brightness=float(cfg["brightness"]), contrast=float(cfg["contrast"])),
    ])
    return lambda arr: _normalize(aug(_to_tensor(arr)))


def load_cache(cache_dir: Path) -> tuple[np.ndarray, pl.DataFrame, dict]:
    """(images uint8 memmap [N, size, size], index with one row per image, cache.json)."""
    cache_dir = Path(cache_dir)
    images = np.load(cache_dir / IMAGES_FILE, mmap_mode="r")
    index = pl.read_parquet(cache_dir / INDEX_FILE)
    meta = json.loads((cache_dir / CACHE_META).read_text(encoding="utf-8"))
    if images.shape[0] != index.height or index["row"].to_list() != list(range(index.height)):
        raise ValueError(f"{cache_dir}: images.npy and index.parquet disagree on the rows")
    return images, index, meta


class PoolDataset(Dataset):
    """(pixels float32 [3, size, size], caption, y int8 [L]) over the eligible rows of one split.

    `train` picks pool-train (True) or pool-val (False) by the cache's `is_val`; rows without a
    PRESENT training concept are dropped (D32). `augment_cfg` None means the eval transform.
    """

    def __init__(self, cache_dir: Path, sampler: CaptionSampler, *, train: bool,
                 augment_cfg: dict | None):
        self.images, self.index, self.meta = load_cache(cache_dir)
        eligible = sampler.eligible()
        if eligible.shape[0] != self.images.shape[0]:
            raise ValueError("the sampler's statuses must have one row per cached image")
        is_val = self.index["is_val"].to_numpy()
        self.rows = np.flatnonzero(eligible & (~is_val if train else is_val))
        self.sampler = sampler
        size = int(self.images.shape[1])
        self.transform = train_transform(augment_cfg, size) if augment_cfg else eval_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, str, torch.Tensor]:
        row = int(self.rows[i])
        text, y = self.sampler.sample(row)
        return self.transform(self.images[row]), text, torch.from_numpy(y)


def reseed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: each worker (and each epoch) draws its own captions."""
    info = get_worker_info()
    info.dataset.sampler.reseed(torch.initial_seed() % 2**32)
