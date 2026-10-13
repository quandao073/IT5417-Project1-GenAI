"""Test doubles shared by the Phase 3 unit tests. No torch, deterministic."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def _vector(text: str, dim: int) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = np.frombuffer(digest[: dim * 4], dtype=np.uint32).astype(np.float32)
    raw = raw / np.float32(2**32) - 0.5
    return raw / np.linalg.norm(raw)


class FakeEncoder:
    """Vectors are a hash of the path or text, so every test can predict them."""

    name = "fake"
    model_id = "fake/model"
    revision = "fakerev0"
    license = "mit"
    preprocessing = {"image_size": 1}
    weights_bytes = 10
    device = "cpu"

    def __init__(self, dim: int = 4, fail_on: str | None = None):
        self._dim = dim
        self._fail_on = fail_on
        self.image_calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        self.image_calls += 1
        for p in paths:
            if self._fail_on and self._fail_on in str(p):
                raise OSError(f"cannot read {p}")
        return np.stack([_vector(Path(p).as_posix(), self._dim) for p in paths])

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([_vector(t, self._dim) for t in texts])
