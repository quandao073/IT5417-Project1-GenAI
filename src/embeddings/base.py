"""The image-text encoder interface every VLM adapter implements.

The embedding dimension is read from real output, never from config (WORKLOG
D6): three sources disagreed on it before anything was measured.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

# Adapters import torch/open_clip/transformers, so they load lazily by name.
_ADAPTERS = {
    "biomedclip": "src.embeddings.biomedclip:BiomedCLIPEncoder",
    "chexzero": "src.embeddings.chexzero:CheXzeroEncoder",
    "xrayclip": "src.embeddings.xrayclip:XrayCLIPEncoder",
    "cxrclip": "src.embeddings.cxrclip:CXRClipEncoder",
}
# A fine-tune is a candidate of its own name (`cxrclip_ft6_lr3e-05`) with its base model's
# adapter, found through models.yaml `framework`, so a new variant needs no code (B16).
_FRAMEWORKS = {"cxrclip_checkpoint": _ADAPTERS["cxrclip"]}


class ImageTextEncoder(Protocol):
    name: str
    model_id: str
    revision: str
    license: str
    preprocessing: dict
    weights_bytes: int
    device: str

    @property
    def dim(self) -> int: ...

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("zero-norm embedding cannot be normalized")
    return x / norms


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def load_encoder(name: str, candidate: dict, device: str = "cpu") -> ImageTextEncoder:
    target = _ADAPTERS.get(name) or _FRAMEWORKS.get(candidate.get("framework"))
    if target is None:
        raise ValueError(f"Unknown encoder {name!r}; known: {sorted(_ADAPTERS)}, or a candidate "
                         f"whose framework is one of {sorted(_FRAMEWORKS)}")
    module, cls = target.split(":")
    return getattr(importlib.import_module(module), cls)(candidate, device)
