"""Real adapters on real weights. Opt in with `uv run pytest -m model`.

Each test skips unless its weights are already downloaded, so a fresh clone
never pulls gigabytes during a test run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.common.config import load_config
from src.embeddings.base import load_encoder

pytestmark = pytest.mark.model
ENCODER_CFG = load_config("models")["image_encoder"]
CANDIDATES = ENCODER_CFG["candidates"]


def _hf_cached(name: str) -> bool:
    from huggingface_hub import scan_cache_dir

    c = CANDIDATES[name]
    for repo in scan_cache_dir().repos:
        if repo.repo_id == c["model_id"]:
            return any(rev.commit_hash == c["revision"] for rev in repo.revisions)
    return False


def _chexzero_cached() -> bool:
    c = CANDIDATES["chexzero"]
    return bool(c["revision"]) and (Path("data/cache/models/chexzero") / c["filename"]).is_file()


AVAILABLE = {
    "biomedclip": lambda: _hf_cached("biomedclip"),
    "chexzero": _chexzero_cached,
    "xrayclip": lambda: _hf_cached("xrayclip"),
}


@pytest.fixture
def xray(tmp_path):
    """A 390x320 grayscale gradient: the shape CheXpert-small images have."""
    path = tmp_path / "xray.jpg"
    Image.fromarray(np.tile(np.linspace(0, 255, 390, dtype=np.uint8), (320, 1))).save(path)
    return path


@pytest.mark.parametrize("name", ["biomedclip", "chexzero", "xrayclip"])
def test_encoder_produces_matching_unit_vectors(name, xray):
    if not AVAILABLE[name]():
        pytest.skip(f"{name} weights not downloaded")
    encoder = load_encoder(name, CANDIDATES[name])
    images = encoder.encode_images([xray, xray])
    texts = encoder.encode_texts(["chest x-ray showing pleural effusion", "normal study"])

    assert images.dtype == texts.dtype == np.float32
    assert images.shape[1] == texts.shape[1] == encoder.dim
    assert np.allclose(np.linalg.norm(images, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(texts, axis=1), 1.0, atol=1e-5)
    assert np.allclose(images[0], images[1], atol=1e-5), "inference must be deterministic"
    assert encoder.revision == CANDIDATES[name]["revision"]
    assert encoder.weights_bytes > 50 * 2**20


def test_selected_preprocessing_matches_the_adapter():
    name = ENCODER_CFG["selected"]
    if not AVAILABLE[name]():
        pytest.skip(f"{name} weights not downloaded")
    actual = load_encoder(name, CANDIDATES[name]).preprocessing
    assert json.loads(json.dumps(actual)) == json.loads(json.dumps(ENCODER_CFG["preprocessing"]))
