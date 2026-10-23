"""Real adapters on real weights. Opt in with `uv run pytest -m model`.

Each test skips unless its weights are already downloaded, so a fresh clone
never pulls gigabytes during a test run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
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


def _cxrclip_cached() -> bool:
    c = CANDIDATES["cxrclip"]
    path = Path(c["weights"])
    if not path.is_file():
        return False
    from src.embeddings.chexzero import sha256_of

    return sha256_of(path) == c["revision"]


AVAILABLE = {
    "biomedclip": lambda: _hf_cached("biomedclip"),
    "chexzero": _chexzero_cached,
    "xrayclip": lambda: _hf_cached("xrayclip"),
    "cxrclip": _cxrclip_cached,
}


@pytest.fixture
def xray(tmp_path):
    """A 390x320 grayscale gradient: the shape CheXpert-small images have."""
    path = tmp_path / "xray.jpg"
    Image.fromarray(np.tile(np.linspace(0, 255, 390, dtype=np.uint8), (320, 1))).save(path)
    return path


@pytest.mark.parametrize("name", ["biomedclip", "chexzero", "xrayclip", "cxrclip"])
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


def _cxrclip_reference(candidate: dict):
    """The authors' recipe (cxrclip/model/clip.py): HF `from_pretrained` towers, linear heads,
    `load_state_dict(ckpt["model"], strict=False)`; downloads the HF weights if absent."""
    import torch
    from huggingface_hub import snapshot_download
    from torch import nn
    from transformers import AutoModel, AutoTokenizer, SwinModel

    swin_id, swin_rev = candidate["swin_config"].split("@")
    bert_id, bert_rev = candidate["text_config"].split("@")
    swin_dir = snapshot_download(swin_id, revision=swin_rev,
                                 allow_patterns=["config.json", "pytorch_model.bin"])
    bert_dir = snapshot_download(bert_id, revision=bert_rev,
                                 allow_patterns=["config.json", "vocab.txt", "pytorch_model.bin"])

    class Reference(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = nn.Module()
            self.image_encoder.image_encoder = SwinModel.from_pretrained(swin_dir)
            self.text_encoder = nn.Module()
            self.text_encoder.text_encoder = AutoModel.from_pretrained(bert_dir)
            self.image_projection = nn.Module()
            self.image_projection.projection = nn.Linear(768, 512)
            self.text_projection = nn.Module()
            self.text_projection.projection = nn.Linear(768, 512)
            self.logit_scale = nn.Parameter(torch.zeros([]))

    model = Reference()
    ckpt = torch.load(candidate["weights"], map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    return model.eval(), AutoTokenizer.from_pretrained(bert_dir)


def test_cxrclip_matches_the_authors_recipe(xray):
    """Adapter and the original code path agree to cosine >= 0.999 on images and templates."""
    if not AVAILABLE["cxrclip"]():
        pytest.skip("cxrclip weights not converted")
    import torch
    from torchvision import transforms

    from src.embeddings.base import load_rgb

    candidate = CANDIDATES["cxrclip"]
    adapter = load_encoder("cxrclip", candidate)
    reference, tokenizer = _cxrclip_reference(candidate)
    images = pl.read_parquet("data/canonical/images.parquet").head(10)
    paths = [Path("data/raw/CheXpert-v1.0-small") / p for p in images["path_to_image"]]
    if not all(p.is_file() for p in paths):
        paths = [xray] * 3
    texts = ["cardiomegaly", "pleural effusion and no pneumothorax", "no finding",
             "an image with lung opacity but without edema", "support devices"]

    pipeline = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5] * 3, [0.5] * 3)])
    with torch.inference_mode():
        pixels = torch.stack([pipeline(load_rgb(p)) for p in paths])
        hidden = reference.image_encoder.image_encoder(pixel_values=pixels)["last_hidden_state"]
        ref_images = reference.image_projection.projection(hidden[:, 0])
        tokens = tokenizer(texts, padding="longest", truncation=True, max_length=256,
                           return_tensors="pt")
        hidden = reference.text_encoder.text_encoder(**tokens)["last_hidden_state"]
        last = tokens["attention_mask"].sum(-1) - 1
        ref_texts = reference.text_projection.projection(hidden[torch.arange(len(texts)), last])
    ref_images = ref_images / ref_images.norm(dim=1, keepdim=True)
    ref_texts = ref_texts / ref_texts.norm(dim=1, keepdim=True)

    ours_images = adapter.encode_images(paths)
    ours_texts = adapter.encode_texts(texts)
    assert ours_images.shape[1] == adapter.dim == 512
    assert np.all(np.sum(ours_images * ref_images.numpy(), axis=1) >= 0.999)
    assert np.all(np.sum(ours_texts * ref_texts.numpy(), axis=1) >= 0.999)
