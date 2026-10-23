"""CXR-CLIP adapter on tiny Swin/BERT configs: architecture, pooling, checkpoint key layout,
preprocessing and the sha256 pin. No weights are downloaded here; the real checkpoint is
checked by tests/integration/test_encoders.py (`uv run pytest -m model`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import BertConfig, SwinConfig

from src.embeddings import cxrclip

MODEL_CFG = {"name": "clip_custom", "temperature": 0.07,
             "image_encoder": {"name": "tiny-swin", "model_type": "swin"},
             "text_encoder": {"name": "tiny-bert", "pooling": "eos"},
             "projection_head": {"name": "linear", "dropout": 0.1, "proj_dim": 6}}


def tiny_configs():
    # Two stages: transformers 4.44's Swin fails on a 1x1 final stage with four tiny stages.
    swin = SwinConfig(image_size=32, patch_size=4, embed_dim=8, depths=[2, 2], num_heads=[1, 1],
                      window_size=4)
    bert = BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                      intermediate_size=32, max_position_embeddings=32)
    return swin, bert


def tiny_model(seed: int = 0) -> cxrclip.CXRClipModule:
    torch.manual_seed(seed)
    return cxrclip.build_model(MODEL_CFG, *tiny_configs()).eval()


def test_build_model_dims_and_logit_scale():
    model = tiny_model()
    with torch.inference_mode():
        image = model.encode_image(torch.zeros(2, 3, 32, 32))
        ids, mask = torch.tensor([[1, 2, 3, 4, 5]] * 2), torch.ones(2, 5, dtype=torch.long)
        text = model.encode_text(ids, mask)
    assert image.shape == (2, 6) and text.shape == (2, 6)
    assert math.isclose(model.logit_scale.item(), math.log(1 / 0.07), rel_tol=1e-6)


def test_eos_pooling_uses_last_attended_token():
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 10]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    with torch.inference_mode():
        padded = model.encode_text(ids, mask)
        alone = model.encode_text(ids[:1, :4], mask[:1, :4])
    assert torch.allclose(padded[0], alone[0], atol=1e-5)


def test_state_dict_uses_the_original_key_layout():
    keys = set(tiny_model().state_dict())
    assert "logit_scale" in keys
    assert "image_projection.projection.weight" in keys
    assert "text_projection.projection.bias" in keys
    assert any(k.startswith("image_encoder.image_encoder.embeddings.") for k in keys)
    assert any(k.startswith("text_encoder.text_encoder.encoder.layer.0.attention.self.query.")
               for k in keys)
    assert "text_encoder.text_encoder.embeddings.position_ids" not in keys


def test_load_state_drops_position_ids_and_is_strict_otherwise():
    source, target = tiny_model(seed=1), tiny_model(seed=2)
    state = {k: v.clone() for k, v in source.state_dict().items()}
    state["text_encoder.text_encoder.embeddings.position_ids"] = torch.arange(32)[None]
    cxrclip.load_state(target, state)
    with torch.inference_mode():
        a = source.encode_image(torch.ones(1, 3, 32, 32))
        b = target.encode_image(torch.ones(1, 3, 32, 32))
    assert torch.allclose(a, b)
    del state["image_projection.projection.weight"]
    with pytest.raises(RuntimeError):
        cxrclip.load_state(tiny_model(seed=3), state)


def test_transform_is_resize_center_crop_half_normalised():
    transform = cxrclip.make_transform(32)
    tensor = transform(Image.new("L", (40, 30), 128))
    assert tuple(tensor.shape) == (3, 32, 32)
    assert torch.allclose(tensor, torch.full((3, 32, 32), (128 / 255 - 0.5) / 0.5), atol=1e-6)
    gradient = Image.fromarray(np.tile(np.linspace(0, 255, 40, dtype=np.uint8), (30, 1)))
    assert cxrclip.make_transform(32)(gradient).min() >= -1.0


def test_encoder_refuses_wrong_sha256(tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"not a checkpoint")
    candidate = {"model_id": "kakaobrain/cxr-clip", "weights": str(weights),
                 "revision": "sha256:" + "0" * 64, "license": "cc-by-nc-4.0",
                 "swin_config": "x@y", "text_config": "x@y"}
    with pytest.raises(ValueError, match="sha256 mismatch"):
        cxrclip.CXRClipEncoder(candidate, "cpu")


def test_plain_config_keeps_only_the_sections_the_adapter_needs():
    raw = {"model": MODEL_CFG, "tokenizer": {"a": 1}, "transform": {"b": 2}, "base": {"c": 3},
           "optimizer": {"drop": 1}, "data_train": {"drop": 2}}
    assert set(cxrclip.plain_config(raw)) == {"model", "tokenizer", "transform", "base"}
