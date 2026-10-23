"""LoRA on both towers and the merged export the `cxrclip` adapter loads (plan Task 11, H3)."""

from __future__ import annotations

import hashlib

import torch
from peft.tuners.lora import LoraLayer
from torch import nn

from src.embeddings.cxrclip import build_model, load_state
from training.lora import apply_lora, export_merged, trainable_fraction, unwrap
from training.model import load_base
from training.tests.conftest import MODEL_CFG, tiny_configs, tiny_model, write_tiny_checkpoint

TARGETS = ("query", "key", "value")


def test_lora_targets_both_towers_and_is_small():
    base = tiny_model()
    linears = {name: m for name, m in base.named_modules()
               if isinstance(m, nn.Linear) and name.rsplit(".", 1)[-1] in TARGETS}
    assert any(n.startswith("image_encoder.") for n in linears)
    assert any(n.startswith("text_encoder.") for n in linears)
    expected = sum(1 * (m.in_features + m.out_features) for m in linears.values())

    peft_model = apply_lora(base, rank=1, alpha=2, dropout=0.0)
    wrapped = {name: m for name, m in unwrap(peft_model).named_modules()
               if isinstance(m, LoraLayer)}
    assert set(wrapped) == set(linears)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    assert trainable == expected
    assert 0 < trainable_fraction(peft_model) < 0.05
    assert not unwrap(peft_model).logit_scale.requires_grad
    assert not unwrap(peft_model).image_projection.projection.weight.requires_grad


def _perturb(peft_model) -> None:
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param, std=0.3)


def test_export_round_trips_through_adapter_loader(tmp_path):
    peft_model = apply_lora(tiny_model(seed=1), rank=2, alpha=4, dropout=0.0)
    _perturb(peft_model)
    peft_model.eval()
    out = tmp_path / "enc" / "model.pt"
    meta = {"variant": "FT1", "lr": 3e-5, "epoch": 2}
    sha = export_merged(peft_model, {"model": MODEL_CFG}, out, meta)

    payload = torch.load(out, map_location="cpu", weights_only=True)
    assert set(payload) == {"model", "config", "finetune"}
    assert payload["finetune"] == meta and payload["config"] == {"model": MODEL_CFG}
    assert not any("lora" in k or "base_layer" in k for k in payload["model"])
    fresh = build_model(MODEL_CFG, *tiny_configs()).eval()
    load_state(fresh, payload["model"])

    x = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    ids = torch.tensor([[2, 12, 13, 3, 0], [2, 5, 7, 6, 3]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    with torch.inference_mode():
        assert torch.allclose(fresh.encode_image(x), unwrap(peft_model).encode_image(x),
                              atol=1e-5)
        assert torch.allclose(fresh.encode_text(ids, mask),
                              unwrap(peft_model).encode_text(ids, mask), atol=1e-5)
        # The merged weights differ from the base: the LoRA perturbation was really folded in.
        assert not torch.allclose(fresh.encode_image(x), tiny_model(seed=1).encode_image(x),
                                  atol=1e-3)
    # The caller's model is untouched and can keep training.
    assert any(isinstance(m, LoraLayer) for m in unwrap(peft_model).modules())
    assert sha == hashlib.sha256(out.read_bytes()).hexdigest()
    assert not out.with_suffix(".pt.part").exists()


def test_export_sha256_matches_file(tmp_path):
    peft_model = apply_lora(tiny_model(), rank=1, alpha=2, dropout=0.0)
    out = tmp_path / "model.pt"
    sha = export_merged(peft_model, {"model": MODEL_CFG}, out, {})
    assert len(sha) == 64 and sha == hashlib.sha256(out.read_bytes()).hexdigest()


def test_load_base_restores_weights_tokenizer_and_config(tmp_path):
    checkpoint, swin_dir, bert_dir, source = write_tiny_checkpoint(tmp_path, seed=4)
    model, tokenizer, config = load_base(checkpoint, swin_dir, bert_dir)
    assert config["model"] == MODEL_CFG and config["base"]["text_max_length"] == 16
    tokens = tokenizer(["pulmonary edema", "cardiomegaly and no pleural effusion"],
                       padding="longest", truncation=True, max_length=16, return_tensors="pt")
    assert tokens["input_ids"].shape[0] == 2 and tokens["attention_mask"][0].sum() == 4
    assert tokens["input_ids"].max() < 50
    x = torch.ones(1, 3, 32, 32)
    with torch.inference_mode():
        assert torch.allclose(model.encode_image(x), source.encode_image(x))
        assert torch.allclose(model.encode_text(tokens["input_ids"], tokens["attention_mask"]),
                              source.encode_text(tokens["input_ids"], tokens["attention_mask"]))
