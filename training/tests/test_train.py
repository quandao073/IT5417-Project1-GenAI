"""The fine-tuning loop on the tiny model, CPU only (plan Task 12): metrics, per-epoch
adapters, and a merged export that the adapter loader reads back."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import polars as pl
import torch

from src.embeddings.cxrclip import build_model, load_state
from src.embeddings.probe import patient_split
from training import train
from training.cache import build_cache
from training.tests.conftest import MODEL_CFG, tiny_configs, write_pool, write_tiny_checkpoint

N_IMAGES = 24


def _pool_with_labels(root: Path) -> tuple[Path, Path, Path]:
    rows = [(f"patient{k + 1:05d}", "study1", "AP" if k % 3 else "PA",
             f"\nImpression {k}: lung opacity.\n", (36 + k % 5, 40 + k % 7))
            for k in range(N_IMAGES)]
    pool_dir, image_root, plus_csv = write_pool(root, rows=rows)
    labels = []
    for k, (patient, study, *_) in enumerate(rows):
        study_id = f"{patient}/{study}"
        labels.append((study_id, "Lung Opacity", "PRESENT"))
        labels.append((study_id, "Edema", "PRESENT" if k % 2 == 0 else "ABSENT"))
        if k % 3 == 0:
            labels.append((study_id, "Cardiomegaly", "PRESENT"))
        if k % 4 == 0:
            labels.append((study_id, "Pleural Effusion", "ABSENT"))
    pl.DataFrame(labels, schema=["study_id", "concept", "status"], orient="row").with_columns(
        pl.lit("CHEXPERT_V1").alias("source")).write_parquet(pool_dir / "pool_labels.parquet")
    return pool_dir, image_root, plus_csv


def test_train_smoke_on_tiny_model_cpu(tmp_path):
    pool_dir, image_root, plus_csv = _pool_with_labels(tmp_path)
    cache = tmp_path / "cache"
    build_cache(pool_dir, image_root, plus_csv, cache, size=32, val_fraction=0.5, seed=3)
    checkpoint, swin_dir, bert_dir, _ = write_tiny_checkpoint(tmp_path / "ckpt")
    patients = pl.read_parquet(cache / "index.parquet")["patient_id"].to_list()
    n_train = int((~patient_split(patients, val_fraction=0.5, seed=3)).sum())
    assert n_train >= 8, "fixture must leave a full batch of training rows"

    out = tmp_path / "run"
    common = ["--lr", "1e-3", "--batch-size", "8", "--seed", "1", "--cache", str(cache),
              "--pool-dir", str(pool_dir), "--checkpoint", str(checkpoint),
              "--swin-config", str(swin_dir), "--bert-config", str(bert_dir),
              "--device", "cpu", "--precision", "fp32", "--num-workers", "0"]
    rc = train.main(["--variant", "FT1", "--epochs", "2", "--subset", "8", "--out", str(out),
                     *common])
    assert rc == 0

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_train"] == 8 and metrics["config"]["variant"] == "FT1"
    assert metrics["config"]["repel_weight"] == 0.3 and metrics["config"]["lr"] == 1e-3
    assert len(metrics["steps"]) == 2 and all(math.isfinite(s) for s in metrics["steps"])
    epochs = metrics["epochs"]
    assert [e["epoch"] for e in epochs] == [1, 2]
    for e in epochs:
        assert math.isfinite(e["train_loss"]) and e["seconds"] >= 0
        assert math.isfinite(e["val_p5_mean"]) and 0.0 <= e["val_p5_mean"] <= 1.0
        assert set(e) >= {"val_p5_single", "val_p5_pair", "val_p5_negation", "val_text_cos",
                          "vram_mib",
                          "train_seconds", "val_seconds", "s_per_step", "data_wait_seconds",
                          "skipped_steps"}
        assert e["skipped_steps"] == 0
        assert (out / f"adapter_epoch{e['epoch']}" / "adapter_model.safetensors").is_file()
    assert metrics["best_epoch"] in (1, 2)
    assert 0 < metrics["trainable_fraction"] < 1

    payload = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    load_state(build_model(MODEL_CFG, *tiny_configs()), payload["model"])
    assert payload["finetune"]["variant"] == "FT1"
    assert payload["finetune"]["best_epoch"] == metrics["best_epoch"]
    sha = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
    assert (out / "model.sha256").read_text(encoding="utf-8").split()[0] == sha
    assert metrics["model_sha256"] == sha


def test_train_ft2_uses_impressions_and_ft3_drops_repel(tmp_path):
    pool_dir, image_root, plus_csv = _pool_with_labels(tmp_path)
    cache = tmp_path / "cache"
    build_cache(pool_dir, image_root, plus_csv, cache, size=32, val_fraction=0.5, seed=3)
    checkpoint, swin_dir, bert_dir, _ = write_tiny_checkpoint(tmp_path / "ckpt")
    common = ["--lr", "1e-3", "--epochs", "1", "--batch-size", "8", "--subset", "8",
              "--seed", "2", "--cache", str(cache), "--pool-dir", str(pool_dir),
              "--checkpoint", str(checkpoint), "--swin-config", str(swin_dir),
              "--bert-config", str(bert_dir), "--device", "cpu", "--precision", "fp32",
              "--num-workers", "0"]
    assert train.main(["--variant", "FT2", "--out", str(tmp_path / "ft2"), *common]) == 0
    ft2 = json.loads((tmp_path / "ft2" / "metrics.json").read_text(encoding="utf-8"))
    assert ft2["config"]["impression_fraction"] == 0.2 and ft2["config"]["repel_weight"] == 0.3
    assert train.main(["--variant", "FT3", "--out", str(tmp_path / "ft3"), *common]) == 0
    ft3 = json.loads((tmp_path / "ft3" / "metrics.json").read_text(encoding="utf-8"))
    assert ft3["config"]["impression_fraction"] == 0.0 and ft3["config"]["repel_weight"] == 0.0
    assert ft3["best_epoch"] == 1 and (tmp_path / "ft3" / "model.pt").is_file()
    assert ft2["config"]["targets"] == ft3["config"]["targets"] == "caption"
    assert train.main(["--variant", "FT6", "--out", str(tmp_path / "ft6"), *common]) == 0
    ft6 = json.loads((tmp_path / "ft6" / "metrics.json").read_text(encoding="utf-8"))
    assert ft6["config"]["targets"] == "image" and ft6["config"]["repel_weight"] == 0.3
    assert all(math.isfinite(s) for s in ft6["steps"])
    # FT6 fixes the temperature at 5 (WORKLOG "FT1 kẹt ở ln 128"); the export carries it.
    assert ft6["config"]["logit_scale"] == 5.0 and ft2["config"]["logit_scale"] is None
    payload = torch.load(tmp_path / "ft6" / "model.pt", map_location="cpu", weights_only=True)
    assert math.isclose(float(payload["model"]["logit_scale"].exp()), 5.0, rel_tol=1e-5)


def _tiny_lora(seed: int = 0):
    from training import lora

    torch.manual_seed(seed)
    net = build_model(MODEL_CFG, *tiny_configs())
    peft_model = lora.apply_lora(net, rank=2, alpha=4, dropout=0.0)
    return peft_model, lora.unwrap(peft_model)


def test_resolve_precision():
    cpu, cuda = torch.device("cpu"), torch.device("cuda")
    assert train.resolve_precision("fp32", cpu) == (None, False)
    assert train.resolve_precision("bf16", cpu) == (torch.bfloat16, False)
    assert train.resolve_precision("fp16", cuda) == (torch.float16, True)
    import pytest

    with pytest.raises(ValueError):
        train.resolve_precision("fp16", cpu)
    with pytest.raises(ValueError):
        train.resolve_precision("int8", cpu)


def test_gradient_checkpointing_keeps_outputs_and_lora_gradients():
    peft_model, net = _tiny_lora()
    peft_model.train()
    pixels = torch.randn(2, 3, 32, 32)
    ids, mask = torch.tensor([[1, 2, 3, 4]] * 2), torch.ones(2, 4, dtype=torch.long)
    torch.manual_seed(5)
    plain = (net.encode_image(pixels), net.encode_text(ids, mask))
    train.enable_gradient_checkpointing(net)
    assert net.image_encoder.image_encoder.is_gradient_checkpointing
    assert net.text_encoder.text_encoder.is_gradient_checkpointing
    torch.manual_seed(5)
    zi, zt = net.encode_image(pixels), net.encode_text(ids, mask)
    assert torch.allclose(zi, plain[0], atol=1e-6) and torch.allclose(zt, plain[1], atol=1e-6)
    (zi.sum() + zt.sum()).backward()
    lora_grads = [p.grad for n, p in peft_model.named_parameters() if "lora_" in n]
    assert lora_grads and all(g is not None for g in lora_grads)
    assert any(float(g.abs().sum()) > 0 for g in lora_grads)


def test_train_step_skips_a_nonfinite_loss_without_touching_weights():
    peft_model, net = _tiny_lora()
    peft_model.train()
    params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    before = [p.detach().clone() for p in params]
    batch = (torch.randn(2, 3, 32, 32), torch.tensor([[1, 2, 3]] * 2),
             torch.ones(2, 3, dtype=torch.long), torch.tensor([[1, 0], [0, 1]], dtype=torch.int8))

    def nan_loss(zi, zt, y, scale, lam):
        return (zi.sum() + zt.sum()) * float("nan")

    common = {"optimizer": optimizer, "scheduler": scheduler, "scaler": None,
              "autocast_dtype": None, "lam": 0.3, "max_grad_norm": 1.0}
    assert train.train_step(net, *batch, loss_fn=nan_loss, **common) is None
    assert all(torch.equal(a, b) for a, b in zip(before, params, strict=True))
    value = train.train_step(net, *batch, **common)
    assert value is not None and math.isfinite(value)
    assert any(not torch.equal(a, b) for a, b in zip(before, params, strict=True))


def test_resume_continues_from_the_last_finished_epoch(tmp_path):
    pool_dir, image_root, plus_csv = _pool_with_labels(tmp_path)
    cache = tmp_path / "cache"
    build_cache(pool_dir, image_root, plus_csv, cache, size=32, val_fraction=0.5, seed=3)
    checkpoint, swin_dir, bert_dir, _ = write_tiny_checkpoint(tmp_path / "ckpt")
    out = tmp_path / "run"
    common = ["--variant", "FT1", "--lr", "1e-3", "--batch-size", "8", "--seed", "1",
              "--subset", "8", "--cache", str(cache), "--pool-dir", str(pool_dir),
              "--checkpoint", str(checkpoint), "--swin-config", str(swin_dir),
              "--bert-config", str(bert_dir), "--device", "cpu", "--precision", "fp32",
              "--num-workers", "0", "--out", str(out)]
    assert train.main([*common, "--epochs", "1"]) == 0
    assert (out / train.STATE_FILE).is_file()
    assert train.main([*common, "--epochs", "2", "--resume"]) == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert [e["epoch"] for e in metrics["epochs"]] == [1, 2]
    assert len(metrics["steps"]) == 2 and metrics["resumed_from_epoch"] == 1
    # Resuming a finished run does nothing and still succeeds.
    assert train.main([*common, "--epochs", "2", "--resume"]) == 0
    again = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert [e["epoch"] for e in again["epochs"]] == [1, 2]


def test_resume_redoes_an_epoch_once_when_drive_kept_a_newer_metrics_json(tmp_path):
    """Drive uploads asynchronously, so a session killed right after an epoch can leave
    metrics.json one epoch ahead of state_last.pt. state_last.pt decides: the epoch is redone
    and recorded once."""
    import shutil

    pool_dir, image_root, plus_csv = _pool_with_labels(tmp_path)
    cache = tmp_path / "cache"
    build_cache(pool_dir, image_root, plus_csv, cache, size=32, val_fraction=0.5, seed=3)
    checkpoint, swin_dir, bert_dir, _ = write_tiny_checkpoint(tmp_path / "ckpt")
    out = tmp_path / "run"
    common = ["--variant", "FT6", "--lr", "1e-3", "--batch-size", "8", "--seed", "1",
              "--subset", "8", "--cache", str(cache), "--pool-dir", str(pool_dir),
              "--checkpoint", str(checkpoint), "--swin-config", str(swin_dir),
              "--bert-config", str(bert_dir), "--device", "cpu", "--precision", "fp32",
              "--num-workers", "0", "--out", str(out), "--epochs", "2"]
    assert train.main([*common, "--max-steps", "1"]) == 0
    assert train.main([*common, "--resume"]) == 0
    stale = tmp_path / "state_epoch1.pt"
    assert train.main([*common[:-2], "--epochs", "1", "--out", str(tmp_path / "run1")]) == 0
    shutil.copy(tmp_path / "run1" / train.STATE_FILE, stale)
    # The lost upload: state_last.pt of epoch 1 next to metrics.json and adapters of epoch 2.
    shutil.copy(stale, out / train.STATE_FILE)
    assert train.main([*common, "--resume"]) == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert [e["epoch"] for e in metrics["epochs"]] == [1, 2]
    assert len(metrics["steps"]) == 2 and metrics["resumed_from_epoch"] == 1


def test_git_commit_comes_from_the_environment_on_colab(monkeypatch):
    monkeypatch.setenv("CXR_GIT_COMMIT", "c" * 40)
    assert train._git_commit() == "c" * 40


def test_emulated_bf16_counts_as_no_bf16(monkeypatch):
    """Colab free's T4: torch says bf16 is supported only through slow emulation."""
    import pytest

    from training import val

    monkeypatch.setattr(torch.cuda, "is_bf16_supported",
                        lambda including_emulation=True: including_emulation)
    with pytest.raises(ValueError, match="no bf16"):
        train.resolve_precision("bf16", torch.device("cuda"))
    assert val._amp_dtype(torch.device("cuda")) == torch.float16
