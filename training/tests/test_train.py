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
              "--device", "cpu", "--no-amp", "--num-workers", "0"]
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
        assert set(e) >= {"val_p5_single", "val_p5_pair", "val_p5_negation", "vram_mib"}
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
              "--bert-config", str(bert_dir), "--device", "cpu", "--no-amp",
              "--num-workers", "0"]
    assert train.main(["--variant", "FT2", "--out", str(tmp_path / "ft2"), *common]) == 0
    ft2 = json.loads((tmp_path / "ft2" / "metrics.json").read_text(encoding="utf-8"))
    assert ft2["config"]["impression_fraction"] == 0.2 and ft2["config"]["repel_weight"] == 0.3
    assert train.main(["--variant", "FT3", "--out", str(tmp_path / "ft3"), *common]) == 0
    ft3 = json.loads((tmp_path / "ft3" / "metrics.json").read_text(encoding="utf-8"))
    assert ft3["config"]["impression_fraction"] == 0.0 and ft3["config"]["repel_weight"] == 0.0
    assert ft3["best_epoch"] == 1 and (tmp_path / "ft3" / "model.pt").is_file()
