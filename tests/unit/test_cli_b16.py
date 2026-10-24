"""B16 tooling: embed only pool-val for selection, register a Colab run as a candidate."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import polars as pl

from src.cli.main import main
from src.common import config
from src.embeddings import embed
from src.embeddings.probe import patient_split
from tests.unit.fakes import FakeEncoder

RUN = "FT6_lr3e-05"


def _pool(tmp_path, n=80):
    pool = pl.DataFrame({
        "image_id": [f"{2000 + i:064x}" for i in range(n)],
        "study_id": [f"poolpatient{i:05d}/study1" for i in range(n)],
        "patient_id": [f"poolpatient{i:05d}" for i in range(n)],
        "path_to_image": [f"train/poolpatient{i:05d}/study1/view1_frontal.jpg" for i in range(n)],
        "projection": ["AP"] * n,
    })
    path = tmp_path / "pool_images.parquet"
    pool.write_parquet(path)
    return pool, path


def test_embed_pool_val_keeps_only_the_pool_val_patients(tmp_path, monkeypatch):
    pool, path = _pool(tmp_path)
    monkeypatch.setattr("src.cli.main._encoder_by_name",
                        lambda name, device: ("fake2", FakeEncoder()))
    assert main(["embed", "--images", str(path), "--out", str(tmp_path / "emb"), "--pool-val",
                 "--build-id", "pv1", "--encoder", "fake2", "--image-root", "root"]) == 0
    split = config.load_config("models")["image_probe"]
    val = patient_split(pool["patient_id"].to_list(), val_fraction=split["val_fraction"],
                        seed=split["seed"])
    shards = embed.load_shards_manifest(tmp_path / "emb" / "pv1")
    assert 0 < shards["embedded"] == int(val.sum()) < pool.height
    _, ids = embed.read_shard(tmp_path / "emb" / "pv1" / shards["shards"][0]["file"])
    assert set(ids) <= set(pool.filter(pl.Series(val))["image_id"])


def _configs(tmp_path, monkeypatch):
    out = tmp_path / "configs"
    shutil.copytree("configs", out)
    monkeypatch.setenv("CONFIG_DIR", str(out))
    return out


def _run_dir(tmp_path, *, recorded=None):
    run = tmp_path / "runs" / RUN
    run.mkdir(parents=True)
    (run / "model.pt").write_bytes(b"merged weights")
    sha = hashlib.sha256(b"merged weights").hexdigest()
    (run / "model.sha256").write_text(f"{recorded or sha}  model.pt\n", encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps({
        "config": {"variant": "FT6", "lr": 3e-5}, "best_epoch": 4, "model_sha256": sha,
        "epochs": [{"epoch": 3, "val_p5_mean": 0.40}, {"epoch": 4, "val_p5_mean": 0.42}]}),
        encoding="utf-8")
    return run, sha


def test_register_run_prints_the_candidate_and_writes_nothing(tmp_path, monkeypatch, capsys):
    cfg = _configs(tmp_path, monkeypatch)
    run, sha = _run_dir(tmp_path)
    before = (cfg / "models.yaml").read_text(encoding="utf-8")
    assert main(["register-run", str(run)]) == 0
    out = capsys.readouterr().out
    assert "cxrclip_ft6_lr3e-05:" in out and f"sha256:{sha}" in out
    assert (cfg / "models.yaml").read_text(encoding="utf-8") == before


def test_register_run_writes_a_candidate_with_the_cxrclip_adapter(tmp_path, monkeypatch):
    cfg = _configs(tmp_path, monkeypatch)
    run, sha = _run_dir(tmp_path)
    assert main(["register-run", str(run), "--write"]) == 0
    encoders = config.load_config("models", cfg)["image_encoder"]
    new, base = encoders["candidates"].pop("cxrclip_ft6_lr3e-05"), encoders["candidates"]["cxrclip"]
    for key in ("model_id", "framework", "variant", "swin_config", "text_config", "license"):
        assert new[key] == base[key], key
    assert "url" not in new and "source_sha256" not in new
    assert new["weights"] == (run / "model.pt").resolve().as_posix()
    assert new["revision"] == f"sha256:{sha}"
    assert new["finetune"] == {"run": run.resolve().as_posix(), "variant": "FT6", "lr": 3e-5,
                               "best_epoch": 4, "val_p5_mean": 0.42,
                               "base_revision": base["revision"]}
    assert encoders == config.load_config("models", Path("configs"))["image_encoder"]
    text = (cfg / "models.yaml").read_text(encoding="utf-8")
    assert "# MIMIC-only" in text and text.index("cxrclip_ft6_lr3e-05:") < text.index(
        "text_prompt_template:")
    scope = config.index_config_scope("cxrclip_ft6_lr3e-05", cfg)
    assert scope[1] == "models:image_encoder.candidates.cxrclip_ft6_lr3e-05"


def test_register_run_refuses_a_checksum_that_does_not_match(tmp_path, monkeypatch, capsys):
    _configs(tmp_path, monkeypatch)
    run, _ = _run_dir(tmp_path, recorded="0" * 64)
    assert main(["register-run", str(run), "--write"]) == 1
    assert "checksum" in capsys.readouterr().err


def test_register_run_refuses_a_name_already_taken_or_with_a_dot(tmp_path, monkeypatch, capsys):
    _configs(tmp_path, monkeypatch)
    run, _ = _run_dir(tmp_path)
    assert main(["register-run", str(run), "--write"]) == 0
    assert main(["register-run", str(run), "--write"]) == 1
    assert "already a candidate" in capsys.readouterr().err
    assert main(["register-run", str(run), "--name", "cxrclip_ft6_lr3.0e-5"]) == 1
    assert "letters, digits" in capsys.readouterr().err
