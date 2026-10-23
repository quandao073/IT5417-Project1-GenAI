"""The Colab driver's checks (training/colab/colab_run.py): what makes setup, check-env and the
gate refuse, and how runs are named and summarised. No GPU, no Colab."""

from __future__ import annotations

import hashlib
import json
import zipfile

from training.colab import colab_run as cr

ZERO = {"val_p5": {"single": 0.667, "pair": 0.196, "negation": 0.267, "mean": 0.377},
        "checkpoint_sha256": "a" * 64, "pool_sha256": "b" * 64}


def _epoch(n: int, mean: float, text_cos: float | None = 0.6, skipped: int = 0) -> dict:
    record = {"epoch": n, "val_p5_single": mean, "val_p5_pair": mean, "val_p5_negation": mean,
              "val_p5_mean": mean, "s_per_step": 0.25, "val_seconds": 12.0,
              "skipped_steps": skipped}
    if text_cos is not None:
        record["val_text_cos"] = text_cos
    return record


def test_run_name():
    assert cr.run_name("FT6", 3e-5, "gate") == "gate_FT6_lr3e-05"
    assert cr.run_name("FT1", 1e-4, None) == "FT1_lr1e-04"


def test_compare_zero_shot_tolerates_one_flip_not_a_broken_environment():
    assert cr.compare_zero_shot(ZERO, ZERO) == []
    one_flip = {**ZERO, "val_p5": {**ZERO["val_p5"], "single": 0.667 - 1 / 60,
                                   "mean": 0.377 - 1 / 180}}
    assert cr.compare_zero_shot(one_flip, ZERO) == []
    broken = {**ZERO, "val_p5": {**ZERO["val_p5"], "pair": 0.10, "mean": 0.345}}
    assert any("pair" in p for p in cr.compare_zero_shot(broken, ZERO))
    assert any("mean" in p for p in cr.compare_zero_shot(broken, ZERO))
    other_weights = {**ZERO, "checkpoint_sha256": "c" * 64}
    assert cr.compare_zero_shot(other_weights, ZERO) == ["checkpoint_sha256 differs"]


def test_gate_fails_a_collapse_not_a_short_run_within_noise_of_zero_shot():
    good = {"epochs": [_epoch(1, 0.36), _epoch(2, 0.41)], "best_epoch": 2}
    assert cr.gate_verdict(good, ZERO) == []
    # The laptop's short FT6 run beat zero-shot by only 0.009: a T4 may land just under it.
    within_noise = {"epochs": [_epoch(1, 0.372), _epoch(2, 0.368)], "best_epoch": 1}
    assert cr.gate_verdict(within_noise, ZERO) == []
    below = {"epochs": [_epoch(1, 0.21), _epoch(2, 0.25)], "best_epoch": 2}    # FT1's collapse
    assert any("below zero-shot" in p for p in cr.gate_verdict(below, ZERO))
    collapsed = {"epochs": [_epoch(1, 0.40, text_cos=0.94)], "best_epoch": 1}
    assert any("collapsed" in p for p in cr.gate_verdict(collapsed, ZERO))
    skipped = {"epochs": [_epoch(1, 0.40, skipped=3)], "best_epoch": 1}
    assert any("non-finite" in p for p in cr.gate_verdict(skipped, ZERO))
    assert cr.gate_verdict({"epochs": []}, ZERO) == ["no finished epoch"]


def _bundle(tmp_path, payload: bytes = b"cache bytes"):
    bundle = tmp_path / "cxr_v11"
    (bundle / "data/train_cache").mkdir(parents=True)
    (bundle / "data/train_cache/images.npy").write_bytes(payload)
    with zipfile.ZipFile(bundle / "code.zip", "w") as zf:
        zf.writestr("training/x.py", "x = 1\n")
    manifest = {"git_head": "0" * 40, "uncommitted": [],
                "code_zip": {"sha256": hashlib.sha256((bundle / "code.zip").read_bytes())
                             .hexdigest()},
                "data": {"data/train_cache/images.npy": {
                    "sha256": hashlib.sha256(b"cache bytes").hexdigest(),
                    "bytes": len(b"cache bytes")}}}
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle


def test_setup_copies_and_verifies_every_file(tmp_path):
    work = tmp_path / "work"
    assert cr.setup(_bundle(tmp_path), work) == 0
    assert (work / "data/train_cache/images.npy").read_bytes() == b"cache bytes"


def test_setup_refuses_a_file_that_differs_from_the_manifest(tmp_path):
    assert cr.setup(_bundle(tmp_path, payload=b"cache bytez"), tmp_path / "work") == 1


def test_summary_picks_the_best_finished_lr_per_variant(tmp_path, capsys):
    bundle = tmp_path / "cxr_v11"
    # Only finished full-data runs compete: unfinished and subset (gate, dry) runs do not.
    for name, lr, mean, done, subset in (("FT6_lr3e-05", 3e-5, 0.45, True, None),
                                         ("FT6_lr1e-04", 1e-4, 0.48, True, None),
                                         ("FT6_lr1e-05", 1e-5, 0.50, False, None),
                                         ("gate_FT6_lr3e-05", 3e-5, 0.60, True, 20000),
                                         ("dry_FT6_lr1e-04", 1e-4, 0.70, True, 2000)):
        run = bundle / "runs" / name
        run.mkdir(parents=True)
        (run / "metrics.json").write_text(json.dumps(
            {"config": {"variant": "FT6", "lr": lr, "subset": subset},
             "epochs": [_epoch(1, mean)],
             "best_epoch": 1}), encoding="utf-8")
        if done:
            (run / "model.pt").write_bytes(b"")
    assert cr.summary(bundle) == 0
    assert "FT6: best lr 1e-04 on pool-val P@5 mean 0.480" in capsys.readouterr().out


def test_machine_flags_pick_fp16_without_checkpointing_on_a_t4(monkeypatch):
    import types

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported",
                        lambda including_emulation=True: including_emulation)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(total_memory=15_828_320_256))
    monkeypatch.setattr(cr.os, "cpu_count", lambda: 2)
    assert cr.machine_flags() == ["--num-workers", "2", "--precision", "fp16",
                                  "--no-grad-checkpointing"]
    # The laptop (8 GB) keeps checkpointing.
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(total_memory=8_585_216_000))
    assert "--no-grad-checkpointing" not in cr.machine_flags()


# --- the Colab environment (WORKLOG T32): never Colab's own Python and packages -------------
REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
LINUX_PY312 = {"sys_platform": "linux", "platform_machine": "x86_64",
               "platform_python_implementation": "CPython", "python_version": "3.12",
               "python_full_version": "3.12.9", "implementation_name": "cpython",
               "platform_system": "Linux", "os_name": "posix"}


def _colab_packages() -> list[dict]:
    """The pylock entries that apply to Colab: Linux x86_64, CPython 3.12."""
    import tomllib

    from packaging.markers import Marker

    lock = tomllib.loads((REPO / "training/colab/pylock.colab.toml").read_text(encoding="utf-8"))
    return [p for p in lock["packages"]
            if "marker" not in p or Marker(p["marker"]).evaluate(LINUX_PY312)]


def test_colab_pylock_is_the_training_lock_for_linux_py312():
    import tomllib

    lock = tomllib.loads((REPO / "training/uv.lock").read_text(encoding="utf-8"))
    locked = {(p["name"], p["version"]) for p in lock["package"]}
    packages = _colab_packages()
    pins = {p["name"]: p["version"] for p in packages}
    assert len(pins) == len(packages), "one version per package on Colab"
    assert set(pins.items()) <= locked
    # The packages whose version decides the numbers (Colab ships torch 2.11, transformers 5.x).
    assert {k: pins[k] for k in ("torch", "torchvision", "transformers", "tokenizers", "peft")} == {
        "torch": "2.4.1+cu124", "torchvision": "0.19.1+cu124", "transformers": "4.44.2",
        "tokenizers": "0.19.1", "peft": "0.14.0"}


def test_colab_pylock_installs_wheels_from_pypi_and_pytorch_only():
    # pypi.nvidia.com timed out in the rehearsal (WORKLOG T32); the lock never used it.
    for p in _colab_packages():
        assert p.get("wheels"), f"{p['name']} would build from source"
        for wheel in p["wheels"]:
            assert wheel["url"].split("/")[2] in {"files.pythonhosted.org",
                                                  "download-r2.pytorch.org",
                                                  "download.pytorch.org"}, wheel["url"]
            assert len(wheel["hashes"]["sha256"]) == 64


def test_notebook_runs_every_step_in_the_py312_venv():
    nb = json.loads((REPO / "training/colab/train_colab.ipynb").read_text(encoding="utf-8"))
    source = "".join(line for cell in nb["cells"] for line in cell["source"])
    assert "--python 3.12" in source and "uv pip sync" in source
    assert "training/colab/pylock.colab.toml" in source
    calls = [line.strip() for line in source.splitlines() if "colab_run.py" in line]
    assert calls and all(line.startswith("!{PY} ") for line in calls), calls
    assert "!pip install" not in source.replace("!pip install -q uv", "")
