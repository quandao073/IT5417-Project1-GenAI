"""Drive the fine-tune on a Colab GPU from the bundle of pack.py (training/colab/README.md).

Run from the work directory where code.zip was unpacked (data is copied next to the code):

    python training/colab/colab_run.py setup     --bundle B   # data to local disk + sha256
    python training/colab/colab_run.py check-env --bundle B   # zero-shot P@5 must match this laptop
    python training/colab/colab_run.py train     --bundle B --variant FT6 --lr 3e-5 [--subset N]
                                                 [--epochs E] [--tag gate]
    python training/colab/colab_run.py gate      --bundle B --run gate_FT6_lr3e-05
    python training/colab/colab_run.py summary   --bundle B

Runs write to <bundle>/runs/<name> on Drive with --resume, so a disconnect loses at most the
epoch in progress: run the same `train` again. Selection reads pool-val only (D33).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# A full run of train.py: pool-train rows / batch 128 (metrics.json:steps_per_epoch of B14).
FULL_STEPS_PER_EPOCH = 618
# check-env: one flipped top-5 image in the 12 single queries moves P@5 by 1/60.
TOL_MEAN, TOL_KIND = 0.01, 0.035
# gate: a run whose template texts are this alike has collapsed (FT1 reached 0.94, base 0.53).
MAX_TEXT_COS = 0.8
# gate: a short run may land within noise of zero-shot (the laptop's FT6 beat it by 0.009); what it
# must not do is fall clearly below it, as FT1's collapse did (0.255 vs 0.377). The full runs decide
# whether FT6 beats zero-shot.
GATE_MARGIN = 0.02
KINDS = ("single", "pair", "negation", "mean")
HF = {"--swin-config": "data/hf/swin", "--bert-config": "data/hf/bert"}
# Per run on Drive: the code it started with and one entry per Colab session that worked on it.
PROVENANCE = "provenance.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _manifest(bundle: Path) -> dict:
    return json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))


def setup(bundle: Path, work: Path) -> int:
    """Copy every data file of the manifest to the local disk and check its sha256."""
    manifest = _manifest(bundle)
    if _sha256(bundle / "code.zip") != manifest["code_zip"]["sha256"]:
        print("code.zip does not match manifest.json: upload the bundle again")
        return 1
    bad = []
    for rel, entry in manifest["data"].items():
        dst = work / rel
        if not dst.is_file() or dst.stat().st_size != entry["bytes"]:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundle / rel, dst)
        ok = _sha256(dst) == entry["sha256"]
        print(f"{'ok ' if ok else 'BAD'} {rel}")
        if not ok:
            bad.append(rel)
    if manifest["uncommitted"]:
        print(f"note: code has {len(manifest['uncommitted'])} uncommitted files "
              f"on top of {manifest['git_head'][:7]}")
    return 1 if bad else 0


def compare_zero_shot(here: dict, reference: dict) -> list[str]:
    """Reasons this machine does not reproduce the laptop's zero-shot run (empty = it does)."""
    problems = []
    for key in ("checkpoint_sha256", "pool_sha256"):
        if here.get(key) != reference.get(key):
            problems.append(f"{key} differs")
    for kind in KINDS:
        delta = abs(here["val_p5"][kind] - reference["val_p5"][kind])
        tol = TOL_MEAN if kind == "mean" else TOL_KIND
        if not delta <= tol:
            problems.append(f"P@5 {kind} off by {delta:.4f} (tolerance {tol})")
    return problems


def check_env(bundle: Path, work: Path) -> int:
    out = work / "zero_shot_colab.json"
    rc = subprocess.call([sys.executable, "training/zero_shot_val.py", "--out", str(out),
                          *[a for kv in HF.items() for a in kv]], cwd=work)
    if rc:
        return rc
    here = json.loads(out.read_text(encoding="utf-8"))
    reference = json.loads((bundle / "zero_shot_val.json").read_text(encoding="utf-8"))
    for kind in KINDS:
        print(f"P@5 {kind:9s} laptop {reference['val_p5'][kind]:.4f}  "
              f"here {here['val_p5'][kind]:.4f}")
    problems = compare_zero_shot(here, reference)
    print("ENV OK: this GPU reproduces the laptop" if not problems
          else "ENV FAIL: " + "; ".join(problems))
    return 1 if problems else 0


def run_name(variant: str, lr: float, tag: str | None) -> str:
    return f"{tag + '_' if tag else ''}{variant}_lr{lr:.0e}"


def machine_flags() -> list[str]:
    """Workers for the vCPUs, bf16 where the GPU has it natively (fp16 + scaler otherwise), no
    gradient checkpointing once VRAM is ample: fp32 without it peaked at 8,661 MiB at batch 128
    (B13 smoke), half precision needs less, so a T4 (14.7 GiB) runs without it."""
    import torch

    flags = ["--num-workers", str(min(os.cpu_count() or 2, 8))]
    if torch.cuda.is_available():
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)    # T4: no
        flags += ["--precision", "bf16" if native_bf16 else "fp16"]
        if torch.cuda.get_device_properties(0).total_memory >= 12 * 2**30:
            flags.append("--no-grad-checkpointing")
    return flags


def resume_problem(recorded: dict | None, code_zip_sha256: str) -> str | None:
    """Why an unfinished run must not continue with this bundle's code, or None."""
    if recorded is None or recorded["code_zip_sha256"] == code_zip_sha256:
        return None
    return (f"this run started with code.zip {recorded['code_zip_sha256'][:12]} (git "
            f"{recorded['git_head'][:7]}), the bundle now has {code_zip_sha256[:12]}. Finish it "
            "with the old code.zip + manifest.json, or start a new run with another --tag")


def _gpu_name() -> str | None:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def chosen_lr(bundle: Path, variant: str, grid: list[float]) -> tuple[float | None, str]:
    """(lr, message): the lr of the D33 grid whose finished full-data run has the best pool-val
    P@5 mean, or (None, why) while any lr of the grid has no model.pt yet."""
    scores, missing = {}, []
    for lr in grid:
        run = bundle / "runs" / run_name(variant, lr, None)
        best = None
        if (run / "model.pt").is_file() and (run / "metrics.json").is_file():
            best = best_record(json.loads((run / "metrics.json").read_text(encoding="utf-8")))
        if best is None:
            missing.append(f"{lr:.0e}")
        else:
            scores[lr] = best["val_p5_mean"]
    if missing:
        return None, (f"{variant} sweep unfinished: lr {', '.join(missing)} has no model.pt yet "
                      "(run cell 6 again)")
    lr = max(grid, key=lambda v: scores[v])    # a tie keeps the grid's first lr
    return lr, (f"{variant}: best lr {lr:.0e} on pool-val P@5 mean {scores[lr]:.3f} "
                f"(grid {', '.join(f'{v:.0e}' for v in grid)})")


def train(bundle: Path, work: Path, variant: str, lr: float | None, subset: int | None,
          epochs: int | None, tag: str | None, lr_from: str | None = None) -> int:
    """Start or continue one run; every session appends itself to the run's provenance.json.
    With `lr_from`, the lr is the one that variant's finished sweep chose (`chosen_lr`)."""
    if lr is None:
        import yaml

        optim = yaml.safe_load((work / "configs" / "training.yaml").read_text(encoding="utf-8"))
        lr, message = chosen_lr(bundle, lr_from, [float(v) for v in optim["optim"]["lr_grid"]])
        print(message)
        if lr is None:
            return 1
    out = bundle / "runs" / run_name(variant, lr, tag)
    if (out / "model.pt").is_file():
        print(f"{out.name}: finished already (model.pt exists)")
        return 0
    manifest = _manifest(bundle)
    prov_path = out / PROVENANCE
    recorded = (json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file()
                else None)
    if problem := resume_problem(recorded, manifest["code_zip"]["sha256"]):
        print(f"REFUSED: {problem}")
        return 1
    flags = machine_flags()
    precision = flags[flags.index("--precision") + 1] if "--precision" in flags else None
    record = recorded or {"code_zip_sha256": manifest["code_zip"]["sha256"],
                          "git_head": manifest["git_head"], "sessions": []}
    if record["sessions"] and record["sessions"][-1]["precision"] != precision:
        print(f"note: precision {record['sessions'][-1]['precision']} -> {precision} in the middle "
              "of this run (another GPU type); recorded in provenance.json")
    record["sessions"].append({"started_at": datetime.now(UTC).isoformat(timespec="seconds"),
                               "gpu": _gpu_name(), "precision": precision})
    out.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    cmd = [sys.executable, "training/train.py", "--variant", variant, "--lr", str(lr),
           "--out", str(out), "--resume", *[a for kv in HF.items() for a in kv], *flags]
    if subset:
        cmd += ["--subset", str(subset)]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    print(" ".join(cmd), flush=True)
    with open(out / "train.log", "a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env={**os.environ, "PYTHONUNBUFFERED": "1",
                                                "CXR_GIT_COMMIT": manifest["git_head"]})
        for line in proc.stdout:
            if "Warning" in line or "warnings.warn" in line:
                continue
            print(line, end="", flush=True)
            log.write(line)
        return proc.wait()


def best_record(metrics: dict) -> dict | None:
    by_epoch = {e["epoch"]: e for e in metrics.get("epochs", [])}
    return by_epoch.get(metrics.get("best_epoch"))


def gate_verdict(metrics: dict, reference: dict) -> list[str]:
    """Reasons a short run is not training as expected (empty = proceed to full runs)."""
    epochs = metrics.get("epochs", [])
    if not epochs:
        return ["no finished epoch"]
    problems = []
    best, last = best_record(metrics), epochs[-1]
    zero = reference["val_p5"]["mean"]
    if not best or not best["val_p5_mean"] >= zero - GATE_MARGIN:
        problems.append(f"best pool-val P@5 mean {best and best['val_p5_mean']:.3f} below "
                        f"zero-shot {zero:.3f} by more than {GATE_MARGIN}")
    text_cos = last.get("val_text_cos")
    if text_cos is not None and not text_cos < MAX_TEXT_COS:
        problems.append(f"template texts collapsed: cosine {text_cos:.3f} >= {MAX_TEXT_COS}")
    if any(e.get("skipped_steps") for e in epochs) or metrics.get("aborted"):
        problems.append("non-finite losses were skipped or the run aborted")
    return problems


def gate(bundle: Path, run: str, cu_per_hour: float | None) -> int:
    metrics = json.loads((bundle / "runs" / run / "metrics.json").read_text(encoding="utf-8"))
    reference = json.loads((bundle / "zero_shot_val.json").read_text(encoding="utf-8"))
    problems = gate_verdict(metrics, reference)
    best = best_record(metrics)
    if best:
        print("best epoch vs zero-shot: " + ", ".join(
            f"{k} {best[f'val_p5_{k}']:.3f} ({best[f'val_p5_{k}'] - reference['val_p5'][k]:+.3f})"
            for k in KINDS) + f", text_cos {best.get('val_text_cos', float('nan')):.3f}")
    last = metrics["epochs"][-1] if metrics.get("epochs") else {}
    if last:
        full_minutes = (FULL_STEPS_PER_EPOCH * last["s_per_step"] + last["val_seconds"]) * 10 / 60
        cu = f", about {full_minutes / 60 * cu_per_hour:.1f} CU" if cu_per_hour else ""
        print(f"{last['s_per_step']} s/step: a full 10-epoch run takes about "
              f"{full_minutes:.0f} min{cu}")
    print("GATE PASS: train the full runs" if not problems else "GATE FAIL: " + "; ".join(problems))
    return 1 if problems else 0


def summary(bundle: Path) -> int:
    rows = []
    for path in sorted((bundle / "runs").glob("*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        best = best_record(m) or {}
        rows.append({"run": path.parent.name, "variant": m["config"]["variant"],
                     "lr": m["config"]["lr"], "subset": m["config"].get("subset"),
                     "epochs": len(m.get("epochs", [])),
                     "best_epoch": m.get("best_epoch"),
                     "done": (path.parent / "model.pt").is_file(),
                     **{k: best.get(f"val_p5_{k}", math.nan) for k in KINDS},
                     "text_cos": best.get("val_text_cos", math.nan)})
    print(f"{'run':28s} {'ep':>3s} {'best':>4s} {'single':>7s} {'pair':>6s} {'neg':>6s} "
          f"{'mean':>6s} {'txtcos':>6s} done")
    for r in rows:
        print(f"{r['run']:28s} {r['epochs']:3d} {str(r['best_epoch']):>4s} {r['single']:7.3f} "
              f"{r['pair']:6.3f} {r['negation']:6.3f} {r['mean']:6.3f} {r['text_cos']:6.3f} "
              f"{'yes' if r['done'] else 'no'}")
    full = [r for r in rows if r["done"] and r["subset"] is None]
    for variant in sorted({r["variant"] for r in full}):
        best = max((r for r in full if r["variant"] == variant), key=lambda r: r["mean"])
        print(f"{variant}: best lr {best['lr']:.0e} on pool-val P@5 mean {best['mean']:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("setup", "check-env", "train", "gate", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", required=True, type=Path)
        if name == "train":
            p.add_argument("--variant", required=True)
            lr = p.add_mutually_exclusive_group(required=True)
            lr.add_argument("--lr", type=float)
            lr.add_argument("--lr-from", metavar="VARIANT",
                            help="the lr this variant's finished sweep chose on pool-val")
            p.add_argument("--subset", type=int)
            p.add_argument("--epochs", type=int)
            p.add_argument("--tag")
        if name == "gate":
            p.add_argument("--run", required=True)
            p.add_argument("--cu-per-hour", type=float,
                           help="from Colab's Resources panel, to project a full run's cost")
    args = parser.parse_args(argv)
    work = Path.cwd()
    if args.cmd == "setup":
        return setup(args.bundle, work)
    if args.cmd == "check-env":
        return check_env(args.bundle, work)
    if args.cmd == "train":
        return train(args.bundle, work, args.variant, args.lr, args.subset, args.epochs, args.tag,
                     lr_from=args.lr_from)
    if args.cmd == "gate":
        return gate(args.bundle, args.run, args.cu_per_hour)
    return summary(args.bundle)


if __name__ == "__main__":
    raise SystemExit(main())
