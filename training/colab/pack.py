"""Bundle what a Colab GPU needs to fine-tune, for Google Drive (training/colab/README.md).

    uv run --project training python training/colab/pack.py --out data/colab_bundle

Writes <out>/cxr_v11/:
  code.zip            src/, training/, configs/ as they are on disk (tracked + new, never ignored
                      files such as .env or .venv)
  data/...            hard links to the pool cache, pool labels, checkpoint and the pinned Swin /
                      BERT configs (no extra disk on the same volume; copied otherwise)
  zero_shot_val.json  this machine's pool-val P@5 of the un-tuned checkpoint (Colab must match it)
  manifest.json       sha256 + size of every file, git HEAD and the uncommitted paths
Upload the cxr_v11 folder to MyDrive as is. The CheXpert data and the CXR-CLIP weights (CC BY-NC)
stay in a private Drive: never share the folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config  # noqa: E402
from training.model import CONFIG_FILES, resolve_config_dir  # noqa: E402

CODE_PATHS = ["src", "training", "configs"]
DATA_FILES = [
    "data/train_cache/pool_224/images.npy",
    "data/train_cache/pool_224/index.parquet",
    "data/train_cache/pool_224/cache.json",
    "data/probe_pool/pool_labels.parquet",
]
BUNDLE = "cxr_v11"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/colab_bundle")
    parser.add_argument("--zero-shot", default="runs/zero_shot_val.json",
                        help="output of training/zero_shot_val.py on this machine")
    args = parser.parse_args(argv)

    zero_shot = ROOT / args.zero_shot
    if not zero_shot.is_file():
        print(f"missing {zero_shot}: run training/zero_shot_val.py first", file=sys.stderr)
        return 1
    bundle = (ROOT / args.out / BUNDLE).resolve()
    if (bundle / "data").exists():
        shutil.rmtree(bundle / "data")    # links only: the originals stay where they are
    bundle.mkdir(parents=True, exist_ok=True)

    # Code: tracked and new files under the code paths; .gitignore keeps .env/.venv out.
    files = [f for f in _git("ls-files", "--cached", "--others", "--exclude-standard", "--",
                             *CODE_PATHS).splitlines() if (ROOT / f).is_file()]
    code_zip = bundle / "code.zip"
    with zipfile.ZipFile(code_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(files):
            zf.write(ROOT / f, f)

    # Data: the cache, labels, checkpoint and the two pinned HF config snapshots.
    candidate = load_config("models")["image_encoder"]["candidates"]["cxrclip"]
    links = {f: ROOT / f for f in DATA_FILES}
    links["data/cache/models/cxrclip/swint_m.pt"] = ROOT / candidate["weights"]
    for name, spec in (("swin", candidate["swin_config"]), ("bert", candidate["text_config"])):
        # The snapshot dir also holds whatever else was cached (pytorch_model.bin): configs only.
        for p in resolve_config_dir(spec).iterdir():
            if p.is_file() and any(fnmatch(p.name, pattern) for pattern in CONFIG_FILES):
                links[f"data/hf/{name}/{p.name}"] = p.resolve()
    entries = {}
    for rel, src in sorted(links.items()):
        _link(src, bundle / rel)
        entries[rel] = {"sha256": sha256(bundle / rel), "bytes": (bundle / rel).stat().st_size}
        print(f"{rel}  {entries[rel]['bytes']:,} B", flush=True)
    shutil.copy2(zero_shot, bundle / "zero_shot_val.json")

    dirty = [line[3:] for line in _git("status", "--porcelain", "--", *CODE_PATHS).splitlines()]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head": _git("rev-parse", "HEAD").strip(), "uncommitted": dirty,
        "code_zip": {"sha256": sha256(code_zip), "files": len(files)},
        "data": entries,
        "zero_shot_val": json.loads(zero_shot.read_text(encoding="utf-8")),
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(e["bytes"] for e in entries.values()) + code_zip.stat().st_size
    print(f"bundle {bundle}  {total / 2**30:.2f} GiB, {len(files)} code files, "
          f"{len(dirty)} uncommitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
