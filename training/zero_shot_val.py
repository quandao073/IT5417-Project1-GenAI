"""Pool-val P@5 of the un-tuned checkpoint: the line every fine-tune must beat, and the number a
new machine (Colab) must reproduce before it trains (same metric as each epoch of train.py).

    uv run --project training python training/zero_shot_val.py --out runs/zero_shot_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# `python training/zero_shot_val.py` puts training/ first on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
import torch

from src.common.config import load_config
from src.data import probe_pool
from src.embeddings.cxrclip import file_sha256
from src.retrieval.templates import training_concepts
from training import data, val
from training.model import load_base, resolve_config_dir


def main(argv: list[str] | None = None) -> int:
    cfg = load_config("training")
    candidate = load_config("models")["image_encoder"]["candidates"]["cxrclip"]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default=cfg["image"]["cache_dir"])
    parser.add_argument("--pool-dir", default="data/probe_pool")
    parser.add_argument("--checkpoint", default=candidate["weights"])
    parser.add_argument("--swin-config", default=candidate["swin_config"],
                        help="a local directory or '<repo>@<sha>'")
    parser.add_argument("--bert-config", default=candidate["text_config"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    images, index, cache_meta = data.load_cache(args.cache)
    resolved = pl.read_parquet(Path(args.pool_dir) / probe_pool.POOL_LABELS)
    statuses = probe_pool.resolved_status_matrix(resolved, index["study_id"].to_list())
    is_val = index["is_val"].to_numpy()
    module, tokenizer, _ = load_base(args.checkpoint, resolve_config_dir(args.swin_config),
                                     resolve_config_dir(args.bert_config))
    module.to(device)
    tick = time.perf_counter()
    p5 = val.val_p5(module, tokenizer, images[is_val], statuses[is_val], training_concepts(cfg),
                    load_config("concepts"), device,
                    text_max_length=int(cfg["optim"]["text_max_length"]),
                    amp=device.type == "cuda")
    result = {"val_p5": p5, "n_val": int(is_val.sum()), "seconds": time.perf_counter() - tick,
              "device": str(device), "torch": torch.__version__,
              "checkpoint_sha256": file_sha256(Path(args.checkpoint)),
              "pool_sha256": cache_meta.get("pool_sha256")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
