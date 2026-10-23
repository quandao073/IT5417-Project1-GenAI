"""LoRA fine-tuning of CXR-CLIP with the label-aware loss (plan Task 12; WORKLOG D30, D33, D34).

    uv run --project training python training/train.py --variant FT1 --lr 3e-5 --out runs/ft1_lr3e-5

Pool-train rows of the image cache are captioned on the fly (training/captions.py), both
towers get LoRA on query/key/value, AdamW with linear warmup then cosine, fp16 autocast with a
GradScaler on CUDA. After every epoch the adapter is saved and pool-val is scored by P@5 of the
S12 single / pair / negation templates (training/val.py); the best epoch by the mean of the
three is exported merged as `model.pt` (+ `model.sha256`), which `src.embeddings.cxrclip`
loads like the original checkpoint. Early stop after `optim.early_stop_patience` epochs
without improvement. Nothing here reads the corpus or `valid/`.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# `python training/train.py` puts training/ first on sys.path, not the repo root, and
# site-packages holds another top-level `training` package (open_clip's). Root first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.nn.functional import normalize
from torch.utils.data import DataLoader

from src.common.config import load_config
from src.data import probe_pool
from src.embeddings.cxrclip import file_sha256
from src.embeddings.embed import write_json_atomic
from src.retrieval.templates import training_concepts
from training import captions, data, lora, loss, val
from training.model import load_base, resolve_config_dir

ADAPTER_FILE = "adapter_model.safetensors"
_LOG_EVERY = 100


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() or None


def _library_versions() -> dict[str, str]:
    import peft
    import torchvision
    import transformers

    return {"torch": torch.__version__, "torchvision": torchvision.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__}


def warmup_cosine(warmup_steps: int, total_steps: int):
    """LambdaLR factor: linear warmup to 1, then cosine to 0 at total_steps."""
    warmup = max(1, min(warmup_steps, total_steps))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, dict]:
    cfg = load_config("training")
    candidate = load_config("models")["image_encoder"]["candidates"]["cxrclip"]
    optim = cfg["optim"]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", required=True, choices=sorted(cfg["variants"]))
    parser.add_argument("--lr", type=float, required=True,
                        help=f"one of training.yaml:optim.lr_grid {optim['lr_grid']} for FT1")
    parser.add_argument("--epochs", type=int, default=optim["epochs"])
    parser.add_argument("--batch-size", type=int, default=optim["batch_size"])
    parser.add_argument("--subset", type=int, default=None,
                        help="train on this many pool-train rows (smoke runs)")
    parser.add_argument("--seed", type=int, default=cfg["split"]["seed"])
    parser.add_argument("--cache", default=cfg["image"]["cache_dir"])
    parser.add_argument("--pool-dir", default="data/probe_pool")
    parser.add_argument("--checkpoint", default=candidate["weights"],
                        help="converted CXR-CLIP checkpoint (cxrclip.convert_checkpoint)")
    parser.add_argument("--swin-config", default=candidate["swin_config"],
                        help="local dir or <repo_id>@<commit sha>")
    parser.add_argument("--bert-config", default=candidate["text_config"],
                        help="local dir or <repo_id>@<commit sha>")
    parser.add_argument("--out", required=True, help="run directory, e.g. runs/ft1_lr3e-5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=bool(optim["amp"]))
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv), cfg


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_args(argv)
    optim, variant, lora_cfg = cfg["optim"], cfg["variants"][args.variant], cfg["lora"]
    concepts_cfg = load_config("concepts")
    concepts = training_concepts(cfg)
    device = torch.device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Data: captions from the pool's resolved labels, aligned with the cache rows.
    images, index, cache_meta = data.load_cache(args.cache)
    resolved = pl.read_parquet(Path(args.pool_dir) / probe_pool.POOL_LABELS)
    statuses = probe_pool.resolved_status_matrix(resolved, index["study_id"].to_list())
    impression_fraction = float(variant["impression_fraction"])
    sampler = captions.CaptionSampler(
        statuses, concepts, concepts_cfg, mix=cfg["captions"]["mix"],
        impressions=index["impression"].to_list() if impression_fraction > 0 else None,
        impression_fraction=impression_fraction, seed=args.seed)
    train_ds = data.PoolDataset(args.cache, sampler, train=True, augment_cfg=cfg["augmentation"])
    if args.subset and args.subset < len(train_ds):
        rng = np.random.default_rng(args.seed)
        train_ds.rows = np.sort(rng.choice(train_ds.rows, args.subset, replace=False))
    if len(train_ds) < args.batch_size:
        print(f"only {len(train_ds)} training rows for batch size {args.batch_size}")
        return 1
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, worker_init_fn=data.reseed_worker,
        persistent_workers=args.num_workers > 0, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed))
    is_val = index["is_val"].to_numpy()
    val_images = np.asarray(images[np.flatnonzero(is_val)])
    val_statuses = statuses[is_val]

    # Model: the converted checkpoint, LoRA on both towers, everything else frozen.
    module, tokenizer, ckpt_config = load_base(
        args.checkpoint, resolve_config_dir(args.swin_config), resolve_config_dir(args.bert_config))
    peft_model = lora.apply_lora(module, rank=lora_cfg["rank"], alpha=lora_cfg["alpha"],
                                 dropout=lora_cfg["dropout"], targets=lora_cfg["targets"])
    peft_model.to(device)
    net = lora.unwrap(peft_model)
    params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=optim["weight_decay"])
    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(optim["warmup_steps"], total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    text_max_length = int(optim["text_max_length"])
    lam = float(variant["repel_weight"])

    metrics: dict = {
        "config": {
            "variant": args.variant, "lr": args.lr, "epochs": args.epochs,
            "batch_size": args.batch_size, "subset": args.subset, "seed": args.seed,
            "repel_weight": lam, "impression_fraction": impression_fraction,
            "mix": cfg["captions"]["mix"], "lora": lora_cfg, "augmentation": cfg["augmentation"],
            "weight_decay": optim["weight_decay"], "warmup_steps": optim["warmup_steps"],
            "early_stop_patience": optim["early_stop_patience"],
            "text_max_length": text_max_length, "concepts": concepts,
            "checkpoint": str(args.checkpoint), "swin_config": str(args.swin_config),
            "bert_config": str(args.bert_config), "cache": str(args.cache),
            "pool_dir": str(args.pool_dir),
        },
        "n_train": len(train_ds), "n_val": int(is_val.sum()), "steps_per_epoch": len(loader),
        "trainable_fraction": lora.trainable_fraction(peft_model), "cache_meta": cache_meta,
        "device": str(device), "amp": use_amp, "library_versions": _library_versions(),
        "git_commit": _git_commit(), "started_at": _now(),
        "steps": [], "epochs": [], "best_epoch": None, "stopped_early": False,
        "model_sha256": None,
    }
    print(f"variant {args.variant} lr {args.lr:g} device {device} amp {use_amp} "
          f"train rows {len(train_ds)} val rows {int(is_val.sum())} "
          f"steps/epoch {len(loader)} trainable {metrics['trainable_fraction']:.4%}", flush=True)

    best_score, best_epoch, since_best = -math.inf, None, 0
    best_record: dict | None = None
    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        peft_model.train()
        losses: list[float] = []
        for step, (pixels, texts, y) in enumerate(loader, 1):
            tokens = tokenizer(list(texts), padding="longest", truncation=True,
                               max_length=text_max_length, return_tensors="pt")
            pixels = pixels.to(device, non_blocking=True)
            y = y.to(device)
            with torch.autocast(device.type, dtype=val._amp_dtype(device), enabled=use_amp):
                zi = net.encode_image(pixels)
                zt = net.encode_text(tokens["input_ids"].to(device),
                                     tokens["attention_mask"].to(device))
            # The loss runs in fp32 whatever autocast did inside the towers.
            zi, zt = normalize(zi.float(), dim=-1), normalize(zt.float(), dim=-1)
            scale = net.logit_scale.exp().clamp(max=100.0)
            step_loss = loss.label_aware_loss(zi, zt, y, scale, lam)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(step_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            value = float(step_loss.detach())
            losses.append(value)
            metrics["steps"].append(value)
            if step % _LOG_EVERY == 0 or step == len(loader):
                print(f"epoch {epoch} step {step}/{len(loader)} loss {value:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e}", flush=True)

        peft_model.save_pretrained(str(out / f"adapter_epoch{epoch}"))
        p5 = val.val_p5(net, tokenizer, val_images, val_statuses, concepts, concepts_cfg, device,
                        text_max_length=text_max_length, batch_size=max(args.batch_size, 64),
                        amp=use_amp)
        record = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "val_p5_single": p5["single"], "val_p5_pair": p5["pair"],
            "val_p5_negation": p5["negation"], "val_p5_mean": p5["mean"],
            "val_n_queries": {k: p5[f"n_{k}"] for k in val.KINDS},
            "seconds": round(time.perf_counter() - tick, 1),
            "vram_mib": (round(torch.cuda.max_memory_allocated(device) / 2**20)
                         if device.type == "cuda" else None),
        }
        metrics["epochs"].append(record)
        # Selection is pool-val P@5 (D33); only a val set too small to score any query falls
        # back to the training loss, so a smoke run still exports something.
        score = p5["mean"] if not math.isnan(p5["mean"]) else -record["train_loss"]
        if score > best_score:
            best_score, best_epoch, best_record, since_best = score, epoch, record, 0
        else:
            since_best += 1
        metrics["best_epoch"] = best_epoch
        write_json_atomic(out / "metrics.json", metrics)
        print(f"epoch {epoch} train_loss {record['train_loss']:.4f} val_p5 "
              f"single {p5['single']:.3f} pair {p5['pair']:.3f} negation {p5['negation']:.3f} "
              f"mean {p5['mean']:.3f} best_epoch {best_epoch} "
              f"({record['seconds']} s, vram {record['vram_mib']} MiB)", flush=True)
        if since_best >= int(optim["early_stop_patience"]):
            metrics["stopped_early"] = True
            print(f"early stop: no improvement for {since_best} epochs", flush=True)
            break

    # Export the best epoch, merged, with the checkpoint's key names (H3).
    best_adapter = out / f"adapter_epoch{best_epoch}" / ADAPTER_FILE
    set_peft_model_state_dict(peft_model, load_file(best_adapter))
    meta = {
        "variant": args.variant, "lr": args.lr, "best_epoch": best_epoch,
        "epochs_run": len(metrics["epochs"]), "val": best_record,
        "repel_weight": lam, "impression_fraction": impression_fraction,
        "lora": lora_cfg, "concepts": concepts, "seed": args.seed,
        "base_checkpoint": Path(args.checkpoint).name,
        "base_sha256": file_sha256(Path(args.checkpoint)),
        "pool_sha256": cache_meta.get("pool_sha256"),
        "git_commit": metrics["git_commit"], "trained_at": _now(),
    }
    sha = lora.export_merged(peft_model, ckpt_config, out / "model.pt", meta)
    (out / "model.sha256").write_text(f"{sha}  model.pt\n", encoding="utf-8")
    metrics["model_sha256"] = sha
    metrics["finished_at"] = _now()
    write_json_atomic(out / "metrics.json", metrics)
    print(f"best epoch {best_epoch}; wrote {out / 'model.pt'} sha256 {sha}", flush=True)
    return 0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


if __name__ == "__main__":
    sys.exit(main())
