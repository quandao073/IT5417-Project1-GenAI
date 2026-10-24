"""LoRA fine-tuning of CXR-CLIP with the label-aware loss (plan Task 12; WORKLOG D30, D33, D34).

    uv run --project training python training/train.py --variant FT1 --lr 3e-5 --out runs/ft1_lr3e-5

Pool-train rows of the image cache are captioned on the fly (training/captions.py), both
towers get LoRA on query/key/value, AdamW with linear warmup then cosine. On the laptop GPU
(8 GB, about 1.5 GB of it taken by the display) the run is kept inside VRAM and stable: bf16
autocast (no loss scaling), gradient checkpointing in both towers (batch 128 stays, so the
number of in-batch negatives does not change), gradient clipping, a non-finite-loss guard,
per-epoch resumable state and a keep-awake request to Windows. After every epoch the adapter is
saved and pool-val is scored by P@5 of the
S12 single / pair / negation templates (training/val.py); the best epoch by the mean of the
three is exported merged as `model.pt` (+ `model.sha256`), which `src.embeddings.cxrclip`
loads like the original checkpoint. Early stop after `optim.early_stop_patience` epochs
without improvement. Nothing here reads the corpus or `valid/`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
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
# training.yaml:variants.<v>.targets -> the loss; "caption" is the paper's K = y yT.
LOSSES = {"caption": loss.label_aware_loss, "image": loss.image_label_loss}
STATE_FILE = "state_last.pt"
PRECISIONS = ("bf16", "fp16", "fp32")
_LOG_EVERY = 100
_WARM_STEPS = 3      # steps left out of s_per_step: CUDA kernels and workers start up there


def _git_commit() -> str | None:
    """CXR_GIT_COMMIT when set (on Colab the work dir is an unzip of the bundle, not a checkout;
    colab_run.py passes the bundle's git HEAD), else the checkout's HEAD."""
    if commit := os.environ.get("CXR_GIT_COMMIT"):
        return commit
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


def resolve_precision(name: str, device: torch.device) -> tuple[torch.dtype | None, bool]:
    """(autocast dtype or None for fp32, whether a GradScaler is needed). bf16 has fp32's range,
    so it needs no loss scaling; fp16 does, and only exists here on CUDA."""
    if name == "fp32":
        return None, False
    if name == "bf16":
        # Native only: torch also says True for a T4 (Colab free), through slow emulation.
        if device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("this GPU has no bf16; use --precision fp16")
        return torch.bfloat16, False
    if name == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 autocast is CUDA only here; use bf16 or fp32 on CPU")
        return torch.float16, True
    raise ValueError(f"precision must be one of {PRECISIONS}, got {name!r}")


def enable_gradient_checkpointing(net) -> None:
    """Recompute both towers' activations in backward. Non-reentrant, so gradients reach the
    LoRA weights even though every input to the frozen blocks has requires_grad=False."""
    kwargs = {"use_reentrant": False}
    for tower in (net.image_encoder.image_encoder, net.text_encoder.text_encoder):
        tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)


def train_step(net, pixels, input_ids, attention_mask, y, *, optimizer, scheduler, scaler,
               autocast_dtype, lam: float, max_grad_norm: float | None,
               loss_fn=loss.label_aware_loss, codes=None) -> float | None:
    """One optimisation step; returns the loss, or None when it was not finite (no update)."""
    device = pixels.device
    with torch.autocast(device.type, dtype=autocast_dtype or torch.float32,
                        enabled=autocast_dtype is not None):
        zi = net.encode_image(pixels)
        zt = net.encode_text(input_ids, attention_mask)
    # The loss runs in fp32 whatever autocast did inside the towers.
    zi, zt = normalize(zi.float(), dim=-1), normalize(zt.float(), dim=-1)
    scale = net.logit_scale.exp().clamp(max=100.0)
    step_loss = loss_fn(zi, zt, y, scale, lam, **({} if codes is None else {"codes": codes}))
    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(step_loss):
        return None
    params = [p for group in optimizer.param_groups for p in group["params"]]
    if scaler is not None:
        scaler.scale(step_loss).backward()
        scaler.unscale_(optimizer)
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        step_loss.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()
    scheduler.step()
    return float(step_loss.detach())


@contextlib.contextmanager
def keep_awake():
    """Ask Windows not to sleep while a run is going (no-op elsewhere or if the call fails)."""
    if sys.platform != "win32":
        yield
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    with contextlib.suppress(AttributeError, OSError):
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        with contextlib.suppress(AttributeError, OSError):
            ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


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
    parser.add_argument("--precision", choices=PRECISIONS, default=optim["precision"],
                        help="autocast dtype on CUDA; bf16 needs no loss scaling")
    parser.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction,
                        default=bool(optim["gradient_checkpointing"]))
    parser.add_argument("--max-grad-norm", type=float, default=float(optim["max_grad_norm"]))
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop each epoch after this many steps (timing runs only)")
    parser.add_argument("--resume", action="store_true",
                        help=f"continue from {STATE_FILE} of the last finished epoch in --out")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv), cfg


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_args(argv)
    with keep_awake():
        return _run(args, cfg)


def _run(args: argparse.Namespace, cfg: dict) -> int:
    optim, variant, lora_cfg = cfg["optim"], cfg["variants"][args.variant], cfg["lora"]
    concepts_cfg = load_config("concepts")
    concepts = training_concepts(cfg)
    device = torch.device(args.device)
    autocast_dtype, needs_scaler = resolve_precision(args.precision, device)
    if device.type == "cpu":
        autocast_dtype = None    # CPU autocast is slow and only the tests run there
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True    # fixed 224 px input
    use_amp = autocast_dtype is not None
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
    fixed_scale = variant.get("logit_scale")
    if fixed_scale is not None:
        # Stays frozen; only its value changes (the cosine ranking does not depend on it).
        with torch.no_grad():
            net.logit_scale.fill_(math.log(float(fixed_scale)))
    if args.grad_checkpointing:
        enable_gradient_checkpointing(net)
    params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=optim["weight_decay"])
    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(optim["warmup_steps"], total_steps))
    scaler = torch.amp.GradScaler("cuda") if needs_scaler else None
    text_max_length = int(optim["text_max_length"])
    lam = float(variant["repel_weight"])
    targets = variant.get("targets", "caption")
    loss_fn = LOSSES[targets]

    metrics: dict = {
        "config": {
            "variant": args.variant, "lr": args.lr, "epochs": args.epochs,
            "batch_size": args.batch_size, "subset": args.subset, "seed": args.seed,
            "repel_weight": lam, "targets": targets, "logit_scale": fixed_scale,
            "impression_fraction": impression_fraction,
            "mix": cfg["captions"]["mix"], "lora": lora_cfg, "augmentation": cfg["augmentation"],
            "weight_decay": optim["weight_decay"], "warmup_steps": optim["warmup_steps"],
            "early_stop_patience": optim["early_stop_patience"],
            "text_max_length": text_max_length, "concepts": concepts,
            "precision": args.precision, "gradient_checkpointing": bool(args.grad_checkpointing),
            "max_grad_norm": args.max_grad_norm, "max_steps": args.max_steps,
            "max_nonfinite_steps": int(optim["max_nonfinite_steps"]),
            "checkpoint": str(args.checkpoint), "swin_config": str(args.swin_config),
            "bert_config": str(args.bert_config), "cache": str(args.cache),
            "pool_dir": str(args.pool_dir),
        },
        "n_train": len(train_ds), "n_val": int(is_val.sum()), "steps_per_epoch": len(loader),
        "trainable_fraction": lora.trainable_fraction(peft_model), "cache_meta": cache_meta,
        "device": str(device), "amp": use_amp,
        "autocast_dtype": str(autocast_dtype).replace("torch.", "") if use_amp else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "library_versions": _library_versions(), "resumed_from_epoch": None,
        "git_commit": _git_commit(), "started_at": _now(),
        "steps": [], "epochs": [], "best_epoch": None, "stopped_early": False,
        "model_sha256": None,
    }
    print(f"variant {args.variant} lr {args.lr:g} device {device} "
          f"precision {metrics['autocast_dtype'] or 'fp32'} "
          f"checkpointing {bool(args.grad_checkpointing)} "
          f"train rows {len(train_ds)} val rows {int(is_val.sum())} "
          f"steps/epoch {len(loader)} trainable {metrics['trainable_fraction']:.4%}", flush=True)

    best_score, best_epoch, since_best = -math.inf, None, 0
    best_record: dict | None = None
    first_epoch = 1
    if args.resume and (out / STATE_FILE).is_file():
        state = torch.load(out / STATE_FILE, map_location="cpu", weights_only=False)
        done = int(state["epoch"])
        set_peft_model_state_dict(
            peft_model, load_file(out / f"adapter_epoch{done}" / ADAPTER_FILE))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        best_score, best_epoch = state["best_score"], state["best_epoch"]
        best_record, since_best = state["best_record"], state["since_best"]
        previous = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        # state_last.pt decides: Drive uploads asynchronously, so after a killed session
        # metrics.json can be one epoch ahead of it. That epoch is redone, and recorded once.
        metrics["steps"] = previous["steps"][:state.get("n_steps", len(previous["steps"]))]
        metrics["epochs"] = [e for e in previous["epochs"] if e["epoch"] <= done]
        metrics["started_at"], metrics["resumed_from_epoch"] = previous["started_at"], done
        metrics["best_epoch"] = best_epoch
        first_epoch = done + 1
        if state.get("stopped_early"):
            metrics["stopped_early"] = True
            first_epoch = args.epochs + 1
        print(f"resumed after epoch {done}; best epoch {best_epoch}", flush=True)
    max_nonfinite = int(optim["max_nonfinite_steps"])
    for epoch in range(first_epoch, args.epochs + 1):
        tick = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # One shuffle per (seed, epoch), so a resumed run sees the order it would have seen.
        loader.generator.manual_seed(args.seed + epoch)
        peft_model.train()
        losses: list[float] = []
        step_seconds: list[float] = []
        data_wait, skipped = 0.0, 0
        n_steps = len(loader) if args.max_steps is None else min(len(loader), args.max_steps)
        fetched = time.perf_counter()
        for step, (pixels, texts, y, codes) in enumerate(loader, 1):
            started = time.perf_counter()
            data_wait += started - fetched
            tokens = tokenizer(list(texts), padding="longest", truncation=True,
                               max_length=text_max_length, return_tensors="pt")
            value = train_step(
                net, pixels.to(device, non_blocking=True), tokens["input_ids"].to(device),
                tokens["attention_mask"].to(device), y.to(device), optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, autocast_dtype=autocast_dtype, lam=lam,
                max_grad_norm=args.max_grad_norm, loss_fn=loss_fn,
                codes=codes.to(device) if targets == "image" else None)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            fetched = time.perf_counter()
            step_seconds.append(fetched - started)
            if value is None:
                skipped += 1
                print(f"epoch {epoch} step {step}: non-finite loss, step skipped", flush=True)
                if skipped > max_nonfinite:
                    metrics["aborted"] = f"{skipped} non-finite losses in epoch {epoch}"
                    write_json_atomic(out / "metrics.json", metrics)
                    print(f"ABORTED: {metrics['aborted']}", flush=True)
                    return 1
            else:
                losses.append(value)
                metrics["steps"].append(value)
            if step % _LOG_EVERY == 0 or step == n_steps:
                last = losses[-1] if losses else float("nan")
                typical = float(np.median(step_seconds[_WARM_STEPS:] or step_seconds))
                print(f"epoch {epoch} step {step}/{n_steps} loss {last:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} {typical:.2f} s/step", flush=True)
            if step >= n_steps:
                break
        train_seconds = time.perf_counter() - tick
        vram_train = (round(torch.cuda.max_memory_allocated(device) / 2**20)
                      if device.type == "cuda" else None)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        peft_model.save_pretrained(str(out / f"adapter_epoch{epoch}"))
        val_tick = time.perf_counter()
        p5 = val.val_p5(net, tokenizer, val_images, val_statuses, concepts, concepts_cfg, device,
                        text_max_length=text_max_length, batch_size=max(args.batch_size, 64),
                        amp=use_amp)
        val_seconds = time.perf_counter() - val_tick
        vram_val = (round(torch.cuda.max_memory_allocated(device) / 2**20)
                    if device.type == "cuda" else None)
        typical = float(np.median(step_seconds[_WARM_STEPS:] or step_seconds))
        record = {
            "epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_p5_single": p5["single"], "val_p5_pair": p5["pair"],
            "val_p5_negation": p5["negation"], "val_p5_mean": p5["mean"],
            "val_text_cos": p5["text_cos"],
            "val_n_queries": {k: p5[f"n_{k}"] for k in val.KINDS},
            "seconds": round(time.perf_counter() - tick, 1),
            "train_seconds": round(train_seconds, 1), "val_seconds": round(val_seconds, 1),
            "s_per_step": round(typical, 3), "data_wait_seconds": round(data_wait, 1),
            "skipped_steps": skipped, "steps_run": len(step_seconds),
            "vram_mib": None if vram_train is None else max(vram_train, vram_val),
            "vram_train_mib": vram_train, "vram_val_mib": vram_val,
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
        stop = since_best >= int(optim["early_stop_patience"])
        metrics["stopped_early"] = stop
        write_json_atomic(out / "metrics.json", metrics)
        _save_state(out, epoch=epoch, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    best_score=best_score, best_epoch=best_epoch, best_record=best_record,
                    since_best=since_best, stopped_early=stop, n_steps=len(metrics["steps"]))
        print(f"epoch {epoch} train_loss {record['train_loss']:.4f} val_p5 "
              f"single {p5['single']:.3f} pair {p5['pair']:.3f} negation {p5['negation']:.3f} "
              f"mean {p5['mean']:.3f} text_cos {p5['text_cos']:.3f} best_epoch {best_epoch} "
              f"(train {record['train_seconds']} s, {record['s_per_step']} s/step, "
              f"data wait {record['data_wait_seconds']} s, val {record['val_seconds']} s, "
              f"vram train {vram_train} / val {vram_val} MiB)", flush=True)
        if stop:
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


def _save_state(out: Path, **state) -> None:
    """Everything a resumed run needs besides the adapter; written after the adapter and
    metrics.json, through a .part file, so it never points at an epoch that is not on disk."""
    payload = {k: v for k, v in state.items() if k not in ("optimizer", "scheduler", "scaler")}
    payload["optimizer"] = state["optimizer"].state_dict()
    payload["scheduler"] = state["scheduler"].state_dict()
    payload["scaler"] = state["scaler"].state_dict() if state["scaler"] is not None else None
    part = out / (STATE_FILE + ".part")
    torch.save(payload, part)
    part.replace(out / STATE_FILE)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


if __name__ == "__main__":
    sys.exit(main())
