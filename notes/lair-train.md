# WP3 + WP4 (`training/`): pool cache, captions, LoRA fine-tune

Built on `feat/lair-train` (merged `7c94d63`, branch deleted 2026-10-24). Everything lives under `training/` and runs in its own uv venv
(`training/pyproject.toml`, lock `training/uv.lock`): torch `2.4.1+cu124`, torchvision
`0.19.1+cu124`, peft 0.14, the main package editable from `..`. The main venv stays `+cpu`
(WORKLOG T14, D34). No compat shim: `training/model.py` imports `src.embeddings.cxrclip`
directly (WP2 merged into this branch at `0d3f9c7`, WP1 at `405635d`); `training/_cxrclip_compat.py` does not exist.

Verified on this machine: `uv run --project training python -c "import torch; print(torch.__version__,
torch.cuda.is_available())"` -> `2.4.1+cu124 True` (RTX 2000 Ada 8 GB);
`uv run python -c "import torch; print(torch.__version__)"` -> `2.4.1+cpu`.

## Modules

| File | Public names |
|---|---|
| `captions.py` | `CaptionSampler(statuses, concepts, concepts_cfg, *, mix, impressions=None, impression_fraction=0.0, seed)`, `.eligible() -> bool[N]`, `.sample(row) -> (text, y int8[L])`, `.sample_kind(row) -> single/pair/negation/impression` (kind drawn, before fallback), `.reseed(seed)`, `.image_codes(row) -> int8[L]`; `label_vector(statuses_row, concepts) -> int8[L]`, `status_codes(statuses_row, concepts) -> int8[L]` (`CODE_PRESENT` 1 / `CODE_ABSENT` -1 / `CODE_UNCERTAIN` 2, NOT_MENTIONED 0) |
| `data.py` | `preprocess_uint8(img, size=224) -> uint8[size,size]`, `eval_transform()`, `train_transform(aug_cfg, size=224)`, `load_cache(dir) -> (images memmap, index DataFrame, cache.json)`, `PoolDataset(cache_dir, sampler, *, train, augment_cfg)` (items: pixels, caption, caption `y`, image codes), `reseed_worker` (DataLoader `worker_init_fn`) |
| `cache.py` | `build_cache(pool_dir, image_root, plus_csv, out_dir, *, size=224, val_fraction, seed) -> dict`; CLI `--pool-dir --image-root --plus-csv --out --size --val-fraction --seed` (defaults from `training.yaml`) |
| `loss.py` | `label_aware_loss(zi, zt, y, logit_scale, lam=0.3)` (roadmap code verbatim, FT1-FT3), `image_targets(y, codes) -> (relevant, conflict)` and `image_label_loss(..., *, codes)` (FT6, D36), `clip_loss(zi, zt, logit_scale)` |
| `model.py` | `load_base(checkpoint, swin_cfg_dir, bert_cfg_dir) -> (CXRClipModule in eval mode, tokenizer, ckpt_config)`, `resolve_config_dir(spec)` (local dir or `<repo>@<sha>` via the HF cache) |
| `lora.py` | `apply_lora(module, *, rank, alpha, dropout, targets=("query","key","value")) -> PeftModel`, `unwrap(peft_model) -> CXRClipModule`, `trainable_fraction(model)`, `export_merged(peft_model, ckpt_config, out, meta) -> sha256` |
| `val.py` | `val_p5(model, tokenizer, images_uint8, statuses, concepts, concepts_cfg, device, *, text_max_length=64, batch_size=256, amp=False, min_relevant=5) -> dict` (P@5 per kind + `text_cos`, the collapse monitor), `p_at_5_by_kind(scores, queries, statuses, *, min_relevant)`, `embed_images`, `encode_texts` |
| `train.py` | CLI, see below; `warmup_cosine(warmup_steps, total_steps)`; `LOSSES` by `variants.<v>.targets`; `variants.<v>.logit_scale` fixes the temperature |
| `zero_shot_val.py` | pool-val P@5 of the un-tuned checkpoint (the line to beat; Colab must reproduce it) |
| `colab/` | `pack.py` (Drive bundle + sha256 manifest), `colab_run.py` (setup / check-env / train / gate / summary), `train_colab.ipynb`, `README.md` |

`val.py` computes only P@5 (no bootstrap, no other k) because it runs after every epoch.
Relevance is `template_bench.relevant_mask(statuses, query, DEFAULT_POLICY)` (WP1, merged at
`405635d`), i.e. the `filter_studies_by_labels` truth table under `exclude_present` /
`EXCLUDE`; queries come from `templates.generate(concepts, ("single", "pair", "negation"))`,
the S12 kinds; queries with fewer than five relevant images are skipped; the mean is over
kinds that still have a scored query. To use `template_bench.evaluate` instead, replace the
body of `val_p5` (the loop reads `single/pair/negation/mean` and `n_<kind>` from its dict).

## Running on the real data (main tree, after merge)

Prerequisites: `data/probe_pool/{pool_images,pool_labels}.parquet` (probe branch),
`data/cache/models/cxrclip/swint_m.pt` (WP2 `convert_checkpoint`), network once for the two HF
config snapshots pinned in `models.yaml` (`swin_config`, `text_config`), Docker not needed.

```bash
# 1. image cache (CPU, one pass over 88,853 JPEGs; writes data/train_cache/pool_224/{images.npy,index.parquet,cache.json})
uv run --project training python training/cache.py --pool-dir data/probe_pool \
    --image-root data/raw/CheXpert-v1.0-small \
    --plus-csv data/raw/chexpert-plus/df_chexpert_plus_240401.csv
#    images.npy is N x 224 x 224 uint8 = ~4.5 GB; expect 10-20 min (cold disk I/O, WORKLOG Phase 1).
#    It raises if any pool study lacks an impression in the Plus CSV.

# 2. smoke run (Task 24 of the plan): 5,000 rows, 1 epoch, timing of the first 100 steps printed
uv run --project training python training/train.py --variant FT1 --lr 3e-5 --epochs 1 \
    --subset 5000 --out runs/smoke_ft1 --num-workers 4

# 3. full runs: on Colab (D36), training/colab/README.md. The FT6 lr sweep (D33 grid), then FT1
#    (the paper's recipe, reference row) at the lr FT6 chose; later variants build on FT6.
```

Defaults come from `configs/training.yaml` (batch 128, 10 epochs, AdamW wd 0.01, warmup 200,
bf16 autocast, patience 3, text max length 64, LoRA 12/24/0.1 on q/k/v) and from
`configs/models.yaml:image_encoder.candidates.cxrclip` (`--checkpoint`, `--swin-config`,
`--bert-config`). `--device` defaults to `cuda` when available. `--precision` (bf16 default, fp16, fp32) and `--grad-checkpointing` come from `training.yaml:optim`; autocast is off on CPU. `--resume` continues from `state_last.pt` after the last finished epoch; `--max-steps` caps an epoch for timing runs.
`--seed` defaults to `training.yaml:split.seed` (20260101) and drives torch, the caption
sampler, the subset and the loader order.

Outputs under `--out`: `metrics.json` (rewritten after every epoch: `steps` losses,
`epochs[]` with `train_loss`, `val_p5_single/pair/negation/mean`, `val_text_cos`, `val_n_queries`, `seconds`,
`vram_mib`; `best_epoch`, `stopped_early`, `trainable_fraction`, `n_train`, `n_val`,
`cache_meta`, `library_versions`, `git_commit`, `model_sha256`), `adapter_epoch{k}/` (peft),
`model.pt` (merged export of the best epoch, `{"model", "config", "finetune"}` with the
original key names) and `model.sha256`. To serve it, add a candidate next to `cxrclip` in
`models.yaml` with `weights: runs/<name>/model.pt` and `revision: sha256:<model.sha256>`
(WP6/WP5).

## Measured (B13c, B14; WORKLOG *Sửa B14*)

Trainable parameters: about 1.0 M of 136.6 M (0.71 %). Laptop RTX 2000 Ada 8 GB, batch 128, bf16 +
gradient checkpointing: 4,616 MiB train / 2,327 MiB val, 1.10-1.13 s/step (B13c smoke: 1.33), a full
epoch (618 steps + pool-val) about 12.6 min, about 2.1 h per 10-epoch run. Without checkpointing, fp32
peaked at 8,661 MiB (B13 smoke). The fp16 + GradScaler path (Colab's T4) ran clean on the laptop GPU.
Full runs moved to Colab (D36): the laptop ran hot.

Trap: both venvs' site-packages contain another top-level package named `training`
(shipped by open_clip_torch 2.24.0; its `data.py` imports `braceexpand`, which is not
installed). `python training/train.py` puts `training/` itself, not the repo root, first on
`sys.path`, so `import training` resolved to open_clip's and crashed. `train.py` and
`cache.py` insert the repo root at `sys.path[0]` before their imports; pytest (root
`pythonpath = ["."]`) and `python -m training.train` from the root are unaffected.

Windows notes: DataLoader workers are spawned processes; `PoolDataset` reopens the memmap in
each worker rather than pickling it (`__getstate__`), captions are reseeded per worker and
epoch (`reseed_worker`). Four workers is a reasonable start. The console is cp1252: all
prints are ASCII.

## Resume

`--resume` continues from `state_last.pt` (optimizer, scheduler, scaler, best epoch, patience)
after the last finished epoch; the Colab driver always passes it. What survives a crash: `metrics.json` after every finished epoch
and `adapter_epoch{k}/` for every finished epoch. To export a specific epoch by hand:

```python
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from training.lora import apply_lora, export_merged
from training.model import load_base, resolve_config_dir
m, tok, cfg = load_base("data/cache/models/cxrclip/swint_m.pt", resolve_config_dir(SWIN), resolve_config_dir(BERT))
p = apply_lora(m, rank=12, alpha=24, dropout=0.1)
set_peft_model_state_dict(p, load_file("runs/ft1_lr3e-5/adapter_epoch4/adapter_model.safetensors"))
export_merged(p, cfg, "runs/ft1_lr3e-5/model.pt", {"variant": "FT1", "best_epoch": 4})
```

## Decisions and deviations (for the main session)

- `PoolDataset(cache_dir, sampler, *, train, augment_cfg)` has no `statuses` argument: the
  sampler already holds them and the dataset checks the row count against the cache.
  `train` selects the split (pool-train / pool-val); `augment_cfg=None` means the eval
  transform. Both splits are filtered by `sampler.eligible()`; the val P@5 path reads the
  memmap directly and does not use `PoolDataset`.
- `CaptionSampler.sample_kind` returns the kind drawn, before the fallback to `single`
  (pair with one PRESENT, negation with no ABSENT). A study whose only PRESENT training
  concept is No Finding always gets the single caption `no finding` (never paired or
  negated, D32); when a pathology is PRESENT as well, No Finding is never mentioned and
  `y[No Finding] = 0` (Review Focus 4).
- `RandomResizedCrop` uses `ratio=(1, 1)`: only the scale is jittered, as the config says.
  Rotation is bilinear, fill 0. Order: crop, rotate, brightness/contrast, normalise.
- `logit_scale` and both projection heads stay frozen (LoRA only); the loss uses
  `logit_scale.exp().clamp(max=100)`. The checkpoint's value is 24.16; FT6 sets it to 5
  (`training.yaml:variants.FT6.logit_scale`, D36), FT1-FT3 keep 24.16. The loss is computed in fp32 outside autocast.
- `load_base` returns the model in eval mode (dropout and drop-path off); the loop calls
  `.train()` per epoch and `val.py` restores the mode it found.
- Checkpoint selection falls back to `-train_loss` only when pool-val cannot score a single
  query (tiny smoke fixtures); on the real pool-val it is always P@5 mean (D33).
- `cache.json` also records `val_fraction` and `seed`; `train.py` trusts the cache's `is_val`
  and does not re-check it against `training.yaml`.
- `training/uv.lock` is committed so the GPU venv is reproducible. The training venv resolved
  polars 2.0.0 (main venv 1.44); the shared `src` code runs on both (main suite 959 passed
  after the WP1 merge, training suite 30 passed in the training venv).
- `training/tests` need peft, so they run only in the training venv:
  `uv run --project training pytest training/tests -q`. The root `pyproject.toml` is untouched;
  `training/pyproject.toml` deliberately has no `[tool.pytest.ini_options]` so pytest picks
  up the root one (`pythonpath = ["."]`, markers).
