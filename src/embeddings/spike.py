"""Phase 2 VLM spike: pick the sample, embed it per candidate, score, gate.

Embedding is slow on CPU, so every candidate's image and prompt embeddings are
cached with the exact inputs that produced them; scoring runs from the cache and
refuses a cache that no longer matches.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from src.data.labels import CHEXPERT_LABELS, resolve
from src.evaluation import zero_shot as zs


def select_sample(canonical_dir: Path, offset: int, n: int) -> pl.DataFrame:
    """Studies at manifest ranks [offset, offset + n), with their image paths.

    Ranks 0-199 are the radiologist-labelled valid/ studies, pinned first. They
    are the test set's only human ground truth, so a sample touching them fails.
    Past them the manifest is stratified-interleaved, so any window keeps the
    label distribution.
    """
    manifest = pl.read_parquet(canonical_dir / "sample_manifest.parquet")
    studies = pl.read_parquet(canonical_dir / "studies.parquet")
    images = pl.read_parquet(canonical_dir / "images.parquet")

    out = (
        manifest.filter(pl.col("rank_in_manifest").is_between(offset, offset + n - 1))
        .join(studies.select("study_id", "selected_image_id"), on="study_id")
        .join(
            images.select("image_id", "local_image_path", "dataset_split"),
            left_on="selected_image_id", right_on="image_id",
        )
        .sort("rank_in_manifest")
    )
    if out.height != n:
        raise ValueError(f"Wanted {n} studies from rank {offset}, found {out.height}")
    if (out["dataset_split"] == "valid").any():
        raise ValueError("Sample contains radiologist-labelled valid/ studies; raise --offset")
    return out.select("study_id", pl.col("selected_image_id").alias("image_id"),
                      "local_image_path")


def status_matrix(
    labels_long: pl.DataFrame,
    study_ids: Sequence[str],
    priority: Sequence[str],
    concepts: Sequence[str] = CHEXPERT_LABELS,
) -> np.ndarray:
    """One resolved status per (study, concept), rows in `study_ids` order.

    A study or concept no source mentions stays NOT_MENTIONED - never ABSENT.
    """
    subset = labels_long.filter(pl.col("study_id").is_in(list(study_ids)))
    wide = resolve(subset, priority).pivot(on="concept", index="study_id", values="status")
    for concept in concepts:
        if concept not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Utf8).alias(concept))
    ordered = pl.DataFrame({"study_id": list(study_ids)}).join(wide, on="study_id", how="left")
    return (
        ordered.select([pl.col(c).fill_null("NOT_MENTIONED") for c in concepts])
        .to_numpy()
        .astype(object)
    )


def build_prompts(
    concepts_cfg: dict, template: str, concepts: Sequence[str] = CHEXPERT_LABELS
) -> list[str]:
    """One prompt per concept from its first English surface form."""
    return [template.format(concepts_cfg["concepts"][c]["en"][0]) for c in concepts]


CORPUS_SIZE = 25_000


def _peak_rss_mb() -> float:
    """Peak working set on Windows; current RSS elsewhere (no peak available)."""
    import psutil

    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 2**20


def _torch_threads() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.get_num_threads()


def embed_candidate(
    encoder,
    image_ids: Sequence[str],
    paths: Sequence[Path],
    prompts: Sequence[str],
    *,
    batch_size: int,
    cache_dir: Path,
    load_seconds: float,
) -> dict:
    """Embed every image and prompt once, timing it; cache what was produced.

    A failing batch is retried image by image so the report names the culprit.
    Any failure means no .npz: a candidate that cannot embed the whole sample
    fails the gate, and a partial matrix would only invite misuse.
    """
    text_vectors = encoder.encode_texts(list(prompts))
    per_query = []
    for prompt in prompts:
        start = time.perf_counter()
        encoder.encode_texts([prompt])
        per_query.append(time.perf_counter() - start)

    chunks, errors, timings = [], [], []
    for start_index in range(0, len(paths), batch_size):
        batch = list(paths[start_index : start_index + batch_size])
        start = time.perf_counter()
        try:
            chunks.append(encoder.encode_images(batch))
        except Exception:
            singles = []
            for path in batch:
                try:
                    singles.append(encoder.encode_images([path]))
                except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                    errors.append({"path": str(path), "error": repr(exc)})
            if singles:
                chunks.append(np.concatenate(singles))
        timings.append((time.perf_counter() - start, len(batch)))

    # The first batch pays one-off warm-up costs; leave it out when there is more.
    warm = timings[1:] or timings
    image_vectors = np.concatenate(chunks) if chunks else np.empty((0, 0), np.float32)
    meta = {
        "name": encoder.name,
        "model_id": encoder.model_id,
        "revision": encoder.revision,
        "license": encoder.license,
        "preprocessing": encoder.preprocessing,
        "image_ids": list(image_ids),
        "prompts": list(prompts),
        "dim_image": int(image_vectors.shape[1]) if image_vectors.size else None,
        "dim_text": int(text_vectors.shape[1]),
        "errors": errors,
        "costs": {
            "device": encoder.device,
            "load_seconds": load_seconds,
            "s_per_image": sum(s for s, _ in warm) / sum(n for _, n in warm),
            "text_ms_per_query": 1000 * float(np.mean(per_query)),
            "peak_rss_mb": _peak_rss_mb(),
            "weights_bytes": encoder.weights_bytes,
            "torch_threads": _torch_threads(),
            "batch_size": batch_size,
        },
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not errors:
        np.savez(cache_dir / f"{encoder.name}.npz", images=image_vectors, texts=text_vectors)
    (cache_dir / f"{encoder.name}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def _load_cache(cache_dir: Path, name: str, candidate: dict, image_ids, prompts) -> dict:
    meta = json.loads((cache_dir / f"{name}.json").read_text(encoding="utf-8"))
    checks = {
        "model_id": meta["model_id"] == candidate["model_id"],
        "revision": meta["revision"] == candidate["revision"],
        "image_ids": meta["image_ids"] == list(image_ids),
        "prompts": meta["prompts"] == list(prompts),
    }
    if bad := [field for field, ok in checks.items() if not ok]:
        raise ValueError(
            f"{name}: stale cache ({', '.join(bad)} differ); re-run without --rescore"
        )
    return meta


def score_candidates(
    cache_dir: Path,
    candidates: dict[str, dict],
    names: Sequence[str],
    image_ids: Sequence[str],
    prompts: Sequence[str],
    statuses: np.ndarray,
    concepts: Sequence[str] = CHEXPERT_LABELS,
    n_boot: int = 1000,
    seed: int = 20260101,
) -> dict:
    """Score every cached candidate against the same labels and apply the gate."""
    if len(prompts) != len(concepts):
        raise ValueError(f"{len(prompts)} prompts for {len(concepts)} concepts")
    if statuses.shape != (len(image_ids), len(concepts)):
        raise ValueError(
            f"statuses shape {statuses.shape} != ({len(image_ids)}, {len(concepts)})"
        )
    eligible = zs.eligible_concepts(statuses)
    models, score_matrices = {}, {}
    for name in names:
        meta = _load_cache(cache_dir, name, candidates[name], image_ids, prompts)
        costs = meta["costs"]
        entry = {
            "model_id": meta["model_id"],
            "revision": meta["revision"],
            "license": meta["license"],
            "preprocessing": meta["preprocessing"],
            "device": costs["device"],
            "errors": len(meta["errors"]),
            "error_samples": meta["errors"][:10],
            "dim_image": meta["dim_image"],
            "dim_text": meta["dim_text"],
            "costs": costs,
            "s_per_image": costs["s_per_image"],
            "forecast_25k_hours": costs["s_per_image"] * CORPUS_SIZE / 3600,
            "vectors_25k_bytes": CORPUS_SIZE * meta["dim_text"] * 4,
            "concepts": None,
            "macro_auroc_strict": None,
            "macro_auroc_lenient": None,
        }
        if not meta["errors"]:
            with np.load(cache_dir / f"{name}.npz") as arrays:
                scores = arrays["images"] @ arrays["texts"].T
            per = {c: zs.concept_metrics(scores[:, j], statuses[:, j])
                   for j, c in enumerate(concepts)}
            lenient = [per[c]["auroc_lenient"] for c in concepts
                       if per[c]["auroc_lenient"] is not None]
            entry |= {
                "concepts": per,
                "macro_auroc_strict": zs.macro([per[concepts[j]]["auroc_strict"]
                                                for j in eligible]),
                "macro_auroc_lenient": zs.macro(lenient),
            }
            score_matrices[name] = scores
        models[name] = entry

    bootstrap = {}
    if score_matrices and eligible:
        reps = zs.bootstrap_macro(score_matrices, statuses, eligible, n_boot=n_boot, seed=seed)
        bootstrap = {"n_boot": n_boot, "seed": seed,
                     "ci95": {n: zs.ci95(r) for n, r in reps.items()}}
        top = sorted(score_matrices, key=lambda n: -models[n]["macro_auroc_strict"])[:2]
        if len(top) == 2:
            bootstrap["top2_diff"] = {
                "models": top, "ci95": zs.ci95(reps[top[0]] - reps[top[1]]),
            }

    winner, reasons = zs.gate(models)
    pos, strict, lenient_mask = zs.label_masks(statuses)
    return {
        "concepts": list(concepts),
        "eligible_concepts": [concepts[j] for j in eligible],
        "label_counts": {
            c: {"pos": int(pos[:, j].sum()), "neg_strict": int(strict[:, j].sum()),
                "neg_lenient": int(lenient_mask[:, j].sum())}
            for j, c in enumerate(concepts)
        },
        "prompts": list(prompts),
        "models": models,
        "bootstrap": bootstrap,
        "gate": {"winner": winner, "reasons": reasons},
    }
