"""Phase 3 measurements: warm/cold latency and zero-shot quality on the full corpus.

The quality numbers reuse the Phase 2 scorer on all 25,000 studies. With 25x the
sample, concepts that were short of negatives in the spike become measurable;
this is still one model, so it is a baseline for hybrid, not a model comparison.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from src.data.labels import CHEXPERT_LABELS
from src.embeddings import embed, spike
from src.evaluation import zero_shot as zs


def bench_queries(concepts_cfg: dict, template: str) -> list[str]:
    singles = spike.build_prompts(concepts_cfg, template)
    terms = [concepts_cfg["concepts"][c]["en"][0] for c in CHEXPERT_LABELS]
    pairs = [template.format(f"{a} and {b}") for a, b in combinations(terms[1:], 2)][:16]
    return singles + pairs


def percentiles(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)),
            "n": int(arr.size)}


def bench_warm(encoder, client, collection: str, queries: Sequence[str], repeats: int,
               *, top_k: int = 10) -> dict:
    """Time the two phases of a vector search; the first repeat of each query is
    warm-up and is dropped."""
    encode, qdrant, total = [], [], []
    for query in queries:
        for r in range(repeats):
            t0 = time.perf_counter()
            vector = encoder.encode_texts([query])[0]
            t1 = time.perf_counter()
            client.query_points(collection, query=vector.tolist(), limit=top_k)
            t2 = time.perf_counter()
            if r == 0:
                continue
            encode.append((t1 - t0) * 1000)
            qdrant.append((t2 - t1) * 1000)
            total.append((t2 - t0) * 1000)
    return {"queries": len(queries), "repeats": repeats, "top_k": top_k,
            "encode_ms": percentiles(encode), "qdrant_ms": percentiles(qdrant),
            "total_ms": percentiles(total)}


def bench_http(api_url: str, queries: Sequence[str], repeats: int, *, top_k: int = 10,
               transport=None) -> dict:
    import httpx

    times = []
    with httpx.Client(base_url=api_url, timeout=30, transport=transport) as http:
        for query in queries:
            for r in range(repeats):
                t0 = time.perf_counter()
                http.post("/api/v1/search",
                          json={"query": query, "top_k": top_k}).raise_for_status()
                if r:
                    times.append((time.perf_counter() - t0) * 1000)
    return {"http_ms": percentiles(times)}


def _all_vectors(shards_dir: Path) -> tuple[np.ndarray, list[str]]:
    manifest = embed.load_shards_manifest(shards_dir)
    vectors, ids = [], []
    for entry in manifest["shards"]:
        v, i = embed.read_shard(shards_dir / entry["file"])
        vectors.append(v)
        ids.extend(i)
    return np.concatenate(vectors), ids


def score_corpus(shards_dir: Path, encoder, labels_long: pl.DataFrame, images: pl.DataFrame,
                 prompts: Sequence[str], priority: Sequence[str], *, n_boot: int = 1000) -> dict:
    vectors, image_ids = _all_vectors(shards_dir)
    study_of = dict(zip(images["image_id"], images["study_id"], strict=True))
    studies = [study_of[i] for i in image_ids]
    statuses = spike.status_matrix(labels_long, studies, priority)
    scores = vectors @ encoder.encode_texts(list(prompts)).T

    eligible = zs.eligible_concepts(statuses)
    per = {c: zs.concept_metrics(scores[:, j], statuses[:, j])
           for j, c in enumerate(CHEXPERT_LABELS)}
    strict = [per[CHEXPERT_LABELS[j]]["auroc_strict"] for j in eligible]
    ci = None
    if eligible:
        reps = zs.bootstrap_macro({"model": scores}, statuses, eligible, n_boot=n_boot)["model"]
        ci = zs.ci95(reps)
    pos, neg_strict, neg_lenient = zs.label_masks(statuses)
    return {
        "n_images": len(studies),
        "n_studies": len(set(studies)),
        "prompts": list(prompts),
        "eligible_concepts": [CHEXPERT_LABELS[j] for j in eligible],
        "concepts": per,
        "macro_auroc_strict": zs.macro(strict),
        "macro_auroc_lenient_same_set": zs.macro(
            [per[CHEXPERT_LABELS[j]]["auroc_lenient"] for j in eligible
             if per[CHEXPERT_LABELS[j]]["auroc_lenient"] is not None]),
        "ci95": ci,
        "n_boot": n_boot,
        "label_counts": {
            c: {"pos": int(pos[:, j].sum()), "neg_strict": int(neg_strict[:, j].sum()),
                "neg_lenient": int(neg_lenient[:, j].sum())}
            for j, c in enumerate(CHEXPERT_LABELS)},
    }


def _read_report(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON; move it aside and re-run") from exc


def check_report_writable(path: Path) -> None:
    """Fail before the expensive work if an existing report cannot be merged into."""
    _read_report(path)


def merge_report(path: Path, section: str, payload: dict, *, git_commit: str | None,
                 library_versions: dict) -> dict:
    """bench-vector and score-corpus share one report file; each owns one key."""
    report = _read_report(path)
    report[section] = payload
    report["git_commit"] = git_commit
    report["library_versions"] = library_versions
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".json.part")
    part.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    part.replace(path)
    return report
