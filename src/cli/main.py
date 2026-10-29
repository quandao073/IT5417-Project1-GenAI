"""Worker CLI. Each command maps to one step of the plan."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from src.common.config import Settings, default_config_dir, load_config
from src.data import canonical, probe_pool
from src.data.audit import CorpusMode, run_audit
from src.data.labels import CHEXPERT_LABELS
from src.embeddings import embed, index, spike, validate
from src.embeddings.probe import patient_split, save_probe
from src.evaluation import parser_benchmark as pb
from src.evaluation import parser_eval as pe
from src.evaluation import vector_baseline as vb
from src.kg import chexpert_candidates, umls_filter, vocab
from src.query.schema import ParseError
from src.retrieval import image_scores as iscores
from src.retrieval import label_filter as lf
from src.retrieval import report_index
from src.retrieval.templates import render


def _cmd_audit(args: argparse.Namespace) -> int:
    report = run_audit(Path(args.plus_csv), Path(args.small_root))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"plus rows      : {report['plus_total']:,}")
    print(f"small images   : {report['small_total']:,}")
    print(f"best strategy  : {report['best_strategy']}")
    print(f"join rate      : {report['join_rate']:.4f}")
    print(f"decision       : {report['decision']}")
    print(f"report written : {out}")

    # The gate must stop the pipeline, not merely warn.
    if report["decision"] == CorpusMode.ABORT.value:
        print(
            "\nGATE FAILED (join rate below 70%). Do not download images. "
            "Inspect unmatched_samples in the report, and consider selective "
            "download from PNG_train on Redivis instead.",
            file=sys.stderr,
        )
        return 1
    return 0


CHEXBERT_FILES = {
    "CHEXBERT_FINDINGS": "findings_fixed.json",
    "CHEXBERT_IMPRESSION": "impression_fixed.json",
    "CHEXBERT_REPORT": "report_fixed.json",
}


def _cmd_build_corpus(args: argparse.Namespace) -> int:
    chexbert_dir = Path(args.chexbert_dir)
    chexbert = {source: chexbert_dir / name for source, name in CHEXBERT_FILES.items()}
    if missing := sorted(str(p) for p in chexbert.values() if not p.is_file()):
        print(f"CheXbert label files not found: {missing}", file=sys.stderr)
        return 1

    build = canonical.build_corpus(
        plus_csv=Path(args.plus_csv),
        small_root=Path(args.small_root),
        chexbert=chexbert,
        limit=args.limit,
        seed=args.seed,
    )

    counts = build.report["counts"]
    # A short corpus is a silent failure: the tables would look fine while the
    # evaluation silently ran on less data than it claims.
    if counts["selected_studies"] < args.limit:
        print(
            f"Only {counts['selected_studies']:,} eligible studies, "
            f"{args.limit:,} requested. Lower --limit or check the eligibility filters.",
            file=sys.stderr,
        )
        return 1

    written = canonical.write_corpus(build, Path(args.out))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(build.report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"plus rows        : {counts['plus_rows']:,}")
    print(f"eligible rows    : {counts['eligible_rows']:,}")
    print(f"eligible studies : {counts['eligible_studies']:,}")
    print(f"selected studies : {counts['selected_studies']:,}")
    print(f"selected patients: {counts['selected_patients']:,}")
    print(f"projection mix   : {build.report['projection_mix']}")
    for path in written:
        print(f"wrote            : {path}")
    print(f"report written   : {report_path}")
    return 0


def _cmd_build_probe_pool(args: argparse.Namespace) -> int:
    chexbert_dir = Path(args.chexbert_dir)
    chexbert = {source: chexbert_dir / name for source, name in CHEXBERT_FILES.items()}
    if missing := sorted(str(p) for p in chexbert.values() if not p.is_file()):
        print(f"CheXbert label files not found: {missing}", file=sys.stderr)
        return 1

    corpus_path = Path(args.canonical_dir) / "images.parquet"
    priority = load_config("retrieval")["label_resolution"]["source_priority"]
    try:
        images, resolved, meta = probe_pool.build_pool(
            Path(args.plus_csv), Path(args.small_root), chexbert,
            pl.read_parquet(corpus_path), priority,
            corpus_images_path=corpus_path, git_commit=_git_commit(),
        )
    except probe_pool.PoolError as exc:
        print(f"POOL FAILED: {exc}", file=sys.stderr)
        return 1

    for path in probe_pool.write_pool(images, resolved, meta, Path(args.out)):
        print(f"wrote      : {path}")
    print(f"studies : {meta['studies']:,}  patients: {meta['patients']:,}  "
          f"pool_sha256: {meta['pool_sha256'][:12]}")
    return 0


SPIKE_MODELS = ["biomedclip", "chexzero", "xrayclip"]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() or None


def _cmd_spike_models(args: argparse.Namespace) -> int:
    encoder_cfg = load_config("models")["image_encoder"]
    candidates = encoder_cfg["candidates"]
    priority = load_config("retrieval")["label_resolution"]["source_priority"]
    prompts = spike.build_prompts(load_config("concepts"), encoder_cfg["text_prompt_template"])

    canonical_path = Path(args.canonical_dir)
    sample = spike.select_sample(canonical_path, args.offset, args.n)
    image_ids = sample["image_id"].to_list()
    cache_dir = Path(args.cache_dir)

    if not args.rescore:
        from src.embeddings.base import load_encoder

        if len(args.models) > 1:
            print("warning: peak RSS is a per-process peak; it is only per-model when "
                  "run one model per process")
        for name in args.models:
            start = time.perf_counter()
            encoder = load_encoder(name, candidates[name], args.device)
            load_seconds = time.perf_counter() - start
            meta = spike.embed_candidate(
                encoder, image_ids, [Path(p) for p in sample["local_image_path"]], prompts,
                batch_size=args.batch_size, cache_dir=cache_dir, load_seconds=load_seconds,
            )
            print(f"{name:<11}: dim {meta['dim_image']}/{meta['dim_text']}, "
                  f"{meta['costs']['s_per_image']:.3f} s/image, {len(meta['errors'])} errors")

    statuses = spike.status_matrix(
        pl.read_parquet(canonical_path / "labels.parquet"), sample["study_id"].to_list(), priority
    )
    report = spike.score_candidates(
        cache_dir, candidates, args.models, image_ids, prompts, statuses
    )
    report["sample"] = {
        "offset": args.offset, "n": args.n,
        "first_study": sample["study_id"][0], "last_study": sample["study_id"][-1],
    }
    report["git_commit"] = _git_commit()
    from importlib.metadata import PackageNotFoundError, version

    def _version(dist: str) -> str | None:
        try:
            return version(dist)
        except PackageNotFoundError:
            return None

    report["library_versions"] = {
        dist: _version(dist)
        for dist in ["torch", "torchvision", "open_clip_torch", "transformers", "numpy"]
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for name, model in report["models"].items():
        auc = model["macro_auroc_strict"]
        print(f"{name:<11}: macro strict AUROC "
              f"{'n/a' if auc is None else f'{auc:.3f}'}, "
              f"25k forecast {model['forecast_25k_hours']:.1f} h")
    for reason in report["gate"]["reasons"]:
        print(f"gate: {reason}")
    print(f"report written : {out}")
    return 0 if report["gate"]["winner"] else 1


def _library_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for dist in ("torch", "torchvision", "open_clip_torch", "transformers", "numpy",
                 "qdrant-client"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = None
    return out


def _selected_encoder(device: str):
    """The pinned encoder from configs/models.yaml, or a clear error."""
    from src.embeddings.base import load_encoder

    cfg = load_config("models")["image_encoder"]
    name = cfg["selected"]
    if not name:
        raise SystemExit("configs/models.yaml: image_encoder.selected is not set")
    encoder = load_encoder(name, cfg["candidates"][name], device)
    if encoder.revision != cfg["revision"]:
        raise SystemExit(
            f"encoder revision {encoder.revision} != models.yaml revision {cfg['revision']}"
        )
    return name, encoder


def _cmd_embed(args: argparse.Namespace) -> int:
    images = pl.read_parquet(
        Path(args.images) if args.images else Path(args.canonical_dir) / "images.parquet"
    )
    name, encoder = _selected_encoder(args.device)
    build_id = args.build_id or embed.make_build_id(name, date.today())
    out_dir = Path(args.out) / build_id
    print(f"build_id   : {build_id}")
    print(f"images     : {images.height:,}  shards of {args.shard_size}  -> {out_dir}")
    try:
        manifest = embed.run_embed(
            images, encoder, Path(args.image_root), out_dir, build_id,
            shard_size=args.shard_size, batch_size=args.batch_size,
            library_versions=_library_versions(), git_commit=_git_commit(),
        )
    except RuntimeError as exc:
        print(f"EMBED FAILED: {exc}", file=sys.stderr)
        return 1
    spi = manifest["s_per_image"]
    print(f"embedded   : {manifest['embedded']:,}  skipped: {manifest['skipped']:,}  "
          f"dim: {manifest['dim']}  "
          f"{'n/a' if spi is None else f'{spi:.3f}'} s/image")
    print(f"manifest   : {out_dir / embed.SHARDS_FILE}")
    return 0


def _resolve_build_dir(embeddings_dir: Path, build_id: str | None) -> Path:
    if build_id:
        return embeddings_dir / build_id
    builds: list[Path] = []
    if embeddings_dir.is_dir():
        builds = sorted(p for p in embeddings_dir.iterdir() if (p / embed.SHARDS_FILE).is_file())
    if len(builds) != 1:
        raise SystemExit(
            f"found {len(builds)} finished builds under {embeddings_dir}; pass --build-id"
        )
    return builds[0]


def _qdrant_client(url: str | None):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url or Settings().qdrant_url, timeout=60)


def _cmd_index(args: argparse.Namespace) -> int:
    from src.common.config import config_hash, index_config_scope

    shards_dir = _resolve_build_dir(Path(args.embeddings), args.build_id)
    images = pl.read_parquet(Path(args.canonical_dir) / "images.parquet")
    data_cfg = load_config("data")["sources"]
    dataset = {"images": data_cfg["images"]["name"],
               "plus_csv": Path(data_cfg["plus_metadata"]["csv"]).name}
    client = _qdrant_client(args.qdrant_url)
    try:
        encoder = index.encoder_of(embed.load_shards_manifest(shards_dir),
                                   load_config("models")["image_encoder"]["candidates"])
        scope = index_config_scope(encoder)
        manifest = index.run_index(
            client, shards_dir, images, Settings().qdrant_collection_prefix,
            recreate=args.recreate, config_hash=config_hash(scope), git_commit=_git_commit(),
            dataset=dataset, library_versions=_library_versions(), encoder=encoder,
            config_scope=scope,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"INDEX FAILED: {exc}", file=sys.stderr)
        return 1
    out = index.manifest_path(Path(args.indexes_dir), manifest["build_id"])
    index.write_manifest(out, manifest)
    print(f"collection : {manifest['collection']}  encoder {manifest['encoder']}")
    print(f"vectors    : {manifest['counts']['vectors']:,}  "
          f"dim {manifest['embedding_model']['dimension']}")
    print(f"manifest   : {out}")
    return 0


def _add_build_args(parser: argparse.ArgumentParser) -> None:
    """Which registered build a command reads; both unset: the selected encoder's build."""
    parser.add_argument("--build-id", default=None, help="a registered build")
    parser.add_argument("--encoder", default=None,
                        help="shorthand for this encoder's only registered build; with "
                             "--build-id, it must be the encoder that embedded that build")


def _registered_build(args: argparse.Namespace, indexes_dir: Path) -> dict | None:
    """The build `--build-id` / `--encoder` name, else the selected encoder's; None after
    printing why there is none."""
    encoder = args.encoder or (
        None if args.build_id else load_config("models")["image_encoder"]["selected"])
    try:
        return index.resolve_build(indexes_dir, build_id=args.build_id, encoder=encoder)
    except LookupError as exc:
        print(f"FAIL manifest    : {exc}", file=sys.stderr)
        return None


def _build_encoder(manifest: dict, device: str = "cpu"):
    """The encoder that embedded `manifest`'s build (index.build_encoder); None after printing
    why it cannot be used."""
    try:
        return index.build_encoder(
            manifest, load_config("models")["image_encoder"]["candidates"], device)
    except ValueError as exc:
        print(f"ENCODER MISMATCH: {exc}", file=sys.stderr)
        return None


def _cmd_validate_indexes(args: argparse.Namespace) -> int:
    manifest = _registered_build(args, Path(args.indexes_dir))
    if manifest is None:
        return 1
    shards_dir = Path(args.embeddings) / manifest["build_id"]
    images = pl.read_parquet(Path(args.canonical_dir) / "images.parquet")
    candidates = load_config("models")["image_encoder"]["candidates"]
    expected_revision = candidates[manifest["encoder"]]["revision"]
    # Hash over the scope this manifest recorded, not today's default, so an
    # index stays checkable across a change of scope (see D16).
    checks = validate.validate_indexes(
        _qdrant_client(args.qdrant_url), manifest, shards_dir, images,
        expected_config_hash=validate.expected_config_hash(manifest),
        expected_revision=expected_revision,
    )
    index_path = (Path(args.report_index) if args.report_index
                  else Path(args.indexes_dir) / "report_search.sqlite")
    checks.append(report_index.validate_report_index(
        index_path, pl.read_parquet(Path(args.canonical_dir) / "reports.parquet")))
    for c in checks:
        print(f"{'PASS' if c.ok else 'FAIL'} {c.name:<12}: {c.detail}")
    return 0 if all(c.ok for c in checks) else 1


def _cmd_build_report_index(args: argparse.Namespace) -> int:
    reports = pl.read_parquet(Path(args.canonical_dir) / "reports.parquet")
    concepts_bytes = (default_config_dir() / "concepts.yaml").read_bytes()
    out = Path(args.out)
    meta = report_index.build_report_index(
        reports, out, concepts_sha256=hashlib.sha256(concepts_bytes).hexdigest(),
        git_commit=_git_commit())
    print(f"rows={reports.height} sha={meta['corpus_sha256'][:12]} out={out}")
    print(json.dumps(meta, ensure_ascii=True))
    return 0


def _merge_json(path: Path, section: str, payload: dict) -> None:
    vb.merge_report(path, section, payload, git_commit=_git_commit(),
                    library_versions=_library_versions())


def _report_is_corrupt(path: Path) -> bool:
    try:
        vb.check_report_writable(path)
    except ValueError as exc:
        print(f"REPORT FILE CORRUPT: {path} ({exc})", file=sys.stderr)
        return True
    return False


def _cmd_bench_vector(args: argparse.Namespace) -> int:
    from src.api.dependencies import build_state

    if _report_is_corrupt(Path(args.out)):
        return 1
    cfg = load_config("models")["image_encoder"]
    queries = vb.bench_queries(
        load_config("concepts"), cfg["text_prompt_template"])[: args.queries]

    t0 = time.perf_counter()
    state = build_state(Settings())
    if not state.ready:
        print(f"API state not ready: {state.not_ready_reason}", file=sys.stderr)
        return 1
    first = state.encoder.encode_texts([queries[0]])[0]
    state.client.query_points(state.collection, query=first.tolist(), limit=10)
    cold_ms = (time.perf_counter() - t0) * 1000

    warm = vb.bench_warm(state.encoder, state.client, state.collection, queries, args.repeats)
    report = {"build_id": state.manifest["build_id"], "cold_ms": round(cold_ms, 1),
              "warm": warm, "torch_threads": spike._torch_threads()}
    if args.api_url:
        report["http"] = vb.bench_http(args.api_url, queries, args.repeats)
    _merge_json(Path(args.out), "latency", report)

    p95 = warm["total_ms"]["p95"]
    print(f"cold       : {cold_ms:.0f} ms (load + first query)")
    print(f"warm p50   : encode {warm['encode_ms']['p50']:.0f}  "
          f"qdrant {warm['qdrant_ms']['p50']:.0f}  total {warm['total_ms']['p50']:.0f} ms")
    print(f"warm p95   : total {p95:.0f} ms  -> {'PASS' if p95 < 1000 else 'FAIL'} (< 1000)")
    if args.api_url:
        print(f"http p95   : {report['http']['http_ms']['p95']:.0f} ms")
    print(f"report     : {args.out}")
    return 0 if p95 < 1000 else 1


def _cmd_score_corpus(args: argparse.Namespace) -> int:
    if _report_is_corrupt(Path(args.out)):
        return 1
    manifest = _registered_build(args, Path(args.indexes_dir))
    if manifest is None:
        return 1
    encoder = _build_encoder(manifest)
    if encoder is None:
        return 1
    shards_dir = Path(args.embeddings) / manifest["build_id"]
    canonical_dir = Path(args.canonical_dir)
    images = pl.read_parquet(canonical_dir / "images.parquet")
    labels = pl.read_parquet(canonical_dir / "labels.parquet")
    cfg = load_config("models")["image_encoder"]
    prompts = spike.build_prompts(load_config("concepts"), cfg["text_prompt_template"])
    priority = load_config("retrieval")["label_resolution"]["source_priority"]
    report = vb.score_corpus(shards_dir, encoder, labels, images, prompts, priority,
                             n_boot=args.n_boot)
    report["build_id"] = embed.load_shards_manifest(shards_dir)["build_id"]
    _merge_json(Path(args.out), "quality", report)
    macro = report["macro_auroc_strict"]
    print(f"images     : {report['n_images']:,}  studies: {report['n_studies']:,}  "
          f"eligible concepts: {len(report['eligible_concepts'])}")
    ci = report["ci95"]
    print(f"macro AUROC: strict {'n/a' if macro is None else f'{macro:.3f}'} "
          f"CI95 {'n/a' if ci is None else [round(x, 3) for x in ci]}")
    for c, v in report["concepts"].items():
        a = v["auroc_strict"]
        print(f"  {c:<28} {'  n/a' if a is None else f'{a:.3f}'}  pos={v['n_pos']:<5} "
              f"neg={v['n_neg_strict']}")
    print(f"report     : {args.out}")
    return 0


def _build_parser_or_exit(need_llm: bool):
    from src.api.dependencies import build_query_parser

    parser, status = build_query_parser(Settings())
    if need_llm and status != "ok":
        print(f"LLM is {status} (model {Settings().llm_model} at {Settings().llm_base_url}); "
              f"start Ollama and pull the model, or drop llm/auto from --parsers",
              file=sys.stderr)
        raise SystemExit(1)
    return parser, status


def _load_pool(args: argparse.Namespace, expect: dict, corpus_images: pl.DataFrame):
    """The pool's vectors, statuses and patient ids in shard order, or an error message.

    `expect` is the corpus build's encoder (`model_id`, `revision`, `dim`). Refuses a pool
    embedded by another encoder, one that shares a patient with the corpus and one that
    holds a valid/ study (D12).
    """
    pool_dir = Path(args.pool_dir)
    images = pl.read_parquet(pool_dir / probe_pool.POOL_IMAGES)
    resolved = pl.read_parquet(pool_dir / probe_pool.POOL_LABELS)
    meta = json.loads((pool_dir / probe_pool.POOL_META).read_text(encoding="utf-8"))
    shards_dir = pool_dir / "embeddings" / args.pool_build_id
    shards = embed.load_shards_manifest(shards_dir)
    got = (shards["model_id"], shards["revision"], shards["dim"])
    want = (expect["model_id"], expect["revision"], expect["dim"])
    if got != want:
        return f"POOL BUILD MISMATCH: pool {got} != corpus {want}", None
    overlap = set(images["patient_id"]) & set(corpus_images["patient_id"])
    if overlap:
        return f"POOL OVERLAPS CORPUS: {len(overlap)} patients", None
    if images.filter(pl.col("path_to_image").str.starts_with("valid/")).height:
        return "POOL CONTAINS valid/ STUDIES", None

    vectors, image_ids = vb._all_vectors(shards_dir)
    study_of = dict(zip(images["image_id"], images["study_id"], strict=True))
    patient_of = dict(zip(images["image_id"], images["patient_id"], strict=True))
    studies = [study_of[i] for i in image_ids]
    statuses = probe_pool.resolved_status_matrix(resolved, studies)
    return None, (vectors, statuses, [patient_of[i] for i in image_ids], meta)


def _corpus_rows(args: argparse.Namespace, build_id: str, studies: pl.DataFrame):
    """Corpus vectors of each study's selected image, in `studies` order."""
    vectors, image_ids = vb._all_vectors(Path(args.embeddings) / build_id)
    row_of = {i: r for r, i in enumerate(image_ids)}
    rows = [row_of[i] for i in studies["selected_image_id"]]
    return vectors[rows]


def _cmd_build_image_scores(args: argparse.Namespace) -> int:
    """The V2 score matrix (src/retrieval/image_scores.py): each label's caption cosine,
    calibrated on pool-val, the patient split of models.yaml:image_probe. Never on pool-train,
    which the fine-tune learned from, nor on valid/ (D12)."""
    report = Path(args.report or f"artifacts/metrics/image_scores_{args.encoder}.json")
    if _report_is_corrupt(report):
        return 1
    # The corpus shards' own manifest, so a build not yet registered in Qdrant can be scored.
    shards = embed.load_shards_manifest(Path(args.embeddings) / args.corpus_build_id)
    build_id = shards["build_id"]
    expect = {"model_id": shards["model_id"], "revision": shards["revision"],
              "dim": shards["dim"]}
    canonical_dir = Path(args.canonical_dir)
    error, pool = _load_pool(args, expect, pl.read_parquet(canonical_dir / "images.parquet"))
    if error:
        print(error, file=sys.stderr)
        return 1
    # The captions go through the build's encoder gate (D40): another revision's text vectors
    # would score these image vectors without an error.
    encoder = _build_encoder({"build_id": build_id, "encoder": args.encoder,
                              "embedding_model": {"revision": expect["revision"]}})
    if encoder is None:
        return 1
    X, pool_statuses, pool_patients, pool_meta = pool
    split = load_config("models")["image_probe"]
    val = patient_split(pool_patients, val_fraction=split["val_fraction"], seed=split["seed"])
    concepts_cfg = load_config("concepts")
    prompts = {c: render([c], [], concepts_cfg) for c in CHEXPERT_LABELS}
    text = encoder.encode_texts(list(prompts.values())).astype("float64")
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    probe, fit = iscores.calibrate(text, X[val], pool_statuses[val])

    studies = pl.read_parquet(canonical_dir / "studies.parquet")
    scores = iscores.corpus_scores(probe, _corpus_rows(args, build_id, studies), studies)
    meta = {
        "build_id": build_id,
        "encoder": {"name": args.encoder, **expect},
        "pool_sha256": pool_meta["pool_sha256"], "pool_build_id": args.pool_build_id,
        "n_val": int(val.sum()), "prompts": prompts, **fit,
        "git_commit": _git_commit(),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out = Path(args.out or f"data/indexes/image_scores/{build_id}")
    save_probe(out, probe, meta, scores)
    _merge_json(report, "image_scores", meta)

    for label, c in fit["calibration"].items():
        auroc = "  n/a" if c["val_auroc"] is None else f"{c['val_auroc']:.3f}"
        print(f"{label:27s} a {c['a']:8.3f}  b {c['b']:8.3f}  pool-val AUROC {auroc}  "
              f"(pos {c['n_pos']}, neg {c['n_neg']})")
    if fit["uncalibrated"]:
        print(f"uncalibrated (< {iscores.MIN_CLASS} per class): {', '.join(fit['uncalibrated'])}")
    if fit["saturated"]:
        print(f"WARNING saturated (|logit| > {iscores.MAX_LOGIT:g} on pool-val, the caption "
              f"separates its slice; that term dominates a sum over labels): "
              f"{', '.join(fit['saturated'])}")
    print(f"matrix : {out}  ({scores.height:,} studies x {len(CHEXPERT_LABELS)} labels, "
          f"build {build_id}, pool-val {int(val.sum()):,} images)")
    print(f"report : {report}")
    return 0


def _cmd_parse(args: argparse.Namespace) -> int:
    parser, status = _build_parser_or_exit(args.parser == "llm")
    result = parser.parse(args.text, mode=args.parser)
    payload = {**result.query.model_dump(), "parser": result.parser,
               "fallback_used": result.fallback_used, "warnings": list(result.warnings),
               "llm_ms": result.llm_ms, "cached": result.cached, "llm_status": status}
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


def _cmd_eval_parser(args: argparse.Namespace) -> int:
    if _report_is_corrupt(Path(args.out)):
        return 1
    rows = pb.load_benchmark(Path(args.benchmark))
    if args.split != "all":
        rows = [r for r in rows if r.split == args.split]
    if not rows:
        print(f"no rows for split {args.split!r}", file=sys.stderr)
        return 1
    need_llm = any(m in ("llm", "auto") for m in args.parsers)
    parser, status = _build_parser_or_exit(need_llm)
    if need_llm and status == "ok":
        # The first call after a server start can exceed the timeout (WORKLOG T16);
        # pay it here so it is not scored against the first benchmark row.
        with contextlib.suppress(ParseError):
            parser.parse("chest x-ray", mode="llm")
    thresholds = load_config("evaluation")["thresholds"]
    report = pe.evaluate_benchmark(rows, parser, args.parsers, thresholds)
    report["benchmark"] = {"file": args.benchmark, "split": args.split,
                           "sha256": hashlib.sha256(Path(args.benchmark).read_bytes()).hexdigest()}
    report["llm"] = {"status": status, "provider": Settings().llm_provider,
                     "model": Settings().llm_model,
                     "ollama_version": parser.llm.version() if parser.llm else None,
                     "prompt_sha256": parser.prompt_sha256}
    _merge_json(Path(args.out), "eval", report)

    for mode, entry in report["parsers"].items():
        m = entry["metrics"]
        print(f"{mode:<5} n={m['n']:<4} valid={m['schema_valid_rate']:.3f} "
              f"concept={_f(m['concept_macro_f1'])} status={_f(m['status_macro_f1'])} "
              f"view={_f(m['view_macro_f1'])} exact={m['exact_match_rate']:.3f} "
              f"p50={entry['latency_ms']['p50']:.0f}ms p95={entry['latency_ms']['p95']:.0f}ms "
              f"fallback={entry['fallback_rate']:.3f}")
        if args.verbose:
            for miss in entry["mismatches"]:
                print("  " + json.dumps(miss, ensure_ascii=True))
    v = report["verdict"]
    print(f"verdict ({report['judged_parser']}, split {args.split}): "
          f"{'PASS' if v['passed'] else 'FAIL'} {json.dumps(v)}")

    if args.label_filter:
        canonical_dir = Path(args.canonical_dir)
        studies = pl.read_parquet(canonical_dir / "studies.parquet")
        cfg = load_config("retrieval")
        matrix = lf.build_label_matrix(
            pl.read_parquet(canonical_dir / "labels.parquet"), studies["study_id"].to_list(),
            cfg["label_resolution"]["source_priority"], seed=cfg["results"]["label_order_seed"])
        counts = pe.label_filter_counts(rows, matrix)
        _merge_json(Path(args.out), "label_filter", {**counts, "benchmark": report["benchmark"]})
        for group, pairs in counts["summary"].items():
            cells = "  ".join(f"{pair}: med {s['median']:.0f} zero {s['zero']}"
                              for pair, s in pairs.items())
            print(f"label {group:<11} {cells}")
    print(f"report     : {args.out}")
    return 0 if v["passed"] else 1


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}ms"


def _print_eval_table(report: dict, verbose: bool) -> None:
    for parser_name, e in report["parsers"].items():
        m, lat = e["metrics"], e["latency_ms"] or {}
        line = (f"{parser_name:<5} n={m['n']:<4} ndcg_bin={_f(m['ndcg_binary@10'])} "
                f"(graded {_f(m['ndcg_graded@10'])}) p@10={_f(m['p@10'])} mrr={_f(m['mrr'])} "
                f"viol={_f(m['exclude_violation@10'])} ret={_f(m['returned@10'])} "
                f"p50={_ms(lat.get('p50'))} p95={_ms(lat.get('p95'))}")
        if e["fallback_rate"] is not None:
            line += f" fallback={e['fallback_rate']:.3f}"
        print(line)
        if verbose:
            for qid, values in e["per_query"].items():
                print("  " + json.dumps({"id": qid, **values}, ensure_ascii=True))


def _cmd_eval_retrieval(args: argparse.Namespace) -> int:
    from src.api.dependencies import build_state
    from src.evaluation import retrieval_eval as rev

    out = Path(args.out)
    if _report_is_corrupt(out):
        return 1
    rows = pb.load_benchmark(Path(args.benchmark))
    if args.split != "all":
        rows = [r for r in rows if r.split == args.split]
    if not rows:
        print(f"no rows for split {args.split!r}", file=sys.stderr)
        return 1
    need_llm = "auto" in args.parsers
    parser, status = _build_parser_or_exit(need_llm)
    if need_llm and status == "ok":
        with contextlib.suppress(ParseError):      # pay the cold start outside the scored rows
            parser.parse("chest x-ray", mode="llm")
    # The flags win over .env: the canonical dir, so the state and the agreement matrix agree,
    # and --build-id / --encoder, whose build the state then serves instead of BUILD_ID's.
    overrides = {"canonical_dir": Path(args.canonical_dir)}
    if args.build_id or args.encoder:
        manifest = _registered_build(args, Settings().indexes_dir)
        if manifest is None:
            return 1
        overrides["build_id"] = manifest["build_id"]
    state = build_state(Settings(**overrides))
    if not state.ready:
        print(f"API state not ready: {state.not_ready_reason}", file=sys.stderr)
        return 1
    if state.report_status != "ok":
        print(f"report index disabled: {state.report_reason}; run cxr build-report-index",
              file=sys.stderr)
        return 1
    # The labels are the answer key only: the search never sees them (the state has none).
    canonical_dir = Path(args.canonical_dir)
    studies = pl.read_parquet(canonical_dir / "studies.parquet")
    labels_long = pl.read_parquet(canonical_dir / "labels.parquet")
    cfg = load_config("retrieval")
    matrix = lf.build_label_matrix(
        labels_long, studies["study_id"].to_list(), cfg["label_resolution"]["source_priority"],
        seed=cfg["results"]["label_order_seed"])
    shared = {"state": state, "parser": parser, "matrix": matrix,
              "agreement": rev.build_agreement_matrix(labels_long, matrix.study_ids),
              "projection_of": {s: state.images[i].projection for s, i in
                                zip(studies["study_id"], studies["selected_image_id"],
                                    strict=True)}}
    digest = hashlib.sha256(Path(args.benchmark).read_bytes()).hexdigest()
    policies = [rev.DEFAULT_POLICY] + ([rev.NEGATION_POLICY]
                                       if args.policies == "negation_both" else [])
    report = rev.evaluate_retrieval(rows, parsers=args.parsers, policies=policies, **shared)
    report.update(benchmark={"file": args.benchmark, "split": args.split, "sha256": digest},
                  build_id=state.manifest["build_id"], encoder=state.manifest["encoder"],
                  report_index=state.report_meta)
    if "auto" in args.parsers:
        report["llm"] = {"status": status, "provider": Settings().llm_provider,
                         "model": Settings().llm_model, "prompt_sha256": parser.prompt_sha256}
    _merge_json(out, "eval", report)

    _print_eval_table(report, args.verbose)
    print(f"report     : {args.out}")
    return 0


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _cmd_kg_filter_umls(args: argparse.Namespace) -> int:
    cfg = load_config("kg")["umls"]
    paths = umls_filter.UmlsPaths()   # UMLS_ZIP, UMLS_META_DIR
    out = Path(args.out_dir) if args.out_dir else Path("data/umls") / cfg["version"]
    try:
        manifest = umls_filter.filter_umls(paths.umls_meta_dir, paths.umls_zip, out, cfg)
    except umls_filter.UmlsFilterError as exc:
        print(f"kg-filter-umls: {exc}", file=sys.stderr)
        return 1
    print(f"UMLS {manifest['umls_version']}, zip md5 {manifest['zip_md5']}")
    for name, entry in manifest["files"].items():
        print(f"{name:8s} {entry['rows_before']:>12,} -> {entry['rows_after']:>12,} rows"
              f"  {entry['bytes'] / 2**20:9.1f} MiB")
    print(f"written: {out}")
    return 0


def _cmd_kg_pool_vocab(args: argparse.Namespace) -> int:
    cfg = load_config("kg")["vocab"]
    pool_dir = Path(args.pool_dir)
    pool = pl.read_parquet(pool_dir / probe_pool.POOL_IMAGES)
    meta = json.loads((pool_dir / probe_pool.POOL_META).read_text(encoding="utf-8"))
    try:
        sections = vocab.pool_sections(pool, Path(args.plus_csv))
    except vocab.VocabError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1
    found, stats = vocab.read_mentions(sections, Path(args.radgraph_dir))
    names = vocab.mrconso_names(Path(args.umls_dir) / "mrconso.parquet")
    found = vocab.apply_rules(found, cfg, names)
    report = {"pool_sha256": meta["pool_sha256"], "pool_studies": pool.height, "radgraph": stats,
              **vocab.summarize(found, cfg)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    found.write_parquet(out / "mentions.parquet")
    vocab.vocab_table(found, "head").write_parquet(out / "vocab_head.parquet")
    vocab.vocab_table(found, "joined").write_parquet(out / "vocab_joined.parquet")
    (out / "coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for section, s in stats.items():
        print(f"radgraph {section:10s} {s['matched']:>7,} / {s['texts']:>7,} pool texts matched")
    for name, c in report["curves"].items():
        levels = "  ".join(f"{float(k):.0%} {v:>6,}" for k, v in c["levels"].items())
        print(f"{name:22s} {c['strings']:>7,} strings  {c['mentions']:>9,} mentions  {levels}")
    for kind, d in report["dropped_heads"].items():
        print(f"heads on the drop lists, {kind}: {d['mentions']:,} of {d['of']:,} mentions")
    gate = report["gate"]
    print(f"gate: {gate['kind']} {gate['coverage']:.0%} of mentions in {gate['strings']} strings "
          f"(max {gate['max_strings']}): {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"written: {out}")
    impression = stats["impression"]
    if impression["matched"] < 0.99 * impression["texts"]:
        print("STOP: under 99% of pool impressions found their RadGraph record", file=sys.stderr)
        return 1
    return 0 if gate["passed"] else 1


def _cmd_kg_chexpert_candidates(args: argparse.Namespace) -> int:
    concepts = load_config("concepts")
    umls = Path(args.umls_dir)
    try:
        found = chexpert_candidates.candidates(concepts, umls / "mrconso.parquet",
                                               umls / "mrsty.parquet")
    except chexpert_candidates.CandidateError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    found.write_csv(out)
    # Label names and project ids only: the candidate names are UMLS content (T38).
    for label in concepts["concepts"]:
        rows = found.filter(pl.col("label") == label)
        kinds = rows["kind"].to_list()
        proposed = next(iter(rows["proposed_id"].drop_nulls().to_list()), "-")
        print(f"{label:27s} exact {kinds.count('cui')}  is_a {kinds.count('is_a')}  "
              f"proposed id {proposed}")
    print(f"written: {out} (holds UMLS names: review it there, then write "
          f"configs/kg/chexpert_cuis.yaml with codes only)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cxr", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Phase 1: measure join rate, decide corpus")
    audit.add_argument("--plus-csv", default="data/raw/chexpert-plus/df_chexpert_plus_240401.csv")
    audit.add_argument("--small-root", default="data/raw/CheXpert-v1.0-small")
    audit.add_argument("--out", default="artifacts/audit/data_audit.json")
    audit.set_defaults(func=_cmd_audit)

    corpus = sub.add_parser("build-corpus", help="Phase 1: build the canonical Parquet tables")
    corpus.add_argument("--plus-csv", default="data/raw/chexpert-plus/df_chexpert_plus_240401.csv")
    corpus.add_argument("--small-root", default="data/raw/CheXpert-v1.0-small")
    corpus.add_argument("--chexbert-dir", default="data/raw/chexpert-plus/labels")
    corpus.add_argument("--limit", type=int, default=25000)
    corpus.add_argument("--seed", type=int, default=20260101)
    corpus.add_argument("--out", default="data/canonical")
    corpus.add_argument("--report", default="artifacts/audit/corpus_report.json")
    corpus.set_defaults(func=_cmd_build_corpus)

    pool = sub.add_parser("build-probe-pool",
                          help="build the out-of-corpus study pool for the image probes")
    pool.add_argument("--plus-csv", default="data/raw/chexpert-plus/df_chexpert_plus_240401.csv")
    pool.add_argument("--small-root", default="data/raw/CheXpert-v1.0-small")
    pool.add_argument("--chexbert-dir", default="data/raw/chexpert-plus/labels")
    pool.add_argument("--canonical-dir", default="data/canonical")
    pool.add_argument("--out", default="data/probe_pool")
    pool.set_defaults(func=_cmd_build_probe_pool)

    spk = sub.add_parser("spike-models", help="Phase 2: embed a sample per VLM, apply the gate")
    spk.add_argument("--models", nargs="+", choices=SPIKE_MODELS, default=SPIKE_MODELS)
    spk.add_argument("--canonical-dir", default="data/canonical")
    # Ranks 0-199 are the radiologist-labelled studies; never tune on them.
    spk.add_argument("--offset", type=int, default=200)
    spk.add_argument("--n", type=int, default=1000)
    spk.add_argument("--batch-size", type=int, default=16)
    spk.add_argument("--cache-dir", default="data/cache/model_spike")
    spk.add_argument("--out", default="artifacts/metrics/model_spike.json")
    spk.add_argument("--device", default="cpu")
    spk.add_argument("--rescore", action="store_true",
                     help="score from the cache only; embed nothing")
    spk.set_defaults(func=_cmd_spike_models)

    emb = sub.add_parser("embed", help="Phase 3: embed the corpus into shard files (resumable)")
    emb.add_argument("--canonical-dir", default="data/canonical")
    emb.add_argument("--images", default=None,
                     help="images parquet to embed instead of <canonical-dir>/images.parquet")
    emb.add_argument("--image-root", default="data/raw/CheXpert-v1.0-small")
    emb.add_argument("--out", default="data/indexes/embeddings")
    emb.add_argument("--shard-size", type=int, default=1000)
    emb.add_argument("--batch-size", type=int, default=32)
    emb.add_argument("--device", default="cpu")
    emb.add_argument("--build-id", default=None,
                     help="override the generated <YYYYMMDD>_chexpert25k_<model>_v1")
    emb.set_defaults(func=_cmd_embed)

    idx = sub.add_parser("index", help="Phase 3: upsert shards into Qdrant, register the build")
    idx.add_argument("--embeddings", default="data/indexes/embeddings")
    idx.add_argument("--build-id", default=None, help="required when several builds exist")
    idx.add_argument("--canonical-dir", default="data/canonical")
    idx.add_argument("--indexes-dir", default="data/indexes")
    idx.add_argument("--qdrant-url", default=None, help="default: QDRANT_URL / Settings")
    idx.add_argument("--recreate", action="store_true")
    idx.set_defaults(func=_cmd_index)

    val = sub.add_parser("validate-indexes", help="Phase 3: cross-check manifest, shards, Qdrant")
    val.add_argument("--indexes-dir", default="data/indexes")
    _add_build_args(val)
    val.add_argument("--embeddings", default="data/indexes/embeddings")
    val.add_argument("--canonical-dir", default="data/canonical")
    val.add_argument("--qdrant-url", default=None)
    val.add_argument("--report-index", default=None,
                     help="default: <indexes-dir>/report_search.sqlite")
    val.set_defaults(func=_cmd_validate_indexes)

    rep = sub.add_parser("build-report-index", help="Phase 5: build the SQLite FTS5 report index")
    rep.add_argument("--canonical-dir", default="data/canonical")
    rep.add_argument("--out", default="data/indexes/report_search.sqlite")
    rep.set_defaults(func=_cmd_build_report_index)

    bench = sub.add_parser("bench-vector", help="Phase 3: cold/warm latency of vector search")
    bench.add_argument("--queries", type=int, default=30)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--api-url", default=None, help="also time POST /api/v1/search over HTTP")
    bench.add_argument("--out", default="artifacts/metrics/vector_baseline.json")
    bench.set_defaults(func=_cmd_bench_vector)

    score = sub.add_parser("score-corpus",
                           help="Phase 3: zero-shot AUROC on all embedded studies")
    score.add_argument("--embeddings", default="data/indexes/embeddings")
    score.add_argument("--indexes-dir", default="data/indexes")
    _add_build_args(score)
    score.add_argument("--canonical-dir", default="data/canonical")
    score.add_argument("--n-boot", type=int, default=1000)
    score.add_argument("--out", default="artifacts/metrics/vector_baseline.json")
    score.set_defaults(func=_cmd_score_corpus)

    bis = sub.add_parser("build-image-scores",
                         help="V2-5: the score matrix, each label's caption cosine calibrated "
                              "on pool-val")
    bis.add_argument("--encoder", required=True, help="models.yaml candidate of the build")
    bis.add_argument("--corpus-build-id", required=True)
    bis.add_argument("--pool-build-id", required=True)
    bis.add_argument("--pool-dir", default="data/probe_pool")
    bis.add_argument("--embeddings", default="data/indexes/embeddings")
    bis.add_argument("--canonical-dir", default="data/canonical")
    bis.add_argument("--out", default=None,
                     help="default: data/indexes/image_scores/<corpus build id>")
    bis.add_argument("--report", default=None,
                     help="default: artifacts/metrics/image_scores_<encoder>.json")
    bis.set_defaults(func=_cmd_build_image_scores)

    prs = sub.add_parser("parse", help="Phase 4: parse one request, print the SearchQuery")
    prs.add_argument("text")
    prs.add_argument("--parser", choices=["auto", "rule", "llm"], default="auto")
    prs.set_defaults(func=_cmd_parse)

    ev = sub.add_parser("eval-parser", help="Phase 4: score the parsers on the benchmark")
    ev.add_argument("--benchmark", default="benchmarks/parser_queries_v1.jsonl")
    ev.add_argument("--split", choices=["heldout", "dev", "all"], default="heldout")
    ev.add_argument("--parsers", nargs="+", choices=["rule", "llm", "auto"],
                    default=["rule", "llm", "auto"])
    ev.add_argument("--out", default="artifacts/metrics/parser_eval.json")
    ev.add_argument("--label-filter", action="store_true",
                    help="also count matching studies per gold parse under every policy pair")
    ev.add_argument("--canonical-dir", default="data/canonical")
    ev.add_argument("--verbose", action="store_true", help="print every mismatch (ASCII)")
    ev.set_defaults(func=_cmd_eval_parser)

    er = sub.add_parser("eval-retrieval",
                        help="score the one search path against the benchmark gold")
    er.add_argument("--benchmark", default="benchmarks/parser_queries_v1.jsonl")
    er.add_argument("--split", choices=["heldout", "dev", "all"], default="heldout")
    er.add_argument("--parsers", nargs="+", choices=["gold", "rule", "auto"],
                    default=["gold", "rule", "auto"])
    er.add_argument("--policies", choices=["default", "negation_both"], default="negation_both")
    _add_build_args(er)   # both unset: BUILD_ID, else the selected encoder's build
    er.add_argument("--canonical-dir", default="data/canonical")
    er.add_argument("--out", default="artifacts/metrics/retrieval_eval.json")
    er.add_argument("--verbose", action="store_true", help="print every per-query row (ASCII)")
    er.set_defaults(func=_cmd_eval_retrieval)

    kgu = sub.add_parser("kg-filter-umls",
                         help="V2-1: filter the UMLS RRF files (UMLS_META_DIR) into Parquet")
    kgu.add_argument("--out-dir", default=None, help="default: data/umls/<umls.version>")
    kgu.set_defaults(func=_cmd_kg_filter_umls)

    kgv = sub.add_parser("kg-pool-vocab",
                         help="V2-2: RadGraph vocabulary of the pool, coverage curves and the gate")
    kgv.add_argument("--pool-dir", default="data/probe_pool")
    kgv.add_argument("--plus-csv", default="data/raw/chexpert-plus/df_chexpert_plus_240401.csv")
    kgv.add_argument("--radgraph-dir", default="data/raw/chexpert-plus/radgraph_xl")
    kgv.add_argument("--umls-dir", default="data/umls/2026AA")
    kgv.add_argument("--out", default="data/kg/vocab")
    kgv.set_defaults(func=_cmd_kg_pool_vocab)

    kgc = sub.add_parser("kg-chexpert-candidates",
                         help="V2-2: up to 3 CUI candidates per CheXpert label, for review")
    kgc.add_argument("--umls-dir", default="data/umls/2026AA")
    kgc.add_argument("--out", default="data/kg/vocab/chexpert_candidates.csv")
    kgc.set_defaults(func=_cmd_kg_chexpert_candidates)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
