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
import yaml

from src.common.config import Settings, default_config_dir, load_config
from src.data import canonical, probe_pool
from src.data.audit import CorpusMode, run_audit
from src.data.labels import CHEXPERT_LABELS
from src.embeddings import embed, index, spike, validate
from src.embeddings.probe import Probe, patient_split, save_probe
from src.evaluation import parser_benchmark as pb
from src.evaluation import parser_eval as pe
from src.evaluation import vector_baseline as vb
from src.kg import (
    chexpert_candidates,
    chexpert_phrases,
    graph,
    graph_llm,
    link_gate,
    linker,
    llm_select,
    neo4j_load,
    quality,
    sapbert,
    umls_filter,
    vocab,
)
from src.query.schema import ParseError
from src.retrieval import image_scores as iscores
from src.retrieval import image_scores_ext as iext
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
    if args.kg:
        checks += _kg_checks(args, manifest)
    for c in checks:
        print(f"{'PASS' if c.ok else 'FAIL'} {c.name:<12}: {c.detail}")
    return 0 if all(c.ok for c in checks) else 1


def _kg_checks(args: argparse.Namespace, manifest: dict) -> list:
    """The four KG checks of the design, read from the loaded Neo4j (src/kg/neo4j_load.py)."""
    matrix = Path(args.kg_matrix)
    meta = json.loads((matrix / "probe.json").read_text(encoding="utf-8"))
    try:
        with _neo4j_driver(Settings()) as driver, driver.session() as session:
            build = neo4j_load.read_build(session)
            if build is None:
                return [validate.Check("kg_loaded", False, "no KGBuild node: run kg-load-neo4j")]
            return neo4j_load.validate(
                session, {"top_n": int(build["top_n"])},
                [c for c in pl.read_parquet(matrix / "corpus_scores.parquet").columns
                 if c not in ("study_id", "image_id")],
                _label_ids(), meta["build_id"], manifest["build_id"])
    except Exception as exc:  # noqa: BLE001 - Neo4j down is a FAIL line, not a traceback
        return [validate.Check("kg_loaded", False, f"neo4j: {type(exc).__name__}: {exc}")]


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


def _kg_parser(prompt_file: str):
    """The KG parser (terms and sides, ambiguous forms left out of the concept list) with the
    given prompt file, for `eval-parser --kg-prompt`."""
    from src.query import llm_parser as llm
    from src.query.parser import QueryParser
    from src.query.rule_parser import RuleParser

    models, concepts = load_config("models"), load_config("concepts")
    key = Settings().anthropic_api_key
    if key is None:
        raise SystemExit("--kg-prompt needs ANTHROPIC_API_KEY")
    model = models["kg_llm"]["parser"]["model"]
    client = llm.AnthropicClient(model, 20, api_key=key.get_secret_value(),
                                 options=llm.request_options(models["llm"], model))
    system = llm.load_prompt(prompt_file, concepts,
                             tuple(load_config("kg")["query"]["ambiguous_forms"]))
    template = models["image_encoder"]["text_prompt_template"]
    return QueryParser(RuleParser(concepts), template=template, llm_client=client,
                       system_prompt=system, with_terms=True), "ok"


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


def _label_ids() -> dict[str, str]:
    mapping = yaml.safe_load((default_config_dir() / "kg" / "chexpert_cuis.yaml")
                             .read_text(encoding="utf-8"))["labels"]
    return {label: entry["id"] for label, entry in mapping.items()}


def _cmd_kg_quality(args: argparse.Namespace) -> int:
    """`quality` of every scorable concept (src/kg/quality.py, D55): the 14 label columns against
    the resolved CheXpert statuses of pool-val, the extended columns as already measured."""
    ecfg = load_config("kg")["scores_ext"]
    matrix = Path(args.matrix)
    meta = json.loads((matrix / "probe.json").read_text(encoding="utf-8"))
    args.pool_build_id = meta["pool_build_id"]
    expect = {k: meta["encoder"][k] for k in ("model_id", "revision", "dim")}
    error, pool = _load_pool(args, expect, pl.read_parquet(Path(args.canonical_dir)
                                                           / "images.parquet"))
    if error:
        print(error, file=sys.stderr)
        return 1
    X, statuses, patients, _ = pool
    split = load_config("models")["image_probe"]
    val = patient_split(patients, val_fraction=split["val_fraction"], seed=split["seed"])
    probe = np.load(matrix / "probe.npz")
    logits = X[val].astype(np.float64) @ probe["W"].T.astype(np.float64) + probe["b"]
    base = quality.base_columns(logits, statuses[val], list(CHEXPERT_LABELS),
                                n_boot=ecfg["bootstrap"]["n"], seed=ecfg["bootstrap"]["seed"])
    table = quality.table(base, _label_ids(), meta["columns"]["extended"], ecfg["quality"])
    out = Path(args.out)
    out.write_text(json.dumps(table, indent=1), encoding="utf-8")
    weak = [c for c, v in table.items() if v["weak"]]
    print(f"{len(table)} concepts, weak (< {ecfg['quality']}): {weak}; written: {out}")
    for label, cui in _label_ids().items():
        v = table[cui]
        print(f"  {label:<28} {v['auroc_lenient']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]  "
              f"present {v['n_present']}")
    return 0


def _neo4j_driver(settings: Settings):
    from neo4j import GraphDatabase

    return GraphDatabase.driver(settings.neo4j_uri, auth=(
        settings.neo4j_user, settings.neo4j_password.get_secret_value()))


def _cmd_kg_load_neo4j(args: argparse.Namespace) -> int:
    """Load the KG tables and the top-N PREDICTED edges into Neo4j (src/kg/neo4j_load.py)."""
    ncfg = load_config("kg")["neo4j"]
    graph_dir, matrix = Path(args.graph_dir), Path(args.matrix)
    top_n = args.top_n or ncfg["top_n"]
    nodes = pl.read_parquet(graph_dir / "nodes.parquet")
    quality_table = json.loads((graph_dir / "quality.json").read_text(encoding="utf-8"))
    scores = pl.read_parquet(matrix / "corpus_scores.parquet")
    images = pl.read_parquet(Path(args.canonical_dir) / "images.parquet")
    kg_id, matrix_version = neo4j_load.build_ids(graph_dir, matrix)
    scorable = nodes.filter(pl.col("scorable"))["cui"].to_list()
    columns = {cui: quality_table[cui]["column"] for cui in scorable}
    started = time.perf_counter()
    with _neo4j_driver(Settings()) as driver, driver.session() as session:
        manifest = neo4j_load.load_all(
            session, nodes=nodes, quality=quality_table,
            types=pl.read_parquet(graph_dir / "semantic_types.parquet"),
            aliases=pl.read_parquet(graph_dir / "aliases.parquet"),
            edges=pl.read_parquet(graph_dir / "edges.parquet"), scores=scores, images=images,
            concept_columns=columns, top_n=top_n, batch=ncfg["batch"],
            build={"kg_build_id": kg_id, "matrix_version": matrix_version})
    manifest |= {"umls": load_config("kg")["umls"]["version"],
                 "graph_manifest_sha256": hashlib.sha256(
                     (graph_dir / "manifest.json").read_bytes()).hexdigest(),
                 "loaded_seconds": round(time.perf_counter() - started, 1),
                 "git_commit": _git_commit()}
    (graph_dir / "neo4j_load.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"kg_build_id {kg_id}; matrix {matrix_version}; top_n {top_n}")
    print(f"nodes {manifest['nodes']}")
    print(f"edges {manifest['edges']}")
    print(f"loaded in {manifest['loaded_seconds']} s; {graph_dir / 'neo4j_load.json'}")
    return 0


def _cmd_kg_bench_scoring(args: argparse.Namespace) -> int:
    """Time `score_images` + top-k on a random matrix of --rows images (src/kg/scoring.py, N4)."""
    from src.kg import scoring

    out = scoring.benchmark(args.rows, [f"c{i}" for i in range(args.columns)], k=args.k,
                            repeats=args.repeats)
    print(json.dumps(out, indent=1))
    return 0


def _cmd_kg_query(args: argparse.Namespace) -> int:
    """Run questions through the KG search with the real tables, matrix, Neo4j and language model
    (no API, no Qdrant): one JSON record per question with the trace, the explanation and the
    time of every step. Questions: arguments, or --file with one per line (`#` starts a comment)."""
    import time

    from src.api.routes.kg import explanation_event
    from src.kg import runtime

    sys.stdout.reconfigure(encoding="utf-8")        # Vietnamese on a cp1252 console
    settings = Settings()
    matrix = Path(args.matrix)
    meta = json.loads((matrix / "probe.json").read_text(encoding="utf-8"))
    images = pl.read_parquet(Path(settings.canonical_dir) / "images.parquet",
                             columns=["image_id", "projection"])
    search, notes = runtime.build_kg_search(
        settings.model_copy(update={"kg_matrix_dir": matrix}), {"build_id": meta["build_id"]},
        images)
    questions = list(args.questions)
    if args.file:
        questions += [ln.strip() for ln in Path(args.file).read_text(encoding="utf-8")
                      .splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    print(json.dumps({"notes": notes, "versions": search.versions}, ensure_ascii=False),
          file=sys.stderr)
    out = open(args.out, "w", encoding="utf-8") if args.out else None  # noqa: SIM115
    for question in questions:
        result = search.search(question, top_k=args.top_k)
        started = time.perf_counter()
        explanation = None if args.no_explain else search.explain(result)
        record = {
            "question": question, "request_id": result.request_id, "language": result.language,
            "parsed": {"include": [c.model_dump() for c in result.query.include],
                       "exclude": result.query.exclude, "views": result.query.views,
                       "terms": [t.model_dump() for t in result.query.terms]},
            "intents": [i.to_dict() for i in result.intents], "warnings": result.warnings_text(),
            "degraded": result.degraded, "trace": result.trace,
            "results": [{k: r[k] for k in ("rank", "image_id", "view", "score")}
                        for r in result.rows],
            "timings_ms": result.timings,
            "explanation": explanation_event(result, explanation),
            "explain_wall_ms": round((time.perf_counter() - started) * 1000, 1)}
        line = json.dumps(record, ensure_ascii=False)
        if out:
            out.write(line + "\n")
        print(f"{result.timings['total_ms']:7.0f} ms  {question}")
    if out:
        out.close()
    return 0


def _cmd_build_image_scores_ext(args: argparse.Namespace) -> int:
    """Extra score-matrix columns (src/retrieval/image_scores_ext.py, D62): candidates from the KG
    core, silver labels from RadGraph on pool-val, the calibration of the 14 columns, the D55
    gate. Writes a new matrix version: the 14 columns unchanged plus the columns that pass."""
    cfg = load_config("kg")
    ecfg = cfg["scores_ext"]
    base_dir = Path(args.base)
    base_meta = json.loads((base_dir / "probe.json").read_text(encoding="utf-8"))
    build_id = base_meta["build_id"]
    graph_dir, link_dir = Path(args.graph_dir), Path(args.link_dir)
    nodes = pl.read_parquet(graph_dir / "nodes.parquet")
    edges = pl.read_parquet(graph_dir / "edges.parquet")
    links = pl.read_parquet(link_dir / "links.parquet")
    pairs = pl.read_parquet(link_dir / "pair_links.parquet")
    strings = iext.pool_strings(links, pairs)
    below = iext.descendants(edges, nodes["cui"].to_list())
    table = iext.candidates(nodes, strings, below, ecfg)
    Path(args.candidates).parent.mkdir(parents=True, exist_ok=True)
    picked = table.filter(pl.col("status") == "candidate")

    canonical_dir = Path(args.canonical_dir)
    args.pool_build_id = base_meta["pool_build_id"]
    expect = {k: base_meta["encoder"][k] for k in ("model_id", "revision", "dim")}
    error, pool = _load_pool(args, expect, pl.read_parquet(canonical_dir / "images.parquet"))
    if error:
        print(error, file=sys.stderr)
        return 1
    encoder = _build_encoder({"build_id": build_id, "encoder": base_meta["encoder"]["name"],
                              "embedding_model": {"revision": expect["revision"]}})
    if encoder is None:
        return 1
    X, _, pool_patients, pool_meta = pool
    _, image_ids = vb._all_vectors(Path(args.pool_dir) / "embeddings" / args.pool_build_id)
    images = pl.read_parquet(Path(args.pool_dir) / probe_pool.POOL_IMAGES)
    study_of = dict(zip(images["image_id"], images["study_id"], strict=True))
    split = load_config("models")["image_probe"]
    val = patient_split(pool_patients, val_fraction=split["val_fraction"], seed=split["seed"])
    val_studies = [study_of[i] for i, v in zip(image_ids, val, strict=True) if v]

    sections = vocab.pool_sections(images, Path(args.plus_csv))
    sections = sections.filter(pl.col("study_id").is_in(val_studies))
    found, _ = vocab.read_mentions(sections, Path(args.radgraph_dir))
    mentions = vocab.apply_rules(found, cfg["vocab"],
                                 vocab.mrconso_names(Path(args.umls_dir) / "mrconso.parquet"))
    mcuis = graph.mention_cuis(mentions, links, pairs)
    concepts = {c: below.get(c, {c}) for c in picked["cui"]}
    statuses = iext.silver_statuses(mcuis, val_studies, concepts)

    captions = {c: cap.split(" | ") for c, cap in picked.select("cui", "captions").iter_rows()}
    text = []
    for cui in concepts:
        vectors = encoder.encode_texts(captions[cui]).astype("float64")
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        mean = vectors.mean(axis=0)
        text.append(mean / np.linalg.norm(mean))
    probe, fit = iscores.calibrate(np.asarray(text), X[val], statuses, concepts=tuple(concepts))
    logits = probe.logits(X[val])
    results = {}
    for j, cui in enumerate(concepts):
        res = iext.lenient_auroc(logits[:, j], statuses[:, j], n_boot=ecfg["bootstrap"]["n"],
                                 seed=ecfg["bootstrap"]["seed"])
        if res["n_present"] < ecfg["min_present"]:
            res["decision"] = f"dropped: {res['n_present']} PRESENT < {ecfg['min_present']}"
        elif not res["auroc_lenient"] >= ecfg["quality"]:
            res["decision"] = f"dropped: lenient AUROC < {ecfg['quality']}"
        else:
            res["decision"] = "kept"
        # kept but borderline when the interval's lower end is under the gate (D62)
        res["borderline"] = res["decision"] == "kept" and res["ci95"][0] < ecfg["quality"]
        results[cui] = {**res, "calibration": fit["calibration"][cui], "captions": captions[cui]}
    kept = [c for c in concepts if results[c]["decision"] == "kept"]

    result_rows = {c: (r["n_present"], r["auroc_lenient"], r["ci95"][0], r["ci95"][1],
                       r["decision"]) for c, r in results.items()}
    table.with_columns(*[pl.col("cui").replace_strict(
        {c: v[i] for c, v in result_rows.items()}, default=None, return_dtype=t).alias(n)
        for i, (n, t) in enumerate([("n_present", pl.Int64), ("auroc_lenient", pl.Float64),
                                    ("ci95_low", pl.Float64), ("ci95_high", pl.Float64),
                                    ("decision", pl.String)])]).write_csv(args.candidates)

    base_probe = np.load(base_dir / "probe.npz")
    keep_idx = [list(concepts).index(c) for c in kept]
    W = np.concatenate([base_probe["W"], probe.W[keep_idx]])
    bias = np.concatenate([base_probe["b"], probe.b[keep_idx]])
    names = (*[str(c) for c in base_probe["concepts"]], *kept)
    full = Probe(W=W.astype(np.float32), b=bias.astype(np.float32), concepts=names)
    studies = pl.read_parquet(canonical_dir / "studies.parquet")
    # the 14 columns are copied, not recomputed, so they stay identical to the base matrix
    extra = Probe(W=probe.W[keep_idx], b=probe.b[keep_idx], concepts=tuple(kept))
    base_scores = pl.read_parquet(base_dir / "corpus_scores.parquet")
    ext_scores = iscores.corpus_scores(extra, _corpus_rows(args, build_id, studies), studies)
    if not base_scores["study_id"].equals(ext_scores["study_id"]):
        print("STOP: the base matrix rows are not in canonical studies order", file=sys.stderr)
        return 1
    scores = base_scores.hstack(ext_scores.drop("study_id", "image_id"))
    out = Path(args.out or f"{base_dir}_ext1")
    meta = {**base_meta, "columns": {
                "base": [str(c) for c in base_probe["concepts"]],
                "extended": {c: {"name": picked.filter(pl.col("cui") == c)["name"].item(),
                                 **results[c]} for c in kept}},
            "extension": {"base_matrix": str(base_dir), "pool_sha256": pool_meta["pool_sha256"],
                          "config": ecfg, "caption_rule": "render: the bare phrase; the most "
                          "frequent pool strings of the concept and its IS_A descendants, "
                          "embeddings averaged", "dropped": {c: results[c] for c in concepts
                                                             if c not in kept}},
            "git_commit": _git_commit(),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds")}
    save_probe(out, full, meta, scores)

    # Counts and codes only (T38): names stay in the files under data/.
    for cui, r in results.items():
        print(f"{cui}  present {r['n_present']:>5}  AUROC {r['auroc_lenient']:.3f}  "
              f"CI [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  {r['decision']}"
              + ("  borderline" if r["borderline"] else ""))
    print(f"kept {len(kept)} of {len(concepts)}; matrix: {out} ({scores.height:,} studies x "
          f"{len(names)} columns); candidates: {args.candidates}")
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
    if args.kg_prompt:
        if args.split != "dev":
            print("--kg-prompt is for the dev split only", file=sys.stderr)
            return 1
        parser, status = _kg_parser(args.kg_prompt)
    else:
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
    usage = getattr(parser.llm, "usage", None)
    price = load_config("models")["llm"].get("prices", {}).get(Settings().llm_model)
    if usage and price and usage["calls"]:
        # one warm-up call and every retry included; cache writes 1.25x, reads 0.1x (list prices)
        usd = (usage["input_tokens"] * price["input"] + usage["output_tokens"] * price["output"]
               + usage["cache_creation_input_tokens"] * price["input"] * 1.25
               + usage["cache_read_input_tokens"] * price["input"] * 0.1) / 1e6
        report["llm"]["usage"] = {**usage, "usd": usd, "usd_per_call": usd / usage["calls"]}
        print(f"LLM usage: {usage}; ${usd:.4f} total, ${1000 * usd / usage['calls']:.3f} "
              f"per 1,000 calls")
    _merge_json(Path(args.out), "eval", report)

    for mode, entry in report["parsers"].items():
        m = entry["metrics"]
        print(f"{mode:<5} n={m['n']:<4} valid={m['schema_valid_rate']:.3f} "
              f"concept={_f(m['concept_macro_f1'])} status={_f(m['status_macro_f1'])} "
              f"view={_f(m['view_macro_f1'])} exact={m['exact_match_rate']:.3f} "
              f"p50={entry['latency_ms']['p50']:.0f}ms p95={entry['latency_ms']['p95']:.0f}ms "
              f"mean={entry['latency_ms']['mean']:.0f}ms "
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


_LLM_MODELS = ("opus", "sonnet")   # configs/kg.yaml linker.llm.models (D56)


def _chooser(name: str, cfg: dict, key: str):
    return llm_select.AnthropicChooser(name, cfg, key)


def _llm_cache(out: Path, name: str) -> llm_select.Cache:
    out.mkdir(parents=True, exist_ok=True)
    return llm_select.Cache(out / f"llm_cache_{name}.jsonl")


def _cmd_kg_link(args: argparse.Namespace) -> int:
    """V2-3 (D56): rules and candidates for the 95% head, the cached choices of both LLMs and the
    adjudication; writes links, candidates, review.csv and the inputs `kg-link-llm` sends."""
    cfg = load_config("kg")
    lcfg = cfg["linker"]
    umls, out = Path(args.umls_dir), Path(args.out)
    try:
        overrides = linker.load_overrides(default_config_dir() / "kg" / "linker_overrides.yaml")
        mapping = yaml.safe_load((default_config_dir() / "kg" / "chexpert_cuis.yaml")
                                 .read_text(encoding="utf-8"))["labels"]
        project_ids = {e["id"] for e in mapping.values() if e["id"].startswith("PRJ:")}
        adjudication = linker.load_adjudication(out / "adjudication.csv", project_ids)
        pool_dir = Path(args.pool_dir)
        meta = json.loads((pool_dir / probe_pool.POOL_META).read_text(encoding="utf-8"))
        sections = vocab.pool_sections(pl.read_parquet(pool_dir / probe_pool.POOL_IMAGES),
                                       Path(args.plus_csv))
    except (linker.LinkerError, vocab.VocabError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1
    found, _ = vocab.read_mentions(sections, Path(args.radgraph_dir))
    mentions = vocab.apply_rules(found, cfg["vocab"], vocab.mrconso_names(umls / "mrconso.parquet"))
    table = vocab.vocab_table(mentions, "joined")
    if not table.equals(pl.read_parquet(Path(args.vocab_dir) / "vocab_joined.parquet")):
        print("STOP: the vocabulary differs from V2-2's vocab_joined.parquet", file=sys.stderr)
        return 1
    heads = linker.scope(table, lcfg)

    started = time.perf_counter()
    domain = linker.domain(umls, lcfg)
    unchecked = linker.check_adjudication(adjudication, umls, lcfg, domain)
    links, cands, outside = linker.link_heads(heads, umls, lcfg, overrides, domain,
                                              lambda: sapbert.SapBert(lcfg["sapbert"]),
                                              Path(args.sapbert_cache))
    link_seconds = time.perf_counter() - started
    observations = heads.filter(pl.col("kind") == "Observation")["string"]
    occurrences = linker.located_pairs(mentions, observations)
    pairs = linker.pair_links(linker.top_pairs(occurrences, lcfg["pairs"]), umls, lcfg, domain)
    shares = linker.pair_share(mentions, occurrences, pairs,
                               ["effusion", "edema", "opacity", "thickening"])

    # Pairs of a descriptive word that no exact name links go through the same selection step
    descriptive = linker.descriptive_pairs(pairs, lcfg, set(heads["string"]))
    pair_cands = linker.candidates_for(linker.pair_rows(descriptive), umls, lcfg, domain,
                                       lambda: sapbert.SapBert(lcfg["sapbert"]),
                                       Path(args.sapbert_cache))
    pair_links_in = descriptive.select(pl.lit("Observation").alias("kind"), "string",
                                       pl.col("n").alias("mentions"), pl.lit("pair").alias("route"))
    context = llm_select.located_context(occurrences) | {
        ("Observation", s): [[a, n]] for a, n, s in descriptive.select("anat", "n", "string")
        .iter_rows()}
    info = linker.describe(set(cands["cui"]) | set(pair_cands["cui"]), umls)
    items = (llm_select.inputs(links, cands, info, context)
             + llm_select.inputs(pair_links_in, pair_cands, info, context))
    system, prompt_sha = llm_select.load_prompt(default_config_dir() / lcfg["llm"]["prompt_file"])
    models = lcfg["llm"]["models"]
    chosen = {m: llm_select.choices(items, _llm_cache(out, m), models[m]["model"], prompt_sha)
              for m in _LLM_MODELS}
    final = linker.decide(links, chosen, adjudication, umls, lcfg, domain)
    pairs, pair_review = linker.decide_pairs(pairs, descriptive, chosen, adjudication)
    pairs = linker.override_pairs(pairs, overrides[linker.PAIRS])
    info = linker.describe(set(cands["cui"]) | set(final["cui"].drop_nulls())
                           | set(final["lateral_from"].drop_nulls()), umls)

    chexpert, mismatched = {"fetched": False}, set()
    phrase_dir = Path(args.chexpert_dir) / lcfg["chexpert_phrases"]["commit"]
    if phrase_dir.is_dir():
        phrases = chexpert_phrases.load(phrase_dir)
        devices = set(pl.read_parquet(umls / "mrsty.parquet", columns=["CUI", "TUI"])
                      .filter(pl.col("TUI").is_in(lcfg["labels"]["Device"]))["CUI"])
        devices &= set(domain.filter(pl.col("kind") == "Observation")["CUI"])
        expected = chexpert_phrases.expected_cuis(mapping, umls, devices,
                                                  lcfg["chexpert_phrases"]["device_label"])
        stats, mismatched = chexpert_phrases.mismatches(final, phrases, expected)
        argmax, _ = chexpert_phrases.mismatches(final, phrases, expected, "argmax_cui")
        chexpert = {"fetched": True, "commit": lcfg["chexpert_phrases"]["commit"],
                    "final": chexpert_phrases.rate(stats),
                    "argmax": chexpert_phrases.rate(argmax), "by_label": stats}
    skipped = {s for entries in overrides.values() for s, v in entries.items() if v == linker.SKIP}
    review = linker.review_rows(final, info, linker.flags(final, info, lcfg, skipped, mismatched))

    out.mkdir(parents=True, exist_ok=True)
    final.write_parquet(out / "links.parquet")
    pairs.write_parquet(out / "pair_links.parquet")
    pair_review.join(info.select("cui", "name"), on="cui", how="left").write_csv(
        out / "pair_review.csv")
    (cands.join(info.select("cui", "name", "semantic_types", "sabs"), on="cui", how="left")
     .sort("kind", "string", "rank").write_csv(out / "candidates.csv"))
    review.write_csv(out / "review.csv")
    llm_select.write_inputs(items, out / llm_select.INPUTS_FILE)
    report = {"pool_sha256": meta["pool_sha256"], "umls": cfg["umls"]["version"],
              "sapbert_revision": lcfg["sapbert"]["revision"], "candidates": lcfg["candidates"],
              "llm": {"prompt_file": lcfg["llm"]["prompt_file"], "prompt_sha256": prompt_sha,
                      **{m: {"model": models[m]["model"], "answered": len(chosen[m])}
                         for m in _LLM_MODELS}},
              "strings": {k: int(n) for k, n in heads.group_by("kind").len().iter_rows()},
              "domain_cuis": linker.domain_sizes(umls, lcfg, domain),
              "methods": linker.mention_counts(final),
              "selection": {"strings": len(items), "candidates": cands.height,
                            "routes": dict(links.group_by("route").len().drop_nulls("route")
                                           .iter_rows())},
              "lateral": {"moved": final["lateral_from"].drop_nulls().len(),
                          "unresolved": int(final["lateral_unresolved"].sum())},
              "outside_label_table": len(outside), "adjudicated": len(adjudication),
              "adjudication_problems": [list(p) for p in unchecked],
              "decisions": {f"{k}/{d}": n for k, d, n in final.group_by("kind", "decision")
                            .len().sort("kind", "decision").iter_rows()},
              "agreement": linker.agreement(final), "coverage": linker.coverage(final),
              "chexpert_phrases": chexpert,
              "pairs": {"tried": pairs.height, "linked": pairs["cui"].drop_nulls().len(),
                        "by_via": dict(pairs.group_by("via").len().drop_nulls("via").iter_rows()),
                        "descriptive_sent": descriptive.height,
                        "descriptive_disagree": int((~pair_review["agree"].fill_null(True)).sum())},
              "pair_share": shares, "review_rows": review.height,
              "link_seconds": round(link_seconds, 1), "git_commit": _git_commit()}
    (out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Counts only: links, candidates and review.csv hold UMLS names (T38).
    for kind, c in report["domain_cuis"].items():
        print(f"{kind} CUIs: label table {c['label_table']:,}, domain {c['domain']:,}")
    for name, c in report["methods"].items():
        print(f"{name:26s} {c['strings']:>5,} strings  {c['mentions']:>9,} mentions")
    print(f"selection step: {len(items)} strings, {cands.height:,} candidates; answered "
          + ", ".join(f"{p} {len(c)}" for p, c in chosen.items()))
    print(f"coverage: {json.dumps(report['coverage'])}")
    print(f"agreement: {json.dumps(report['agreement'])}")
    print(f"decisions: {json.dumps(report['decisions'])}")
    print(f"adjudicated {len(adjudication)}; CUIs failing the checks: {len(unchecked)}")
    for kind, string, cui, problem in unchecked:
        print(f"  {kind} {string!r} {cui}: {problem}")
    if chexpert["fetched"]:
        print(f"CheXpert phrases: final {chexpert['final']}, argmax {chexpert['argmax']}")
    print(f"lateral CUIs moved to a parent {report['lateral']['moved']}, "
          f"unresolved {report['lateral']['unresolved']}; strings named only outside the "
          f"label table {len(outside)}")
    print(f"pairs tried {pairs.height:,}, linked {report['pairs']['linked']:,} "
          f"{report['pairs']['by_via']}; descriptive pairs sent to the LLMs "
          f"{descriptive.height}, disagree {report['pairs']['descriptive_disagree']}")
    for s, v in shares.items():
        print(f"pair links, {s:10s}: {v['pair_linked']:>7,} of {v['mentions']:>7,} mentions "
              f"({v['located']:,} located)")
    print(f"review.csv: {review.height} rows; linking took {link_seconds:,.0f} s")
    print(f"written: {out}")
    return 0


def _cmd_kg_link_llm(args: argparse.Namespace) -> int:
    """V2-3 (D56): ask one model about the strings of llm_inputs.jsonl it has not answered
    under the current prompt and model; answers go to its cache. Fold them in with kg-link."""
    lcfg = load_config("kg")["linker"]
    out = Path(args.out)
    items = llm_select.read_inputs(out / llm_select.INPUTS_FILE)
    key = llm_select.api_key("ANTHROPIC_API_KEY")
    if not key:
        print("STOP: ANTHROPIC_API_KEY is not set (environment or .env)", file=sys.stderr)
        return 1
    system, prompt_sha = llm_select.load_prompt(default_config_dir() / lcfg["llm"]["prompt_file"])
    chooser = _chooser(args.model, lcfg["llm"]["models"][args.model], key)
    stats = llm_select.run(items, chooser, system, prompt_sha, _llm_cache(out, args.model),
                           None if args.all else args.limit, lcfg["llm"]["workers"])
    print(json.dumps({"name": args.model, "model": chooser.model, **stats}, indent=2))
    return 0


def _cmd_kg_link_sample(args: argparse.Namespace) -> int:
    """V2-3 gate (D56): write sample.csv; --judge grades it with the LLM judge (silver standard)
    into sample_judged.csv; --score <csv> computes the gate from a graded sample."""
    lcfg = load_config("kg")["linker"]
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if args.score:
        result = link_gate.gate(pl.read_csv(args.score, infer_schema_length=0),
                                manifest["coverage"], lcfg["gate"])
        (out / "gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    if args.judge:
        graded = pl.read_csv(out / "sample.csv", infer_schema_length=0).fill_null("")
        jcfg = lcfg["judge"]
        key = llm_select.api_key("ANTHROPIC_API_KEY")
        if not key:
            print("STOP: ANTHROPIC_API_KEY is not set (environment or .env)", file=sys.stderr)
            return 1
        system, prompt_sha = llm_select.load_prompt(default_config_dir() / jcfg["prompt_file"])
        items = link_gate.judge_items(graded)
        cache = _llm_cache(out, "judge")
        chooser = _chooser("judge", jcfg["model"], key)
        stats = llm_select.run(items, chooser, system, prompt_sha, cache, None, jcfg["workers"],
                               schema=llm_select.GRADE_SCHEMA, parse=llm_select.parse_grade,
                               refused=llm_select.REFUSED_GRADE)
        grades = llm_select.choices(items, cache, chooser.model, prompt_sha)
        link_gate.apply_grades(graded, grades, f"llm:{chooser.model} (silver)").write_csv(
            out / "sample_judged.csv")
        print(json.dumps({"judge": chooser.model, **stats}, indent=2))
        return 0
    if (out / "sample.csv").exists():
        print("STOP: sample.csv exists; grades in it would be lost", file=sys.stderr)
        return 1
    links = pl.read_parquet(out / "links.parquet")
    context = {(i["kind"], i["string"]): i["input"].get("located_at")
               or i["input"].get("observations_located_here", [])
               for i in llm_select.read_inputs(out / llm_select.INPUTS_FILE)}
    cuis = set(links["cui"].drop_nulls()) | set(links["argmax_cui"].drop_nulls())
    sample = link_gate.sample(links, linker.describe(cuis, Path(args.umls_dir)), context,
                              lcfg["gate"])
    sample.write_csv(out / "sample.csv")
    print(f"sample.csv: {sample.height} strings (seed {lcfg['gate']['seed']})")
    return 0


def _cmd_kg_build_graph(args: argparse.Namespace) -> int:
    """V2-4 (D57, D58): nodes and edges of the knowledge part of the KG, weights on the pool."""
    cfg = load_config("kg")
    lcfg, gcfg = cfg["linker"], cfg["graph"]
    umls, link_dir, out = Path(args.umls_dir), Path(args.link_dir), Path(args.out)
    mapping = yaml.safe_load((default_config_dir() / "kg" / "chexpert_cuis.yaml")
                             .read_text(encoding="utf-8"))["labels"]
    pool_dir = Path(args.pool_dir)
    try:
        meta = json.loads((pool_dir / probe_pool.POOL_META).read_text(encoding="utf-8"))
        sections = vocab.pool_sections(pl.read_parquet(pool_dir / probe_pool.POOL_IMAGES),
                                       Path(args.plus_csv))
    except vocab.VocabError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1
    found, _ = vocab.read_mentions(sections, Path(args.radgraph_dir))
    mentions = vocab.apply_rules(found, cfg["vocab"], vocab.mrconso_names(umls / "mrconso.parquet"))
    if not vocab.vocab_table(mentions, "joined").equals(
            pl.read_parquet(Path(args.vocab_dir) / "vocab_joined.parquet")):
        print("STOP: the vocabulary differs from V2-2's vocab_joined.parquet", file=sys.stderr)
        return 1
    links = pl.read_parquet(link_dir / "links.parquet")
    pairs = pl.read_parquet(link_dir / "pair_links.parquet")
    domain = linker.domain(umls, lcfg)

    grown = graph.expand(graph.core(links, pairs, mapping, gcfg), umls, lcfg, gcfg, domain)
    mcuis = graph.mention_cuis(mentions, links, pairs)
    nodes = graph.nodes_table(grown, mapping, umls, lcfg, gcfg, graph.n_pool(mcuis))
    # a concept with a column in the extended score matrix is scorable too (D62)
    matrix_meta = Path(args.image_scores) / "probe.json"
    extended = (list(json.loads(matrix_meta.read_text(encoding="utf-8"))["columns"]["extended"])
                if matrix_meta.is_file() else [])
    nodes = nodes.with_columns(pl.col("scorable") | pl.col("cui").is_in(extended))
    phrase_dir = Path(args.chexpert_dir) / lcfg["chexpert_phrases"]["commit"]
    device_phrases = chexpert_phrases.load(phrase_dir)["Support Devices"]
    groups = graph.device_groups(links, grown, umls, lcfg, device_phrases, gcfg["devices_parent"],
                                 mapping["Support Devices"].get("children", []))
    cx, missing = graph.chexpert_edges(mapping, gcfg, links, groups)
    suggests, s_all = graph.weighted(mcuis, "suggests", "Observation", graph.SUGGESTS,
                                     gcfg["min_n"])
    located, t_all = graph.weighted(mcuis, "located_at", "Anatomy", graph.TYPICALLY_AT,
                                    gcfg["min_n"])
    s_count = {"pairs": s_all.height, "kept": suggests.height}
    t_count = {"pairs": t_all.height, "kept": located.height}
    # SUGGESTS ends at a Disease (schema); every weighted edge needs both ends among the nodes
    disease = set(nodes.filter(pl.col("labels").str.contains("Disease"))["cui"])
    suggests = suggests.filter(pl.col("dst").is_in(list(disease)))
    stypes, st_edges = graph.semantic_types(umls)
    edges = pl.concat([graph.umls_edges(nodes, umls, lcfg), cx, suggests, located, st_edges,
                       graph.manual_edges(gcfg)])
    known = set(nodes["cui"]) | set(stypes["tui"])
    dropped = edges.filter(~pl.col("src").is_in(list(known)) | ~pl.col("dst").is_in(list(known)))
    edges = edges.join(dropped.select("src", "rel", "dst"), on=["src", "rel", "dst"], how="anti")
    if (out / "suggests_review.csv").exists():
        edges = graph.apply_review(edges, pl.read_csv(out / "suggests_review.csv",
                                                      infer_schema_length=0).fill_null(""))
    reach = graph.scorable_reach(nodes, edges)
    nodes = nodes.join(reach, on="cui")
    alias_rows = None
    if (out / "alias_vi_review.csv").exists():
        alias_rows = graph.aliases(nodes, umls, load_config("concepts")["concepts"], mapping,
                                   pl.read_csv(out / "alias_vi_review.csv",
                                               infer_schema_length=0).fill_null(""),
                                   yaml.safe_load((default_config_dir() / "kg" /
                                                   "aliases_manual.yaml").read_text(encoding="utf-8")))

    out.mkdir(parents=True, exist_ok=True)
    nodes.write_parquet(out / "nodes.parquet")
    stypes.write_parquet(out / "semantic_types.parquet")
    edges.write_parquet(out / "edges.parquet")
    if alias_rows is not None:
        alias_rows.write_parquet(out / "aliases.parquet")
    pl.concat([s_all, t_all]).write_parquet(out / "weights_all.parquet")   # before the n cut
    in_core = nodes.filter(pl.col("in_core"))
    chf = sorted(set(links.filter(pl.col("kind") == "Observation", pl.col("string").is_in(
        ["chf", "heart failure", "congestive heart failure"]))["cui"].drop_nulls()))
    report = {
        "pool_sha256": meta["pool_sha256"], "umls": cfg["umls"]["version"],
        "include_snomed": cfg["umls"]["include_snomed"], "rel_sabs": cfg["umls"]["rel_sabs"],
        "links_sha256": hashlib.sha256((link_dir / "links.parquet").read_bytes()).hexdigest(),
        "nodes": {"all": nodes.height, "core": in_core.height,
                  "by_label": dict(nodes.group_by("labels").len().sort("labels").iter_rows())},
        "edges": dict(edges.group_by("rel").len().sort("rel").iter_rows()),
        "edges_by_source": {f"{r}/{s}": n for r, s, n in edges.filter(pl.col("rel") == graph.ISA)
                            .group_by("rel", "source").len().sort("source").iter_rows()
                            if s in ("chexpert", "STN")},
        "weighted": {"SUGGESTS": {**s_count, "to_disease": suggests.height},
                     "TYPICALLY_AT": t_count, "min_n": gcfg["min_n"]},
        "dropped_edges_without_nodes": dropped.height,
        "scorable_reach": _reach_report(in_core, links),
        "suggests_review": dict(edges.filter(pl.col("rel") == graph.SUGGESTS)
                                .group_by("review").len().fill_null("none").iter_rows()),
        "aliases": None if alias_rows is None else dict(
            alias_rows.group_by("lang", "source").len()
            .select(pl.concat_str("lang", "source", separator="/"), "len").iter_rows()),
        "devices": {k: len(v) for k, v in groups.items()}, "generic_missing": missing,
        "chf_cuis": chf, "scorable_extended": len(extended),
        "image_scores": str(args.image_scores), "git_commit": _git_commit()}
    (out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "devices.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")

    # Counts and codes only (T38).
    print(f"nodes {nodes.height:,} (core {in_core.height:,}); "
          f"by label {report['nodes']['by_label']}")
    print(f"edges {json.dumps(report['edges'])}; chexpert/STN IS_A {report['edges_by_source']}")
    print(f"weighted (n >= {gcfg['min_n']}): SUGGESTS {s_count['pairs']:,} pairs -> "
          f"{s_count['kept']:,} kept -> {suggests.height:,} to a Disease; TYPICALLY_AT "
          f"{t_count['pairs']:,} -> {t_count['kept']:,}")
    reach = report["scorable_reach"]
    print(f"core concepts reaching a scorable one (D59): by mentions {reach['by_mentions']:.3f}, "
          f"by concepts {reach['concepts']} of {reach['of']}; by label {reach['by_label']}")
    print(f"unreached, most mentioned: {reach['unreached_top20']}")
    print(f"SUGGESTS review {report['suggests_review']}; aliases {report['aliases']}")
    print(f"devices: CheXpert phrases {len(groups['support_devices'])}, other "
          f"{len(groups['other'])}; generic strings without a CUI: {missing}")
    print(f"chf / heart failure CUIs {chf}; written: {out}")
    return 0


def _reach_report(core: pl.DataFrame, links: pl.DataFrame) -> dict:
    """D59: the share of the core's pool mentions (n_pool) that reach a scorable concept, the
    count of concepts, per label, and the 20 most mentioned concepts that do not."""
    core = core.with_columns(pl.col("n_pool").cast(pl.Int64))
    share = (core["n_pool"] * core["reaches"]).sum() / core["n_pool"].sum()
    by_label = {label: {"concepts": f"{int(r)}/{n}", "by_mentions": round(w, 3)}
                for label, n, r, w in core.group_by("labels").agg(
                    pl.len(), pl.col("reaches").sum(),
                    (pl.col("n_pool") * pl.col("reaches")).sum() / pl.col("n_pool").sum())
                .sort("labels").iter_rows()}
    named = dict(links.filter(pl.col("cui").is_not_null()).group_by("cui")
                 .agg(pl.col("string").sort_by("mentions", descending=True).first()).iter_rows())
    top = (core.filter(~pl.col("reaches")).sort("n_pool", descending=True).head(20)
           .select("cui", "labels", "n_pool").rows())
    return {"by_mentions": share, "concepts": int(core["reaches"].sum()), "of": core.height,
            "by_label": by_label,
            "unreached_top20": [[named.get(c, ""), c, label, n] for c, label, n in top]}


def _cmd_kg_graph_llm(args: argparse.Namespace) -> int:
    """V2-4 (D58): Opus reviews the strongest SUGGESTS edges, or proposes Vietnamese aliases;
    answers are cached and written to a review CSV for Claude in the chat app to adjudicate."""
    gcfg = load_config("kg")["graph"]
    mcfg, out, umls = gcfg["llm"], Path(args.out), Path(args.umls_dir)
    key = llm_select.api_key("ANTHROPIC_API_KEY")
    if not key:
        print("STOP: ANTHROPIC_API_KEY is not set (environment or .env)", file=sys.stderr)
        return 1
    nodes = pl.read_parquet(out / "nodes.parquet")
    concepts = load_config("concepts")
    existing_vi = {label: entry.get("vi", []) for label, entry in concepts["concepts"].items()}
    if args.task == "suggests":
        edges = pl.read_parquet(out / "edges.parquet")
        top = (edges.filter(pl.col("rel") == graph.SUGGESTS)
               .sort(["n", "p"], descending=True).head(mcfg["suggests_top"]))
        cuis = set(top["src"]) | set(top["dst"])
    elif args.cuis:
        cuis = set(args.cuis)      # e.g. the concepts that just got a column (D62)
    else:
        cuis = set(nodes.filter(pl.col("scorable"))["cui"]) | set(
            nodes.filter(~pl.col("scorable")).sort("n_pool", descending=True)
            .head(mcfg["alias_top"])["cui"])
    umls_cuis = [c for c in cuis if not c.startswith("PRJ:")]
    described = linker.describe(umls_cuis, umls)
    info = {c: (n, t or "") for c, n, t in described.select("cui", "name", "semantic_types")
            .iter_rows()}
    info |= {c: (n, "") for c, n in nodes.filter(pl.col("cui").str.starts_with("PRJ:"))
             .select("cui", "name").iter_rows()}
    if args.task == "suggests":
        items = graph_llm.suggest_items(edges, info, mcfg["suggests_top"])
        prompt_file, schema = mcfg["suggests_prompt"], graph_llm.SUGGEST_SCHEMA
        parse, refused = graph_llm.parse_review, graph_llm.REFUSED_REVIEW
    else:
        english = {}
        for cui, text in (pl.scan_parquet(umls / "mrconso.parquet")
                          .filter(pl.col("CUI").is_in(umls_cuis)).select("CUI", "STR")
                          .unique().sort("CUI", "STR").collect().iter_rows()):
            english.setdefault(cui, []).append(text)
        items = graph_llm.alias_items(nodes.filter(pl.col("cui").is_in(list(cuis))), info,
                                      english, existing_vi, mcfg["alias_top"])
        prompt_file, schema = mcfg["alias_prompt"], graph_llm.ALIAS_SCHEMA
        parse, refused = graph_llm.parse_aliases, graph_llm.REFUSED_ALIASES
    system, prompt_sha = llm_select.load_prompt(default_config_dir() / prompt_file)
    cache = _llm_cache(out, f"{args.task}_opus")
    chooser = _chooser("opus", mcfg["model"], key)
    stats = llm_select.run(items, chooser, system, prompt_sha, cache,
                           None if args.all else args.limit, mcfg["workers"], schema=schema,
                           parse=parse, refused=refused)
    answers = llm_select.choices(items, cache, chooser.model, prompt_sha)

    if args.task == "suggests":
        rows = [{"rank": i, "src": it["input"]["source"]["cui"],
                 "src_name": it["input"]["source"]["name"], "dst": it["input"]["target"]["cui"],
                 "dst_name": it["input"]["target"]["name"],
                 "n": it["input"]["n_reports_with_both"],
                 "n_src": it["input"]["n_reports_with_source"], "p": it["input"]["p"],
                 "opus_decision": answers.get(("SUGGESTS", it["string"]), {}).get("decision", ""),
                 "opus_reason": answers.get(("SUGGESTS", it["string"]), {}).get("reason", ""),
                 "decision": "", "note": ""} for i, it in enumerate(items, start=1)]
        # an edge already adjudicated keeps its decision; a new one takes Opus's proposal for now
        path = out / "suggests_review.csv"
        done = ({(r["src"], r["dst"]): (r["decision"], r["note"]) for r in
                 pl.read_csv(path, infer_schema_length=0).fill_null("").iter_rows(named=True)
                 if r["decision"]} if path.exists() else {})
        for r in rows:
            r["decision"], r["note"] = done.get((r["src"], r["dst"]), (
                r["opus_decision"], "opus proposal applied; chat review pending"))
        pl.DataFrame(rows).write_csv(path)
    else:
        blocked = [*concepts["blocked_phrases"]["vi"], *gcfg["alias_common_phrases"]]
        label_of = dict(nodes.filter(pl.col("chexpert_label").is_not_null())
                        .select("cui", "chexpert_label").iter_rows())
        # rows already in the file keep their adjudication; only concepts not yet in it are added
        path = out / "alias_vi_review.csv"
        old = (pl.read_csv(path, infer_schema_length=0).fill_null("") if path.exists()
               else pl.DataFrame(schema={c: pl.String for c in (
                   "cui", "name", "chexpert_label", "n_pool", "alias", "opus_reason", "flags",
                   "decision", "note")}))
        rows = []
        for it in items:
            answer = answers.get(("alias", it["string"]))
            if answer is None or it["string"] in set(old["cui"]):
                continue
            proposed = list(dict.fromkeys(a.strip().lower() for a in answer["aliases"]
                                          if a.strip()))[:mcfg["alias_max"]]
            for alias in proposed or [""]:
                rows.append({"cui": it["string"], "name": it["input"]["name"],
                             "chexpert_label": it["input"]["chexpert_label"] or "",
                             "n_pool": str(it["input"]["n_pool_reports"]), "alias": alias,
                             "opus_reason": answer["reason"]})
        named = [r for r in rows if r["alias"]]
        everyone = [(c, a) for c, a in old.filter(pl.col("alias") != "")
                    .select("cui", "alias").iter_rows()] + [(r["cui"], r["alias"]) for r in named]
        flagged = graph_llm.alias_flags(everyone, blocked, existing_vi, label_of)[-len(named):]
        flags = iter(flagged if named else [])
        for r in rows:
            r["flags"] = next(flags) if r["alias"] else "none_proposed"
            r["decision"], r["note"] = "", ""
        new = pl.DataFrame(rows, schema={c: pl.String for c in old.columns}) if rows else None
        (pl.concat([old, new.select(old.columns)]) if new is not None else old).write_csv(path)
        print(f"alias_vi_review.csv: {len(rows)} new rows for "
              f"{len({r['cui'] for r in rows})} concepts, decision empty")
    print(json.dumps({"task": args.task, "model": chooser.model, "prompt_sha256": prompt_sha,
                      **stats}, indent=2))
    return 0


def _cmd_kg_fetch_chexpert_phrases(args: argparse.Namespace) -> int:
    """V2-3 (D56): the CheXpert labeler's mention phrases at the pinned commit, into data/."""
    target = chexpert_phrases.fetch(load_config("kg")["linker"]["chexpert_phrases"],
                                    Path(args.out))
    print(f"written: {target}")
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
    val.add_argument("--kg", action="store_true",
                     help="also run the four KG checks (needs the kg profile of compose)")
    val.add_argument("--kg-matrix", default="data/indexes/image_scores/"
                     "20261024_chexpert25k_cxrclip_ft6_lr1e-04_v1_ext1")
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

    kgq = sub.add_parser("kg-quality", help="`quality` (lenient AUROC) of every scorable concept")
    kgq.add_argument("--matrix", default="data/indexes/image_scores/"
                     "20261024_chexpert25k_cxrclip_ft6_lr1e-04_v1_ext1")
    kgq.add_argument("--pool-dir", default="data/probe_pool")
    kgq.add_argument("--canonical-dir", default="data/canonical")
    kgq.add_argument("--out", default="data/kg/graph/quality.json")
    kgq.set_defaults(func=_cmd_kg_quality)

    kgb = sub.add_parser("kg-bench-scoring",
                         help="time score_images + top-k on a random matrix (target < 100 ms)")
    kgb.add_argument("--rows", type=int, default=1_000_000)
    kgb.add_argument("--columns", type=int, default=37)
    kgb.add_argument("--k", type=int, default=10)
    kgb.add_argument("--repeats", type=int, default=15)
    kgb.set_defaults(func=_cmd_kg_bench_scoring)

    kgq = sub.add_parser("kg-query", help="run questions through the KG search (real data)")
    kgq.add_argument("questions", nargs="*")
    kgq.add_argument("--file", default=None, help="one question per line")
    kgq.add_argument("--matrix", default="data/indexes/image_scores/"
                     "20261024_chexpert25k_cxrclip_ft6_lr1e-04_v1_ext1")
    kgq.add_argument("--top-k", type=int, default=10)
    kgq.add_argument("--no-explain", action="store_true")
    kgq.add_argument("--out", default=None, help="JSONL, one record per question")
    kgq.set_defaults(func=_cmd_kg_query)

    kgn = sub.add_parser("kg-load-neo4j", help="load the KG and the top-N PREDICTED edges")
    kgn.add_argument("--graph-dir", default="data/kg/graph")
    kgn.add_argument("--matrix", default="data/indexes/image_scores/"
                     "20261024_chexpert25k_cxrclip_ft6_lr1e-04_v1_ext1")
    kgn.add_argument("--canonical-dir", default="data/canonical")
    kgn.add_argument("--top-n", type=int, default=None, help="default: kg.yaml neo4j.top_n")
    kgn.set_defaults(func=_cmd_kg_load_neo4j)

    bie = sub.add_parser("build-image-scores-ext",
                         help="extra score-matrix columns for core findings without one")
    bie.add_argument("--base", default="data/indexes/image_scores/"
                                       "20261024_chexpert25k_cxrclip_ft6_lr1e-04_v1")
    bie.add_argument("--pool-dir", default="data/probe_pool")
    bie.add_argument("--plus-csv", default="data/raw/chexpert-plus/df_chexpert_plus_240401.csv")
    bie.add_argument("--radgraph-dir", default="data/raw/chexpert-plus/radgraph_xl")
    bie.add_argument("--umls-dir", default="data/umls/2026AA")
    bie.add_argument("--link-dir", default="data/kg/link")
    bie.add_argument("--graph-dir", default="data/kg/graph")
    bie.add_argument("--embeddings", default="data/indexes/embeddings")
    bie.add_argument("--canonical-dir", default="data/canonical")
    bie.add_argument("--candidates", default="data/kg/scores_ext/candidates.csv")
    bie.add_argument("--out", default=None, help="default: <base>_ext1")
    bie.set_defaults(func=_cmd_build_image_scores_ext)

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
    ev.add_argument("--kg-prompt", default=None, metavar="PROMPT_FILE",
                    help="score the KG parser prompt (path under configs/), dev split only")
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

    kgl = sub.add_parser("kg-link",
                         help="V2-3: link the vocabulary head to CUIs from rules, candidates and "
                              "the cached LLM choices; write review.csv")
    kgl.add_argument("--pool-dir", default="data/probe_pool")
    kgl.add_argument("--plus-csv", default="data/raw/chexpert-plus/df_chexpert_plus_240401.csv")
    kgl.add_argument("--radgraph-dir", default="data/raw/chexpert-plus/radgraph_xl")
    kgl.add_argument("--umls-dir", default="data/umls/2026AA")
    kgl.add_argument("--vocab-dir", default="data/kg/vocab")
    kgl.add_argument("--sapbert-cache", default="data/kg/sapbert")
    kgl.add_argument("--chexpert-dir", default="data/kg/chexpert_phrases")
    kgl.add_argument("--out", default="data/kg/link")
    kgl.set_defaults(func=_cmd_kg_link)

    kgm = sub.add_parser("kg-link-llm",
                         help="V2-3: ask one LLM about the selection-step strings")
    kgm.add_argument("--model", choices=_LLM_MODELS, required=True)
    budget = kgm.add_mutually_exclusive_group(required=True)
    budget.add_argument("--limit", type=int, help="at most this many uncached strings")
    budget.add_argument("--all", action="store_true", help="every uncached string")
    kgm.add_argument("--out", default="data/kg/link")
    kgm.set_defaults(func=_cmd_kg_link_llm)

    kgs = sub.add_parser("kg-link-sample", help="V2-3 gate: sample, LLM judge, score")
    kgs.add_argument("--judge", action="store_true",
                     help="grade sample.csv with the LLM judge (silver standard, costs money)")
    kgs.add_argument("--score", default=None, help="a graded sample CSV: compute the gate")
    kgs.add_argument("--umls-dir", default="data/umls/2026AA")
    kgs.add_argument("--out", default="data/kg/link")
    kgs.set_defaults(func=_cmd_kg_link_sample)

    kgg = sub.add_parser("kg-build-graph",
                         help="V2-4: nodes and edges of the KG, SUGGESTS and TYPICALLY_AT weights")
    kgg.add_argument("--pool-dir", default="data/probe_pool")
    kgg.add_argument("--plus-csv", default="data/raw/chexpert-plus/df_chexpert_plus_240401.csv")
    kgg.add_argument("--radgraph-dir", default="data/raw/chexpert-plus/radgraph_xl")
    kgg.add_argument("--umls-dir", default="data/umls/2026AA")
    kgg.add_argument("--vocab-dir", default="data/kg/vocab")
    kgg.add_argument("--link-dir", default="data/kg/link")
    kgg.add_argument("--chexpert-dir", default="data/kg/chexpert_phrases")
    kgg.add_argument("--image-scores", default="data/indexes/image_scores/"
                     "20261024_chexpert25k_cxrclip_ft6_lr1e-04_v1_ext1",
                     help="score matrix whose extended columns make their concepts scorable")
    kgg.add_argument("--out", default="data/kg/graph")
    kgg.set_defaults(func=_cmd_kg_build_graph)

    kgr = sub.add_parser("kg-graph-llm",
                         help="V2-4: Opus reviews SUGGESTS edges or proposes Vietnamese aliases")
    kgr.add_argument("--task", choices=["suggests", "alias"], required=True)
    budget = kgr.add_mutually_exclusive_group(required=True)
    budget.add_argument("--limit", type=int, help="at most this many uncached items")
    budget.add_argument("--all", action="store_true", help="every uncached item")
    kgr.add_argument("--cuis", nargs="+", default=None,
                     help="alias task: only these concepts (rows already in the file are kept)")
    kgr.add_argument("--umls-dir", default="data/umls/2026AA")
    kgr.add_argument("--out", default="data/kg/graph")
    kgr.set_defaults(func=_cmd_kg_graph_llm)

    kgf = sub.add_parser("kg-fetch-chexpert-phrases",
                         help="V2-3: CheXpert labeler mention phrases at the pinned commit")
    kgf.add_argument("--out", default="data/kg/chexpert_phrases")
    kgf.set_defaults(func=_cmd_kg_fetch_chexpert_phrases)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
