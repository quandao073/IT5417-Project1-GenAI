"""Worker CLI. Each command maps to one step of the plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import polars as pl

from src.common.config import Settings, load_config
from src.data import canonical
from src.data.audit import CorpusMode, run_audit
from src.embeddings import embed, index, spike, validate
from src.evaluation import parser_benchmark as pb
from src.evaluation import parser_eval as pe
from src.evaluation import vector_baseline as vb
from src.retrieval import label_filter as lf


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
    images = pl.read_parquet(Path(args.canonical_dir) / "images.parquet")
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
    from src.common.config import config_hash

    shards_dir = _resolve_build_dir(Path(args.embeddings), args.build_id)
    images = pl.read_parquet(Path(args.canonical_dir) / "images.parquet")
    data_cfg = load_config("data")["sources"]
    dataset = {"images": data_cfg["images"]["name"],
               "plus_csv": Path(data_cfg["plus_metadata"]["csv"]).name}
    client = _qdrant_client(args.qdrant_url)
    try:
        manifest = index.run_index(
            client, shards_dir, images, Settings().qdrant_collection_prefix,
            recreate=args.recreate, config_hash=config_hash(), git_commit=_git_commit(),
            dataset=dataset, library_versions=_library_versions(),
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"INDEX FAILED: {exc}", file=sys.stderr)
        return 1
    out = Path(args.indexes_dir) / index.MANIFEST_FILE
    index.write_manifest(out, manifest)
    print(f"collection : {manifest['collection']}")
    print(f"vectors    : {manifest['counts']['vectors']:,}  "
          f"dim {manifest['embedding_model']['dimension']}")
    print(f"manifest   : {out}")
    return 0


def _cmd_validate_indexes(args: argparse.Namespace) -> int:
    manifest_path = Path(args.indexes_dir) / index.MANIFEST_FILE
    if not manifest_path.is_file():
        print(f"FAIL manifest    : {manifest_path} not found; run cxr index", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards_dir = Path(args.embeddings) / manifest["build_id"]
    images = pl.read_parquet(Path(args.canonical_dir) / "images.parquet")
    expected_revision = load_config("models")["image_encoder"]["revision"]
    # Hash over the scope this manifest recorded, not today's default, so an
    # index stays checkable across a change of scope (see D16).
    checks = validate.validate_indexes(
        _qdrant_client(args.qdrant_url), manifest, shards_dir, images,
        expected_config_hash=validate.expected_config_hash(manifest),
        expected_revision=expected_revision,
    )
    for c in checks:
        print(f"{'PASS' if c.ok else 'FAIL'} {c.name:<12}: {c.detail}")
    return 0 if all(c.ok for c in checks) else 1


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
    shards_dir = _resolve_build_dir(Path(args.embeddings), args.build_id)
    canonical = Path(args.canonical_dir)
    images = pl.read_parquet(canonical / "images.parquet")
    labels = pl.read_parquet(canonical / "labels.parquet")
    cfg = load_config("models")["image_encoder"]
    _, encoder = _selected_encoder("cpu")
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
    thresholds = load_config("evaluation")["thresholds"]
    report = pe.evaluate_benchmark(rows, parser, args.parsers, thresholds)
    report["benchmark"] = {"file": args.benchmark, "split": args.split,
                           "sha256": hashlib.sha256(Path(args.benchmark).read_bytes()).hexdigest()}
    report["llm"] = {"status": status, "model": Settings().llm_model,
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
        canonical = Path(args.canonical_dir)
        studies = pl.read_parquet(canonical / "studies.parquet")
        cfg = load_config("retrieval")
        matrix = lf.build_label_matrix(
            pl.read_parquet(canonical / "labels.parquet"), studies["study_id"].to_list(),
            cfg["label_resolution"]["source_priority"], seed=cfg["results"]["label_order_seed"])
        counts = pe.label_filter_counts(rows, matrix)
        _merge_json(Path(args.out), "label_filter", counts)
        for group, pairs in counts["summary"].items():
            cells = "  ".join(f"{pair}: med {s['median']:.0f} zero {s['zero']}"
                              for pair, s in pairs.items())
            print(f"label {group:<11} {cells}")
    print(f"report     : {args.out}")
    return 0 if v["passed"] else 1


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


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
    emb.add_argument("--image-root", default="data/raw/CheXpert-v1.0-small")
    emb.add_argument("--out", default="data/indexes/embeddings")
    emb.add_argument("--shard-size", type=int, default=1000)
    emb.add_argument("--batch-size", type=int, default=32)
    emb.add_argument("--device", default="cpu")
    emb.add_argument("--build-id", default=None,
                     help="override the generated <YYYYMMDD>_chexpert25k_<model>_v1")
    emb.set_defaults(func=_cmd_embed)

    idx = sub.add_parser("index", help="Phase 3: upsert shards into Qdrant, write manifest")
    idx.add_argument("--embeddings", default="data/indexes/embeddings")
    idx.add_argument("--build-id", default=None, help="required when several builds exist")
    idx.add_argument("--canonical-dir", default="data/canonical")
    idx.add_argument("--indexes-dir", default="data/indexes")
    idx.add_argument("--qdrant-url", default=None, help="default: QDRANT_URL / Settings")
    idx.add_argument("--recreate", action="store_true")
    idx.set_defaults(func=_cmd_index)

    val = sub.add_parser("validate-indexes", help="Phase 3: cross-check manifest, shards, Qdrant")
    val.add_argument("--indexes-dir", default="data/indexes")
    val.add_argument("--embeddings", default="data/indexes/embeddings")
    val.add_argument("--canonical-dir", default="data/canonical")
    val.add_argument("--qdrant-url", default=None)
    val.set_defaults(func=_cmd_validate_indexes)

    bench = sub.add_parser("bench-vector", help="Phase 3: cold/warm latency of vector search")
    bench.add_argument("--queries", type=int, default=30)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--api-url", default=None, help="also time POST /api/v1/search over HTTP")
    bench.add_argument("--out", default="artifacts/metrics/vector_baseline.json")
    bench.set_defaults(func=_cmd_bench_vector)

    score = sub.add_parser("score-corpus",
                           help="Phase 3: zero-shot AUROC on all embedded studies")
    score.add_argument("--embeddings", default="data/indexes/embeddings")
    score.add_argument("--build-id", default=None)
    score.add_argument("--canonical-dir", default="data/canonical")
    score.add_argument("--n-boot", type=int, default=1000)
    score.add_argument("--out", default="artifacts/metrics/vector_baseline.json")
    score.set_defaults(func=_cmd_score_corpus)

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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
