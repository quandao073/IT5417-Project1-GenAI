"""Worker CLI. Each command maps to one step of the plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

from src.common.config import load_config
from src.data import canonical
from src.data.audit import CorpusMode, run_audit
from src.embeddings import spike


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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
