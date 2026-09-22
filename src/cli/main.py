"""Worker CLI. Each command maps to one step of the plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data import canonical
from src.data.audit import CorpusMode, run_audit


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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
