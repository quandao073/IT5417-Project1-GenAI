"""Worker CLI. Each command maps to one step of the plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cxr_retrieval.data.audit import CorpusMode, run_audit


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cxr", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Phase 1: measure join rate, decide corpus")
    audit.add_argument("--plus-csv", default="data/raw/chexpert-plus/df_chexpert_plus_240401.csv")
    audit.add_argument("--small-root", default="data/raw/CheXpert-v1.0-small")
    audit.add_argument("--out", default="artifacts/data_audit.json")
    audit.set_defaults(func=_cmd_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
