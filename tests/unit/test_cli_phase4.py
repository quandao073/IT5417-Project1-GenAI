"""Parser contracts for the Phase 4 commands; the pieces are tested elsewhere."""

from __future__ import annotations

from src.cli.main import build_parser


def test_parse_defaults():
    args = build_parser().parse_args(["parse", "tim to"])
    assert args.text == "tim to" and args.parser == "auto"


def test_eval_parser_defaults():
    args = build_parser().parse_args(["eval-parser"])
    assert args.benchmark == "benchmarks/parser_queries_v1.jsonl" and args.split == "heldout"
    assert args.parsers == ["rule", "llm", "auto"]
    assert args.out == "artifacts/metrics/parser_eval.json"
    assert args.label_filter is False and args.canonical_dir == "data/canonical"
    assert args.verbose is False


def test_eval_parser_accepts_a_subset_of_parsers():
    args = build_parser().parse_args(["eval-parser", "--parsers", "rule", "--split", "all"])
    assert args.parsers == ["rule"] and args.split == "all"
