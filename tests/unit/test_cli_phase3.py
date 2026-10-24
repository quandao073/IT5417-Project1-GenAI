"""Parser contracts for the Phase 3 commands; the pieces are tested elsewhere."""

from __future__ import annotations

from src.cli.main import build_parser


def test_embed_defaults():
    args = build_parser().parse_args(["embed"])
    assert args.shard_size == 1000 and args.batch_size == 32 and args.device == "cpu"
    assert args.out == "data/indexes/embeddings" and args.build_id is None


def test_index_defaults():
    args = build_parser().parse_args(["index"])
    assert args.embeddings == "data/indexes/embeddings" and args.indexes_dir == "data/indexes"
    assert args.build_id is None and args.recreate is False and args.qdrant_url is None


def test_validate_indexes_defaults():
    args = build_parser().parse_args(["validate-indexes"])
    assert args.indexes_dir == "data/indexes" and args.embeddings == "data/indexes/embeddings"
    assert args.build_id is None   # the selected encoder's registered build


def test_bench_vector_defaults():
    args = build_parser().parse_args(["bench-vector"])
    assert args.queries == 30 and args.repeats == 5 and args.api_url is None
    assert args.out == "artifacts/metrics/vector_baseline.json"


def test_score_corpus_defaults():
    args = build_parser().parse_args(["score-corpus"])
    assert args.n_boot == 1000 and args.out == "artifacts/metrics/vector_baseline.json"
