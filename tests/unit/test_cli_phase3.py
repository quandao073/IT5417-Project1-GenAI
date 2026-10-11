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
