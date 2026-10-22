"""CLI wiring for the image-concept probe: build-probe-pool and embed --images."""

from __future__ import annotations

from src.cli.main import build_parser


def test_embed_accepts_images_override():
    args = build_parser().parse_args(["embed", "--images", "x.parquet"])
    assert args.images == "x.parquet"


def test_embed_default_images_is_none():
    assert build_parser().parse_args(["embed"]).images is None


def test_build_probe_pool_defaults_do_not_touch_main_paths():
    args = build_parser().parse_args(["build-probe-pool"])
    assert args.out == "data/probe_pool"
    assert args.canonical_dir == "data/canonical"
    assert args.plus_csv == "data/raw/chexpert-plus/df_chexpert_plus_240401.csv"
    assert args.small_root == "data/raw/CheXpert-v1.0-small"
    assert args.chexbert_dir == "data/raw/chexpert-plus/labels"
