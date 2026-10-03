"""The spike-models command wires the spike together; the pieces are tested in
test_spike.py, so this only checks the parser contract."""

from __future__ import annotations

from src.cli.main import build_parser


def test_defaults_match_the_spec():
    args = build_parser().parse_args(["spike-models"])
    assert (args.offset, args.n) == (200, 1000)
    assert args.models == ["biomedclip", "chexzero", "xrayclip"]
    assert args.device == "cpu"
    assert args.rescore is False


def test_one_model_at_a_time():
    """Each model embeds in its own process so peak RSS is its own."""
    args = build_parser().parse_args(["spike-models", "--models", "chexzero"])
    assert args.models == ["chexzero"]


def test_rescore_flag():
    assert build_parser().parse_args(["spike-models", "--rescore"]).rescore is True
