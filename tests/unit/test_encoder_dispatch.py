"""A fine-tune is a models.yaml candidate of its own name that reuses its base model's adapter
through `framework` (B16): `cxrclip_ft6_lr3e-05` must load like `cxrclip`, with no code change."""

from __future__ import annotations

import pytest

from src.common.config import load_config
from src.embeddings import base


class EchoEncoder:
    def __init__(self, candidate, device):
        self.candidate, self.device = candidate, device


def test_a_fine_tune_candidate_loads_through_its_framework(monkeypatch):
    monkeypatch.setitem(base._FRAMEWORKS, "echo_checkpoint",
                        "tests.unit.test_encoder_dispatch:EchoEncoder")
    encoder = base.load_encoder("cxrclip_ft6_lr3e-05",
                                {"framework": "echo_checkpoint", "weights": "w"}, "cpu")
    assert isinstance(encoder, EchoEncoder) and encoder.candidate["weights"] == "w"


def test_the_cxrclip_checkpoint_framework_is_the_cxrclip_adapter():
    assert base._FRAMEWORKS["cxrclip_checkpoint"] == base._ADAPTERS["cxrclip"]


def test_a_name_without_adapter_or_known_framework_is_refused():
    with pytest.raises(ValueError, match="Unknown encoder 'mystery'"):
        base.load_encoder("mystery", {"framework": "nope"}, "cpu")


def test_every_models_yaml_candidate_resolves_to_an_adapter():
    for name, candidate in load_config("models")["image_encoder"]["candidates"].items():
        assert name in base._ADAPTERS or candidate.get("framework") in base._FRAMEWORKS, name
