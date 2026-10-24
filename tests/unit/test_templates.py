"""The one function that turns parsed constraints into the string the text encoder reads."""

from __future__ import annotations

import pytest

from src.common.config import load_config
from src.retrieval.templates import render


@pytest.fixture(scope="module")
def concepts():
    return load_config("concepts")


def test_includes_then_negated_excludes_joined_by_and(concepts):
    assert render(["Cardiomegaly", "Edema"], ["Pleural Effusion"], concepts) == (
        "cardiomegaly and pulmonary edema and no pleural effusion")


def test_the_order_is_canonical_whatever_order_the_query_named_them_in(concepts):
    assert render(["Edema", "Cardiomegaly"], [], concepts) == render(
        ["Cardiomegaly", "Edema"], [], concepts) == "cardiomegaly and pulmonary edema"
    assert render([], ["Pleural Effusion", "Pneumothorax"], concepts) == (
        "no pneumothorax and no pleural effusion")


def test_a_repeated_concept_is_rendered_once(concepts):
    assert render(["Edema", "Edema"], [], concepts) == "pulmonary edema"


def test_only_negations(concepts):
    assert render([], ["Pleural Effusion"], concepts) == "no pleural effusion"


def test_no_finding_reads_normal_chest_not_a_negation(concepts):
    """concepts.yaml spells No Finding "no finding", which an encoder reads as a negation."""
    assert render(["No Finding"], [], concepts) == "normal chest"
    assert render(["No Finding", "Support Devices"], [], concepts) == (
        "normal chest and support device")


def test_every_other_concept_uses_its_first_english_phrase(concepts):
    assert render(["Pleural Other"], [], concepts) == "pleural thickening"


@pytest.mark.parametrize(("include", "exclude", "match"), [
    ([], [], "at least one concept"),
    ([], ["No Finding"], "No Finding cannot be negated"),
    (["Edema"], ["Edema"], r"both included and excluded: \['Edema'\]"),
])
def test_refuses_what_no_template_can_say(concepts, include, exclude, match):
    with pytest.raises(ValueError, match=match):
        render(include, exclude, concepts)
