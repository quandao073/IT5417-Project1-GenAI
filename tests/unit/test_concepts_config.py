"""configs/concepts.yaml drives both parsers; keep it consistent with the label table."""

from __future__ import annotations

import pytest

from src.common.config import load_config
from src.data.labels import CHEXPERT_LABELS


@pytest.fixture(scope="module")
def cfg():
    return load_config("concepts")


def test_the_fourteen_concepts_match_the_data_config(cfg):
    assert list(cfg["concepts"]) == list(load_config("data")["labels"]["pathologies"])
    assert set(cfg["concepts"]) == set(CHEXPERT_LABELS)


def test_every_concept_has_non_empty_lowercase_forms_in_both_languages(cfg):
    for concept, forms in cfg["concepts"].items():
        for lang in ("en", "vi"):
            assert forms[lang], f"{concept}.{lang} is empty"
            for form in forms[lang]:
                assert form == form.casefold().strip(), f"{concept}.{lang}: {form!r}"


def test_no_surface_form_belongs_to_two_concepts(cfg):
    owner: dict[str, str] = {}
    for concept, forms in cfg["concepts"].items():
        for form in forms["en"] + forms["vi"]:
            assert form not in owner, f"{form!r} under both {owner[form]} and {concept}"
            owner[form] = concept


def test_views_cover_ap_pa_and_frontal(cfg):
    assert set(cfg["views"]) == {"AP", "PA", "FRONTAL"}


@pytest.mark.parametrize("section", ["negation_cues", "negation_cues_post", "uncertainty_cues"])
def test_cue_sections_exist_in_both_languages_and_never_equal_a_concept_form(cfg, section):
    forms = {f for v in cfg["concepts"].values() for f in v["en"] + v["vi"]}
    assert cfg[section]["en"] and cfg[section]["vi"]
    for cue in cfg[section]["en"] + cfg[section]["vi"]:
        assert cue == cue.casefold().strip()
        assert cue not in forms, f"cue {cue!r} is also a concept surface form"


def test_the_parser_config_keys_exist():
    llm = load_config("models")["llm"]
    for key in ("prompt_file", "num_predict", "seed", "keep_alive", "cache_size", "max_retries"):
        assert key in llm, key
    assert isinstance(load_config("retrieval")["results"]["label_order_seed"], int)


FTS_SYNTAX = set('"*():^')


def test_every_concept_has_lowercase_report_terms_without_fts_syntax(cfg):
    for concept, forms in cfg["concepts"].items():
        assert forms["report"], concept
        for term in forms["report"]:
            assert term == term.lower() == " ".join(term.split()), (concept, term)
            assert not set(term) & FTS_SYNTAX, (concept, term)


def test_report_terms_are_unique_across_concepts(cfg):
    # A term may repeat the `en` list of its own concept; two concepts may not share one.
    owner: dict[str, str] = {}
    for concept, forms in cfg["concepts"].items():
        for term in forms["report"]:
            assert owner.setdefault(term, concept) == concept, (term, owner[term], concept)


def test_blocked_phrases_have_both_languages(cfg):
    assert cfg["blocked_phrases"]["vi"] == ["phù hợp"]
    assert cfg["blocked_phrases"]["en"] == []
