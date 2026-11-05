"""lookup_concept and expand_concepts over the tiny KG (tests/kg/tiny_kg.py, FakeStore)."""

from __future__ import annotations

import pytest

from src.common.config import load_config
from src.kg import expand, lookup
from src.kg.expand import ABSENT, NOISY_OR, PRESENT, WEIGHTED
from tests.kg.fake_store import FakeStore

QCFG = load_config("kg")["query"]
LOOKUP_CFG = QCFG
EXPAND_CFG = {**QCFG["expand"], "weak_or_children": ["C10"]}


@pytest.fixture
def store():
    return FakeStore()


def codes(intent):
    return [w.code for w in intent.warnings]


def test_lookup_finds_an_alias_exactly_and_cuts_the_side(store):
    got = lookup.lookup_concept(store, "Tràn dịch màng phổi trái", LOOKUP_CFG)
    assert (got.core, got.sides) == ("tràn dịch màng phổi", ["trái"])
    assert [(c.cui, c.match) for c in got.candidates] == [("C1", "exact")]
    both = lookup.lookup_concept(store, "giảm thể tích phổi hai bên", LOOKUP_CFG)
    assert both.sides == ["hai bên"]
    # one alias, two concepts: both come back, the more mentioned one first
    assert [c.cui for c in both.candidates] == ["C12", "C13"]


def test_lookup_reads_a_plural_as_its_singular_and_falls_back_to_shared_words(store):
    plural = lookup.lookup_concept(store, "pleural effusions", LOOKUP_CFG)
    assert [(c.cui, c.match) for c in plural.candidates] == [("C1", "exact")]
    near = lookup.lookup_concept(store, "pleural", LOOKUP_CFG)       # one word, in three aliases
    assert {c.cui for c in near.candidates} == {"C1", "C7", "C9"} and all(
        c.match == "fulltext" and c.score >= 0.5 for c in near.candidates)
    unknown = lookup.lookup_concept(store, "xyz qwerty", LOOKUP_CFG)
    assert unknown.candidates == [] and unknown.core == "xyz qwerty"


def test_a_scorable_concept_is_direct_and_a_weak_one_carries_the_warning(store):
    got = expand.expand_concepts(store, "C1", PRESENT, EXPAND_CFG, "tràn dịch")
    assert (got.kind, got.combine, [c.column for c in got.columns]) == (
        "direct", "single", ["Pleural Effusion"]) and codes(got) == []
    weak = expand.expand_concepts(store, "C4", PRESENT, EXPAND_CFG)
    assert weak.kind == "direct" and codes(weak) == ["weak_quality"]


def test_a_weak_label_named_by_d63_is_noisy_ored_with_its_scorable_children(store):
    got = expand.expand_concepts(store, "C10", PRESENT, EXPAND_CFG)
    assert (got.combine, [c.cui for c in got.columns]) == (NOISY_OR, ["C10", "C11"])
    assert codes(got) == ["weak_quality"]
    plain = expand.expand_concepts(store, "C10", PRESENT, {**EXPAND_CFG, "weak_or_children": []})
    assert [c.cui for c in plain.columns] == ["C10"]


def test_wider_narrower_and_the_negation_of_a_narrow_concept(store):
    wide = expand.expand_concepts(store, "C7", PRESENT, EXPAND_CFG)
    assert (wide.kind, wide.combine, [c.cui for c in wide.columns]) == ("wider", NOISY_OR, ["C1"])
    assert codes(wide) == [] and wide.path == ["Pleural disease ← Pleural effusion"]
    narrow = expand.expand_concepts(store, "C5", PRESENT, EXPAND_CFG)
    assert narrow.kind == "narrower" and [c.cui for c in narrow.columns] == ["C4"]
    assert codes(narrow) == ["narrower", "weak_quality"]    # a weak ancestor warns twice
    negated = expand.expand_concepts(store, "C5", ABSENT, EXPAND_CFG)
    assert negated.columns == [] and codes(negated) == ["narrower", "negation_skipped"]


def test_a_disease_is_inferred_from_the_findings_that_suggest_it(store):
    got = expand.expand_concepts(store, "C6", PRESENT, EXPAND_CFG, "suy tim")
    assert got.kind == "disease" and got.combine == WEIGHTED
    # C2 (p 0.3) and C3 (p 0.1) count; C4 is weak and C1's edge was dropped on review
    assert [(c.cui, round(c.weight, 2)) for c in got.columns] == [("C2", 0.75), ("C3", 0.25)]
    assert codes(got) == ["disease_inferred"] and got.path[0].startswith("Cardiomegaly")


def test_an_anatomy_region_goes_through_its_parts_to_the_typical_findings(store):
    got = expand.expand_concepts(store, "C8", PRESENT, EXPAND_CFG, "màng phổi")
    assert got.kind == "anatomy" and [c.cui for c in got.columns] == ["C1"]   # C2: n 5 < 20
    assert codes(got) == ["anatomy_region"] and "Pleural space" in got.path[0]


def test_a_concept_with_no_route_and_an_unknown_one_are_not_scored(store):
    none = expand.expand_concepts(store, "C12", PRESENT, EXPAND_CFG, "giảm thể tích phổi")
    assert (none.kind, none.columns, codes(none)) == ("none", [], ["no_path"])
    ghost = expand.expand_concepts(store, "C999", PRESENT, EXPAND_CFG, "?")
    assert ghost.kind == "none" and codes(ghost) == ["unrecognized"]


def test_a_strong_concept_comes_before_the_weak_label_that_covers_it(store):
    got = lookup.lookup_concept(store, "broken bone", LOOKUP_CFG)
    assert [c.cui for c in got.candidates] == ["C11", "C10"]        # C10 is weak, C11 is not


def test_the_sapbert_stage_runs_only_when_no_alias_is_the_phrase(store):
    class Index:
        def search(self, norm, k):
            return [{"cui": "C3", "text": "edema", "score": 0.91},
                    {"cui": "C2", "text": "cardiomegaly", "score": 0.5}]

    cfg = {**LOOKUP_CFG, "lookup": {**LOOKUP_CFG["lookup"],
                                    "sapbert": {"enabled": True, "top": 5, "min_cos": 0.8}}}
    near = lookup.lookup_concept(store, "fluid in the lungs", cfg, Index())
    assert [(c.cui, c.match, c.score) for c in near.candidates] == [("C3", "sapbert", 0.91)]
    exact = lookup.lookup_concept(store, "edema", cfg, Index())
    assert [(c.cui, c.match) for c in exact.candidates] == [("C3", "exact")]
    assert lookup.lookup_concept(store, "xyz qwerty", cfg).candidates == []   # no index, no stage


def test_a_hand_written_suggests_edge_counts_without_a_pool_count(store):
    got = expand.expand_concepts(store, "C14", PRESENT, EXPAND_CFG, "interstitial disease")
    assert got.kind == "disease" and got.combine == WEIGHTED
    assert [(c.cui, round(c.weight, 2)) for c in got.columns] == [("C11", 0.5), ("C3", 0.5)]


def test_a_weak_finding_takes_no_slot_of_the_region_s_top_findings(store):
    # C4 (weak) has the strongest TYPICALLY_AT edge of the region: it must not push C1 out
    got = expand.expand_concepts(store, "C9", PRESENT, {**EXPAND_CFG, "anatomy_top": 1}, "space")
    assert got.kind == "anatomy" and [c.cui for c in got.columns] == ["C1"]


def test_sapbert_runs_last_after_an_empty_fulltext_and_modifiers_are_skipped(store):
    calls = []

    class Index:
        def search(self, norm, k):
            calls.append(norm)
            return [{"cui": "C3", "text": "edema", "score": 0.95}]

    cfg = {**LOOKUP_CFG, "modifiers": ["small", "diffuse"],
           "lookup": {**LOOKUP_CFG["lookup"], "sapbert": {"enabled": True, "top": 5,
                                                          "min_cos": 0.8}}}
    near = lookup.lookup_concept(store, "pleural", cfg, Index())      # fulltext finds it
    assert {c.match for c in near.candidates} == {"fulltext"} and calls == []
    for words in ("small", "small diffuse"):
        got = lookup.lookup_concept(store, words, cfg, Index())
        assert got.modifier and got.candidates == [] and calls == []
    assert not lookup.lookup_concept(store, "small effusion", cfg, Index()).modifier


def test_modifier_words_come_from_the_linker_and_the_vocabulary():
    words = lookup.modifier_words(load_config("kg"), {"observation": {"diffuse": "skip",
                                                                      "opacity": "C1"},
                                                      "anatomy": {"zone": "skip"}})
    assert {"diffuse", "zone", "small", "left"} <= set(words) and "opacity" not in words


def test_a_region_alone_is_a_noisy_or_weighted_by_how_typical_each_finding_is(store):
    got = expand.expand_concepts(store, "C9", PRESENT, EXPAND_CFG, "space")
    assert got.combine == expand.WEIGHTED_OR
    assert [(c.cui, c.weight) for c in got.columns] == [("C1", 0.25)]
