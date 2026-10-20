"""Exact-sentence report evidence: span invariant, rule-parser polarity, selection."""

from __future__ import annotations

import pytest

from src.common.config import load_config
from src.query.schema import Constraint, SearchQuery
from src.retrieval import evidence
from src.retrieval.evidence import Evidence, EvidenceError, EvidenceMatcher, Span, sentences


@pytest.fixture(scope="module")
def matcher():
    return EvidenceMatcher(load_config("concepts"))


def _include(concept, status="PRESENT"):
    return SearchQuery(include=[Constraint(concept=concept, status=status)], language="en")


def test_sentences_split_on_punctuation_not_newlines():
    text = "Low lung volumes limit evaluation.  There \nis prominence of the cardiac shadow."
    spans = sentences(text)
    assert [s.text for s in spans] == [
        "Low lung volumes limit evaluation.", "There \nis prominence of the cardiac shadow."]
    assert all(text[s.start:s.end] == s.text for s in spans)


def test_enumerators_are_merged_into_the_next_sentence():
    assert [s.text for s in sentences("1. Mild edema. 2. No effusion.")] == [
        "1. Mild edema.", "2. No effusion."]


def test_spans_are_exact_slices_on_messy_text():
    text = "No acute\r\nprocess.   Stable  lines. 2."
    spans = sentences(text)
    assert [s.text for s in spans] == ["No acute\r\nprocess.", "Stable  lines."]
    assert all(text[s.start:s.end] == s.text for s in spans)


def test_polarity_from_cues(matcher):
    q = SearchQuery(include=[Constraint(concept="Edema", status="PRESENT")],
                    exclude=["Pleural Effusion"], language="en")
    ev = matcher.build("Mild pulmonary edema.  No \npleural effusion.", q)
    assert [(e.concept, e.polarity, e.text) for e in ev] == [
        ("Edema", "affirmed", "Mild pulmonary edema."),
        ("Pleural Effusion", "negated", "No \npleural effusion.")]


def test_report_terms_reach_the_matcher(matcher):
    ev = matcher.build("There is a 5 mm nodule in the right upper lobe.", _include("Lung Lesion"))
    assert ev[0].polarity == "affirmed"


def test_no_finding_phrase_is_affirmed(matcher):
    ev = matcher.build("No acute cardiopulmonary abnormality.", _include("No Finding"))
    assert ev[0].polarity == "affirmed"


def test_uncertain_and_preferred_polarity_first(matcher):
    ev = matcher.build("Possible edema.  Edema has worsened.", _include("Edema"))
    assert ev[0].text == "Edema has worsened." and ev[0].polarity == "affirmed"
    ev = matcher.build("Possible edema.  Edema has worsened.", _include("Edema", "UNCERTAIN"))
    assert ev[0].polarity == "uncertain"


def test_fallback_to_any_polarity_then_nothing(matcher):
    q = _include("Edema")
    assert matcher.build("No edema.", q)[0].polarity == "negated"
    assert matcher.build("Normal heart.", q) == []


def test_free_text_picks_the_sentence_with_most_matched_tokens(matcher):
    text = "Lungs are clear.  Small hiatal hernia is noted."
    ev = matcher.build(text, SearchQuery(language="en"),
                       free_text_tokens=["hiatal", "hernia", "noted"])
    assert ev == [Evidence(concept=None, text="Small hiatal hernia is noted.", start=18, end=47,
                           polarity="mention")]


def test_missing_text_gives_no_evidence(matcher):
    assert matcher.build("", _include("Edema")) == []


def test_verify_catches_a_bad_span(monkeypatch, matcher):
    monkeypatch.setattr(evidence, "sentences", lambda text: [Span(0, 4, "Mild")])
    with pytest.raises(EvidenceError):
        matcher.build("Edema.", _include("Edema"))
    assert matcher.build("Edema.", _include("Edema"), verify=False) == []
