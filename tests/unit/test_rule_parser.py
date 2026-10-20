"""The deterministic parser: baseline for the ablation and fallback for the LLM."""

from __future__ import annotations

import copy

import pytest

from src.common.config import load_config
from src.query.rule_parser import RuleParser, match, tokenize


@pytest.fixture(scope="module")
def rp():
    return RuleParser(load_config("concepts"))


def parsed(rp, text):
    q = rp.parse(text)
    return ([(c.concept, c.status) for c in q.include], q.exclude, q.views, q.language)


class TestTokenize:
    def test_casefolds_nfc_and_keeps_clause_punctuation_as_tokens(self):
        assert tokenize("Edema, no  Effusion.") == ["edema", ",", "no", "effusion", "."]

    def test_vietnamese_diacritics_survive(self):
        assert tokenize("Tràn Dịch") == ["tràn", "dịch"]

    def test_stray_punctuation_is_dropped_but_hyphens_stay(self):
        assert tokenize("Is there pneumothorax?") == ["is", "there", "pneumothorax"]
        assert tokenize("(edema): x-ray") == ["edema", "x-ray"]


class TestMatch:
    def test_longest_form_wins_and_hits_do_not_overlap(self):
        forms = [(("tràn", "khí", "màng", "phổi"), "Pneumothorax", "vi"),
                 (("tràn", "khí"), "Pneumothorax", "vi"), (("phổi",), "X", "vi")]
        hits = match(tokenize("tràn khí màng phổi"), forms)
        assert [(h.start, h.end, h.key) for h in hits] == [(0, 4, "Pneumothorax")]


class TestSingleAndMulti:
    @pytest.mark.parametrize("text, concept", [
        ("Find chest X-rays with cardiomegaly.", "Cardiomegaly"),
        ("enlarged heart", "Cardiomegaly"),
        ("show me pneumothorax cases", "Pneumothorax"),
        ("tìm ảnh có tràn khí màng phổi", "Pneumothorax"),
        ("phim có bóng tim to", "Cardiomegaly"),
    ])
    def test_one_concept_is_included_as_present(self, rp, text, concept):
        include, exclude, views, _ = parsed(rp, text)
        assert include == [(concept, "PRESENT")] and exclude == [] and views == []

    def test_a_trailing_question_mark_does_not_hide_the_concept(self, rp):
        assert parsed(rp, "Is there pneumothorax?")[0] == [("Pneumothorax", "PRESENT")]

    def test_several_concepts_keep_mention_order(self, rp):
        include, _, _, _ = parsed(rp, "cardiomegaly with pulmonary edema and atelectasis")
        assert include == [("Cardiomegaly", "PRESENT"), ("Edema", "PRESENT"),
                           ("Atelectasis", "PRESENT")]

    def test_no_finding_is_a_concept_not_a_negation(self, rp):
        include, exclude, _, _ = parsed(rp, "no finding")
        assert include == [("No Finding", "PRESENT")] and exclude == []
        include, exclude, _, _ = parsed(rp, "no acute abnormality")
        assert include == [("No Finding", "PRESENT")] and exclude == []

    def test_nothing_known_parses_to_an_empty_query(self, rp):
        assert parsed(rp, "dense white area near the hilum") == ([], [], [], "en")


class TestNegation:
    @pytest.mark.parametrize("text", [
        "edema with no pleural effusion",
        "edema without pleural effusion",
        "edema, negative for pleural effusion",
        "edema and no evidence of effusion",
        "edema; pleural effusion ruled out",
        "edema but effusion is absent",
        "phù phổi nhưng không có tràn dịch màng phổi",
        "phù phổi, không kèm tràn dịch",
        "phù phổi, tràn dịch đã hết",
    ])
    def test_negated_concept_goes_to_exclude_and_the_other_stays(self, rp, text):
        assert parsed(rp, text)[:2] == ([("Edema", "PRESENT")], ["Pleural Effusion"])

    def test_scope_crosses_or_and_and(self, rp):
        assert parsed(rp, "no pleural effusion or pneumothorax")[1] == [
            "Pleural Effusion", "Pneumothorax"]
        assert parsed(rp, "không có tràn dịch và tràn khí")[1] == [
            "Pleural Effusion", "Pneumothorax"]

    @pytest.mark.parametrize("text", [
        "no pleural effusion but cardiomegaly",
        "no pleural effusion, cardiomegaly",
        "no pleural effusion with cardiomegaly",
        "không thấy tràn dịch, có tim to",
    ])
    def test_scope_stops_at_a_terminator(self, rp, text):
        include, exclude, _, _ = parsed(rp, text)
        assert include == [("Cardiomegaly", "PRESENT")] and exclude == ["Pleural Effusion"]

    def test_cue_too_far_before_does_not_count(self, rp):
        include, exclude, _, _ = parsed(rp, "no one two three four five six cardiomegaly")
        assert include == [("Cardiomegaly", "PRESENT")] and exclude == []

    def test_post_cue_does_not_reach_backwards_over_a_concept(self, rp):
        include, exclude, _, _ = parsed(rp, "cardiomegaly without effusion")
        assert include == [("Cardiomegaly", "PRESENT")] and exclude == ["Pleural Effusion"]

    def test_rule_out_is_a_negation_cue(self, rp):
        assert parsed(rp, "rule out pneumothorax")[:2] == ([], ["Pneumothorax"])
        assert parsed(rp, "edema, rule out pleural effusion")[:2] == (
            [("Edema", "PRESENT")], ["Pleural Effusion"])

    def test_last_mention_wins(self, rp):
        assert parsed(rp, "effusion present, actually no effusion")[:2] == (
            [], ["Pleural Effusion"])
        assert parsed(rp, "no effusion, actually effusion")[:2] == (
            [("Pleural Effusion", "PRESENT")], [])


class TestUncertainty:
    @pytest.mark.parametrize("text", [
        "possible pneumonia", "suspected pneumonia", "pneumonia is uncertain",
        "pneumonia cannot be excluded", "cannot exclude pneumonia", "questionable pneumonia",
        "nghi ngờ viêm phổi", "viêm phổi chưa rõ", "không loại trừ viêm phổi",
    ])
    def test_uncertain_concept_is_included_with_status_uncertain(self, rp, text):
        assert parsed(rp, text)[:2] == ([("Pneumonia", "UNCERTAIN")], [])

    def test_nearest_cue_wins_between_uncertainty_and_negation(self, rp):
        include, exclude, _, _ = parsed(rp, "possible pneumonia and no pleural effusion")
        assert include == [("Pneumonia", "UNCERTAIN")] and exclude == ["Pleural Effusion"]
        include, exclude, _, _ = parsed(rp, "no effusion, pneumonia cannot be excluded")
        assert include == [("Pneumonia", "UNCERTAIN")] and exclude == ["Pleural Effusion"]

    def test_uncertainty_does_not_leak_past_a_terminator(self, rp):
        include, _, _, _ = parsed(rp, "possible pneumonia with cardiomegaly")
        assert include == [("Pneumonia", "UNCERTAIN"), ("Cardiomegaly", "PRESENT")]


class TestViewsAndLanguage:
    @pytest.mark.parametrize("text, views", [
        ("show frontal PA images with lung opacity", ["PA"]),
        ("AP view with edema", ["AP"]),
        ("anteroposterior or posteroanterior films with edema", ["AP", "PA"]),
        ("frontal chest x-ray with edema", []),
        ("phim thẳng tư thế PA có phù phổi", ["PA"]),
        ("ảnh chụp thẳng có phù phổi", []),
        ("AP/PA films with edema", ["AP", "PA"]),
    ])
    def test_views(self, rp, text, views):
        assert parsed(rp, text)[2] == views

    @pytest.mark.parametrize("text, language", [
        ("find cardiomegaly", "en"), ("tìm tim to", "vi"), ("tim to", "vi"), ("PA", "en"),
    ])
    def test_language(self, rp, text, language):
        assert parsed(rp, text)[3] == language

    def test_never_raises_on_garbage(self, rp):
        for text in ["", "   ", "!!!", "no no no", "with with", ",,,"]:
            assert rp.parse(text).include == []

    def test_semantic_query_is_left_empty_for_the_parser_layer(self, rp):
        assert rp.parse("cardiomegaly").semantic_query == ""


class TestBlockedPhrases:
    def test_phu_inside_phu_hop_is_not_edema(self, rp):
        q = rp.parse("tìm ca phù hợp viêm phổi")
        assert [c.concept for c in q.include] == ["Pneumonia"] and q.exclude == []

    def test_phu_phoi_before_phu_hop_still_matches(self, rp):
        q = rp.parse("phù phổi phù hợp với suy tim")
        assert [c.concept for c in q.include] == ["Edema"]

    def test_blocked_mask_leaves_a_form_that_only_touches_it_alone(self):
        forms = [(("phù",), "Edema", "vi")]
        tokens = tokenize("phù hợp phù")
        assert [h.start for h in match(tokens, forms, blocked=[("phù", "hợp")])] == [2]
        assert [h.start for h in match(tokens, forms)] == [0, 2]

    def test_config_without_blocked_phrases_still_parses(self):
        cfg = copy.deepcopy(load_config("concepts"))
        del cfg["blocked_phrases"]
        assert RuleParser(cfg).parse("phù").include[0].concept == "Edema"
