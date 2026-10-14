"""parse(): cache -> LLM (retry once) -> rule parser, and the semantic_query rule (spec 4.2)."""

from __future__ import annotations

import json

import pytest

from src.common.config import load_config
from src.query import llm_parser as llm
from src.query.parser import QueryParser, clean
from src.query.rule_parser import RuleParser
from src.query.schema import ParseError
from tests.unit.fakes import FakeOllama

TEMPLATE = "chest x-ray showing {}"


def output(findings="", include=(), exclude=(), views=(), language="en") -> str:
    return json.dumps({"findings": findings,
                       "include": [{"concept": c, "status": s} for c, s in include],
                       "exclude": list(exclude), "views": list(views), "language": language})


@pytest.fixture(scope="module")
def rule():
    return RuleParser(load_config("concepts"))


def parser(rule, *replies, **kwargs) -> QueryParser:
    client = FakeOllama("ok", replies) if replies else None
    return QueryParser(rule, template=TEMPLATE, llm_client=client, system_prompt="SYS", **kwargs)


def concepts(result):
    return [c.concept for c in result.query.include]


class TestClean:
    def test_keeps_case_collapses_whitespace(self):
        assert clean("  Find  PA\tfilms ") == "Find PA films"


class TestLLMPath:
    def test_valid_reply_is_used_and_semantic_query_comes_from_the_template(self, rule):
        p = parser(rule, output(include=[("edema", "PRESENT")], exclude=["Pleural Effusion"]))
        text = "Find edema with no pleural effusion"
        r = p.parse(text)
        assert r.parser == "llm" and not r.fallback_used and r.warnings == ()
        assert concepts(r) == ["Edema"] and r.query.exclude == ["Pleural Effusion"]
        assert r.query.semantic_query == "chest x-ray showing pulmonary edema"
        assert r.llm_ms is not None and r.llm_ms >= 0
        assert p.llm.calls[0] == [{"role": "system", "content": "SYS"},
                                  {"role": "user", "content": text}]

    def test_invalid_reply_is_retried_once_with_the_error_fed_back(self, rule):
        p = parser(rule, "{not json", output(include=[("Cardiomegaly", "PRESENT")]))
        r = p.parse("cardiomegaly")
        assert r.parser == "llm" and r.warnings == ("llm retried 1x",)
        second = p.llm.calls[1]
        assert second[2] == {"role": "assistant", "content": "{not json"}
        assert second[3]["role"] == "user" and "not JSON" in second[3]["content"]

    def test_two_invalid_replies_fall_back_to_the_rule_parser(self, rule):
        p = parser(rule, "{bad", output(include=[("Cardiomegalyy", "PRESENT")]))
        r = p.parse("cardiomegaly")
        assert r.parser == "rule" and r.fallback_used and concepts(r) == ["Cardiomegaly"]
        assert r.warnings[0].startswith("llm output invalid after 2 attempts")
        assert "unknown concept" in r.warnings[0] and len(p.llm.calls) == 2

    def test_malformed_chat_body_is_retried_then_used(self, rule):
        p = parser(rule, llm.LLMInvalidOutput("malformed chat response: not json"),
                   output(include=[("Edema", "PRESENT")]))
        r = p.parse("edema")
        assert r.parser == "llm" and r.warnings == ("llm retried 1x",)
        assert len(p.llm.calls) == 2 and p.llm.calls[1] == p.llm.calls[0]

    def test_two_malformed_chat_bodies_fall_back(self, rule):
        bad = llm.LLMInvalidOutput("malformed chat response: not json")
        r = parser(rule, bad, bad).parse("cardiomegaly")
        assert r.parser == "rule" and r.fallback_used
        assert r.warnings[0].startswith("llm output invalid after 2 attempts")

    def test_a_concept_in_both_lists_counts_as_invalid(self, rule):
        both = output(include=[("Edema", "PRESENT")], exclude=["edema"])
        r = parser(rule, both, both).parse("edema")
        assert r.parser == "rule" and r.fallback_used
        assert "both include and exclude" in r.warnings[0]

    def test_transport_failure_falls_back_without_retry(self, rule):
        p = parser(rule, llm.LLMError("ConnectError: refused"))
        r = p.parse("cardiomegaly")
        assert r.parser == "rule" and r.fallback_used and len(p.llm.calls) == 1
        assert r.warnings == ("llm unreachable: ConnectError: refused",)

    def test_mode_llm_raises_instead_of_falling_back(self, rule):
        with pytest.raises(ParseError, match="invalid"):
            parser(rule, "{bad", "{bad").parse("cardiomegaly", mode="llm")
        with pytest.raises(ParseError, match="unreachable"):
            parser(rule, llm.LLMError("down")).parse("cardiomegaly", mode="llm")
        with pytest.raises(ParseError, match="not configured"):
            parser(rule).parse("cardiomegaly", mode="llm")

    def test_mode_rule_never_calls_the_llm(self, rule):
        p = parser(rule, output(include=[("Edema", "PRESENT")]))
        r = p.parse("cardiomegaly", mode="rule")
        assert r.parser == "rule" and not r.fallback_used and p.llm.calls == []
        assert concepts(r) == ["Cardiomegaly"]

    def test_overlong_findings_are_truncated_not_rejected(self, rule):
        r = parser(rule, output(findings="x" * 400)).parse("something unknown")
        assert r.parser == "llm" and len(r.query.semantic_query) == 300


class TestNoLLM:
    def test_rule_only_with_the_startup_warning(self, rule):
        p = parser(rule, startup_warning="llm unreachable at startup; rule parser only")
        r = p.parse("cardiomegaly")
        assert r.parser == "rule" and not r.fallback_used
        assert r.warnings == ("llm unreachable at startup; rule parser only",)
        assert p.parse("cardiomegaly", mode="rule").warnings == ()
        assert parser(rule).parse("cardiomegaly").warnings == ()


class TestSemanticQuery:
    def test_concepts_join_with_and_in_mention_order(self, rule):
        r = parser(rule).parse("cardiomegaly and pulmonary edema but no effusion")
        assert r.query.semantic_query == "chest x-ray showing cardiomegaly and pulmonary edema"

    def test_uncertain_concepts_are_part_of_the_phrase(self, rule):
        assert parser(rule).parse("possible pneumonia").query.semantic_query == (
            "chest x-ray showing pneumonia")

    def test_llm_findings_are_used_only_when_nothing_was_included(self, rule):
        r = parser(rule, output(findings="a hazy left lung base")).parse("hazy left lung base")
        assert r.query.semantic_query == "a hazy left lung base" and r.warnings == ()
        r = parser(rule, output(findings="whatever", include=[("Edema", "PRESENT")])).parse("x")
        assert r.query.semantic_query == "chest x-ray showing pulmonary edema"

    def test_llm_with_no_findings_and_no_include_warns(self, rule):
        r = parser(rule, output(findings="", exclude=["Pleural Effusion"])).parse("no effusion")
        assert r.query.semantic_query == "" and "vector leg skipped" in r.warnings[0]

    def test_rule_english_without_a_concept_falls_back_to_the_raw_text(self, rule):
        r = parser(rule).parse("  dense  white area near the hilum ")
        assert r.query.semantic_query == "dense white area near the hilum"
        assert r.warnings == ("no known concept; vector leg encodes the raw text",)

    def test_rule_english_exclude_only_warns_about_no_positive_finding(self, rule):
        r = parser(rule).parse("no pleural effusion")
        assert r.query.exclude == ["Pleural Effusion"] and r.query.include == []
        assert r.query.semantic_query == "no pleural effusion"
        assert r.warnings == ("no positive finding; vector leg encodes the raw text",)

    def test_rule_vietnamese_without_a_concept_leaves_it_empty(self, rule):
        r = parser(rule).parse("vùng mờ gần rốn phổi")
        assert r.query.semantic_query == "" and r.query.language == "vi"
        assert "Vietnamese" in r.warnings[0]


class TestCache:
    def test_same_text_is_served_from_the_cache(self, rule):
        p = parser(rule, output(include=[("Edema", "PRESENT")]))
        first, second = p.parse("edema"), p.parse("  edema ")
        assert not first.cached and second.cached and len(p.llm.calls) == 1
        assert second.query == first.query and second.parser == "llm"

    def test_fallback_results_are_not_cached(self, rule):
        # A transient LLM failure must not pin the rule parse for that text.
        p = parser(rule, llm.LLMError("down"), output(include=[("Edema", "PRESENT")]))
        first = p.parse("edema")
        assert first.parser == "rule" and first.fallback_used
        second = p.parse("edema")
        assert second.parser == "llm" and not second.cached and len(p.llm.calls) == 2

    def test_cache_hits_report_no_llm_time(self, rule):
        p = parser(rule, output(include=[("Edema", "PRESENT")]))
        first, second = p.parse("edema"), p.parse("edema")
        assert first.llm_ms is not None
        assert second.cached and second.llm_ms is None

    def test_mode_is_part_of_the_key(self, rule):
        p = parser(rule, output(include=[("Edema", "PRESENT")]))
        p.parse("edema", mode="rule")
        assert not p.parse("edema").cached and len(p.llm.calls) == 1

    def test_cache_is_bounded(self, rule):
        p = parser(rule, cache_size=2)
        for text in ("a", "b", "c"):
            p.parse(text)
        assert not p.parse("a").cached and p.parse("c").cached

    def test_prompt_hash_is_exposed_for_reports(self, rule):
        assert len(parser(rule).prompt_sha256) == 64
