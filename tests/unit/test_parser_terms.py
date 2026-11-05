"""`terms` of the KG parser prompt (query_parser_v2): schema, query model, parser, renderer."""

from __future__ import annotations

import json

import pytest

from src.common.config import load_config
from src.query import llm_parser as llm
from src.query.parser import QueryParser
from src.query.rule_parser import RuleParser
from src.query.schema import SearchQuery, Term


def test_the_schema_has_terms_only_when_asked():
    assert "terms" not in llm.output_schema()["properties"]
    schema = llm.output_schema(with_terms=True)
    assert "terms" in schema["required"]
    assert schema["properties"]["terms"]["items"]["properties"]["polarity"]["enum"] == [
        "PRESENT", "ABSENT"]


def test_a_query_keeps_distinct_terms_and_refuses_a_bad_polarity():
    q = SearchQuery(language="vi", terms=[
        {"text": "suy  tim", "polarity": "PRESENT"}, {"text": "Suy tim", "polarity": "PRESENT"},
        {"text": "suy tim", "polarity": "ABSENT"}])
    assert [(t.text, t.polarity) for t in q.terms] == [("suy tim", "PRESENT"),
                                                       ("suy tim", "ABSENT")]
    with pytest.raises(ValueError):
        Term(text="x", polarity="MAYBE")
    assert SearchQuery(language="en").terms == []


class Scripted:
    model = "fake"

    def __init__(self, reply: dict):
        self.reply = reply

    def chat(self, messages, schema):
        self.schema = schema
        return json.dumps(self.reply), 1.0


REPLY = {"findings": "", "include": [], "exclude": ["Pneumothorax"], "views": ["PA"],
         "terms": [{"text": "suy tim", "polarity": "PRESENT"}], "language": "vi"}


def parser(with_terms: bool) -> QueryParser:
    return QueryParser(RuleParser(load_config("concepts")), template="x {}",
                       llm_client=Scripted(REPLY), system_prompt="p", with_terms=with_terms)


def test_the_kg_parser_passes_terms_through_and_the_plain_one_drops_them():
    kg = parser(True)
    got = kg.parse("nghi suy tim, không tràn khí, phim PA", mode="llm")
    assert [(t.text, t.polarity) for t in got.query.terms] == [("suy tim", "PRESENT")]
    assert got.query.exclude == ["Pneumothorax"] and got.query.views == ["PA"]
    assert "terms" in kg.llm.schema["properties"]
    plain = parser(False).parse("nghi suy tim, không tràn khí, phim PA", mode="llm")
    assert plain.query.terms == []


def test_ambiguous_forms_leave_the_concept_list_but_not_their_neighbours():
    cfg = load_config("concepts")
    text = llm.render_prompt("{{CONCEPTS}}", cfg, ("giảm thể tích phổi",))
    atelectasis = next(line for line in text.splitlines() if line.startswith("- Atelectasis:"))
    assert "giảm thể tích phổi" not in atelectasis and "xẹp phổi" in atelectasis
    assert "giảm thể tích phổi" in llm.render_prompt("{{CONCEPTS}}", cfg)


def test_the_kg_prompt_renders_and_every_example_parses_to_the_terms_schema():
    prompt = llm.load_prompt(load_config("models")["llm"]["kg_prompt_file"],
                             load_config("concepts"), ("giảm thể tích phổi",))
    assert "{{CONCEPTS}}" not in prompt and "Terms (phrases outside" in prompt
    json_lines = [ln for ln in prompt.splitlines() if ln.startswith('{"findings"')]
    assert len(json_lines) == len(llm.prompt_examples(prompt)) >= 20
    for line in json_lines:
        out = llm.parse_output(line)
        assert out.terms is not None
