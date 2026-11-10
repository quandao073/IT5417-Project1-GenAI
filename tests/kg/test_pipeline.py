"""KGSearch over the tiny KG: the routes, and what survives a lost language model or Neo4j."""

from __future__ import annotations

import time

from src.query.llm_parser import LLMError
from tests.kg.search_kit import Chat, DownStore, make_search, parse_reply


def codes(result):
    return [w.code for w in [*result.warnings, *[w for i in result.intents for w in i.warnings]]]


def test_a_label_and_a_term_are_scored_together_and_the_trace_says_what_was_decided():
    reply = parse_reply(include=["Cardiomegaly"], terms=[("suy tim", "PRESENT")])
    search = make_search(parser_chat=Chat(reply))
    got = search.search("tim to, nghi suy tim", top_k=3)
    assert [i.term for i in got.intents] == ["Cardiomegaly", "suy tim"]
    assert got.intents[1].kind == "disease" and len(got.rows) == 3
    assert [r["rank"] for r in got.rows] == [1, 2, 3]
    assert got.rows[0]["score"] >= got.rows[1]["score"] >= got.rows[2]["score"]
    assert got.trace["terms"][0]["choice"]["via"] == "deterministic"  # no chooser configured
    assert {"parse_ms", "labels_ms", "terms_ms", "score_ms", "total_ms"} <= set(got.timings)
    assert len(got.request_id) == 32 and not got.degraded


def test_a_phrase_the_graph_does_not_know_is_dropped_with_a_warning():
    reply = parse_reply(include=["Cardiomegaly"], terms=[("xyz qwerty", "PRESENT")])
    got = make_search(parser_chat=Chat(reply)).search("tim to xyz qwerty")
    assert "unrecognized" in codes(got) and len(got.rows) > 0
    assert [i.term for i in got.intents if i.columns] == ["Cardiomegaly"]


def test_a_repeated_concept_counts_once():
    reply = parse_reply(include=["Pleural Effusion"], terms=[("tràn dịch màng phổi", "PRESENT")])
    got = make_search(parser_chat=Chat(reply)).search("tràn dịch")
    assert len([i for i in got.intents if i.columns]) == 1


def test_a_lost_neo4j_still_returns_the_label_results_with_warnings():
    reply = parse_reply(include=["Cardiomegaly"], terms=[("suy tim", "PRESENT")])
    for store in (None, DownStore()):
        got = make_search(parser_chat=Chat(reply), store=store).search("tim to, suy tim")
        assert [i.term for i in got.intents] == ["Cardiomegaly"] and got.rows
        assert got.intents[0].kind == "direct"
        assert {"kg_unavailable", "terms_dropped"} <= set(codes(got))
        assert any("knowledge graph unavailable" in d for d in got.degraded)


def test_neo4j_failing_midway_is_handled_like_a_lost_neo4j():
    class Dies(DownStore):
        def __getattribute__(self, name):
            if name == "aliases_exact":
                return super().__getattribute__(name)      # the term lookup dies
            return object.__getattribute__(self, name)

    reply = parse_reply(include=["Cardiomegaly"], terms=[("suy tim", "PRESENT")])
    got = make_search(parser_chat=Chat(reply), store=Dies()).search("tim to, suy tim")
    assert got.rows and "kg_unavailable" in codes(got)


def test_when_the_language_model_fails_the_rule_parser_answers_in_under_a_second():
    for failure in (ConnectionError("refused"), LLMError("HTTP 503"), 30):   # 30: hangs
        search = make_search(parser_chat=Chat(failure), timeout=0.3)
        started = time.perf_counter()
        got = search.search("pleural effusion")
        assert time.perf_counter() - started < 1.0
        assert [i.term for i in got.intents] == ["Pleural Effusion"] and got.rows
        assert "llm_fallback" in codes(got) and got.degraded[0] == "parser: rule parser"


def test_after_three_failures_the_model_is_skipped_at_once():
    search = make_search(parser_chat=Chat(ConnectionError("refused")), timeout=0.3)
    for i in range(3):
        search.search(f"pleural effusion {i}")
    started = time.perf_counter()
    got = search.search("pleural effusion again")
    assert (time.perf_counter() - started) * 1000 < 200 and "llm_fallback" in codes(got)


def test_a_failing_chooser_keeps_the_search_and_says_so():
    reply = parse_reply(terms=[("giảm thể tích phổi hai bên", "PRESENT")])
    search = make_search(parser_chat=Chat(reply), choose_chat=Chat(LLMError("HTTP 500")))
    got = search.search("giảm thể tích phổi hai bên")
    assert got.intents[0].cui == "C12" and "chosen_deterministic" in codes(got)
    assert "side_ignored" in codes(got) and got.trace["terms"][0]["choice"]["error"]


def test_a_chooser_pick_outside_the_candidates_is_never_used():
    reply = parse_reply(terms=[("giảm thể tích phổi", "PRESENT")])
    chat = Chat({"choice": "C1", "reason": "x"})            # a real concept, but not offered
    got = make_search(parser_chat=Chat(reply), choose_chat=chat).search("giảm thể tích phổi")
    assert got.intents[0].cui == "C12"


def test_explain_is_separate_and_none_without_an_explainer():
    reply = parse_reply(include=["Cardiomegaly"])
    search = make_search(parser_chat=Chat(reply))
    assert search.explain(search.search("tim to")) is None


def test_the_explanation_gets_the_ranked_results_and_a_failing_model_does_not_raise():
    reply = parse_reply(include=["Cardiomegaly"])
    chat = Chat(LLMError("HTTP 529"))
    search = make_search(parser_chat=Chat(reply), explain_chat=chat)
    result = search.search("tim to", top_k=4)
    got = search.explain(result, top=2)
    assert got.status == "failed"
    sent = chat.calls[0][1]["content"]
    assert '"rank":2' in sent and '"rank":3' not in sent


def test_a_side_alone_is_not_a_term_and_a_page_without_results_is_not_explained():
    reply = parse_reply(include=["Cardiomegaly"], terms=[("bên phải", "PRESENT")])
    got = make_search(parser_chat=Chat(reply)).search("tim to bên phải")
    assert "unrecognized" not in codes(got) and "side_ignored" in codes(got)
    assert [i.term for i in got.intents] == ["Cardiomegaly"]
    chat = Chat({"summary": "x", "images": []})
    nothing = make_search(parser_chat=Chat(parse_reply(terms=[("xyz qwerty", "PRESENT")])),
                          explain_chat=chat)
    result = nothing.search("xyz qwerty")
    assert result.rows == [] and nothing.explain(result) is None and chat.calls == []


def test_a_location_beside_a_finding_only_warns_and_a_location_alone_is_scored():
    both = parse_reply(include=["Cardiomegaly"], terms=[("pleural space", "PRESENT")])
    got = make_search(parser_chat=Chat(both)).search("tim to ở khoang màng phổi")
    assert [i.term for i in got.intents] == ["Cardiomegaly"]
    assert "location_ignored" in codes(got) and got.trace["location_ignored"] == ["Pleural space"]
    alone = make_search(parser_chat=Chat(parse_reply(terms=[("pleural space", "PRESENT")])))
    got = alone.search("khoang màng phổi")
    assert [i.kind for i in got.intents] == ["anatomy"] and got.rows
    assert "location_ignored" not in codes(got)


def test_the_side_warning_comes_from_the_parser_or_from_unambiguous_words():
    reply = {**parse_reply(include=["Cardiomegaly"]), "sides": ["left"]}
    got = make_search(parser_chat=Chat(reply)).search("tim to")          # the question names none
    assert "side_ignored" in codes(got) and got.trace["parser"]["sides"] == ["left"]
    plain = make_search(parser_chat=Chat(parse_reply(include=["Cardiomegaly"])))
    assert "side_ignored" in codes(plain.search("tim to hai bên"))
    assert "side_ignored" not in codes(plain.search("phim không phải tim to"))   # "phải" alone
    rule = make_search().search("pleural effusion on the left")           # the 14-label route
    assert "side_ignored" in codes(rule) and rule.rows


def test_modifier_terms_are_listed_in_one_not_considered_warning():
    reply = parse_reply(include=["Cardiomegaly"], terms=[("small", "PRESENT"),
                                                         ("loculated", "PRESENT")])
    search = make_search(parser_chat=Chat(reply))
    search.cfg["query"]["modifiers"] = ["small", "loculated"]
    got = search.search("small loculated cardiomegaly")
    assert [i.term for i in got.intents] == ["Cardiomegaly"]
    assert codes(got).count("not_considered") == 1 and "unrecognized" not in codes(got)
    assert "Không xét: small, loculated." in got.warnings_text()           # a "vi" question


class _Encoder:
    """Every text points at the last matrix row."""

    def __init__(self, n):
        self.n, self.texts = n, []

    def encode_texts(self, texts):
        import numpy as np

        self.texts += texts
        return np.eye(self.n, dtype="float32")[[self.n - 1]]


def _fallback(search_matrix):
    import numpy as np

    from src.common.config import load_config
    from src.kg.text_fallback import TextFallback

    n = len(search_matrix.study_ids)
    return TextFallback(_Encoder(n), np.eye(n, dtype="float32"), search_matrix.views,
                        load_config("concepts"))


def test_a_question_the_graph_does_not_understand_is_ranked_by_text():
    from tests.kg.search_kit import matrix

    fb = _fallback(matrix())
    got = make_search(parser_chat=Chat(parse_reply(terms=[("xyz qwerty", "PRESENT")])),
                      text_fallback=fb).search("xyz qwerty", top_k=3)
    assert "text_fallback" in codes(got) and "unrecognized" in codes(got)
    assert got.rows and got.rows[0]["row"] == len(matrix().study_ids) - 1
    assert got.trace["fallback"]["route"] == "text"
    assert fb.encoder.texts == [got.query.semantic_query]       # no label: the free text


def test_the_text_route_is_not_taken_when_something_wanted_was_scored_or_only_negated():
    from tests.kg.search_kit import matrix

    reply = parse_reply(include=["Cardiomegaly"], terms=[("xyz qwerty", "PRESENT")])
    got = make_search(parser_chat=Chat(reply), text_fallback=_fallback(matrix())).search("x")
    assert "text_fallback" not in codes(got) and "fallback" not in got.trace
    only_no = parse_reply(exclude=["Pleural Effusion"])
    got = make_search(parser_chat=Chat(only_no), text_fallback=_fallback(matrix())).search("y")
    assert "text_fallback" not in codes(got) and got.rows
