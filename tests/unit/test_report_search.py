"""The FTS5 report leg: MATCH builder, bm25 inside an allowed set, verbatim text fetch."""

from __future__ import annotations

import threading

import pytest

from src.common.config import load_config
from src.query.schema import Constraint, SearchQuery
from src.retrieval.report_index import build_report_index, open_report_index
from src.retrieval.report_search import (
    ReportSearchError,
    build_match,
    fetch_texts,
    free_text_tokens,
    report_terms_from_config,
    search_reports,
)
from tests.unit.test_report_index import reports_frame

TERMS = {"Edema": ["pulmonary edema", "edema"],
         "Pleural Effusion": ["pleural effusion", "effusion"]}
TEXT_A = "Large effusion.  The effusion is new and a third effusion is seen.\nNo edema."
TEXT_B = "Small effusion on the right."
TEXT_C = "Clear lungs.  Normal heart size."


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "r.sqlite"
    build_report_index(reports_frame([("A", TEXT_A), ("B", TEXT_B), ("C", TEXT_C)]), path,
                       concepts_sha256="c" * 64, git_commit=None)
    con = open_report_index(path)
    yield con, threading.Lock()
    con.close()


def test_match_quotes_every_include_term_with_or():
    q = SearchQuery(include=[Constraint(concept="Edema", status="PRESENT")],
                    exclude=["Pleural Effusion"], language="en")
    assert build_match(q, TERMS) == '"pulmonary edema" OR "edema"'   # exclude never appears


def test_match_from_free_text_quotes_tokens_and_drops_short_ones():
    q = SearchQuery(semantic_query="studies showing a hiatal hernia", language="en")
    assert build_match(q, TERMS) == '"studies" OR "showing" OR "hiatal" OR "hernia"'


def test_free_text_tokens_are_the_quoted_tokens_in_order_without_duplicates():
    q = SearchQuery(semantic_query="studies showing a hiatal hernia", language="en")
    assert free_text_tokens(q) == ["studies", "showing", "hiatal", "hernia"]
    dup = SearchQuery(semantic_query="Hernia, hiatal HERNIA. hiatal", language="en")
    assert free_text_tokens(dup) == ["hernia", "hiatal"]
    assert build_match(dup, TERMS) == '"hernia" OR "hiatal"'


def test_match_neutralises_fts_syntax(index):
    con, lock = index
    q = SearchQuery(semantic_query='edema AND NEAR(x) OR "quote" col:val star*', language="en")
    m = build_match(q, TERMS)
    assert m is not None
    assert m.count('"') == 2 * m.count(" OR ") + 2          # every term is exactly one "..."
    assert not {"AND", "NEAR", "NOT"} & set(m.replace('"', " ").split())  # no bare operator
    assert [t for t in m.split(" OR ")] == [f'"{t}"' for t in free_text_tokens(q)]
    assert [h.study_id for h in search_reports(con, lock, m, None, 10)] == ["A"]


def test_match_is_none_when_no_token_survives():
    assert build_match(SearchQuery(semantic_query="AND OR *", language="en"), TERMS) is None
    assert build_match(SearchQuery(semantic_query="x y", language="en"), TERMS) is None
    assert build_match(SearchQuery(language="vi"), TERMS) is None


def test_search_ranks_by_bm25_and_breaks_ties_by_study_id(index):
    con, lock = index
    hits = search_reports(con, lock, '"effusion"', None, 10)
    assert [h.study_id for h in hits] == ["A", "B"]
    assert hits[0].score > hits[1].score and [h.rank for h in hits] == [1, 2]


def test_search_ties_resolve_by_study_id(tmp_path):
    path = tmp_path / "t.sqlite"
    build_report_index(reports_frame([("Z", "small effusion"), ("M", "small effusion"),
                                      ("Q", "small effusion")]), path,
                       concepts_sha256="c" * 64, git_commit=None)
    con = open_report_index(path)
    hits = search_reports(con, threading.Lock(), '"effusion"', None, 10)
    assert [h.study_id for h in hits] == ["M", "Q", "Z"]
    con.close()


def test_search_inside_allowed_set_only(index):
    con, lock = index
    assert [h.study_id for h in search_reports(con, lock, '"effusion"', ["B", "C"], 10)] == ["B"]
    assert search_reports(con, lock, '"effusion"', [], 10) == []
    # ranks are 1-based inside the restricted result, and reusing the connection is safe
    both = search_reports(con, lock, '"effusion"', ["A", "B"], 10)
    assert [(h.study_id, h.rank) for h in both] == [("A", 1), ("B", 2)]
    assert [h.study_id for h in search_reports(con, lock, '"effusion"', ["B"], 10)] == ["B"]


def test_search_candidate_k_and_malformed_match(index):
    con, lock = index
    assert len(search_reports(con, lock, '"effusion"', None, 1)) == 1
    with pytest.raises(ReportSearchError):
        search_reports(con, lock, "NEAR(", None, 10)
    with pytest.raises(ReportSearchError):
        search_reports(con, lock, "NEAR(", ["A"], 10)


def test_fetch_texts_returns_verbatim_and_skips_unknown(index):
    con, lock = index
    assert fetch_texts(con, lock, ["A", "nope"]) == {"A": TEXT_A}
    assert fetch_texts(con, lock, []) == {}


def test_report_terms_from_config_merges_nothing_and_keeps_order():
    cfg = load_config("concepts")
    terms = report_terms_from_config(cfg)
    assert list(terms) == list(cfg["concepts"]) and len(terms) == 14
    for concept, forms in cfg["concepts"].items():
        assert terms[concept] == list(forms["report"])
