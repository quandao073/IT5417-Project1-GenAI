"""What evidence reads from the report index: free-text tokens and the verbatim text."""

from __future__ import annotations

import threading

import pytest

from src.query.schema import SearchQuery
from src.retrieval.report_index import build_report_index, open_report_index
from src.retrieval.report_search import ReportSearchError, fetch_texts, free_text_tokens
from tests.unit.test_report_index import reports_frame

TEXT_A = "Large effusion.  The effusion is new and a third effusion is seen.\nNo edema."


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "r.sqlite"
    build_report_index(reports_frame([("A", TEXT_A), ("B", "Small effusion on the right.")]),
                       path, concepts_sha256="c" * 64, git_commit=None)
    con = open_report_index(path)
    yield con, threading.Lock()
    con.close()


def test_free_text_tokens_are_in_order_without_duplicates_or_short_words():
    q = SearchQuery(semantic_query="studies showing a hiatal hernia", language="en")
    assert free_text_tokens(q) == ["studies", "showing", "hiatal", "hernia"]
    dup = SearchQuery(semantic_query="Hernia, hiatal HERNIA. hiatal", language="en")
    assert free_text_tokens(dup) == ["hernia", "hiatal"]


def test_free_text_tokens_drop_words_that_carry_nothing():
    q = SearchQuery(semantic_query='edema AND NEAR(x) OR "quote" not', language="en")
    assert free_text_tokens(q) == ["edema", "quote"]
    assert free_text_tokens(SearchQuery(language="vi")) == []


def test_fetch_texts_returns_verbatim_and_skips_unknown(index):
    con, lock = index
    assert fetch_texts(con, lock, ["A", "nope"]) == {"A": TEXT_A}
    assert fetch_texts(con, lock, []) == {}


def test_a_closed_index_is_a_report_search_error(index):
    con, lock = index
    con.close()
    with pytest.raises(ReportSearchError):
        fetch_texts(con, lock, ["A"])
