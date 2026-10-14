"""One real parse through Ollama. `uv run pytest -m llm`; skipped by default."""

from __future__ import annotations

import pytest

from src.api.dependencies import build_query_parser
from src.common.config import Settings

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def parser():
    parser, status = build_query_parser(Settings(_env_file=None))
    if status != "ok":
        pytest.skip(f"ollama {status}")
    return parser


@pytest.mark.xfail(strict=True, reason="WORKLOG T17: prompt c975e4ca returns include Edema-first "
                                       "and language 'en'")
def test_use_case_six_through_the_llm(parser):
    text = "Tìm ảnh X-quang thẳng có tim to và phù phổi nhưng không có tràn dịch màng phổi."
    result = parser.parse(text, mode="llm")
    assert [c.concept for c in result.query.include] == ["Cardiomegaly", "Edema"]
    assert all(c.status == "PRESENT" for c in result.query.include)
    assert result.query.exclude == ["Pleural Effusion"]
    assert result.query.views == [] and result.query.language == "vi"
    assert result.query.semantic_query == "chest x-ray showing cardiomegaly and pulmonary edema"
    assert result.llm_ms is not None and result.llm_ms < 20_000


def test_uncertainty_and_view_through_the_llm(parser):
    result = parser.parse("PA films where pneumonia cannot be excluded", mode="llm")
    assert [(c.concept, c.status) for c in result.query.include] == [("Pneumonia", "UNCERTAIN")]
    assert result.query.views == ["PA"] and result.query.exclude == []
