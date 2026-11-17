"""The translator of the KG search on Vietnamese typed without tone marks, through the real model
(demo review 2026-11-17: the plain prompt read "suy tim" as "find"). Needs the KG aliases under
data/kg/graph and ANTHROPIC_API_KEY:
`uv run pytest -m llm tests/integration/test_translate_live.py`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.common.config import Settings, default_config_dir, load_config
from src.kg.guard import Guard
from src.kg.runtime import vi_ascii_phrases
from src.kg.translate import Translator, model_ask
from src.query import llm_parser as llm

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def translator():
    settings = Settings()
    aliases = Path(settings.kg_graph_dir) / "aliases.parquet"
    if settings.anthropic_api_key is None or not aliases.is_file():
        pytest.skip("needs ANTHROPIC_API_KEY and data/kg/graph")
    models = load_config("models")
    kl = models["kg_llm"]
    c = kl["translate"]
    client = llm.AnthropicClient(c["model"], c["timeout_seconds"],
                                 api_key=settings.anthropic_api_key.get_secret_value(),
                                 max_tokens=c["max_tokens"], max_retries=0,
                                 connect_timeout_s=kl["connect_timeout_seconds"],
                                 options=llm.request_options(models["llm"], c["model"]))
    guard = Guard(3, 60.0)

    def ask(key: str):
        prompt = (default_config_dir() / c[key]).read_text(encoding="utf-8")
        return model_ask(client, guard, prompt, 10)

    nodes = pl.read_parquet(aliases.parent / "nodes.parquet", columns=["cui", "name"])
    return Translator(ask("prompt_file"), vi_ascii_phrases(pl.read_parquet(aliases), nodes),
                      ask("unaccented_prompt_file"))


@pytest.mark.parametrize(("question", "want"), [("suy tim", "heart failure"),
                                                ("tran dich mang phoi", "pleural effusion")])
def test_unaccented_vietnamese_is_translated_by_meaning(translator, question, want):
    got = translator.english(question)
    assert got.called and got.error is None
    assert want in got.text.lower(), got.text
