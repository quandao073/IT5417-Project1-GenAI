"""The comparison screen of the demo shows the evaluation's ranks: B0 and B1 of `/api/v1/search`
against `Runner.run_b0` / `Runner.run_b1` of `cxr eval-kg`, on three dev questions (one in
Vietnamese).

Needs the served build, Qdrant (`docker compose up -d qdrant`), Neo4j and an Anthropic key:
`uv run pytest -m "integration and llm" tests/integration/test_demo_matches_eval.py`.
The two sides share one parser object, so B1 compares one parse (a language model gives another
parse on another call, T18). B0's translation is the API's, handed to the evaluation's caller: the
test checks that the API translates exactly the questions the benchmark marks as not English, and
that the same English ranks the same studies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.integration, pytest.mark.llm]

QUESTIONS = ("kg-control-01", "kg-control-04", "kg-synonym_vi-01")   # en, vi, en
K = 10


class _Ledger:
    def add_client_delta(self, *a, **k) -> dict:
        return {}


class _Caller:
    """The evaluation's translator caller, answering with the English the API used."""

    def __init__(self):
        self.english: dict[str, str] = {}
        self.asked: list[str] = []

    def call(self, step, system, sha, payload, user, schema) -> dict:
        self.asked.append(payload["question"])
        return {"text": self.english[payload["question"]]}


@pytest.fixture(scope="module")
def world():
    from src.api.dependencies import build_state
    from src.api.main import create_app
    from src.common.config import Settings
    from src.evaluation import kg_eval as ke
    from src.evaluation.vector_baseline import _all_vectors

    settings = Settings()
    state = build_state(settings)
    if not state.ready:
        pytest.skip(f"API not ready: {state.not_ready_reason}")
    if state.parser.llm is None or state.kg is None or state.kg.translator is None:
        pytest.skip("needs the B1 language model and the KG translator")
    vectors, ids = _all_vectors(Path(settings.indexes_dir) / "embeddings"
                                / state.manifest["build_id"])
    images = pl.read_parquet(Path(settings.canonical_dir) / "images.parquet")
    study, view = (dict(zip(images["image_id"], images[c], strict=True))
                   for c in ("study_id", "projection"))
    corpus = ke.Corpus(ke.unit_rows(vectors), ids, [study[i] for i in ids],
                       np.array([view[i] for i in ids]))
    rows = {r["id"]: r for r in map(json.loads, Path("benchmarks/kg_queries_v1.jsonl")
                                    .read_text(encoding="utf-8").splitlines())}
    return TestClient(create_app(state)), state, corpus, rows


def _runner(state, corpus, caller, tmp_path):
    from src.evaluation import kg_eval_run as kr

    return kr.Runner(rows=[], split="dev", ecfg={"eval": {"top_k": K}}, work=tmp_path,
                     ledger=_Ledger(), corpus=corpus, encoder=state.encoder,
                     b1_parser=state.parser, search=None, translator=caller, t_text="t",
                     t_sha="s", label_matrix=None, label_of_cui=None, labels_cfg=None,
                     concepts_cfg=state.concepts_cfg)


@pytest.mark.parametrize("qid", QUESTIONS)
def test_b0_and_b1_of_the_demo_rank_like_the_evaluation(world, qid, tmp_path):
    client, state, corpus, rows = world
    row = rows[qid]
    caller = _Caller()
    runner = _runner(state, corpus, caller, tmp_path)

    b1 = client.post("/api/v1/search", json={"query": row["question"], "top_k": K}).json()
    want = runner.run_b1(row)
    assert b1["query_text"] == want["text"]
    assert [r["study_id"] for r in b1["results"]] == want["studies"]

    b0 = client.post("/api/v1/search", json={"query": row["question"], "top_k": K,
                                             "system": "B0"}).json()
    translated = b0["query_text"] != row["question"]
    assert translated == (row["lang"] != "en"), "the API translates what the benchmark marks VI"
    caller.english[row["question"]] = b0["query_text"]
    want = runner.run_b0(row)
    assert caller.asked == ([row["question"]] if row["lang"] != "en" else [])
    assert b0["query_text"] == want["text"]
    assert [r["study_id"] for r in b0["results"]] == want["studies"]
