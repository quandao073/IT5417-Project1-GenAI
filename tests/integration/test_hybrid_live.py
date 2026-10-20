"""Live Phase 5 checks on the real corpus: a 50-image hybrid round trip and evidence faithfulness.

Run with `uv run pytest -m integration` while `docker compose up -d qdrant` is up. The round trip
reuses the throwaway 50-image build of test_vector_index.py (same build id, so its `qdrant`
fixture deletes the collection afterwards) plus a temporary report index over those 50 studies.
The faithfulness check reads data/canonical only; `-s` shows its per-concept coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from src.api.dependencies import build_state
from src.common.config import Settings, config_hash, load_config
from src.data.labels import CHEXPERT_LABELS, resolve
from src.embeddings import embed, index
from src.query.schema import Constraint, SearchQuery
from src.retrieval.evidence import EvidenceMatcher
from src.retrieval.label_filter import satisfies
from src.retrieval.pipeline import run_search
from src.retrieval.report_index import build_report_index
from tests.integration.test_vector_index import BUILD, qdrant  # noqa: F401 - pytest fixture

pytestmark = pytest.mark.integration
CANONICAL = Path("data/canonical")
POLICY = {"absence_policy": "exclude_present", "uncertainty_policy": "EXCLUDE"}


def _subset_world(client, tmp_path: Path, encoder) -> tuple[Path, Path, pl.DataFrame]:
    """Embed and index the first 50 images; write their canonical rows and a manifest."""
    images = pl.read_parquet(CANONICAL / "images.parquet").sort("image_id").head(50)
    shards = tmp_path / BUILD
    embed.run_embed(images, encoder, Path("data/raw/CheXpert-v1.0-small"), shards, BUILD,
                    shard_size=20, batch_size=16, library_versions={}, git_commit=None)
    built = index.run_index(client, shards, images, "cxr_images", recreate=True,
                            config_hash=config_hash(), git_commit=None, dataset={},
                            library_versions={})
    canonical, indexes = tmp_path / "canonical", tmp_path / "indexes"
    canonical.mkdir()
    indexes.mkdir()
    (indexes / index.MANIFEST_FILE).write_text(json.dumps(built), encoding="utf-8")
    images.write_parquet(canonical / "images.parquet")
    studies = images["study_id"].unique().to_list()
    for table in ("studies", "labels", "reports"):
        (pl.read_parquet(CANONICAL / f"{table}.parquet")
         .filter(pl.col("study_id").is_in(studies))
         .write_parquet(canonical / f"{table}.parquet"))
    return canonical, indexes, pl.read_parquet(canonical / "reports.parquet")


def test_hybrid_fifty_images_round_trip(qdrant, tmp_path):  # noqa: F811 - the fixture
    from src.embeddings.base import load_encoder

    cfg = load_config("models")["image_encoder"]
    encoder = load_encoder(cfg["selected"], cfg["candidates"][cfg["selected"]], "cpu")
    canonical, indexes, reports = _subset_world(qdrant, tmp_path, encoder)
    fts = tmp_path / "report_search.sqlite"
    build_report_index(reports, fts, concepts_sha256="itest", git_commit=None)
    settings = Settings(_env_file=None, qdrant_url=Settings().qdrant_url, canonical_dir=canonical,
                        indexes_dir=indexes, fts_db_path=fts, llm_disabled=True)
    state = build_state(settings, encoder_loader=lambda *_: encoder)
    try:
        assert state.ready, state.not_ready_reason
        assert state.report_status == "ok", state.report_reason
        query = state.parser.parse("edema with no pleural effusion", mode="rule").query
        assert [c.concept for c in query.include] == ["Edema"]
        assert query.exclude == ["Pleural Effusion"]

        outcome = run_search(state, query, "hybrid", top_k=50, views=query.views, **POLICY)
        expected = [s for s in state.labels.study_ids.tolist()
                    if satisfies(state.labels, s, query.include, query.exclude, **POLICY)]
        assert expected, "no edema-without-effusion study among the 50"
        assert outcome.total_matches == len(expected)
        ids = [h.study_id for h in outcome.hits]
        assert all(satisfies(state.labels, s, query.include, query.exclude, **POLICY)
                   for s in ids)
        # vector_top_k (100) covers the whole 50-study collection, so every allowed study
        # comes back ranked.
        assert sorted(ids) == sorted(expected)
        text_of = dict(zip(reports["study_id"], reports["search_text"], strict=True))
        quotes = [(h.study_id, e) for h in outcome.hits for e in h.evidence]
        assert quotes, "no hit carries evidence"
        assert all(text_of[s][e.start:e.end] == e.text for s, e in quotes)
        assert outcome.warnings == []
    finally:
        if state.report_db is not None:
            state.report_db.close()


def test_evidence_is_faithful_on_a_thousand_real_reports():
    reports = pl.read_parquet(CANONICAL / "reports.parquet").head(1000)
    ids = reports["study_id"].to_list()
    labels = pl.read_parquet(CANONICAL / "labels.parquet").filter(pl.col("study_id").is_in(ids))
    priority = load_config("retrieval")["label_resolution"]["source_priority"]
    present = (resolve(labels, priority).filter(pl.col("status") == "PRESENT")
               .group_by("study_id").agg(pl.col("concept")))
    text_of = dict(zip(ids, reports["search_text"].to_list(), strict=True))
    matcher = EvidenceMatcher(load_config("concepts"))
    counts = {concept: [0, 0] for concept in CHEXPERT_LABELS}   # [present, with evidence]
    for study_id, concepts in present.iter_rows():
        text = text_of[study_id]
        query = SearchQuery(include=[Constraint(concept=c, status="PRESENT") for c in concepts],
                            language="en")
        evidence = matcher.build(text, query)
        assert all(text[e.start:e.end] == e.text for e in evidence), study_id
        quoted = {e.concept for e in evidence}
        for concept in concepts:
            counts[concept][0] += 1
            counts[concept][1] += concept in quoted
    for concept, (n, m) in counts.items():
        print(f"concept={concept} present={n} with_evidence={m}")
    assert sum(n for n, _ in counts.values()) > 0
