"""End to end against a real Qdrant: embed 50 images, index, validate, search.

Run with `uv run pytest -m integration` while `docker compose up -d qdrant` is up.
Uses a throwaway build id and deletes its collection afterwards.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.common.config import Settings, config_hash, load_config
from src.embeddings import embed, index, validate
from src.retrieval.image_search import search_image_embeddings

pytestmark = pytest.mark.integration
BUILD = "19700101_chexpert25k_biomedclip_itest"


@pytest.fixture
def qdrant():
    from qdrant_client import QdrantClient

    client = QdrantClient(url=Settings().qdrant_url, timeout=30)
    try:
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"qdrant not reachable: {exc}")
    yield client
    name = index.collection_name("cxr_images", BUILD)
    if client.collection_exists(name):
        client.delete_collection(name)


def test_fifty_images_round_trip(qdrant, tmp_path):
    from src.embeddings.base import load_encoder

    cfg = load_config("models")["image_encoder"]
    encoder = load_encoder(cfg["selected"], cfg["candidates"][cfg["selected"]], "cpu")
    images = pl.read_parquet("data/canonical/images.parquet").sort("image_id").head(50)
    out = tmp_path / BUILD
    manifest = embed.run_embed(images, encoder, Path("data/raw/CheXpert-v1.0-small"), out, BUILD,
                               shard_size=20, batch_size=16, library_versions={}, git_commit=None)
    assert manifest["dim"] == encoder.dim and manifest["embedded"] == 50

    built = index.run_index(qdrant, out, images, "cxr_images", recreate=True,
                            config_hash=config_hash(), git_commit=None, dataset={},
                            library_versions={}, encoder=cfg["selected"])
    # The in-memory client ignores payload indexes, so only a real server can prove this one.
    assert "projection" in qdrant.get_collection(built["collection"]).payload_schema
    checks = validate.validate_indexes(qdrant, built, out, images,
                                       expected_config_hash=config_hash(),
                                       expected_revision=cfg["revision"], sample_size=10)
    assert [c.name for c in checks if not c.ok] == []

    hits = search_image_embeddings("chest x-ray showing pleural effusion", None, 5,
                                   encoder=encoder, client=qdrant, collection=built["collection"])
    by_id = dict(zip(images["image_id"], images["study_id"], strict=True))
    assert len(hits) == 5 and all(by_id[h.image_id] == h.study_id for h in hits)
    # A cosine, not a distance: inside [-1, 1], best first. FT6 cosines are mostly negative (T40).
    assert all(-1.0 <= h.score <= 1.0 for h in hits)
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:], strict=False))
