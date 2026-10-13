"""The vector leg: encode a query, nearest neighbours, optional AP/PA filter (G3)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.embeddings import embed, index
from src.retrieval import image_search
from tests.unit.fakes import FakeEncoder
from tests.unit.test_embed import images_frame

BUILD = "20261011_chexpert25k_fake_v1"


@pytest.fixture
def indexed(tmp_path):
    images = images_frame(12)
    out = tmp_path / BUILD
    encoder = FakeEncoder()
    embed.run_embed(images, encoder, Path("root"), out, BUILD, shard_size=5, batch_size=4,
                    library_versions={}, git_commit=None)
    client = QdrantClient(":memory:")
    manifest = index.run_index(client, out, images, "cxr_images", recreate=False,
                               config_hash="c", git_commit=None, dataset={}, library_versions={})
    return encoder, client, manifest["collection"], images


def _search(indexed, query="pleural effusion", views=None, k=5, lock=None):
    encoder, client, collection, _ = indexed
    return image_search.search_image_embeddings(
        query, views, k, encoder=encoder, client=client, collection=collection, lock=lock)


def _all_vectors(shards_dir: Path):
    vectors, ids = [], []
    for path in sorted(shards_dir.glob("shard_*.npz")):
        v, i = embed.read_shard(path)
        vectors.append(v)
        ids.extend(i)
    return np.concatenate(vectors), ids


def test_returns_k_candidates_ranked_by_descending_cosine(indexed, tmp_path):
    encoder, client, collection, images = indexed
    hits = _search(indexed, k=5)
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:], strict=False))
    vectors, ids = _all_vectors(tmp_path / BUILD)
    scores = vectors @ encoder.encode_texts(["pleural effusion"])[0]
    expected = [ids[i] for i in np.argsort(-scores)[:5]]
    assert [h.image_id for h in hits] == expected
    assert all(abs(h.score - scores[ids.index(h.image_id)]) < 1e-5 for h in hits)


def test_candidates_carry_study_id_from_payload(indexed):
    _, _, _, images = indexed
    by_id = dict(zip(images["image_id"], images["study_id"], strict=True))
    assert all(h.study_id == by_id[h.image_id] for h in _search(indexed))


def test_views_filter_restricts_to_projection(indexed):
    _, _, _, images = indexed
    pa = set(images.filter(images["projection"] == "PA")["image_id"])
    hits = _search(indexed, views=["PA"], k=12)
    assert {h.image_id for h in hits} == pa


def test_empty_views_means_no_filter(indexed):
    assert len(_search(indexed, views=[], k=12)) == 12
    assert image_search.views_filter([]) is None and image_search.views_filter(None) is None


def test_k_larger_than_corpus_returns_everything(indexed):
    assert len(_search(indexed, k=50)) == 12


def test_lock_is_held_while_encoding(indexed):
    encoder, client, collection, _ = indexed
    lock = threading.Lock()
    seen = []
    original = encoder.encode_texts

    def spy(texts):
        seen.append(lock.locked())
        return original(texts)

    encoder.encode_texts = spy
    _search(indexed, lock=lock)
    assert seen == [True] and not lock.locked()
