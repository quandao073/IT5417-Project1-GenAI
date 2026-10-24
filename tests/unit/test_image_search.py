"""The vector leg: encode a query, nearest neighbours, optional AP/PA filter (G3)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from src.common.ids import qdrant_point_id
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
                               config_hash="c", git_commit=None, dataset={}, library_versions={},
                               encoder="fake")
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


def test_view_facet_restricts_to_projection(indexed):
    _, _, _, images = indexed
    pa = set(images.filter(images["projection"] == "PA")["image_id"])
    hits = _search(indexed, views=["PA"], k=12)
    assert {h.image_id for h in hits} == pa


def test_empty_views_means_no_filter(indexed):
    assert len(_search(indexed, views=[], k=12)) == 12
    assert image_search.points_filter([], None) is None
    assert image_search.points_filter(None, None) is None


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


class _RecordingClient:
    """Records the kwargs of query_points and returns no hits."""

    def __init__(self):
        self.calls = []

    def query_points(self, collection, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"points": []})()


@pytest.fixture
def fake_client():
    return _RecordingClient()


def _filter_sent(client, views, point_ids):
    image_search.search_image_embeddings(
        "x", views, 5, encoder=FakeEncoder(), client=client, collection="c",
        point_ids=point_ids)
    return client.calls[0]["query_filter"]


def test_point_ids_become_a_has_id_condition_combined_with_views(fake_client):
    flt = _filter_sent(fake_client, ["PA"], ["id1", "id2"])
    assert isinstance(flt.must[0], models.FieldCondition)
    assert isinstance(flt.must[1], models.HasIdCondition) and flt.must[1].has_id == ["id1", "id2"]


def test_point_ids_alone_make_a_single_has_id_condition(fake_client):
    flt = _filter_sent(fake_client, None, ["id1"])
    assert len(flt.must) == 1 and isinstance(flt.must[0], models.HasIdCondition)


def test_no_point_ids_keeps_the_old_filter(fake_client):
    flt = _filter_sent(fake_client, ["PA"], None)
    assert len(flt.must) == 1 and isinstance(flt.must[0], models.FieldCondition)
    assert _filter_sent(_RecordingClient(), None, None) is None
    assert image_search.points_filter([], None) is None


def test_empty_point_ids_is_an_allow_list_of_nothing():
    flt = image_search.points_filter([], [])
    assert len(flt.must) == 1 and flt.must[0].has_id == []


def test_empty_point_ids_returns_nothing_without_touching_qdrant():
    class ExplodingEncoder:
        def encode_texts(self, texts):
            raise AssertionError("encoder must not be called")

    client = _RecordingClient()
    hits = image_search.search_image_embeddings(
        "x", ["PA"], 5, encoder=ExplodingEncoder(), client=client, collection="c", point_ids=[])
    assert hits == [] and client.calls == []


def test_point_ids_restrict_a_real_search(indexed):
    encoder, client, collection, images = indexed
    wanted = [str(qdrant_point_id(p)) for p in images["path_to_image"][:3]]
    hits = image_search.search_image_embeddings(
        "x", None, 12, encoder=encoder, client=client, collection=collection, point_ids=wanted)
    assert {h.image_id for h in hits} == set(images["image_id"][:3])
