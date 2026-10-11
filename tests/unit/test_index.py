"""Shards -> Qdrant collection -> build_manifest.json, in memory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.common.ids import qdrant_point_id
from src.embeddings import embed, index
from tests.unit.fakes import FakeEncoder
from tests.unit.test_embed import images_frame

BUILD = "20261011_chexpert25k_fake_v1"
DATASET = {"images": "chexpert-v1.0-small", "plus_csv": "df_chexpert_plus_240401.csv"}


@pytest.fixture
def shards(tmp_path):
    images = images_frame(10)
    out = tmp_path / BUILD
    embed.run_embed(images, FakeEncoder(), Path("root"), out, BUILD, shard_size=4,
                    batch_size=3, library_versions={}, git_commit=None)
    return images, out


def _index(client, shards_dir, images, recreate=False):
    return index.run_index(
        client, shards_dir, images, "cxr_images", recreate=recreate, config_hash="cfg",
        git_commit="abc", dataset=DATASET, library_versions={"torch": "x"},
    )


class TestRunIndex:
    def test_creates_the_collection_with_the_shard_dim_and_all_points(self, shards):
        images, out = shards
        client = QdrantClient(":memory:")
        manifest = _index(client, out, images)
        name = index.collection_name("cxr_images", BUILD)
        assert client.collection_exists(name)
        assert client.get_collection(name).config.params.vectors.size == 4
        assert client.count(name).count == 10
        assert manifest["collection"] == name and manifest["build_id"] == BUILD
        assert manifest["embedding_model"]["dimension"] == 4
        assert manifest["embedding_model"]["revision"] == "fakerev0"
        assert manifest["counts"] == {"studies": 5, "images": 10, "reports": 0, "vectors": 10}
        assert manifest["config_hash"] == "cfg" and manifest["dataset"] == DATASET

    def test_point_ids_follow_d7_and_payload_has_five_fields(self, shards):
        images, out = shards
        client = QdrantClient(":memory:")
        _index(client, out, images)
        row = images.row(3, named=True)
        [point] = client.retrieve(index.collection_name("cxr_images", BUILD),
                                  ids=[str(qdrant_point_id(row["path_to_image"]))],
                                  with_vectors=True)
        fields = ("image_id", "study_id", "patient_id", "projection", "path_to_image")
        assert point.payload == {k: row[k] for k in fields}
        vectors, ids = embed.read_shard(embed.shard_path(out, 0))
        assert np.allclose(point.vector, vectors[ids.index(row["image_id"])], atol=1e-6)

    def test_refuses_an_existing_collection_without_recreate(self, shards):
        images, out = shards
        client = QdrantClient(":memory:")
        _index(client, out, images)
        with pytest.raises(ValueError, match="--recreate"):
            _index(client, out, images)
        _index(client, out, images, recreate=True)
        assert client.count(index.collection_name("cxr_images", BUILD)).count == 10

    def test_a_refused_recreate_keeps_the_existing_collection(self, shards):
        images, out = shards
        client = QdrantClient(":memory:")
        _index(client, out, images)
        name = index.collection_name("cxr_images", BUILD)
        vectors, ids = embed.read_shard(embed.shard_path(out, 2))
        _, ids0 = embed.read_shard(embed.shard_path(out, 0))
        ids[0] = ids0[0]
        embed.write_shard(embed.shard_path(out, 2), vectors, ids)
        with pytest.raises(ValueError, match="duplicate"):
            _index(client, out, images, recreate=True)
        assert client.collection_exists(name) is True
        assert client.count(name).count == 10

    def test_refuses_a_shard_whose_hash_changed(self, shards):
        images, out = shards
        vectors, ids = embed.read_shard(embed.shard_path(out, 1))
        embed.write_shard(embed.shard_path(out, 1), vectors + 1.0, ids)
        with pytest.raises(ValueError, match="sha256"):
            _index(QdrantClient(":memory:"), out, images)

    def test_unknown_image_id_is_refused_before_any_write(self, shards):
        images, out = shards
        # The sha256 covers the vectors only, so rewriting the ids alone keeps
        # shards.json valid and the failure comes from the id check.
        vectors, ids = embed.read_shard(embed.shard_path(out, 2))
        ids[0] = "f" * 64
        embed.write_shard(embed.shard_path(out, 2), vectors, ids)
        client = QdrantClient(":memory:")
        with pytest.raises(ValueError, match="not in images.parquet"):
            _index(client, out, images)
        assert client.collection_exists(index.collection_name("cxr_images", BUILD)) is False

    def test_duplicate_image_id_is_refused_before_any_write(self, shards):
        images, out = shards
        vectors, ids = embed.read_shard(embed.shard_path(out, 2))
        _, ids0 = embed.read_shard(embed.shard_path(out, 0))
        ids[0] = ids0[0]
        embed.write_shard(embed.shard_path(out, 2), vectors, ids)
        client = QdrantClient(":memory:")
        with pytest.raises(ValueError, match="duplicate"):
            _index(client, out, images)
        assert client.collection_exists(index.collection_name("cxr_images", BUILD)) is False

    def test_requests_a_keyword_payload_index_on_projection(self, shards):
        """:memory: Qdrant ignores payload indexes, so assert the request, not the schema;
        tests/integration/test_vector_index.py checks payload_schema on the real server."""
        from qdrant_client import models

        images, out = shards
        client = QdrantClient(":memory:")
        calls = []
        original = client.create_payload_index

        def spy(collection_name, field_name, field_schema, **kwargs):
            calls.append((collection_name, field_name, field_schema))
            return original(collection_name, field_name=field_name, field_schema=field_schema,
                            **kwargs)

        client.create_payload_index = spy
        _index(client, out, images)
        assert calls == [(index.collection_name("cxr_images", BUILD), "projection",
                          models.PayloadSchemaType.KEYWORD)]


def test_write_manifest_round_trips(tmp_path):
    index.write_manifest(tmp_path / index.MANIFEST_FILE, {"build_id": "x", "counts": {"a": 1}})
    written = json.loads((tmp_path / index.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert written["build_id"] == "x"
    assert not list(tmp_path.glob("*.part"))
