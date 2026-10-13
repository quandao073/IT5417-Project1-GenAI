"""Each of the six validate-indexes checks has a way to fail, and each is tested."""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from src.embeddings import embed, index, validate
from tests.unit.fakes import FakeEncoder
from tests.unit.test_embed import images_frame

BUILD = "20261011_chexpert25k_fake_v1"


@pytest.fixture
def built(tmp_path):
    images = images_frame(10)
    out = tmp_path / BUILD
    embed.run_embed(images, FakeEncoder(), Path("root"), out, BUILD, shard_size=4,
                    batch_size=3, library_versions={}, git_commit=None)
    client = QdrantClient(":memory:")
    manifest = index.run_index(client, out, images, "cxr_images", recreate=False,
                               config_hash="cfg", git_commit=None, dataset={},
                               library_versions={})
    return client, manifest, out, images


def _run(built, **over):
    client, manifest, out, images = built
    kwargs = {"expected_config_hash": "cfg", "expected_revision": "fakerev0"} | over
    return validate.validate_indexes(client, manifest, out, images, **kwargs)


def _failed(checks):
    return [c.name for c in checks if not c.ok]


class TestExpectedConfigHash:
    @pytest.fixture
    def config_dir(self, tmp_path):
        (tmp_path / "data.yaml").write_text("corpus:\n  target_studies: 1\n", encoding="utf-8")
        (tmp_path / "models.yaml").write_text(
            "image_encoder:\n  revision: abc\nllm:\n  model: small\n", encoding="utf-8"
        )
        (tmp_path / "retrieval.yaml").write_text("fusion:\n  rrf_k: 60\n", encoding="utf-8")
        (tmp_path / "concepts.yaml").write_text("concepts: {}\n", encoding="utf-8")
        return tmp_path

    def test_uses_the_scope_the_manifest_recorded(self, config_dir):
        from src.common.config import config_hash

        manifest = {"config_scope": ["data"]}
        assert validate.expected_config_hash(manifest, config_dir) == config_hash(
            ("data",), config_dir
        )

    def test_a_manifest_without_a_scope_uses_the_phase_0a_four_configs(self, config_dir):
        """Manifests indexed before config_scope existed hashed all four configs."""
        from src.common.config import config_hash

        assert validate.expected_config_hash({}, config_dir) == config_hash(
            ("data", "models", "retrieval", "concepts"), config_dir
        )


def test_a_consistent_build_passes_all_six(built):
    checks = _run(built)
    assert len(checks) == 6 and _failed(checks) == []


def test_config_hash_drift_fails_check_1(built):
    assert _failed(_run(built, expected_config_hash="other")) == ["config_hash"]


def test_count_mismatch_fails_check_2(built):
    client, manifest, out, images = built
    client.delete(manifest["collection"], points_selector=models.PointIdsList(
        points=[validate.all_point_ids(client, manifest["collection"]).pop()]))
    failed = _failed(_run(built))
    assert "counts" in failed and "orphans" in failed


def test_orphan_point_fails_check_3(built):
    client, manifest, out, images = built
    client.upsert(manifest["collection"], points=[models.PointStruct(
        id="00000000-0000-0000-0000-000000000001", vector=[1.0, 0, 0, 0], payload={})])
    assert "orphans" in _failed(_run(built))


def test_dimension_mismatch_fails_check_4(built):
    client, manifest, out, images = built
    manifest = {**manifest, "embedding_model": {**manifest["embedding_model"], "dimension": 8}}
    assert "dimension" in _failed(validate.validate_indexes(
        client, manifest, out, images, expected_config_hash="cfg", expected_revision="fakerev0"))


def test_vector_or_payload_drift_fails_check_5(built):
    client, manifest, out, images = built
    ids = sorted(validate.all_point_ids(client, manifest["collection"]))
    [point] = client.retrieve(manifest["collection"], ids=[ids[0]], with_vectors=True)
    client.upsert(manifest["collection"], points=[models.PointStruct(
        id=ids[0], vector=[0.0, 0.0, 0.0, 1.0], payload=point.payload)])
    assert "sample" in _failed(_run(built, sample_size=10))


def test_revision_drift_fails_check_6(built):
    assert _failed(_run(built, expected_revision="otherrev")) == ["revision"]


def test_all_point_ids_pages_through_everything(built):
    client, manifest, out, images = built
    assert len(validate.all_point_ids(client, manifest["collection"], page=3)) == 10


def test_an_exception_inside_a_check_becomes_a_fail_line(built):
    client, manifest, out, images = built

    class BrokenCount:
        def __getattr__(self, name):
            return getattr(client, name)

        def count(self, *args, **kwargs):
            raise ConnectionError("qdrant went away")

    checks = validate.validate_indexes(
        BrokenCount(), manifest, out, images, expected_config_hash="cfg",
        expected_revision="fakerev0")
    assert len(checks) == 6 and _failed(checks) == ["counts"]
    [counts] = [c for c in checks if c.name == "counts"]
    assert counts.detail == "ConnectionError: qdrant went away"
