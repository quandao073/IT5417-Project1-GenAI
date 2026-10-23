"""Shard writing, resume and the shards manifest."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.embeddings import embed
from tests.unit.fakes import FakeEncoder


def images_frame(n: int) -> pl.DataFrame:
    return pl.DataFrame({
        "image_id": [f"{i:064x}" for i in range(n)],
        "study_id": [f"patient{i // 2:05d}/study1" for i in range(n)],
        "patient_id": [f"patient{i // 2:05d}" for i in range(n)],
        "path_to_image": [f"train/patient{i // 2:05d}/study1/view{i % 2 + 1}_frontal.jpg"
                          for i in range(n)],
        "projection": ["AP" if i % 3 else "PA" for i in range(n)],
    })


class ShortEncoder(FakeEncoder):
    """Returns one row fewer than it was given."""

    def encode_images(self, paths):
        return super().encode_images(paths)[:-1]


class TestHelpers:
    def test_build_id_format(self):
        assert embed.make_build_id("biomedclip", date(2026, 10, 11)) == (
            "20261011_chexpert25k_biomedclip_v1"
        )

    def test_build_id_takes_a_scope(self):
        assert embed.make_build_id("cxrclip", date(2026, 10, 23), scope="pool88k") == (
            "20261023_pool88k_cxrclip_v1"
        )

    def test_shard_bounds_cover_everything_once(self):
        assert embed.shard_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
        assert embed.shard_bounds(8, 4) == [(0, 4), (4, 8)]
        assert embed.shard_bounds(0, 4) == []

    def test_shard_bounds_rejects_nonpositive_size(self):
        with pytest.raises(ValueError):
            embed.shard_bounds(10, 0)

    def test_shard_path_is_zero_padded(self, tmp_path):
        assert embed.shard_path(tmp_path, 7).name == "shard_0007.npz"


class TestShardFiles:
    def test_write_then_read_round_trips_and_hashes(self, tmp_path):
        vectors = np.arange(6, dtype=np.float32).reshape(2, 3)
        path = embed.shard_path(tmp_path, 0)
        digest = embed.write_shard(path, vectors, ["a", "b"])
        back, ids = embed.read_shard(path)
        assert np.array_equal(back, vectors) and back.dtype == np.float32
        assert ids == ["a", "b"]
        assert digest == embed.vectors_sha256(vectors)
        assert not path.with_suffix(".npz.part").exists()
        assert not list(tmp_path.glob("*.part.npz")), "numpy must not append .npz to the temp name"

    def test_part_file_is_not_a_shard(self, tmp_path):
        """A killed run leaves shard_0000.npz.part; resume must not count it."""
        path = embed.shard_path(tmp_path, 0)
        path.with_suffix(".npz.part").write_bytes(b"half written")
        assert embed.shard_is_complete(path, ["a", "b"]) is False

    def test_corrupt_shard_is_redone(self, tmp_path):
        path = embed.shard_path(tmp_path, 0)
        path.write_bytes(b"not a zip file")
        assert embed.shard_is_complete(path, ["a"]) is False

    def test_shard_with_wrong_ids_is_not_complete(self, tmp_path):
        path = embed.shard_path(tmp_path, 0)
        embed.write_shard(path, np.zeros((2, 3), np.float32), ["a", "b"])
        assert embed.shard_is_complete(path, ["a", "b"]) is True
        assert embed.shard_is_complete(path, ["a", "c"]) is False
        assert embed.shard_is_complete(path, ["a"]) is False


class TestRunEmbed:
    def _run(self, tmp_path, n=10, shard_size=4, encoder=None):
        encoder = encoder or FakeEncoder()
        images = images_frame(n)
        out = tmp_path / "20261011_chexpert25k_fake_v1"
        manifest = embed.run_embed(
            images, encoder, Path("root"), out, "20261011_chexpert25k_fake_v1",
            shard_size=shard_size, batch_size=3, library_versions={"torch": "x"}, git_commit="abc",
        )
        return manifest, out, encoder

    def test_writes_every_shard_and_the_manifest(self, tmp_path):
        manifest, out, _ = self._run(tmp_path)
        assert sorted(p.name for p in out.glob("shard_*.npz")) == [
            "shard_0000.npz", "shard_0001.npz", "shard_0002.npz"
        ]
        assert manifest["dim"] == 4 and manifest["n_images"] == 10
        assert [s["n"] for s in manifest["shards"]] == [4, 4, 2]
        assert manifest["embedded"] == 10 and manifest["skipped"] == 0
        on_disk = json.loads((out / embed.SHARDS_FILE).read_text(encoding="utf-8"))
        assert on_disk["build_id"] == "20261011_chexpert25k_fake_v1"
        assert on_disk["model_id"] == "fake/model" and on_disk["revision"] == "fakerev0"
        assert on_disk["library_versions"] == {"torch": "x"} and on_disk["git_commit"] == "abc"
        assert on_disk["s_per_image"] >= 0
        assert not list(out.glob("*.part"))

    def test_last_shard_is_short(self, tmp_path):
        manifest, out, _ = self._run(tmp_path, n=10, shard_size=4)
        vectors, ids = embed.read_shard(embed.shard_path(out, 2))
        assert vectors.shape == (2, 4) and len(ids) == 2

    def test_vectors_follow_image_id_order_and_use_image_root(self, tmp_path):
        encoder = FakeEncoder()
        shuffled = images_frame(5).sample(fraction=1.0, shuffle=True, seed=1)
        out = tmp_path / "out"
        embed.run_embed(
            shuffled, encoder, Path("root"), out, "b",
            shard_size=5, batch_size=3, library_versions={}, git_commit=None,
        )
        vectors, ids = embed.read_shard(embed.shard_path(out, 0))
        images = images_frame(5)
        assert ids == sorted(images["image_id"].to_list())
        expected = encoder.encode_images(
            [Path("root") / p for p in images.sort("image_id")["path_to_image"].to_list()]
        )
        assert np.allclose(vectors, expected)

    def test_short_encoder_output_is_rejected_before_writing(self, tmp_path):
        with pytest.raises(RuntimeError, match="rows"):
            self._run(tmp_path, encoder=ShortEncoder())
        out = tmp_path / "20261011_chexpert25k_fake_v1"
        assert not embed.shard_path(out, 0).exists()
        assert not (out / embed.SHARDS_FILE).exists()

    def test_failure_names_the_image_not_the_batch_start(self, tmp_path):
        # view2 images are odd indices; index 1 is the first to fail, inside batch 0-2.
        encoder = FakeEncoder(fail_on="view2_frontal")
        with pytest.raises(RuntimeError, match=r"failed to embed \S*view2_frontal\.jpg: "):
            self._run(tmp_path, encoder=encoder)

    def test_stale_manifest_is_removed_when_a_rerun_fails(self, tmp_path):
        _, out, _ = self._run(tmp_path)
        embed.shard_path(out, 1).write_bytes(b"corrupt")
        with pytest.raises(RuntimeError, match="patient00002"):
            self._run(tmp_path, encoder=FakeEncoder(fail_on="patient00002"))
        assert not (out / embed.SHARDS_FILE).exists()

    def test_rerun_of_a_finished_build_keeps_the_manifest_untouched(self, tmp_path):
        first, out, _ = self._run(tmp_path)
        before = (out / embed.SHARDS_FILE).read_bytes()

        class OtherEncoder(FakeEncoder):
            device = "cuda"

        encoder = OtherEncoder()
        images = images_frame(10)
        again = embed.run_embed(
            images, encoder, Path("root"), out, "20261011_chexpert25k_fake_v1",
            shard_size=4, batch_size=3, library_versions={"torch": "other"}, git_commit="def",
        )
        assert (out / embed.SHARDS_FILE).read_bytes() == before
        assert again == first and again["embedded"] == 10
        assert encoder.image_calls == 0
        assert not list(out.glob("*.part"))

    def test_rerun_with_another_build_id_rewrites_the_manifest(self, tmp_path):
        _, out, _ = self._run(tmp_path)
        manifest = embed.run_embed(
            images_frame(10), FakeEncoder(), Path("root"), out, "other_build",
            shard_size=4, batch_size=3, library_versions={}, git_commit=None,
        )
        assert manifest["build_id"] == "other_build" and manifest["embedded"] == 0
        assert manifest["skipped"] == 10

    def test_resume_skips_complete_shards_and_redoes_broken_ones(self, tmp_path):
        _, out, _ = self._run(tmp_path)
        embed.shard_path(out, 1).write_bytes(b"corrupt")
        encoder = FakeEncoder()
        manifest, _, _ = self._run(tmp_path, encoder=encoder)
        assert manifest["skipped"] == 6 and manifest["embedded"] == 4  # shards 0 (4) + 2 (2) kept
        assert encoder.image_calls == 2  # one shard of 4 images, batch size 3
        assert (out / embed.SHARDS_FILE).is_file()

    def test_a_broken_image_stops_the_run_and_names_it(self, tmp_path):
        encoder = FakeEncoder(fail_on="patient00002")
        with pytest.raises(RuntimeError, match="patient00002"):
            self._run(tmp_path, encoder=encoder)
        assert not (tmp_path / "20261011_chexpert25k_fake_v1" / embed.SHARDS_FILE).exists()

    def test_manifest_shard_hashes_match_files(self, tmp_path):
        manifest, out, _ = self._run(tmp_path)
        for entry in manifest["shards"]:
            vectors, _ = embed.read_shard(out / entry["file"])
            assert embed.vectors_sha256(vectors) == entry["sha256"]

    def test_load_shards_manifest_hints_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="cxr embed"):
            embed.load_shards_manifest(tmp_path)


def test_manifest_records_the_encoder_name(tmp_path):
    kwargs = {"shard_size": 4, "batch_size": 4, "library_versions": {}, "git_commit": None}
    default = embed.run_embed(images_frame(4), FakeEncoder(), Path("root"), tmp_path / "a", "a",
                              **kwargs)
    named = embed.run_embed(images_frame(4), FakeEncoder(), Path("root"), tmp_path / "b", "b",
                            encoder_name="fake2", **kwargs)
    assert default["encoder"] == "fake" and named["encoder"] == "fake2"
