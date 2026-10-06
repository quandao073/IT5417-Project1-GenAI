"""CheXzero: the checkpoint pin and the original preprocessing."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image

from src.embeddings import chexzero


class TestSha256Pin:
    def test_matching_hash_passes(self, tmp_path):
        path = tmp_path / "w.pt"
        path.write_bytes(b"weights")
        chexzero.verify_sha256(path, "sha256:" + hashlib.sha256(b"weights").hexdigest())

    def test_mismatch_stops_the_run(self, tmp_path):
        path = tmp_path / "w.pt"
        path.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="sha256 mismatch"):
            chexzero.verify_sha256(path, "sha256:" + hashlib.sha256(b"weights").hexdigest())

    def test_an_unpinned_checkpoint_is_refused(self, tmp_path):
        path = tmp_path / "w.pt"
        path.write_bytes(b"weights")
        with pytest.raises(ValueError, match="not pinned"):
            chexzero.verify_sha256(path, None)

    def test_sha256_of_uses_the_prefix(self, tmp_path):
        path = tmp_path / "w.pt"
        path.write_bytes(b"x")
        assert chexzero.sha256_of(path) == "sha256:" + hashlib.sha256(b"x").hexdigest()


class TestPixels:
    def test_letterboxes_a_wide_image_to_320_square(self, tmp_path):
        path = tmp_path / "wide.jpg"
        Image.new("L", (480, 240), 200).save(path)
        pixels = chexzero.chexzero_pixels(path)
        assert pixels.shape == (3, 320, 320) and pixels.dtype == np.float32
        assert pixels[0, 0, 160] == 0.0                   # top padding is black
        assert abs(pixels[0, 160, 160] - 200) <= 2        # content keeps 0-255 scale
        assert np.array_equal(pixels[0], pixels[2])       # grey repeated to 3 channels
