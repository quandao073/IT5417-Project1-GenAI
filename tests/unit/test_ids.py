"""Stable identifiers derived from image paths.

The two source CSVs spell the same image differently: CheXpert-small's `Path`
carries the dataset-root prefix, CheXpert Plus's `path_to_image` does not.
Everything downstream keys on these ids, so the two spellings must collapse to
one value. That is what most of this module guards.
"""

from __future__ import annotations

import uuid

import pytest

from src.common import ids

# The same image, as each source CSV spells it.
SMALL_CSV = "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
PLUS_CSV = "train/patient00001/study1/view1_frontal.jpg"


class TestNormalizeImagePath:
    def test_strips_the_dataset_root_prefix(self):
        assert ids.normalize_image_path(SMALL_CSV) == PLUS_CSV

    def test_leaves_an_already_normalized_path_alone(self):
        assert ids.normalize_image_path(PLUS_CSV) == PLUS_CSV

    def test_converts_backslashes_to_slashes(self):
        assert ids.normalize_image_path(r"train\patient00001\study1\view1_frontal.jpg") == PLUS_CSV

    def test_collapses_duplicate_separators(self):
        messy = "train//patient00001///study1/view1_frontal.jpg"
        assert ids.normalize_image_path(messy) == PLUS_CSV

    def test_strips_surrounding_whitespace(self):
        assert ids.normalize_image_path(f"  {PLUS_CSV}\n") == PLUS_CSV

    def test_strips_a_leading_current_directory(self):
        assert ids.normalize_image_path(f"./{PLUS_CSV}") == PLUS_CSV

    def test_is_idempotent(self):
        once = ids.normalize_image_path(SMALL_CSV)
        assert ids.normalize_image_path(once) == once

    def test_rejects_an_empty_path(self):
        with pytest.raises(ValueError):
            ids.normalize_image_path("   ")


class TestImageId:
    def test_is_64_lowercase_hex_characters(self):
        value = ids.image_id(PLUS_CSV)
        assert len(value) == 64
        assert value == value.lower()
        int(value, 16)  # raises if it is not hex

    def test_both_csv_spellings_yield_the_same_id(self):
        assert ids.image_id(SMALL_CSV) == ids.image_id(PLUS_CSV)

    def test_different_images_yield_different_ids(self):
        other = "train/patient00001/study1/view2_frontal.jpg"
        assert ids.image_id(PLUS_CSV) != ids.image_id(other)


class TestStudyAndPatientIds:
    def test_study_id_is_patient_and_study(self):
        assert ids.study_id_from_path(PLUS_CSV) == "patient00001/study1"

    def test_study_id_ignores_the_root_prefix(self):
        assert ids.study_id_from_path(SMALL_CSV) == ids.study_id_from_path(PLUS_CSV)

    def test_patient_id_is_the_patient_segment(self):
        assert ids.patient_id_from_path(PLUS_CSV) == "patient00001"

    def test_valid_split_patients_are_parsed_too(self):
        assert ids.study_id_from_path("valid/patient64541/study1/view1_frontal.jpg") == (
            "patient64541/study1"
        )

    @pytest.mark.parametrize(
        "path",
        [
            "train/patient00001/view1_frontal.jpg",  # no study segment
            "train/study1/view1_frontal.jpg",  # no patient segment
            "some/other/file.jpg",
        ],
    )
    def test_a_path_without_patient_and_study_is_an_error(self, path):
        with pytest.raises(ValueError):
            ids.study_id_from_path(path)


class TestQdrantPointId:
    def test_is_a_uuid(self):
        assert isinstance(ids.qdrant_point_id(PLUS_CSV), uuid.UUID)

    def test_is_deterministic(self):
        assert ids.qdrant_point_id(PLUS_CSV) == ids.qdrant_point_id(PLUS_CSV)

    def test_both_csv_spellings_yield_the_same_point_id(self):
        assert ids.qdrant_point_id(SMALL_CSV) == ids.qdrant_point_id(PLUS_CSV)

    def test_different_images_yield_different_point_ids(self):
        other = "train/patient00001/study1/view2_frontal.jpg"
        assert ids.qdrant_point_id(PLUS_CSV) != ids.qdrant_point_id(other)


def test_root_prefix_matches_the_data_config():
    """ids.py must not drift from configs/data.yaml:join.images.strip_root_prefix."""
    import yaml

    with open("configs/data.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["join"]["images"]["strip_root_prefix"] == ids.ROOT_PREFIX
