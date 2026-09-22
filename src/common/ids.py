"""Stable identifiers derived from image paths.

CheXpert-small's `Path` column spells an image as
`CheXpert-v1.0-small/train/patientNNNNN/studyN/viewN_frontal.jpg`; CheXpert
Plus's `path_to_image` spells the same image without the dataset-root prefix.
Every id here is computed from the normalized form, so the two sources agree.

Normalization is deterministic only - no fuzzy or approximate matching.
See plan section 8.5 and configs/data.yaml:join.images.
"""

from __future__ import annotations

import hashlib
import re
import uuid

ROOT_PREFIX = "CheXpert-v1.0-small/"

_DUPLICATE_SEPARATORS = re.compile(r"/{2,}")
_PATIENT_STUDY = re.compile(r"(patient\d+)/(study\d+)")


def normalize_image_path(path: str) -> str:
    """Reduce either CSV's spelling of an image path to one canonical form."""
    cleaned = path.replace("\\", "/").strip()
    cleaned = _DUPLICATE_SEPARATORS.sub("/", cleaned)
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if cleaned.startswith(ROOT_PREFIX):
        cleaned = cleaned[len(ROOT_PREFIX) :]
    if not cleaned:
        raise ValueError(f"Image path is empty after normalization: {path!r}")
    return cleaned


def image_id(path: str) -> str:
    """Primary key for an image: sha256 of its normalized path."""
    return hashlib.sha256(normalize_image_path(path).encode("utf-8")).hexdigest()


def study_id_from_path(path: str) -> str:
    """Return `patientNNNNN/studyN`, the study a given image belongs to."""
    match = _PATIENT_STUDY.search(normalize_image_path(path))
    if match is None:
        raise ValueError(f"Path holds no patient/study pair: {path!r}")
    return f"{match.group(1)}/{match.group(2)}"


def patient_id_from_path(path: str) -> str:
    """Return `patientNNNNN`, the patient a given image belongs to."""
    return study_id_from_path(path).split("/", 1)[0]


def qdrant_point_id(path: str) -> uuid.UUID:
    """Point id for Qdrant, which accepts a UUID or a uint64 but not a hex digest."""
    return uuid.uuid5(uuid.NAMESPACE_URL, normalize_image_path(path))
