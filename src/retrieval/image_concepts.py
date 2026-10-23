"""Image-only ranking signal: per-concept probe logits, scored per query.

`cxr train-probe` writes one logit per (corpus study, concept). At query time an
include constraint adds log P(concept present) = -softplus(-z) and an exclude adds
log P(absent) = -softplus(z), so a study scores high when the probe is confident
of every include and of no exclude. Both terms are <= 0 and finite at any logit.
The status of an include (PRESENT or UNCERTAIN) does not change its image term;
the probe has no notion of uncertainty, and the pipeline warns about it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src.data.labels import CHEXPERT_LABELS
from src.embeddings import probe
from src.embeddings.validate import Check
from src.query.schema import Constraint


class ImageProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageScores:
    study_ids: np.ndarray                 # str, in the order given to build_image_scores
    row_of: dict[str, int]
    concepts: tuple[str, ...]
    col_of: dict[str, int]
    logits: np.ndarray                    # float32 [n, len(concepts)]


def build_image_scores(
    scores: pl.DataFrame,
    study_ids: Sequence[str],
    concepts: Sequence[str] = CHEXPERT_LABELS,
) -> ImageScores:
    ids = list(study_ids)
    row_of = {s: i for i, s in enumerate(ids)}
    scored = scores["study_id"].to_list()
    scored_set = set(scored)
    if len(scored_set) != len(scored):
        dup = len(scored) - len(scored_set)
        raise ImageProbeError(f"{dup} duplicate study_id rows in corpus_scores")
    if missing := sum(1 for s in ids if s not in scored_set):
        raise ImageProbeError(f"{missing} corpus studies have no image score")
    if extra := sum(1 for s in scored if s not in row_of):
        raise ImageProbeError(f"{extra} scored studies are not in the corpus")
    for concept in concepts:
        if concept not in scores.columns:
            raise ImageProbeError(f"corpus_scores is missing column {concept!r}")
    # Placed by study id: the parquet's row order is not the corpus order.
    logits = np.zeros((len(ids), len(concepts)), dtype=np.float32)
    logits[[row_of[s] for s in scored]] = scores.select(list(concepts)).to_numpy()
    return ImageScores(
        study_ids=np.asarray(ids, dtype=object),
        row_of=row_of,
        concepts=tuple(concepts),
        col_of={c: j for j, c in enumerate(concepts)},
        logits=logits,
    )


def image_score(
    scores: ImageScores, include: Sequence[Constraint], exclude: Sequence[str]
) -> np.ndarray:
    """float64 [n]; higher is a better match."""
    if not include and not exclude:
        raise ValueError("image score needs at least one constraint")
    total = np.zeros(len(scores.study_ids), dtype=np.float64)
    for c in include:
        z = scores.logits[:, scores.col_of[c.concept]].astype(np.float64)
        total -= np.logaddexp(0.0, -z)
    for concept in exclude:
        z = scores.logits[:, scores.col_of[concept]].astype(np.float64)
        total -= np.logaddexp(0.0, z)
    return total


def rank_studies(
    scores: ImageScores,
    include: Sequence[Constraint],
    exclude: Sequence[str],
    *,
    allowed: Sequence[str] | None,
    limit: int,
) -> list[tuple[str, float]]:
    """Best `limit` studies by image score, ties by study_id. `allowed=None` ranks
    the whole matrix; a list ranks only those studies (an unknown id is a KeyError)."""
    total = image_score(scores, include, exclude)
    if allowed is None:
        ids = scores.study_ids
        values = total
    else:
        rows = [scores.row_of[s] for s in allowed]
        ids = scores.study_ids[rows]
        values = total[rows]
    order = np.lexsort((ids.astype(str), -values))[:limit]
    return [(str(ids[i]), float(values[i])) for i in order]


def _read(path: Path, load):
    """`load(path)`, with any failure to read or parse it reported as an ImageProbeError."""
    try:
        return load(path)
    except ImageProbeError:
        raise
    except Exception as exc:
        raise ImageProbeError(f"{path}: {type(exc).__name__}: {exc}") from exc


def open_image_probe(
    probe_dir: Path, *, build_id: str, revision: str, study_ids: Sequence[str],
    encoder_name: str | None = None,
) -> tuple[ImageScores, dict]:
    """Load the saved corpus scores, refusing an artifact trained for another build or encoder.

    `encoder_name` is the manifest's adapter name; it is compared only when given and the
    artifact recorded one (artifacts older than `encoder.name` pass on build and revision)."""
    for name in (probe.PROBE_FILE, probe.SCORES_FILE, probe.META_FILE):
        if not (probe_dir / name).is_file():
            raise ImageProbeError(f"{probe_dir / name} not found; run cxr train-probe")
    meta = _read(probe_dir / probe.META_FILE,
                 lambda path: json.loads(path.read_text(encoding="utf-8")))
    _read(probe_dir / probe.PROBE_FILE, lambda _: probe.load_probe(probe_dir))
    try:
        meta_build, meta_revision = meta["build_id"], meta["encoder"]["revision"]
    except (KeyError, TypeError) as exc:
        raise ImageProbeError(
            f"{probe_dir / probe.META_FILE}: missing key {exc}; run cxr train-probe") from exc
    if meta_build != build_id:
        raise ImageProbeError(
            f"image probe was trained for build {meta_build}, manifest says {build_id}; "
            "run cxr train-probe"
        )
    if meta_revision != revision:
        raise ImageProbeError(
            f"image probe was trained on encoder revision {meta_revision}, "
            f"manifest says {revision}; run cxr train-probe"
        )
    meta_name = meta["encoder"].get("name")
    if encoder_name is not None and meta_name is not None and meta_name != encoder_name:
        raise ImageProbeError(
            f"image scores were built with encoder {meta_name}, manifest says {encoder_name}"
        )
    scores = _read(probe_dir / probe.SCORES_FILE,
                   lambda path: build_image_scores(pl.read_parquet(path), study_ids))
    return scores, meta


def validate_image_probe(
    probe_dir: Path, *, build_id: str, revision: str, study_ids: Sequence[str],
    encoder_name: str | None = None, name: str = "image_probe",
) -> Check:
    """A `cxr validate-indexes` check (`image_probe`, or `image_scores` for the decomposed
    artifact); an exception is a FAIL line."""
    try:
        scores, _ = open_image_probe(
            probe_dir, build_id=build_id, revision=revision, study_ids=study_ids,
            encoder_name=encoder_name,
        )
        n = len(scores.study_ids)
        return Check(name, True, f"{n} studies scored, build {build_id}")
    except Exception as exc:
        return Check(name, False, f"{type(exc).__name__}: {exc}")
