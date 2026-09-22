"""Stratified, interleaved ordering of the eligible studies.

The manifest orders *every* eligible study; `--limit` takes a prefix. That only
works if any prefix carries the pool's label distribution, so studies are not
shuffled globally: each stratum is spread evenly across the whole ordering.

A study's stratum is its rarest PRESENT label. Rarest, because a study showing
both Cardiomegaly (common) and Fracture (rare) is far more useful to the corpus
as a Fracture example - keying on the common label would leave the rare ones
under-sampled.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Sequence

import polars as pl

NO_FINDING_STRATUM = "none"


def stratum_priority(resolved: pl.DataFrame) -> tuple[str, ...]:
    """Order concepts from rarest to commonest by how often they are PRESENT.

    Computed once over the whole eligible pool and written into the corpus
    report, so the stratum assignment can be reproduced exactly.
    """
    counts = (
        resolved.filter(pl.col("status") == "PRESENT")
        .group_by("concept")
        .len()
        # Concept name breaks count ties, so the order never depends on
        # the row order polars happened to produce.
        .sort("len", "concept")
    )
    return tuple(counts.get_column("concept").to_list())


def assign_strata(resolved: pl.DataFrame, priority: Sequence[str]) -> pl.DataFrame:
    """Give each study exactly one stratum: its rarest PRESENT label, else `none`."""
    rank = {concept: index for index, concept in enumerate(priority)}

    present = resolved.filter(pl.col("status") == "PRESENT")
    if unknown := sorted(set(present.get_column("concept").unique()) - set(rank)):
        raise ValueError(f"Concepts missing from the stratum priority: {unknown}")

    rarest = (
        present.with_columns(
            pl.col("concept").replace_strict(rank, return_dtype=pl.Int32).alias("_rank")
        )
        .sort("_rank")
        .group_by("study_id")
        .first()
        .select("study_id", pl.col("concept").alias("stratum"))
    )
    return (
        resolved.select("study_id")
        .unique()
        .join(rarest, on="study_id", how="left")
        .with_columns(pl.col("stratum").fill_null(NO_FINDING_STRATUM))
        .sort("study_id")
    )


def build_manifest(
    strata: pl.DataFrame,
    seed: int,
    pinned: Collection[str] = (),
) -> pl.DataFrame:
    """Order every study once, interleaving strata so each prefix stays proportional.

    `pinned` studies are placed at the head regardless of stratum. That is for the
    234 radiologist-labelled valid/ studies, which must be in the corpus at any
    --limit because they are the only human ground truth available.
    """
    known = set(strata.get_column("study_id").to_list())
    if missing := sorted(set(pinned) - known):
        raise ValueError(f"Pinned studies are not in the pool: {missing}")

    pinned_set = set(pinned)
    ordered = strata.with_columns(
        pl.col("study_id")
        .map_elements(
            lambda study: hashlib.sha256(f"{seed}:{study}".encode()).hexdigest(),
            return_dtype=pl.Utf8,
        )
        .alias("_shuffle"),
        pl.col("study_id").is_in(pinned_set).alias("_pinned"),
    )

    # Within a stratum the order is a seeded shuffle; across strata, placing
    # member j of a stratum of n at (j + 0.5) / n spreads every stratum evenly
    # over the whole ordering, which is what keeps each prefix proportional.
    ordered = (
        ordered.sort("_shuffle")
        .with_columns(pl.int_range(pl.len()).over("stratum").alias("_index"))
        .with_columns(pl.len().over("stratum").alias("_size"))
        .with_columns(
            ((pl.col("_index") + 0.5) / pl.col("_size")).alias("_spread"),
        )
        .sort("_pinned", "_spread", "_shuffle", descending=[True, False, False])
        .with_columns(
            pl.int_range(pl.len(), dtype=pl.Int64).alias("rank_in_manifest"),
            pl.lit(seed, dtype=pl.Int64).alias("selection_seed"),
        )
    )
    return ordered.select("study_id", "stratum", "rank_in_manifest", "selection_seed")
