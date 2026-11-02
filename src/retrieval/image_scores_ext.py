"""Extra columns for the score matrix: findings and diseases of the KG core that have no column
among the 14 labels (WORKLOG D60, D62).

Each candidate is captioned the way the 14 labels are (`render`: the bare English phrase, no
template around it), by its most frequent pool strings, the embeddings averaged. Its silver
labels on pool-val come from RadGraph: PRESENT when a report mentions the concept or an IS_A
descendant as definitely present, UNCERTAIN (dropped) when only as uncertain, ABSENT when only
as absent, NOT_MENTIONED otherwise; the AUROC is the lenient one of D55 (PRESENT against ABSENT
or NOT_MENTIONED). Calibration is that of the 14 columns (`image_scores.calibrate`). The corpus
is only scored, never labelled.
"""

from __future__ import annotations

import re
from collections.abc import Collection

import numpy as np
import polars as pl

from src.evaluation import zero_shot as zs

STATUSES = ("PRESENT", "UNCERTAIN", "ABSENT", "NOT_MENTIONED")
_NORMAL = re.compile(r"\bnormal\b", re.IGNORECASE)   # a word: "abnormality" is no normal statement


def descendants(edges: pl.DataFrame, cuis: Collection[str]) -> dict[str, set[str]]:
    """{cui: the cui and every concept below it along IS_A (child -> parent edges)}."""
    children: dict[str, list[str]] = {}
    for child, parent in edges.filter(pl.col("rel") == "IS_A").select("src", "dst").iter_rows():
        children.setdefault(parent, []).append(child)
    out = {}
    for cui in cuis:
        seen, stack = {cui}, [cui]
        while stack:
            for child in children.get(stack.pop(), ()):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        out[cui] = seen
    return out


def pool_strings(links: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """cui, string, mentions: the pool strings linked to each CUI, single and pair links (a pair
    is written "observation anatomy", or its exact MRCONSO phrase when it has one)."""
    single = (links.filter(pl.col("cui").is_not_null())
              .select("cui", "string", pl.col("mentions").cast(pl.Int64)))
    paired = (pairs.filter(pl.col("cui").is_not_null())
              .select("cui", pl.coalesce("phrase", pl.concat_str("obs", "anat", separator=" "))
                      .alias("string"), pl.col("n").cast(pl.Int64).alias("mentions")))
    return (pl.concat([single, paired]).group_by("cui", "string").agg(pl.col("mentions").sum())
            .sort(["cui", "mentions", "string"], descending=[False, True, False]))


def candidates(nodes: pl.DataFrame, strings: pl.DataFrame, below: dict[str, set[str]],
               cfg: dict) -> pl.DataFrame:
    """cui, name, labels, n_pool, captions, status (candidate or excluded), reason: the core
    findings and diseases without a column, by pool mentions. Excluded: normal statements,
    generic words (`cfg.generic`), anatomy and devices. `cfg.forced` CUIs are always weighed; the
    others fill the list up to `cfg.max_candidates`."""
    pool = (nodes.filter(pl.col("in_core"), ~pl.col("scorable"))
            .sort(["n_pool", "cui"], descending=[True, False]))
    rows = []
    for cui, name, labels, n in pool.select("cui", "name", "labels", "n_pool").iter_rows():
        own = (strings.filter(pl.col("cui").is_in(list(below.get(cui, {cui}))))
               .group_by("string").agg(pl.col("mentions").sum())
               .sort(["mentions", "string"], descending=[True, False]))
        captions = own["string"].head(cfg["captions"]).to_list()
        kinds = set((labels or "").split(","))
        if kinds & {"Anatomy", "Device"} or not kinds & {"Finding", "Disease"}:
            reason = "anatomy or device"
        elif any(_NORMAL.search(t) for t in [name or "", *captions]):
            reason = "normal statement"
        elif cui in cfg["generic"]:
            reason = f"generic: {cfg['generic'][cui]}"
        elif not captions:
            reason = "no pool string to caption it"
        else:
            reason = ""
        rows.append((cui, name, labels, int(n), " | ".join(captions), reason))
    table = pl.DataFrame(rows, schema=["cui", "name", "labels", "n_pool", "captions", "reason"],
                         orient="row")
    kept = table.filter(pl.col("reason") == "")
    forced = kept.filter(pl.col("cui").is_in(cfg["forced"]))
    rest = kept.filter(~pl.col("cui").is_in(cfg["forced"])).head(
        max(cfg["max_candidates"] - forced.height, 0))
    chosen = set(forced["cui"]) | set(rest["cui"])
    return table.with_columns(
        pl.when(pl.col("cui").is_in(list(chosen))).then(pl.lit("candidate"))
        .otherwise(pl.lit("excluded")).alias("status"),
        pl.when(pl.col("cui").is_in(list(chosen)) | (pl.col("reason") != "")).then("reason")
        .otherwise(pl.lit(f"beyond the first {cfg['max_candidates']}")).alias("reason"))


def silver_statuses(mcuis: pl.DataFrame, studies: list[str],
                    concepts: dict[str, set[str]]) -> np.ndarray:
    """[len(studies), len(concepts)] statuses of each study's report for each concept (with its
    IS_A descendants): PRESENT over UNCERTAIN over ABSENT, NOT_MENTIONED when silent."""
    want = set(studies)
    obs = mcuis.filter(pl.col("kind") == "Observation", pl.col("study_id").is_in(list(want)))
    out = np.full((len(studies), len(concepts)), "NOT_MENTIONED", dtype=object)
    row = {s: i for i, s in enumerate(studies)}
    rank = {"present": 0, "uncertain": 1, "absent": 2}
    for j, (_, cuis) in enumerate(concepts.items()):
        best = (obs.filter(pl.col("cui").is_in(list(cuis)), pl.col("polarity").is_in(list(rank)))
                .with_columns(pl.col("polarity").replace_strict(rank, return_dtype=pl.Int64)
                              .alias("r"))
                .group_by("study_id").agg(pl.col("r").min()))
        for study, r in best.iter_rows():
            if study in row:
                out[row[study], j] = STATUSES[r]
    return out


def lenient_auroc(logits: np.ndarray, statuses: np.ndarray, *, n_boot: int,
                  seed: int) -> dict:
    """D55: PRESENT against ABSENT or NOT_MENTIONED, UNCERTAIN dropped; the point estimate and
    a 95% percentile interval over `n_boot` resamples of the images."""
    pos, _, lenient = zs.label_masks(statuses)
    value = zs.auroc(logits, pos, lenient)
    keep = np.flatnonzero(pos | lenient)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        pick = rng.choice(keep, size=keep.size, replace=True)
        rep = zs.auroc(logits[pick], pos[pick], lenient[pick])
        if not np.isnan(rep):
            reps.append(rep)
    lo, hi = (np.percentile(reps, [2.5, 97.5]) if reps else (np.nan, np.nan))
    return {"auroc_lenient": value, "ci95": [float(lo), float(hi)],
            "n_present": int(pos.sum()), "n_negative": int(lenient.sum())}
