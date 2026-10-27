"""Candidate CUIs for the 14 CheXpert labels (TIP V2-2 step 5). No LLM, no SapBERT.

A label's name and its `en` forms in configs/concepts.yaml are matched exactly, after
`vocab.normalize`, against MRCONSO names. Up to three CUIs per label are kept, ranked by whether
the label's own name matched, then how many forms matched, then whether a matched name is preferred
(TS = P, ISPREF = Y), then how many sources name the CUI. A label with no exact match is proposed
as PRJ:<name>, with up to three candidates for its is_a, found the same way from the head word
(last word, singular) of its forms. No Finding is the special project ID PRJ:no_finding, with no
candidates and no is_a.

The candidate file holds UMLS names, so it stays under data/ (T38). The mapping written after the
user's review, configs/kg/chexpert_cuis.yaml, holds codes only; `check_mapping` validates it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from src.data.labels import CHEXPERT_LABELS
from src.kg import vocab

NO_FINDING = "No Finding"
NO_FINDING_ID = "PRJ:no_finding"
_CUI = re.compile(r"C\d{7}")
_PRJ = re.compile(r"PRJ:[a-z0-9_]+")
_SAB_ORDER = ["MSH", "NCI", "FMA", "HPO"]   # which source's preferred name to show


class CandidateError(RuntimeError):
    """A label has neither an exact CUI nor an is_a candidate."""


def prj_id(label: str) -> str:
    return "PRJ:" + vocab.normalize(label).replace(" ", "_")


def label_forms(concepts_cfg: dict) -> dict[str, list[str]]:
    """Each label's normalized name, then its normalized `en` forms, without repeats."""
    return {label: list(dict.fromkeys(vocab.normalize(f) for f in [label, *forms["en"]]))
            for label, forms in concepts_cfg["concepts"].items()}


def _rank(forms_by_label: Mapping[str, Sequence[str]], mrconso: Path, mrsty: Path,
          k: int) -> pl.DataFrame:
    """label, rank, cui, name, sabs, semantic_types, matched_forms for the top `k` CUIs."""
    forms = pl.DataFrame([(label, form, form == vocab.normalize(label))
                          for label, fs in forms_by_label.items() for form in fs],
                         schema={"label": pl.String, "form": pl.String, "own": pl.Boolean},
                         orient="row")
    norm = (pl.col("STR").str.to_lowercase().str.replace_all(r"[^a-z0-9]+", " ")
            .str.strip_chars())
    conso = pl.scan_parquet(mrconso)
    hits = (conso.select("CUI", norm.alias("form"), "TS", "ISPREF")
            .join(forms.lazy(), on="form", how="inner").collect(engine="streaming"))
    if hits.is_empty():
        return pl.DataFrame()
    per = hits.group_by("label", "CUI").agg(
        pl.col("own").any().alias("own_name"),
        pl.col("form").n_unique().alias("n_forms"),
        ((pl.col("TS") == "P") & (pl.col("ISPREF") == "Y")).any().alias("preferred"),
        pl.col("form").unique().sort().str.join("; ").alias("matched_forms"))
    cuis = per["CUI"].unique().to_list()
    rows = conso.filter(pl.col("CUI").is_in(cuis)).select("CUI", "STR", "SAB", "TS", "ISPREF",
                                                          "STT").collect()
    sabs = rows.group_by("CUI").agg(pl.col("SAB").unique().sort().alias("sabs"))
    order = {sab: i for i, sab in enumerate(_SAB_ORDER)}
    names = (rows.filter(pl.col("TS") == "P", pl.col("ISPREF") == "Y", pl.col("STT") == "PF")
             .with_columns(pl.col("SAB").replace_strict(order, default=len(order),
                                                        return_dtype=pl.Int64).alias("o"))
             .sort("o", "SAB").group_by("CUI", maintain_order=True).first()
             .select("CUI", pl.col("STR").alias("name")))
    sty = (pl.scan_parquet(mrsty).filter(pl.col("CUI").is_in(cuis)).collect()
           .group_by("CUI").agg(pl.concat_str("TUI", "STY", separator=" ").unique().sort()
                                .str.join("; ").alias("semantic_types")))
    return (per.join(sabs, on="CUI").join(names, on="CUI", how="left")
            .join(sty, on="CUI", how="left")
            .with_columns(pl.col("sabs").list.len().alias("n_sabs"))
            .sort(["label", "own_name", "n_forms", "preferred", "n_sabs", "CUI"],
                  descending=[False, True, True, True, True, False])
            .with_columns((pl.int_range(pl.len()).over("label") + 1).alias("rank"))
            .filter(pl.col("rank") <= k)
            .select("label", "rank", pl.col("CUI").alias("cui"), "name",
                    pl.col("sabs").list.join("; "), "semantic_types", "matched_forms"))


def candidates(concepts_cfg: dict, mrconso: Path, mrsty: Path, k: int = 3) -> pl.DataFrame:
    """One row per candidate: kind "cui" (exact match), "is_a" (for a PRJ id) or "prj"
    (No Finding). Raises CandidateError for a label with neither kind of candidate."""
    forms = label_forms(concepts_cfg)
    exact = _rank({label: f for label, f in forms.items() if label != NO_FINDING}, mrconso,
                  mrsty, k)
    matched = set(exact["label"]) if not exact.is_empty() else set()
    missing = [label for label in forms if label != NO_FINDING and label not in matched]
    words = {label: [f.rsplit(" ", 1)[-1] for f in forms[label]] for label in missing}
    lookup = vocab.mrconso_names(mrconso)
    known = lookup({c for ws in words.values() for w in ws if (c := vocab.singular_candidate(w))})
    heads = {label: list(dict.fromkeys(vocab.singular(w, known) for w in ws))
             for label, ws in words.items()}
    is_a = _rank(heads, mrconso, mrsty, k) if heads else pl.DataFrame()
    found = set(is_a["label"]) if not is_a.is_empty() else set()
    if empty := [label for label in missing if label not in found]:
        raise CandidateError(f"no CUI and no is_a candidate for: {', '.join(empty)}")

    parts = []
    if not exact.is_empty():
        parts.append(exact.with_columns(pl.lit("cui").alias("kind"),
                                        pl.lit(None, pl.String).alias("proposed_id")))
    if not is_a.is_empty():
        parts.append(is_a.with_columns(
            pl.lit("is_a").alias("kind"),
            pl.col("label").map_elements(prj_id, return_dtype=pl.String).alias("proposed_id")))
    if NO_FINDING in forms:
        parts.append(pl.DataFrame({"label": [NO_FINDING], "kind": ["prj"],
                                   "proposed_id": [NO_FINDING_ID]}))
    columns = ["label", "kind", "rank", "cui", "name", "sabs", "semantic_types", "matched_forms",
               "proposed_id"]
    return pl.concat(parts, how="diagonal_relaxed").select(columns)


def check_mapping(mapping: dict, labels: Sequence[str] = CHEXPERT_LABELS) -> list[str]:
    """Problems with configs/kg/chexpert_cuis.yaml: every label once; `id` a CUI, or PRJ:<name>
    with an `is_a` CUI and optional `children` (narrower CUIs that get IS_A to it, D53); No Finding
    exactly PRJ:no_finding without is_a or children. [] means valid."""
    entries = mapping.get("labels") or {}
    problems = [f"{label}: missing" for label in labels if label not in entries]
    problems += [f"{label}: not a CheXpert label" for label in entries if label not in labels]
    for label, entry in entries.items():
        entry = entry or {}
        id_, is_a, children = str(entry.get("id")), entry.get("is_a"), entry.get("children")
        if label == NO_FINDING:
            if id_ != NO_FINDING_ID or is_a is not None or children is not None:
                problems.append(f"No Finding: must be {NO_FINDING_ID} without is_a or children")
        elif _CUI.fullmatch(id_):
            if is_a is not None or children is not None:
                problems.append(f"{label}: a CUI id takes no is_a or children")
        elif _PRJ.fullmatch(id_):
            if not (isinstance(is_a, str) and _CUI.fullmatch(is_a)):
                problems.append(f"{label}: a PRJ id needs an is_a CUI")
            if children is not None and not (isinstance(children, list) and children and all(
                    isinstance(c, str) and _CUI.fullmatch(c) for c in children)):
                problems.append(f"{label}: children must be a list of CUIs")
        else:
            problems.append(f"{label}: id {id_!r} is neither a CUI nor PRJ:<name>")
    return problems
