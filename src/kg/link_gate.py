"""Gate of the V2-3 linker (D56): a seeded sample of LLM links that a physician grades, or an LLM
judge as a silver standard; precision with its Wilson interval; coverage by mentions comes from
`linker.coverage` in the manifest. The sample also carries the SapBERT argmax CUI of each string,
so the same grading compares "SapBERT argmax" with "SapBERT + LLM".
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from src.kg.linker import CUI, LLM

GRADES = ("correct", "incorrect", "unsure")
SAME = "same"   # argmax_grade when the argmax CUI is the link itself


def _context_text(pairs: list[list]) -> str:
    return "; ".join(f"{other} ({n})" for other, n in pairs)


def sample(links: pl.DataFrame, info: pl.DataFrame, context: dict, cfg: dict) -> pl.DataFrame:
    """`cfg.sample` strings drawn with `cfg.seed` from the links both LLMs agreed on (never an
    adjudicated or rule-decided string), with empty grade columns."""
    pool = (links.filter(pl.col("method") == LLM, pl.col("decision") == CUI)
            .sort("kind", "string"))
    picked = np.random.default_rng(cfg["seed"]).choice(
        pool.height, size=min(cfg["sample"], pool.height), replace=False)
    names = {c: (n, t) for c, n, t in info.select("cui", "name", "semantic_types").iter_rows()}
    rows = []
    for i, row in enumerate(pool[sorted(picked.tolist())].iter_rows(named=True), start=1):
        link, argmax = row["cui"], row["argmax_cui"]
        rows.append({
            "sample_id": i, "kind": row["kind"], "string": row["string"],
            "mentions": row["mentions"],
            "located": _context_text(context.get((row["kind"], row["string"]), [])),
            "link_cui": link, "link_name": names.get(link, ("", ""))[0],
            "link_types": names.get(link, ("", ""))[1],
            "argmax_cui": argmax, "argmax_name": names.get(argmax, ("", ""))[0],
            "argmax_types": names.get(argmax, ("", ""))[1],
            "link_grade": "", "argmax_grade": SAME if argmax == link else "",
            "grader": "", "note": ""})
    return pl.DataFrame(rows)


def judge_items(graded: pl.DataFrame) -> list[dict]:
    """One judge item per link and per differing argmax CUI of the sample."""
    items = []
    for row in graded.iter_rows(named=True):
        for which in ("link", "argmax"):
            if which == "argmax" and row["argmax_grade"] == SAME:
                continue
            items.append({"kind": row["kind"], "string": f"{row['string']}#{which}",
                          "input": {"string": row["string"], "kind": row["kind"],
                                    "mentions": row["mentions"], "located": row["located"],
                                    "concept": {"cui": row[f"{which}_cui"],
                                                "name": row[f"{which}_name"],
                                                "semantic_types": row[f"{which}_types"]}}})
    return items


def apply_grades(graded: pl.DataFrame, grades: dict[tuple[str, str], dict],
                 grader: str) -> pl.DataFrame:
    """The sample with the judge's grades filled in where it answered."""
    def grade(kind: str, string: str, which: str, current: str) -> str:
        answer = grades.get((kind, f"{string}#{which}"))
        return answer["grade"] if answer else current

    return graded.with_columns(
        pl.Series("link_grade", [grade(k, s, "link", g) for k, s, g in
                                 graded.select("kind", "string", "link_grade").iter_rows()]),
        pl.Series("argmax_grade", [grade(k, s, "argmax", g) for k, s, g in
                                   graded.select("kind", "string", "argmax_grade").iter_rows()]),
        pl.lit(grader).alias("grader"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n))


def precision(graded: pl.DataFrame, column: str) -> dict:
    """Correct over correct + incorrect (unsure left out), with the Wilson 95% interval. An
    argmax graded `same` takes the link's grade."""
    grades = graded.select(pl.when(pl.col(column) == SAME).then(pl.col("link_grade"))
                           .otherwise(pl.col(column)).alias("g"))["g"].to_list()
    k, n = grades.count("correct"), grades.count("correct") + grades.count("incorrect")
    lo, hi = wilson(k, n)
    weights = [int(m) for m in graded["mentions"]] if "mentions" in graded.columns else [1] * len(
        grades)
    right = sum(w for g, w in zip(grades, weights, strict=True) if g == "correct")
    judged = sum(w for g, w in zip(grades, weights, strict=True) if g in ("correct", "incorrect"))
    return {"correct": k, "graded": n, "unsure": grades.count("unsure"),
            "precision": k / n if n else None, "wilson95": [lo, hi],
            "precision_by_mentions": right / judged if judged else None}


def gate(graded: pl.DataFrame, coverage: dict, cfg: dict) -> dict:
    link = precision(graded, "link_grade")
    decided = coverage["all"]["decided"]
    return {"link": link, "argmax": precision(graded, "argmax_grade"),
            "coverage_decided": decided,
            "precision_passed": link["precision"] is not None
            and link["precision"] >= cfg["precision"],
            "coverage_passed": decided >= cfg["coverage_by_mentions"]}
