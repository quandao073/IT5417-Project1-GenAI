"""A tiny invented KG with the shape of data/kg/graph/*.parquet, a score matrix and `quality`:
shared by the loader, store and tool tests. Concept ids are made up (C-numbers below), names are
plain words, nothing comes from UMLS.

    effusion C1, cardiomegaly C2, edema C3: scorable columns of the 14-label kind
    pneumonia C4 (weak), fracture C10 (weak) with a scorable child C11 (extended column)
    lobar pneumonia C5 below C4; heart failure C6 suggested by C2, C3, C4 (weak) and C1 (dropped)
    pleural disease C7 above C1; pleura C8 with part pleural space C9, where C1 is typical
"""

from __future__ import annotations

import numpy as np
import polars as pl

COLUMNS = {"C1": "Pleural Effusion", "C2": "Cardiomegaly", "C3": "Edema", "C4": "Pneumonia",
           "C10": "Fracture", "C11": "C11"}
NAMES = {"C1": "Pleural effusion", "C2": "Cardiomegaly", "C3": "Edema", "C4": "Pneumonia",
         "C5": "Lobar pneumonia", "C6": "Heart failure", "C7": "Pleural disease",
         "C8": "Pleura", "C9": "Pleural space", "C10": "Fracture", "C11": "Rib fracture",
         "C12": "Lung volume loss", "C13": "Low lung volumes"}
LABELS = {"C1": "Finding", "C2": "Finding", "C3": "Finding", "C4": "Disease", "C5": "Disease",
          "C6": "Disease", "C7": "Disease", "C8": "Anatomy", "C9": "Anatomy", "C10": "Finding",
          "C11": "Finding", "C12": "Finding", "C13": "Finding"}
LABEL_OF = {"C1": "Pleural Effusion", "C2": "Cardiomegaly", "C3": "Edema", "C4": "Pneumonia",
            "C10": "Fracture"}
QUALITY = {"C1": 0.83, "C2": 0.71, "C3": 0.73, "C4": 0.63, "C10": 0.62, "C11": 0.74}


def nodes() -> pl.DataFrame:
    return pl.DataFrame([{
        "cui": c, "name": n, "labels": LABELS[c], "sty": "T047", "chexpert_label": LABEL_OF.get(c),
        "scorable": c in COLUMNS, "n_pool": 100 - i, "in_core": True, "reaches": True}
        for i, (c, n) in enumerate(NAMES.items())])


def _edge(src, rel, dst, **kw):
    return {"src": src, "rel": rel, "dst": dst, "source": kw.get("source", "test"),
            "n": kw.get("n"), "n_src": kw.get("n_src"), "p": kw.get("p"),
            "review": kw.get("review")}


def edges() -> pl.DataFrame:
    rows = [_edge("C5", "IS_A", "C4"), _edge("C1", "IS_A", "C7"), _edge("C11", "IS_A", "C10"),
            _edge("C9", "PART_OF", "C8"), _edge("C1", "HAS_STY", "T047", source="MRSTY"),
            _edge("T047", "IS_A", "T000", source="STN"),
            _edge("C2", "SUGGESTS", "C6", n=30, n_src=100, p=0.3, review="keep"),
            _edge("C3", "SUGGESTS", "C6", n=25, n_src=250, p=0.1, review="keep"),
            _edge("C4", "SUGGESTS", "C6", n=30, n_src=100, p=0.3, review="keep"),
            _edge("C1", "SUGGESTS", "C6", n=40, n_src=400, p=0.1, review="drop"),
            _edge("C1", "TYPICALLY_AT", "C9", n=50, n_src=200, p=0.25),
            _edge("C2", "TYPICALLY_AT", "C9", n=5, n_src=100, p=0.05)]
    return pl.DataFrame(rows, schema={"src": pl.String, "rel": pl.String, "dst": pl.String,
                                      "source": pl.String, "n": pl.Int64, "n_src": pl.Int64,
                                      "p": pl.Float64, "review": pl.String})


def aliases() -> pl.DataFrame:
    rows = [("C1", "pleural effusion", "en"), ("C1", "tràn dịch màng phổi", "vi"),
            ("C2", "cardiomegaly", "en"), ("C2", "tim to", "vi"), ("C3", "edema", "en"),
            ("C3", "phù phổi", "vi"), ("C4", "pneumonia", "en"), ("C4", "viêm phổi", "vi"),
            ("C5", "lobar pneumonia", "en"), ("C6", "heart failure", "en"),
            ("C6", "suy tim", "vi"), ("C7", "pleural disease", "en"),
            ("C8", "pleura", "en"), ("C9", "pleural space", "en"), ("C10", "fracture", "en"),
            ("C11", "rib fracture", "en"),
            ("C12", "giảm thể tích phổi", "vi"), ("C13", "giảm thể tích phổi", "vi"),
            ("C13", "thể tích phổi thấp", "vi")]
    return pl.DataFrame([{"cui": c, "text": t, "norm": t, "lang": lang, "source": "test"}
                         for c, t, lang in rows])


def semantic_types() -> pl.DataFrame:
    return pl.DataFrame({"tui": ["T047", "T000"], "name": ["Disease", "Entity"],
                         "stn": ["B2.2", "B2"]})


def quality() -> dict:
    out = {}
    for cui, col in COLUMNS.items():
        a = QUALITY[cui]
        out[cui] = {"column": col, "auroc_lenient": a, "ci95": [a - 0.02, a + 0.02],
                    "n_present": 100, "weak": a < 0.65, "borderline": a - 0.02 < 0.65}
    return out


def scores(n: int = 8) -> pl.DataFrame:
    """n images: image i is 'effusion-like' when i % 2 == 0, 'cardiomegaly-like' when i % 3 == 0."""
    rng = np.random.default_rng(1)
    cols = {}
    for col in COLUMNS.values():
        cols[col] = rng.normal(-1.0, 0.5, n)
    cols["Pleural Effusion"][::2] += 4.0
    cols["Cardiomegaly"][::3] += 4.0
    return pl.DataFrame({"study_id": [f"s{i}" for i in range(n)],
                         "image_id": [f"i{i}" for i in range(n)], **cols})


def images(n: int = 8) -> pl.DataFrame:
    return pl.DataFrame({"image_id": [f"i{i}" for i in range(n)],
                         "study_id": [f"s{i}" for i in range(n)],
                         "view": ["PA" if i % 4 == 0 else "AP" for i in range(n)]})
