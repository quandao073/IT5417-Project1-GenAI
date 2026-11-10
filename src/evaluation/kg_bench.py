"""The KG search benchmark (V2-7): benchmarks/kg_queries_v1.jsonl, 8 groups x 9 questions,
half English and half Vietnamese, split 36 / 36 into dev and heldout.

The questions and their gold are written by one model (`llm.author`), group by group, from a
catalog of graph concepts; a second model (`llm.reviewer`) reviews every question on its own;
an automatic check expands each gold concept through the graph and records where the graph
disagrees with the gold, without changing the gold. A question the reviewer disagrees with is
kept and marked `gold_disputed`. The file holds codes and question text only (CLAUDE.md §5).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

WARNING_CODES = {
    "weak_quality": "a weak score column is used (unreliable image scores)",
    "narrower": "the concept is narrower than any column and is scored at the column above it",
    "negation_skipped": "a negated narrow concept cannot be excluded (nothing to exclude it by)",
    "disease_inferred": "a disease is inferred from the signs that suggest it",
    "anatomy_region": "a region is matched through the findings typically seen there",
    "location_ignored": "a region next to a finding is not ranked by; the finding is",
    "unrecognized": "a phrase matches no concept of the graph",
    "no_path": "a concept of the graph has no route to any score column",
    "side_ignored": "a side (left, right, bilateral) or a severity is not ranked by",
    "uncertain_as_present": "a 'possible' finding is searched as present",
    "not_considered": "a phrase made only of qualifying words is not considered",
}
VIEWS = ("AP", "PA")
POLARITIES = ("PRESENT", "ABSENT")


def _strict(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


_STR = {"type": "string"}
_TERM = _strict({"text": _STR, "cui": _STR,
                 "polarity": {"type": "string", "enum": list(POLARITIES)}})
_SCORABLE = _strict({"cui": _STR, "weak": {"type": "boolean"}})
QUESTION_SCHEMA = _strict({
    "question": _STR, "lang": {"type": "string", "enum": ["en", "vi"]},
    "translation_of": {"type": "integer"}, "query_en": _STR,
    "views": {"type": "array", "items": {"type": "string", "enum": list(VIEWS)}},
    "terms": {"type": "array", "items": _TERM},
    "expected_scorable": {"type": "array", "items": _SCORABLE},
    "exclude": {"type": "array", "items": _STR},
    "expect_warnings": {"type": "array", "items": {"type": "string",
                                                   "enum": list(WARNING_CODES)}},
    "note": _STR})
AUTHOR_SCHEMA = _strict({"questions": {"type": "array", "items": QUESTION_SCHEMA}})
REVIEW_SCHEMA = _strict({"reviews": {"type": "array", "items": _strict({
    "index": {"type": "integer"}, "verdict": {"type": "string", "enum": ["agree", "disagree"]},
    "reason": _STR})}})


def normalize(text: str) -> str:
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFC", text).lower()).strip()


def catalog(nodes, quality: dict, extra_cuis: Sequence[str]) -> tuple[list[dict], list[dict]]:
    """(score_columns, catalog): every scorable concept with its weak flag, and the core
    concepts plus `extra_cuis` (ends of hand-written edges) as cui, kind, name, scorable."""
    import polars as pl

    names = dict(nodes.select("cui", "name").iter_rows())
    columns = [{"cui": c, "name": names.get(c, c), "weak": bool(quality[c]["weak"])}
               for c in sorted(quality)]
    keep = nodes.filter(pl.col("in_core") | pl.col("cui").is_in(list(extra_cuis)))
    rows = [{"cui": c, "kind": label.split(";")[0] if label else "", "name": name,
             "scorable": bool(s)}
            for c, label, name, s in keep.sort("cui").select(
                "cui", "labels", "name", "scorable").iter_rows()]
    return columns, rows


def lang_counts(group_index: int, per_group: int) -> tuple[int, int]:
    """(n_en, n_vi) of a group: 5/4 and 4/5 alternate, so the whole set is half and half."""
    big, small = per_group - per_group // 2, per_group // 2
    return (big, small) if group_index % 2 == 0 else (small, big)


def structure_problems(questions: Sequence[dict], n_en: int, n_vi: int,
                       known_cuis: set[str], used: set[str]) -> list[str]:
    """What a group of authored questions breaks of the rules the prompt states."""
    out = []
    langs = [q["lang"] for q in questions]
    if len(questions) != n_en + n_vi:
        out.append(f"{len(questions)} questions, not {n_en + n_vi}")
    if langs.count("en") != n_en or langs.count("vi") != n_vi:
        out.append(f"{langs.count('en')} en / {langs.count('vi')} vi, not {n_en} / {n_vi}")
    pairs = [q for q in questions if q["translation_of"] >= 0]
    if len(pairs) < 2:
        out.append(f"{len(pairs)} translation pairs, need at least 2")
    for i, q in enumerate(questions):
        j = q["translation_of"]
        if j >= 0 and (j >= len(questions) or j == i or questions[j]["translation_of"] >= 0
                       or questions[j]["lang"] == q["lang"]):
            out.append(f"question {i}: translation_of {j} is not the first of a pair")
        bad = ({t["cui"] for t in q["terms"] if t["cui"]}
               | {c["cui"] for c in q["expected_scorable"]} | set(q["exclude"])) - known_cuis
        if bad:
            out.append(f"question {i}: CUIs not in the catalog {sorted(bad)}")
        if normalize(q["question"]) in used:
            out.append(f"question {i}: reuses an existing question")
    if not any(q["views"] for q in questions):
        out.append("no question names a view")
    if not any(any(not t["cui"] for t in q["terms"]) for q in questions):
        out.append("no question has an unscorable phrase (a term with cui '')")
    return out


def assign_ids(group: str, questions: Sequence[dict]) -> list[dict]:
    """Rows of the benchmark file: id, group, pair (the id of the first of a translation pair,
    the same on both), and the gold fields."""
    ids = [f"kg-{group}-{i + 1:02d}" for i in range(len(questions))]
    rows = []
    for i, q in enumerate(questions):
        j = q["translation_of"]
        pair = ids[j] if j >= 0 else (ids[i] if any(
            o["translation_of"] == i for o in questions) else None)
        rows.append({"id": ids[i], "group": group, "lang": q["lang"], "pair": pair,
                     "question": q["question"], "query_en": q["query_en"],
                     "views": sorted(set(q["views"])), "terms": q["terms"],
                     "expected_scorable": q["expected_scorable"], "exclude": q["exclude"],
                     "expect_warnings": sorted(set(q["expect_warnings"])), "note": q["note"]})
    return rows


def split(rows: Sequence[dict], dev_per_group: Sequence[int], seed: int) -> dict[str, str]:
    """{id: dev | heldout}. A translation pair is one unit. In each group (in order) the dev
    share alternates over `dev_per_group`; among the subsets of units with that many questions,
    the one closest to half English is taken, ties broken by a seeded shuffle."""
    out: dict[str, str] = {}
    groups = list(dict.fromkeys(r["group"] for r in rows))
    rng = random.Random(seed)
    for gi, group in enumerate(groups):
        members = [r for r in rows if r["group"] == group]
        units: dict[str, list[dict]] = {}
        for r in members:
            units.setdefault(r["pair"] or r["id"], []).append(r)
        keys = sorted(units)
        rng.shuffle(keys)
        want = dev_per_group[gi % len(dev_per_group)]
        best = None
        for n in range(len(keys) + 1):
            for combo in itertools.combinations(keys, n):
                chosen = [r for k in combo for r in units[k]]
                if len(chosen) != want:
                    continue
                en = sum(r["lang"] == "en" for r in chosen)
                score = abs(2 * en - want)
                if best is None or score < best[0]:
                    best = (score, set(combo))
        if best is None:
            raise ValueError(f"group {group}: no set of units has {want} questions")
        for k in keys:
            for r in units[k]:
                out[r["id"]] = "dev" if k in best[1] else "heldout"
    return out


def expand_gold(store, row: dict, expand_cfg: dict, expand) -> dict:
    """The graph's columns for the gold concepts, combined like the search does (a region
    beside a finding is dropped), against the gold columns. Recorded, never applied."""
    intents = [expand.expand_concepts(store, t["cui"], t["polarity"], expand_cfg, t["text"])
               for t in row["terms"] if t["cui"]]
    if any(i.kind == "anatomy" for i in intents) and any(
            i.columns and i.kind != "anatomy" for i in intents):
        intents = [i for i in intents if i.kind != "anatomy"]
    got_inc = sorted({c.cui for i in intents if i.polarity == "PRESENT" for c in i.columns})
    got_exc = sorted({c.cui for i in intents if i.polarity == "ABSENT" for c in i.columns})
    want_inc = sorted({c["cui"] for c in row["expected_scorable"]})
    want_exc = sorted(set(row["exclude"]))
    return {"columns_match": got_inc == want_inc and got_exc == want_exc,
            "graph_include": got_inc, "graph_exclude": got_exc,
            "missing": sorted((set(want_inc) - set(got_inc)) | (set(want_exc) - set(got_exc))),
            "extra": sorted((set(got_inc) - set(want_inc)) | (set(got_exc) - set(want_exc))),
            "kinds": [i.kind for i in intents],
            "warnings": sorted({w.code for i in intents for w in i.warnings})}


def weak_problems(row: dict, quality: dict) -> list[str]:
    return [f"{c['cui']} weak={c['weak']} but quality says {bool(quality[c['cui']]['weak'])}"
            for c in row["expected_scorable"]
            if c["cui"] in quality and bool(quality[c["cui"]]["weak"]) != c["weak"]]


def write_jsonl(path: Path, rows: Sequence[dict]) -> str:
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def split_sha256(rows: Sequence[dict], which: str) -> str:
    """sha256 of one split's rows, in id order, as written (for WORKLOG)."""
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                   for r in sorted(rows, key=lambda r: r["id"]) if r["split"] == which)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
