"""Pool vocabulary for the KG (TIP V2-2; docs/v2-kg-design.md, "Dựng KG offline" step 2; D52).

The pool's impression and findings are joined to their RadGraph-XL records by normalized text,
never by position (T29). Each entity that modifies nothing is a mention: its head, plus the
entities that `modify` it. A modifier entity belongs to its head's phrase and is not a mention of
its own. The joined string drops the modifiers on the closed lists of configs/kg.yaml (all four
groups for an Observation, only laterality for an Anatomy) and keeps the rest only when the joined
phrase is an MRCONSO name; otherwise the head stands alone.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from pathlib import Path

import polars as pl

from src.common.ids import normalize_image_path

SECTIONS = {"impression": "section_impression", "findings": "section_findings"}
KINDS = ("Observation", "Anatomy")
_POLARITY = {"definitely present": "present", "definitely absent": "absent",
             "uncertain": "uncertain"}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_KEEP_ENDINGS = ("as", "ss", "us", "is")   # "gas", "mass", "status", "atelectasis" stay as they are

Names = Callable[[Iterable[str]], set[str]]   # candidate strings -> those that are MRCONSO names


class VocabError(RuntimeError):
    """An input does not line up (pool vs Plus CSV); nothing is measured."""


def normalize(text: str) -> str:
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def text_key(text: str) -> str:
    """The join key between a report section and its RadGraph record (T29)."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def singular_candidate(phrase: str) -> str | None:
    rest, _, last = phrase.rpartition(" ")
    if not last.endswith("s") or last.endswith(_KEEP_ENDINGS):
        return None
    last = last[:-3] + "y" if last.endswith("ies") else last[:-1]
    return f"{rest} {last}" if rest else last


def singular(phrase: str, known: Collection[str]) -> str:
    """The singular form when it is known (a pool head or an MRCONSO name), else `phrase`."""
    candidate = singular_candidate(phrase)
    return candidate if candidate is not None and candidate in known else phrase


def drop_sets(cfg: dict) -> dict[str, frozenset[str]]:
    groups = cfg["drop_modifiers"]
    return {"Observation": frozenset(word for group in groups.values() for word in group),
            "Anatomy": frozenset(groups["laterality"])}


def pool_sections(pool_images: pl.DataFrame, plus_csv: Path) -> pl.DataFrame:
    """study_id, section, text for every non-empty impression and findings of the pool (T1, T2)."""
    def norm(column: str) -> pl.Expr:
        return pl.col(column).map_elements(normalize_image_path, return_dtype=pl.String)

    plus = pl.read_csv(plus_csv, columns=["path_to_image", *SECTIONS.values()])
    joined = pool_images.select("study_id", norm("path_to_image")).join(
        plus.with_columns(norm("path_to_image")), on="path_to_image", how="inner")
    if joined.height != pool_images.height or joined["study_id"].n_unique() != pool_images.height:
        raise VocabError(f"{joined.height} Plus CSV rows for {pool_images.height} pool studies")
    parts = [joined.select("study_id", pl.lit(section).alias("section"),
                           pl.col(column).str.strip_chars().alias("text"))
             for section, column in SECTIONS.items()]
    return pl.concat(parts).filter(pl.col("text").is_not_null() & (pl.col("text") != ""))


def _mentions(entities: dict) -> Iterator[tuple[str, str | None, list[str], int]]:
    """(kind, polarity, tokens in text order, head index) for each entity that modifies nothing."""
    modifiers: dict[str, list[str]] = {}
    for eid, entity in entities.items():
        for relation, target in entity["relations"]:
            if relation == "modify":
                modifiers.setdefault(target, []).append(eid)
    for eid, entity in entities.items():
        if any(relation == "modify" for relation, _ in entity["relations"]):
            continue
        kind, *_, status = entity["label"].split("::")
        parts = sorted([eid, *modifiers.get(eid, [])], key=lambda i: entities[i]["start_ix"])
        polarity = _POLARITY.get(status) if kind == "Observation" else None
        yield kind, polarity, [normalize(entities[i]["tokens"]) for i in parts], parts.index(eid)


def read_mentions(sections: pl.DataFrame, radgraph_dir: Path) -> tuple[pl.DataFrame, dict]:
    """One row per mention and study, and per section how many texts found their RadGraph record."""
    rows, stats = [], {}
    for section in SECTIONS:
        part = sections.filter(pl.col("section") == section)
        want: dict[str, list[str]] = {}
        for study, text in part.select("study_id", "text").iter_rows():
            want.setdefault(text_key(text), []).append(study)
        with open(radgraph_dir / f"section_{section}.json", encoding="utf-8") as handle:
            records = json.load(handle)
        matched = 0
        for record in records:
            graph = record["0"]
            studies = want.pop(text_key(graph["text"]), None)   # equal text, equal graph
            if not studies:
                continue
            matched += len(studies)
            for kind, polarity, tokens, head in _mentions(graph["entities"]):
                rows.extend((study, section, kind, polarity, tokens, head) for study in studies)
        stats[section] = {"texts": part.height, "matched": matched}
    schema = {"study_id": pl.String, "section": pl.String, "kind": pl.String, "polarity": pl.String,
              "tokens": pl.List(pl.String), "head_index": pl.Int64}
    return pl.DataFrame(rows, schema=schema, orient="row"), stats


def _candidates(tokens: Sequence[str], head_index: int, head: str,
                drop: frozenset[str]) -> list[str]:
    """Joined strings to try, in order: every kept modifier with the head, then each one alone."""
    kept = [i for i, token in enumerate(tokens) if i != head_index and token and token not in drop]
    if not kept:
        return []
    words = [*tokens]
    words[head_index] = head
    phrases = [" ".join(words[i] for i in sorted([*kept, head_index]))]
    phrases += [" ".join(words[i] for i in sorted([k, head_index])) for k in kept]
    return list(dict.fromkeys(phrases))


def apply_rules(mentions: pl.DataFrame, cfg: dict, names: Names) -> pl.DataFrame:
    """Add `head` (singular), `joined` (rule of step 3) and `dropped_head` (head on a drop list)."""
    drops = drop_sets(cfg)
    kinds, tokens, indexes = (mentions[c].to_list() for c in ("kind", "tokens", "head_index"))
    raw = [t[i] for t, i in zip(tokens, indexes, strict=True)]
    known = set(raw) | names({c for h in set(raw) if (c := singular_candidate(h))})
    single = {h: singular(h, known) for h in set(raw)}

    keys = [(k, tuple(t), i) for k, t, i in zip(kinds, tokens, indexes, strict=True)]
    tries = {key: _candidates(key[1], key[2], single[key[1][key[2]]], drops[key[0]])
             for key in set(keys)}
    hits = names({c for cands in tries.values() for c in cands})
    joined = {key: next((c for c in cands if c in hits), single[key[1][key[2]]])
              for key, cands in tries.items()}
    return mentions.with_columns(
        pl.Series("head", [single[h] for h in raw], dtype=pl.String),
        pl.Series("joined", [joined[key] for key in keys], dtype=pl.String),
        pl.Series("dropped_head", [h in drops[k] for h, k in zip(raw, kinds, strict=True)],
                  dtype=pl.Boolean),
    ).filter(pl.col("head") != "")


def mrconso_names(mrconso: Path) -> Names:
    """Which candidate strings equal an MRCONSO name after `normalize`."""
    def lookup(candidates: Iterable[str]) -> set[str]:
        wanted = list(candidates)
        if not wanted:
            return set()
        norm = (pl.col("STR").str.to_lowercase().str.replace_all(r"[^a-z0-9]+", " ")
                .str.strip_chars())
        hits = (pl.scan_parquet(mrconso).select(norm.alias("name"))
                .filter(pl.col("name").is_in(wanted)).unique().collect(engine="streaming"))
        return set(hits["name"].to_list())
    return lookup


def vocab_table(mentions: pl.DataFrame, column: str) -> pl.DataFrame:
    """kind, string, mentions, studies for `column` ("head" or "joined"); heads on the drop
    lists are left out."""
    return (mentions.filter(~pl.col("dropped_head"))
            .group_by("kind", pl.col(column).alias("string"))
            .agg(pl.len().alias("mentions"), pl.col("study_id").n_unique().alias("studies"))
            .sort(["kind", "mentions", "string"], descending=[False, True, False]))


def curve(counts: Sequence[int], levels: Sequence[float]) -> dict[float, int]:
    """For each coverage level, how many of the most frequent strings it takes to reach it."""
    ordered, pending = sorted(counts, reverse=True), sorted(levels)
    total, acc, out = sum(ordered), 0, {}
    for n, count in enumerate(ordered, start=1):
        acc += count
        while pending and acc >= pending[0] * total - 1e-9:
            out[pending.pop(0)] = n
    return out


def summarize(mentions: pl.DataFrame, cfg: dict) -> dict:
    """Both coverage curves per kind, the gate, and the mentions whose head is on the drop lists."""
    curves = {}
    for column, name in (("head", "head_only"), ("joined", "rule")):
        table = vocab_table(mentions, column)
        for kind in KINDS:
            counts = table.filter(pl.col("kind") == kind)["mentions"].to_list()
            curves[f"{kind}/{name}"] = {
                "strings": len(counts), "mentions": sum(counts),
                "levels": {str(level): n
                           for level, n in curve(counts, cfg["coverage_levels"]).items()}}
    gate = cfg["gate"]
    needed = curves[f"{gate['kind']}/rule"]["levels"].get(str(gate["coverage"]))
    dropped = {kind: {"mentions": mentions.filter(pl.col("kind") == kind,
                                                  pl.col("dropped_head")).height,
                      "of": mentions.filter(pl.col("kind") == kind).height} for kind in KINDS}
    return {"curves": curves, "dropped_heads": dropped,
            "gate": {**gate, "strings": needed,
                     "passed": needed is not None and needed <= gate["max_strings"]}}
