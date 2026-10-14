"""Parser benchmark loader (spec 7): 200 labelled requests, dev/heldout split.

Each line: {"id", "split", "language", "group", "query", "gold": {"include",
"exclude", "views"}}. `group` is the author's primary group; the loader derives
the full group set from the gold and refuses a row whose primary group is not
among them. The gold is validated as a SearchQuery, so an unknown concept
cannot enter the benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from src.query.schema import SearchQuery

GROUPS = ("single", "multi", "negation", "uncertainty", "view", "other")
SPLITS = ("dev", "heldout")
LANGUAGES = ("en", "vi")
# Per language. Primary groups are assigned by the author (spec 7).
PRIMARY_QUOTA = {"single": 25, "multi": 25, "negation": 25, "uncertainty": 10, "view": 10,
                 "other": 5}
DEV_QUOTA = {"single": 5, "multi": 5, "negation": 5, "uncertainty": 2, "view": 2, "other": 1}
# The primary group of index i (1..100, per language) is PATTERN[(i - 1) % 20]; one cycle is
# exactly DEV_QUOTA, so the dev split (001-020) covers every group.
PATTERN = ("single", "multi", "negation", "single", "uncertainty", "multi", "negation", "single",
           "view", "multi", "negation", "single", "other", "multi", "negation", "single",
           "uncertainty", "negation", "view", "multi")
REQUIRED_KEYS = ("id", "split", "language", "group", "query", "gold")
GOLD_KEYS = frozenset({"include", "exclude", "views"})


@dataclass(frozen=True)
class BenchmarkQuery:
    id: str
    split: str
    language: str
    group: str                  # primary group, author-assigned
    groups: tuple[str, ...]     # every group the gold implies
    query: str
    gold: SearchQuery           # semantic_query is "" and never scored


def derive_groups(gold: SearchQuery) -> tuple[str, ...]:
    n = len(gold.include) + len(gold.exclude)
    groups: list[str] = []
    if n == 1 and not gold.exclude:
        groups.append("single")
    if n >= 2:
        groups.append("multi")
    if gold.exclude:
        groups.append("negation")
    if any(c.status == "UNCERTAIN" for c in gold.include):
        groups.append("uncertainty")
    if gold.views:
        groups.append("view")
    if n == 0:
        groups.append("other")
    return tuple(groups)


def load_benchmark(path: Path) -> list[BenchmarkQuery]:
    rows: list[BenchmarkQuery] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {number}: not JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {number}: not a JSON object")
            rid = record.get("id", f"line {number}")
            for key in REQUIRED_KEYS:
                if key not in record:
                    raise ValueError(f"{rid}: missing key {key!r}")
            if rid in seen:
                raise ValueError(f"{path}: duplicate id {rid}")
            seen.add(rid)
            if record.get("split") not in SPLITS:
                raise ValueError(f"{rid}: split must be one of {SPLITS}")
            if record.get("language") not in LANGUAGES:
                raise ValueError(f"{rid}: language must be one of {LANGUAGES}")
            if record.get("group") not in GROUPS:
                raise ValueError(f"{rid}: group must be one of {GROUPS}")
            # SearchQuery ignores unknown fields, so a typo ("includes") would load as empty gold.
            unknown = set(record["gold"]) - GOLD_KEYS if isinstance(record["gold"], dict) else ()
            if unknown:
                raise ValueError(f"{rid}: unknown gold key(s) {sorted(unknown)}")
            try:
                gold = SearchQuery(language=record["language"], **record["gold"])
            except (ValidationError, TypeError) as exc:
                raise ValueError(f"{rid}: invalid gold: {exc}") from exc
            groups = derive_groups(gold)
            if record["group"] not in groups:
                raise ValueError(f"{rid}: primary group {record['group']!r} is not implied by "
                                 f"the gold {groups}")
            rows.append(BenchmarkQuery(rid, record["split"], record["language"],
                                       record["group"], groups, record["query"], gold))
    return rows
