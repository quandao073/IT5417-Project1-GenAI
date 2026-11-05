"""`lookup_concept` (docs/v2-kg-design.md, "Truy vấn online" step 2): a phrase of the question to
at most five candidate concepts of the graph: by exact alias (the phrase or its singular), and
only when none matches, by fulltext alias.

The side of the question ("trái", "left") is not part of a concept (schema decision 3): it is cut
before the lookup and reported, so "giảm thể tích phổi trái" finds the aliases of
"giảm thể tích phổi". A fulltext hit counts only when its alias shares enough words with the
phrase (Dice coefficient); the SapBERT stage of the design is left out here: its dictionary
covers UMLS names, not the Vietnamese aliases most queries need. When nothing is found the phrase
is unrecognised and the caller drops it with a warning.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

from src.kg.vocab import singular_candidate

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def normalize(text: str) -> str:
    """The `norm` of an alias: NFC, lower case, words separated by one space."""
    return _NON_WORD.sub(" ", unicodedata.normalize("NFC", text).lower()).strip()


def strip_sides(norm: str, sides: list[str]) -> tuple[str, list[str]]:
    """The phrase without side words (longest first, whole words) and the sides found."""
    found, words = [], f" {norm} "
    for side in sorted((normalize(s) for s in sides), key=len, reverse=True):
        if f" {side} " in words:
            found.append(side)
            words = words.replace(f" {side} ", " ")
    return " ".join(words.split()), found


def dice(a: list[str], b: list[str]) -> float:
    shared = len(set(a) & set(b))
    return 2 * shared / (len(set(a)) + len(set(b))) if a and b else 0.0


@dataclass(frozen=True)
class Candidate:
    cui: str
    name: str
    label: str | None
    scorable: bool
    parents: list[str]
    match: str            # exact | fulltext
    alias: str
    score: float
    n_pool: int
    weak: bool | None


@dataclass
class Lookup:
    term: str
    core: str
    sides: list[str] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    timings: dict = field(default_factory=dict)       # ms of each step, set by the pipeline

    def to_dict(self) -> dict:
        return {"term": self.term, "core": self.core, "sides": self.sides,
                "candidates": [asdict(c) for c in self.candidates]}


def lookup_concept(store, term: str, cfg: dict) -> Lookup:
    """`cfg` is configs/kg.yaml: query (lookup limits, sides)."""
    qcfg = cfg["lookup"]
    core, sides = strip_sides(normalize(term), cfg["sides"])
    result = Lookup(term=term, core=core, sides=sides)
    if not core:
        return result
    hits: dict[str, tuple[str, str, float]] = {}
    for form in dict.fromkeys([core, singular_candidate(core) or core]):
        for row in store.aliases_exact(form):
            hits.setdefault(row["cui"], ("exact", row["text"], 1.0))
    if not hits:                          # fulltext only when no alias is the phrase itself
        tokens = core.split()
        for row in store.aliases_fulltext(core, qcfg["fulltext_limit"]):
            score = dice(tokens, normalize(row["text"]).split())
            if score >= qcfg["min_dice"] and (
                    row["cui"] not in hits or hits[row["cui"]][2] < score):
                hits[row["cui"]] = ("fulltext", row["text"], round(score, 3))
    info = store.concepts(list(hits))
    ranked = sorted(
        (c for c in hits if c in info),
        key=lambda c: (hits[c][0] != "exact", -hits[c][2], not info[c]["scorable"],
                       -(info[c]["n_pool"] or 0), c))
    for cui in ranked[:qcfg["max_candidates"]]:
        match, alias, score = hits[cui]
        i = info[cui]
        result.candidates.append(Candidate(
            cui=cui, name=i["name"], label=i["label"], scorable=bool(i["scorable"]),
            parents=i["parents"], match=match, alias=alias, score=score,
            n_pool=int(i["n_pool"] or 0), weak=i["weak"]))
    return result
