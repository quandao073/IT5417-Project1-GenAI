"""`lookup_concept` (docs/v2-kg-design.md, "Truy vấn online" step 2): a phrase of the question to
at most five candidate concepts of the graph. Three steps always run and their hits are merged,
one per concept: exact alias (the phrase or its singular), fulltext alias, and SapBERT over the
aliases (`lookup.sapbert`, WORKLOG D66, D68). An exact hit comes first; the others are ranked by
their score (Dice of a fulltext hit, cosine of a SapBERT one), except that a fulltext hit whose
alias has under `min_coverage` of the phrase's words comes after every SapBERT hit: "mediastinum
looks wider" shares one word with "mediastinum", which must not hide what SapBERT finds. Of equal
matches a scorable concept comes first, a weak one (D55) after the strong ones: the specific
concept with its own column before the weak label that covers it.

The side of the question ("trái", "left") is not part of a concept (schema decision 3): it is cut
before the lookup and reported, so "giảm thể tích phổi trái" finds the aliases of
"giảm thể tích phổi". A fulltext hit counts only when its alias shares enough words with the
phrase (Dice coefficient). A phrase made only of modifier words (the linker's qualitative words,
the vocabulary's dropped modifiers and the query's own) names no concept: it is not looked up, and
the caller lists it in one "not considered" warning. When
nothing is found the phrase is unrecognised and the caller drops it with a warning.
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


def modifier_words(kg_cfg: dict, overrides: dict) -> list[str]:
    """Words that only qualify a concept: the linker's `skip` strings (configs/kg/
    linker_overrides.yaml, observation and anatomy), the vocabulary's dropped modifiers
    (kg.yaml vocab.drop_modifiers) and the query's own (kg.yaml query.extra_modifiers), normalised.
    The first two lists are frozen with the linker and the vocabulary; additions go to the third."""
    words = {s for kind in ("observation", "anatomy")
             for s, v in (overrides.get(kind) or {}).items() if v == "skip"}
    words |= {w for group in kg_cfg["vocab"]["drop_modifiers"].values() for w in group}
    words |= set(kg_cfg["query"].get("extra_modifiers", []))
    return sorted({normalize(w) for w in words})


def dice(a: list[str], b: list[str]) -> float:
    shared = len(set(a) & set(b))
    return 2 * shared / (len(set(a)) + len(set(b))) if a and b else 0.0


def coverage(phrase: list[str], alias: list[str]) -> float:
    """The share of the phrase's words that the alias has."""
    return len(set(phrase) & set(alias)) / len(set(phrase)) if phrase else 0.0


@dataclass(frozen=True)
class Candidate:
    cui: str
    name: str
    label: str | None
    scorable: bool
    parents: list[str]
    match: str            # exact | sapbert | fulltext
    alias: str
    score: float
    n_pool: int
    weak: bool | None
    coverage: float | None = None     # fulltext only: the share of the phrase's words matched


@dataclass
class Lookup:
    term: str
    core: str
    sides: list[str] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    modifier: bool = False                             # only modifier words: not looked up
    timings: dict = field(default_factory=dict)       # ms of each step, set by the pipeline

    @property
    def single_exact(self) -> bool:
        """One alias is the phrase itself: the fuzzy hits ranked after it do not compete."""
        return [c.match for c in self.candidates].count("exact") == 1

    def to_dict(self) -> dict:
        return {"term": self.term, "core": self.core, "sides": self.sides,
                "modifier": self.modifier,
                "candidates": [asdict(c) for c in self.candidates]}


def lookup_concept(store, term: str, cfg: dict, semantic=None) -> Lookup:
    """`cfg` is configs/kg.yaml: query (lookup limits, sides, and `modifiers`, the words of
    `modifier_words`). `semantic` is the optional SapBERT stage (sapbert.AliasIndex), the last."""
    qcfg = cfg["lookup"]
    core, sides = strip_sides(normalize(term), cfg["sides"])
    result = Lookup(term=term, core=core, sides=sides)
    if not core:
        return result
    modifiers = set(cfg.get("modifiers", ()))
    if core in modifiers or all(w in modifiers for w in core.split()):
        result.modifier = True
        return result
    # cui -> (tier, match, alias, score, coverage): tier 0 exact; 1 SapBERT, or fulltext covering
    # enough of the phrase; 2 fulltext that does not. The best hit of a concept is kept.
    hits: dict[str, tuple[int, str, str, float, float | None]] = {}

    def add(cui: str, hit: tuple) -> None:
        if cui not in hits or (hit[0], -hit[3]) < (hits[cui][0], -hits[cui][3]):
            hits[cui] = hit

    for form in dict.fromkeys([core, singular_candidate(core) or core]):
        for row in store.aliases_exact(form):
            add(row["cui"], (0, "exact", row["text"], 1.0, None))
    tokens = core.split()
    for row in store.aliases_fulltext(core, qcfg["fulltext_limit"]):
        words = normalize(row["text"]).split()
        score = dice(tokens, words)
        if score >= qcfg["min_dice"]:
            cover = round(coverage(tokens, words), 3)
            add(row["cui"], (1 if cover >= qcfg["min_coverage"] else 2, "fulltext", row["text"],
                             round(score, 3), cover))
    if semantic is not None:
        floor = qcfg["sapbert"]["min_cos"]
        for row in semantic.search(core, qcfg["sapbert"]["top"]):
            if row["score"] >= floor:
                add(row["cui"], (1, "sapbert", row["text"], round(row["score"], 3), None))
    info = store.concepts(list(hits))
    ranked = sorted(
        (c for c in hits if c in info),
        key=lambda c: (hits[c][0], -hits[c][3], not info[c]["scorable"],
                       bool(info[c]["weak"]), -(info[c]["n_pool"] or 0), c))
    for cui in ranked[:qcfg["max_candidates"]]:
        _, match, alias, score, cover = hits[cui]
        i = info[cui]
        result.candidates.append(Candidate(
            cui=cui, name=i["name"], label=i["label"], scorable=bool(i["scorable"]),
            parents=i["parents"], match=match, alias=alias, score=score,
            n_pool=int(i["n_pool"] or 0), weak=i["weak"], coverage=cover))
    return result
