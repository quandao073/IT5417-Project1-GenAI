"""`expand_concepts` (docs/v2-kg-design.md, "Truy vấn online" step 3 and its table): a concept
and a polarity to the score columns that stand for it, how they combine, the warnings the
answer must carry, and the path in the graph.

    scorable           itself, weight 1; a weak one (D55) with a warning, and a weak one that
                       D63 names (`weak_or_children`) noisy-ORed with its scorable children
    non-scorable, with scorable concepts below it (IS_A)   those, noisy-OR (wider)
    otherwise, a scorable ancestor (IS_A)                  that, with a warning (narrower)
    otherwise a Disease     the scorable findings that SUGGESTS it, weights from p
    otherwise an Anatomy    the scorable findings usually seen there (TYPICALLY_AT), noisy-OR
                            weighted by how often each is seen there (p of the edge, divided by
                            the largest p of the region, so the most typical finding counts fully)
    otherwise               nothing: "cannot be scored"

A weak concept is never a target of the wider, disease and anatomy routes (D55); a narrower route
may end at one, with its warning. "No X" negates the intent's probability, except when X is
narrower than every scorable concept: negating the ancestor would exclude far more than asked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.kg.messages import Warn

PRESENT, ABSENT = "PRESENT", "ABSENT"
SINGLE, NOISY_OR, WEIGHTED = "single", "noisy_or", "weighted"
WEIGHTED_OR = "weighted_noisy_or"   # 1 - prod(1 - w p): a finding counts as much as it is typical
ANATOMY_FETCH = 50      # TYPICALLY_AT findings read for a region before the weak ones are dropped


@dataclass(frozen=True)
class Column:
    cui: str
    column: str
    weight: float
    quality: float | None
    weak: bool
    borderline: bool
    name: str = ""


@dataclass
class Intent:
    term: str
    cui: str
    name: str
    polarity: str
    kind: str                       # direct | wider | narrower | disease | anatomy | none
    combine: str = SINGLE
    columns: list[Column] = field(default_factory=list)
    warnings: list[Warn] = field(default_factory=list)
    path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"term": self.term, "cui": self.cui, "name": self.name,
                "polarity": self.polarity, "kind": self.kind, "combine": self.combine,
                "columns": [asdict(c) for c in self.columns], "path": self.path,
                "warnings": [w.code for w in self.warnings]}


def _column(info: dict, weight: float = 1.0) -> Column:
    return Column(cui=info["cui"], column=info["column"], weight=weight,
                  quality=info["quality"], weak=bool(info["weak"]),
                  borderline=bool(info["borderline"]), name=info["name"])


def _weak_warning(info: dict) -> Warn:
    return Warn("weak_quality", {"name": info["name"], "auroc": f"{info['quality']:.2f}"})


def expand_concepts(store, cui: str, polarity: str, cfg: dict, term: str = "") -> Intent:
    """`cfg` is configs/kg.yaml: query.expand and graph.min_n (as `min_n`)."""
    info = store.concepts([cui]).get(cui)
    if info is None:
        return Intent(term, cui, cui, polarity, "none",
                      warnings=[Warn("unrecognized", {"term": term or cui})])
    info = {**info, "cui": cui}
    name, label, down = info["name"], info["label"], cfg["down_depth"]
    intent = Intent(term or name, cui, name, polarity, "none")
    nothing = Warn("no_path", {"term": term or name, "name": name})

    if info["scorable"]:
        intent.kind, intent.columns, intent.path = "direct", [_column(info)], [name]
        if info["weak"]:
            intent.warnings.append(_weak_warning(info))
            if cui in cfg["weak_or_children"]:
                kids = _strong(store, [r["cui"] for r in store.scorable_below(cui, down)
                                       if r["cui"] != cui])
                if kids:
                    intent.combine = NOISY_OR
                    intent.columns += [_column(k) for k in kids]
                    intent.path = [f"{name} + {', '.join(k['name'] for k in kids)}"]
        return intent

    below = store.scorable_below(cui, down)
    strong = _strong(store, [r["cui"] for r in below])
    if strong:
        via = {r["cui"]: r["via"] for r in below}
        intent.kind, intent.combine = "wider", NOISY_OR
        intent.columns = [_column(k) for k in strong]
        intent.path = [" ← ".join(via[k["cui"]]) for k in strong]
        return intent

    above = store.scorable_above(cui, cfg["up_depth"])
    if above:
        ancestor = store.concepts([above[0]["cui"]])[above[0]["cui"]] | {"cui": above[0]["cui"]}
        intent.kind, intent.path = "narrower", [" → ".join(above[0]["via"])]
        intent.warnings.append(Warn("narrower", {"ancestor": ancestor["name"], "name": name}))
        if polarity == ABSENT:
            intent.warnings.append(Warn("negation_skipped", {"name": name}))
        else:
            intent.columns = [_column(ancestor)]
            if ancestor["weak"]:
                intent.warnings.append(_weak_warning(ancestor))
        return intent

    if label == "Disease":
        rows = store.suggesting(cui, cfg["suggests_depth"], cfg["min_n"])
        found = _strong(store, [r["cui"] for r in rows])
        if found:
            p = {r["cui"]: r["p"] for r in rows}
            total = sum(p[k["cui"]] for k in found)
            intent.kind, intent.combine = "disease", WEIGHTED
            intent.columns = [_column(k, p[k["cui"]] / total) for k in found]
            intent.path = [f"{k['name']} —SUGGESTS(p={p[k['cui']]:.4f})→ "
                           f"{next(r['target'] for r in rows if r['cui'] == k['cui'])}"
                           for k in found]
            intent.warnings.append(Warn("disease_inferred", {"name": name}))
            return intent
    if label == "Anatomy":
        # the best `anatomy_top` findings that are not weak: ask for more, a weak one takes no slot
        rows = store.anatomy_findings(cui, cfg["part_depth"], cfg["min_n"], ANATOMY_FETCH)
        found = _strong(store, [r["cui"] for r in rows])[:cfg["anatomy_top"]]
        if found:
            region = {r["cui"]: r["region"] for r in rows}
            p = {r["cui"]: r["p"] for r in rows}
            top = max(p[k["cui"]] for k in found)
            intent.kind, intent.combine = "anatomy", WEIGHTED_OR
            intent.columns = [_column(k, p[k["cui"]] / top) for k in found]
            intent.path = [f"{k['name']} —TYPICALLY_AT→ {region[k['cui']]}" for k in found]
            intent.warnings.append(Warn("anatomy_region", {"region": name}))
            return intent
    intent.warnings.append(nothing)
    return intent


def _strong(store, cuis: list[str]) -> list[dict]:
    """The scorable, not weak concepts among `cuis`, in order, with their `cui` key."""
    if not cuis:
        return []
    info = store.concepts(list(dict.fromkeys(cuis)))
    return [{**info[c], "cui": c} for c in dict.fromkeys(cuis)
            if c in info and info[c]["scorable"] and not info[c]["weak"]]
