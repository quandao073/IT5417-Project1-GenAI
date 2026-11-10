"""Knowledge for the explanation (RAG over the KG): short definitions of the concepts, made
offline, and the edges of the path a search used, written as sentences by templates.

Offline (`cxr kg-definitions`):
    concept_set     the concepts a search can show: every scorable concept, the diseases with a
                    route to a column (wider or SUGGESTS anywhere in the graph, narrower only for
                    core diseases: a narrower route reaches two thousand diseases), and the regions
                    whose findings (TYPICALLY_AT, through PART_OF) include a column that is not weak
    mrdef_level0    their UMLS definitions from the Level 0 sources (data/ only, T38)
    gloss / check   claude-opus-5-5 writes one sentence in EN and VI per concept, claude-sonnet-5-5
                    reviews it independently; batches of `batch` concepts per call, cached
    table           definitions.parquet and a review CSV; a gloss is used only when Sonnet agrees,
                    no automatic flag is raised, or the chat's adjudication accepts it

At query time `KnowledgeBase.items` turns the intents of a search into numbered items (K1, K2…):
the gloss of each intent's concept and of its columns, and the edges from the concept to each
column, rebuilt from edges.parquet with the rules of `expand_concepts` (same tables as Neo4j).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, ValidationError

from src.kg import llm_select

DISEASE, ANATOMY = "Disease", "Anatomy"
_WORD = re.compile(r"\w+", re.UNICODE)


# ---- the concept set --------------------------------------------------------------------------

class Graph:
    """The IS_A, PART_OF, SUGGESTS and TYPICALLY_AT edges of edges.parquet in RAM."""

    def __init__(self, edges: pl.DataFrame):
        self.up: dict[str, set[str]] = defaultdict(set)       # child -> parents
        self.down: dict[str, set[str]] = defaultdict(set)     # parent -> children
        self.parts: dict[str, set[str]] = defaultdict(set)    # whole -> parts
        self.suggests: dict[str, list[dict]] = defaultdict(list)       # disease -> edges in
        self.typical: dict[str, list[dict]] = defaultdict(list)        # region -> edges in
        for row in edges.filter(pl.col("rel").is_in(["IS_A", "PART_OF", "SUGGESTS",
                                                     "TYPICALLY_AT"])).iter_rows(named=True):
            s, rel, d = row["src"], row["rel"], row["dst"]
            if rel == "IS_A":
                self.up[s].add(d)
                self.down[d].add(s)
            elif rel == "PART_OF":
                self.parts[d].add(s)
            elif rel == "SUGGESTS":
                self.suggests[d].append(row)
            else:
                self.typical[d].append(row)

    @staticmethod
    def reach(start: str, adj: dict[str, set[str]], depth: int) -> dict[str, list[str]]:
        """Every node within `depth` steps of `start`, with one shortest path (start first);
        ties go to the smallest id, so the path is the same on every run."""
        paths = {start: [start]}
        frontier = [start]
        for _ in range(depth):
            nxt = []
            for node in frontier:
                for other in sorted(adj.get(node, ())):
                    if other not in paths:
                        paths[other] = [*paths[node], other]
                        nxt.append(other)
            frontier = nxt
        return paths

    def suggests_into(self, cui: str, depth: int, min_n: int, sources: Collection[str]):
        """The kept SUGGESTS edges from `sources` into `cui` or a concept up to `depth` IS_A steps
        below it (the filter of `Neo4jStore.suggesting`)."""
        below = self.reach(cui, self.down, depth)
        return [(e, below[e["dst"]]) for t in below for e in self.suggests.get(t, ())
                if e["src"] in sources and (e["source"] == "manual" or (e["n"] or 0) >= min_n)
                and (e["review"] or "keep") != "drop"]

    def typical_in(self, cui: str, depth: int, min_n: int, sources: Collection[str]):
        """The TYPICALLY_AT edges from `sources` into `cui` or a part of it (PART_OF*)."""
        parts = self.reach(cui, self.parts, depth)
        return [(e, parts[e["dst"]]) for r in parts for e in self.typical.get(r, ())
                if e["src"] in sources and (e["n"] or 0) >= min_n]


def concept_set(nodes: pl.DataFrame, graph: Graph, quality: dict, expand: dict) -> pl.DataFrame:
    """cui, name, label, route of every concept whose definition a search can show. `expand` is
    configs/kg.yaml query.expand; `quality` is quality.json (weak columns)."""
    scorable = set(nodes.filter(pl.col("scorable"))["cui"])
    strong = scorable - {c for c, q in quality.items() if q["weak"]}
    rows = []
    for cui, name, label, core in nodes.select("cui", "name", "labels", "in_core").iter_rows():
        route = None
        if cui in scorable:
            route = "column"
        elif label == DISEASE:
            if set(graph.reach(cui, graph.down, expand["down_depth"])) & strong:
                route = "wider"
            elif graph.suggests_into(cui, expand["suggests_depth"], expand["min_n"], strong):
                route = "suggests"
            elif core and (set(graph.reach(cui, graph.up, expand["up_depth"])) - {cui}) & scorable:
                route = "narrower"
        elif label == ANATOMY and graph.typical_in(cui, expand["part_depth"], expand["min_n"],
                                                   strong):
            route = "anatomy"
        if route:
            rows.append({"cui": cui, "name": name, "label": label, "route": route})
    return pl.DataFrame(rows, schema={"cui": pl.String, "name": pl.String, "label": pl.String,
                                      "route": pl.String}).sort("cui")


# ---- UMLS definitions -------------------------------------------------------------------------

def mrdef_level0(meta: Path, umls_dir: Path, cuis: Collection[str], order: Sequence[str],
                 per_concept: int) -> pl.DataFrame:
    """cui, sab, definition: unsuppressed MRDEF rows of the Level 0 sources (MRSAB SRL 0, current
    version), at most `per_concept` per concept, sources in `order` first, then shortest."""
    from src.kg import umls_filter

    level0 = set(pl.read_parquet(umls_dir / "mrsab.parquet")
                 .filter((pl.col("SRL") == "0") & (pl.col("CURVER") == "Y"))["RSAB"])
    columns = umls_filter.read_mrfiles(meta)["MRDEF"]["columns"]
    rank = {sab: i for i, sab in enumerate(order)}
    rows = (umls_filter.scan_rrf(meta, "MRDEF", columns)
            .filter(pl.col("CUI").is_in(list(cuis)) & pl.col("SAB").is_in(list(level0))
                    & (pl.col("SUPPRESS") == "N"))
            .select(pl.col("CUI").alias("cui"), pl.col("SAB").alias("sab"),
                    pl.col("DEF").alias("definition")).unique().collect())
    return (rows.with_columns(pl.col("sab").replace_strict(rank, default=len(rank),
                                                           return_dtype=pl.Int64).alias("o"),
                              pl.col("definition").str.len_chars().alias("len"))
            .sort("cui", "o", "len", "definition").group_by("cui", maintain_order=True)
            .head(per_concept).drop("o", "len"))


# ---- the two language-model passes ------------------------------------------------------------

class Gloss(BaseModel):
    cui: str
    en: str
    vi: str
    note: str


class Glosses(BaseModel):
    glosses: list[Gloss]


class Verdict(BaseModel):
    cui: str
    verdict: Literal["agree", "disagree"]
    reason: str


class Verdicts(BaseModel):
    verdicts: list[Verdict]


GLOSS_SCHEMA = {"type": "object", "properties": {"glosses": {
    "type": "array", "items": llm_select._schema(Gloss)}}, "required": ["glosses"],
    "additionalProperties": False}
CHECK_SCHEMA = {"type": "object", "properties": {"verdicts": {
    "type": "array", "items": llm_select._schema(Verdict)}}, "required": ["verdicts"],
    "additionalProperties": False}


def concept_inputs(concepts: pl.DataFrame, nodes: pl.DataFrame, graph: Graph,
                   aliases: pl.DataFrame, defs: pl.DataFrame, names_per_lang: int) -> list[dict]:
    """What the models see of each concept: cui, name, kind, IS_A parents (names), a few EN and
    VI names of the graph, and its definitions."""
    name = dict(nodes.select("cui", "name").iter_rows())
    shown: dict[tuple[str, str], list[str]] = defaultdict(list)
    for cui, text, lang in (aliases.filter(pl.col("cui").is_in(concepts["cui"].to_list()))
                            .with_columns(pl.col("text").str.len_chars().alias("len"))
                            .sort("cui", "lang", "len", "text").select("cui", "text", "lang")
                            .iter_rows()):
        if len(shown[(cui, lang)]) < names_per_lang and text not in shown[(cui, lang)]:
            shown[(cui, lang)].append(text)
    by_cui: dict[str, list[dict]] = defaultdict(list)
    for cui, sab, text in defs.select("cui", "sab", "definition").iter_rows():
        by_cui[cui].append({"source": sab, "text": text})
    return [{"cui": cui, "name": n, "kind": label,
             "parents": sorted(name.get(p, p) for p in graph.up.get(cui, ()))[:3],
             "english_names": shown[(cui, "en")], "vietnamese_names": shown[(cui, "vi")],
             "definitions": by_cui.get(cui, [])}
            for cui, n, label in concepts.select("cui", "name", "label").iter_rows()]


def batches(inputs: Sequence[dict], size: int, kind: str) -> list[dict]:
    """llm_select items of `size` concepts each, in cui order."""
    return [{"kind": kind, "string": f"{kind}-{i // size:03d}",
             "input": {"concepts": list(inputs[i:i + size])}}
            for i in range(0, len(inputs), size)]


def check_batches(gloss_items: Sequence[dict], glosses: dict[str, dict]) -> list[dict]:
    """The same batches with each concept's glosses added, for the reviewer; a batch whose
    glosses are not all there yet is left out."""
    out = []
    for item in gloss_items:
        concepts = item["input"]["concepts"]
        if all(c["cui"] in glosses for c in concepts):
            out.append({"kind": "check", "string": item["string"].replace("gloss", "check"),
                        "input": {"concepts": [{**c, "gloss": {
                            "en": glosses[c["cui"]]["en"], "vi": glosses[c["cui"]]["vi"]}}
                            for c in concepts]}})
    return out


def _answers(model: type[BaseModel], field: str):
    def parse(content: str, item: dict) -> dict:
        try:
            rows = getattr(model.model_validate_json(content), field)
        except ValidationError as exc:
            raise llm_select.LLMSelectError(
                f"reply does not fit the schema: {exc.errors()[:1]}") from exc
        wanted = [c["cui"] for c in item["input"]["concepts"]]
        got = {r.cui: r.model_dump() for r in rows}
        if set(got) != set(wanted) or len(rows) != len(wanted):
            raise llm_select.LLMSelectError(
                f"cuis {sorted(set(got) ^ set(wanted))[:3]} missing or extra")
        return {field: [got[c] for c in wanted]}
    return parse


parse_glosses, parse_verdicts = _answers(Glosses, "glosses"), _answers(Verdicts, "verdicts")


# ---- automatic flags and the table ------------------------------------------------------------

def words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def copied(gloss: str, definitions: Sequence[str], run: int) -> bool:
    """True when the gloss repeats `run` consecutive words of a definition."""
    g = words(gloss)
    grams = {tuple(g[i:i + run]) for i in range(len(g) - run + 1)}
    for text in definitions:
        d = words(text)
        if any(tuple(d[i:i + run]) in grams for i in range(len(d) - run + 1)):
            return True
    return False


def flags(gloss: dict, definitions: Sequence[str], cfg: dict, diagnostic: Sequence[str]) -> str:
    """`;`-joined: too_long (over max_words of the language; Vietnamese counts syllables),
    copied (a run of copy_run words of a definition), diagnostic (a word of
    explain.diagnostic), empty."""
    found = []
    for lang in ("en", "vi"):
        text = gloss[lang].strip()
        if not text:
            found.append(f"empty_{lang}")
            continue
        if len(text.split()) > cfg["max_words"][lang]:
            found.append(f"too_long_{lang}")
        if copied(text, definitions, cfg["copy_run"]):
            found.append(f"copied_{lang}")
        if any(d.casefold() in text.casefold() for d in diagnostic):
            found.append(f"diagnostic_{lang}")
    return ";".join(found)


REVIEW_COLUMNS = ("cui", "name", "label", "route", "source", "gloss_en", "gloss_vi", "flags",
                  "opus_note", "sonnet_verdict", "sonnet_reason", "decision", "fixed_en",
                  "fixed_vi", "note")


def table(concepts: pl.DataFrame, defs: pl.DataFrame, glosses: dict[str, dict],
          verdicts: dict[str, dict], adjudicated: dict[str, dict], cfg: dict,
          diagnostic: Sequence[str]) -> pl.DataFrame:
    """One row per concept. `adjudicated`: cui -> the review CSV row of the chat (decision
    accept | reject, optional fixed_en / fixed_vi). `usable` and the final glosses follow the
    rule of the module docstring."""
    by_cui: dict[str, list[str]] = defaultdict(list)
    for cui, text in defs.select("cui", "definition").iter_rows():
        by_cui[cui].append(text)
    rows = []
    for cui, name, label, route in concepts.select("cui", "name", "label", "route").iter_rows():
        g, v = glosses.get(cui), verdicts.get(cui, {})
        a = adjudicated.get(cui, {})
        f = flags(g, by_cui.get(cui, []), cfg, diagnostic) if g else "no_gloss"
        agree = bool(g) and v.get("verdict") == "agree" and not f
        decision = a.get("decision", "")
        en = (a.get("fixed_en") or (g or {}).get("en", "")).strip()
        vi = (a.get("fixed_vi") or (g or {}).get("vi", "")).strip()
        usable = decision == "accept" or (agree and decision != "reject")
        rows.append({"cui": cui, "name": name, "label": label, "route": route,
                     "source": "mrdef" if by_cui.get(cui) else "llm",
                     "gloss_en": (g or {}).get("en", ""), "gloss_vi": (g or {}).get("vi", ""),
                     "flags": f, "opus_note": (g or {}).get("note", ""),
                     "sonnet_verdict": v.get("verdict", ""), "sonnet_reason": v.get("reason", ""),
                     "agree": agree, "decision": decision, "usable": usable,
                     "final_en": en if usable else "", "final_vi": vi if usable else ""})
    return pl.DataFrame(rows)


def review_rows(definitions: pl.DataFrame, adjudicated: dict[str, dict]) -> pl.DataFrame:
    """The review CSV: every concept, the ones not agreed first; earlier decisions kept."""
    rows = []
    for r in definitions.sort(["agree", "cui"]).iter_rows(named=True):
        a = adjudicated.get(r["cui"], {})
        rows.append({**{k: r[k] for k in REVIEW_COLUMNS if k in r},
                     "decision": a.get("decision", ""), "fixed_en": a.get("fixed_en", ""),
                     "fixed_vi": a.get("fixed_vi", ""), "note": a.get("note", "")})
    return pl.DataFrame(rows, schema={c: pl.String for c in REVIEW_COLUMNS})


def read_adjudication(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return {r["cui"]: r for r in pl.read_csv(path, infer_schema_length=0).fill_null("")
            .iter_rows(named=True) if r["decision"]}


def cache_records(path: Path) -> list[dict]:
    """Every record of an llm_select cache file, errors included (they are billed too)."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def usd(records: Sequence[dict], prices: dict) -> float:
    """The cost of cache records from their usage and the list prices (USD per million)."""
    total = 0.0
    for r in records:
        price, use = prices.get(r.get("model")), r.get("usage")
        if price and use:
            total += (use["input_tokens"] * price["input"]
                      + use["output_tokens"] * price["output"]) / 1e6
    return total


# ---- the knowledge items of a search ----------------------------------------------------------

TEMPLATES = {
    "vi": {"gloss": "{x}: {gloss}",
           "suggests": "{x} thường được ghi nhận cùng {y} trong report, p = {p}.",
           "suggests_manual": "{x} được nối với {y} bằng một cạnh thêm tay, không có p từ report.",
           "is_a": "{x} là một dạng của {y}.",
           "typically_at": "{x} thường thấy ở vùng {y}.",
           "part_of": "{x} là một phần của {y}."},
    "en": {"gloss": "{x}: {gloss}",
           "suggests": "{x} is often recorded together with {y} in reports, p = {p}.",
           "suggests_manual": "{x} is linked to {y} by an edge added by hand, with no p from "
                              "reports.",
           "is_a": "{x} is a kind of {y}.",
           "typically_at": "{x} is usually seen in the region {y}.",
           "part_of": "{x} is part of {y}."}}


def _p(p: float, language: str) -> str:
    text = f"{p:.2f}" if round(p, 2) > 0 else f"{p:.3f}"
    return text.replace(".", ",") if language == "vi" else text


class KnowledgeBase:
    """Glosses (cui -> {en, vi}, the usable ones of definitions.parquet), the graph and a display
    name per concept and language: the graph's hand-made alias first, then the shortest, else
    the concept's name."""

    def __init__(self, glosses: dict[str, dict[str, str]], graph: Graph,
                 names: dict[str, dict[str, str]], fallback: dict[str, str], expand: dict,
                 max_items: int):
        self.glosses, self.graph, self.names, self.fallback = glosses, graph, names, fallback
        self.expand, self.max_items = expand, max_items

    @classmethod
    def load(cls, definitions: Path, edges: pl.DataFrame, aliases: pl.DataFrame,
             nodes: pl.DataFrame, expand: dict, max_items: int) -> KnowledgeBase:
        glosses: dict[str, dict[str, str]] = {}
        if definitions.is_file():
            for cui, en, vi in (pl.read_parquet(definitions).filter(pl.col("usable"))
                                .select("cui", "final_en", "final_vi").iter_rows()):
                glosses[cui] = {"en": en, "vi": vi}
        names: dict[str, dict[str, str]] = defaultdict(dict)
        ranked = (aliases.with_columns((pl.col("source") == "mrconso").alias("umls"),
                                       pl.col("text").str.len_chars().alias("len"))
                  .sort("cui", "lang", "umls", "len", "text"))
        for cui, lang, text in ranked.select("cui", "lang", "text").iter_rows():
            names[cui].setdefault(lang, text)
        return cls(glosses, Graph(edges), dict(names),
                   dict(nodes.select("cui", "name").iter_rows()), expand, max_items)

    def name(self, cui: str, language: str) -> str:
        return self.names.get(cui, {}).get(language) or self.fallback.get(cui, cui)

    def path_edges(self, intent) -> list[tuple]:
        """(rel, src, dst, edge row or None) of the path from the intent's concept to each of
        its columns, strongest column first; the edges `expand_concepts` walked."""
        g, cfg, cui = self.graph, self.expand, intent.cui
        out: list[tuple] = []

        def chain(path: list[str], rel: str, upward: bool) -> None:
            # path runs from the intent's concept outwards; upward: each node IS_A the next
            for a, b in zip(path, path[1:], strict=False):
                out.append((rel, a, b, None) if upward else (rel, b, a, None))

        for col in sorted(intent.columns, key=lambda c: -c.weight):
            if col.cui == cui:
                continue
            if intent.kind in ("direct", "wider"):
                chain(g.reach(cui, g.down, cfg["down_depth"]).get(col.cui, []), "is_a", False)
            elif intent.kind == "narrower":
                chain(g.reach(cui, g.up, cfg["up_depth"]).get(col.cui, []), "is_a", True)
            elif intent.kind == "disease":
                found = g.suggests_into(cui, cfg["suggests_depth"], cfg["min_n"], {col.cui})
                if found:
                    edge, below = max(found, key=lambda f: (f[0]["p"] or 0, f[0]["dst"]))
                    out.append(("suggests", col.cui, edge["dst"], edge))
                    chain(below, "is_a", False)
            elif intent.kind == "anatomy":
                found = g.typical_in(cui, cfg["part_depth"], cfg["min_n"], {col.cui})
                if found:
                    edge, parts = max(found, key=lambda f: (f[0]["p"] or 0, f[0]["dst"]))
                    out.append(("typically_at", col.cui, edge["dst"], edge))
                    chain(parts, "part_of", False)
        return _unique(out)

    def items(self, intents, language: str) -> list[dict]:
        """K1, K2…: glosses of the intents' concepts, the path edges, glosses of the columns;
        at most `max_items`. Each item: id, kind, text (in `language`), cuis."""
        lang = "vi" if language == "vi" else "en"
        t = TEMPLATES[lang]
        found: list[dict] = []
        seen: set = set()

        def add(key, kind: str, text: str, cuis: list[str]) -> None:
            if key not in seen:
                seen.add(key)
                found.append({"kind": kind, "text": text, "cuis": cuis})

        def gloss(cui: str) -> None:
            text = self.glosses.get(cui, {}).get(lang)
            if text:
                add(("gloss", cui), "gloss",
                    t["gloss"].format(x=self.name(cui, lang), gloss=text), [cui])

        scored = [it for it in intents if it.columns]
        for it in scored:
            gloss(it.cui)
        for it in scored:
            for rel, src, dst, edge in self.path_edges(it):
                x, y = self.name(src, lang), self.name(dst, lang)
                if rel == "suggests" and (edge["source"] == "manual" or edge["p"] is None):
                    text = t["suggests_manual"].format(x=x, y=y)
                elif rel == "suggests":
                    text = t["suggests"].format(x=x, y=y, p=_p(edge["p"], lang))
                else:
                    text = t[rel].format(x=x, y=y)
                add((rel, src, dst), rel, text, [src, dst])
        for it in scored:
            for col in it.columns:
                gloss(col.cui)
        return [{"id": f"K{i}", **item}
                for i, item in enumerate(found[:self.max_items], start=1)]


def _unique(rows: list[tuple]) -> list[tuple]:
    seen, out = set(), []
    for row in rows:
        key = row[:3]
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out
