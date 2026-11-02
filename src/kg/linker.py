"""Vocabulary strings -> CUIs (TIP V2-3, D56; docs/v2-kg-design.md, "Dựng KG offline" step 3).

Every string of the 95% head of the pool vocabulary (D52) is decided by the first rule that
fires: the normal group, a hand override (a CUI or `skip`), a descriptive word on its own, a name
whose CUIs all lie outside the label table (`out_of_scope`). Any other string goes to the
selection step with its candidates: its exact MRCONSO CUIs (of an anatomy adjective, those of the
noun) and the CUIs of its nearest SapBERT names, all within the kind's domain (`domain`). Two
LLMs choose among them (src/kg/llm_select.py); a hand adjudication wins over both. A CUI named
with a side moves to its unlateralised parent (design, schema decision 3).

In parallel each Observation is tried with the Anatomy it is `located_at`: RadGraph splits
"pleural effusion" into Anatomy "pleural" and Observation "effusion", so the single string
"effusion" never reaches the pleural effusion CUI, while the pair "pleural" + "effusion" does.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

import polars as pl
import yaml

from src.kg import sapbert, vocab

NORMAL, SKIP, OVERRIDE, DESCRIPTIVE = "normal", "skip", "override", "descriptive"
EXACT, ADJECTIVE, OUT_OF_SCOPE, SAPBERT = "exact", "adjective", "out_of_scope", "sapbert"
SELECT, ADJUDICATED, LLM, LLM_DISAGREE = "select", "adjudicated", "llm", "llm_disagree"
CUI, NONE, PENDING, REJECTED = "cui", "none", "pending", "rejected"   # decisions
UNDECIDED = (NONE, PENDING)   # the only decisions coverage counts as missing (D56)
PAIRS = "pairs"   # the overrides file's section for located_at pairs
_CUI = re.compile(r"C\d{7}")
_SAB_ORDER = ["MSH", "NCI", "FMA", "HPO"]   # whose preferred name to show
_NORM = pl.col("STR").str.to_lowercase().str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()


class LinkerError(RuntimeError):
    """An input the linker cannot trust, such as an override that is neither a CUI nor skip."""


def load_overrides(path: Path) -> dict[str, dict[str, str]]:
    """{kind: {string: CUI or "skip"}} from configs/kg/linker_overrides.yaml, plus "pairs":
    {"observation + anatomy": CUI} for located_at pairs."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    kinds = {kind.lower(): kind for kind in vocab.KINDS}
    pairs = raw.pop(PAIRS, None) or {}
    for key, value in pairs.items():
        obs, sep, anat = key.partition(" + ")
        if not sep or vocab.normalize(obs) != obs or vocab.normalize(anat) != anat or not (
                _CUI.fullmatch(str(value))):
            raise LinkerError(f"{path.name}: pairs {key!r}: {value!r} is not 'obs + anat': CUI")
    if unknown := set(raw) - set(kinds):
        raise LinkerError(f"{path.name}: unknown sections {sorted(unknown)}")
    out = {}
    for key, kind in kinds.items():
        entries = raw.get(key) or {}
        for string, value in entries.items():
            valid = value == SKIP or _CUI.fullmatch(str(value))
            if vocab.normalize(string) != string or not valid:
                raise LinkerError(f"{path.name}: {key} {string!r}: {value!r} is not a CUI or skip")
        out[kind] = dict(entries)
    out[PAIRS] = dict(pairs)
    return out


def override_pairs(pairs: pl.DataFrame, overrides: dict[str, str]) -> pl.DataFrame:
    """Pair links with the hand overrides of `pairs` ("obs + anat": CUI) applied, `via` override."""
    hand = {tuple(k.split(" + ")): v for k, v in overrides.items()}
    picked = [hand.get((o, a)) for o, a in pairs.select("obs", "anat").iter_rows()]
    hit = pl.Series(picked, dtype=pl.String)
    return pairs.with_columns(
        pl.when(hit.is_not_null()).then(hit).otherwise(pl.col("cui")).alias("cui"),
        pl.when(hit.is_not_null()).then(pl.lit(OVERRIDE)).otherwise(pl.col("via")).alias("via"))


def load_adjudication(path: Path, project_ids: Collection[str] = ()) -> dict[tuple[str, str], dict]:
    """{(kind, string): {decision, cui, reason}} from data/kg/link/adjudication.csv (columns
    string, kind, decision, cui, reason); none when the file does not exist yet. A `cui` is a
    CUI or one of `project_ids` (PRJ:<name> of configs/kg/chexpert_cuis.yaml)."""
    if not path.is_file():
        return {}
    out = {}
    for row in pl.read_csv(path, infer_schema_length=0).fill_null("").iter_rows(named=True):
        decision, cui = row["decision"].strip(), row["cui"].strip()
        is_id = bool(_CUI.fullmatch(cui)) or cui in project_ids
        if row["kind"] not in vocab.KINDS or decision not in (CUI, SKIP, OUT_OF_SCOPE, NONE) or (
                decision == CUI) != is_id:
            raise LinkerError(f"{path.name}: {row['kind']!r} {row['string']!r}: decision "
                              f"{decision!r} with cui {cui!r}")
        out[(row["kind"], row["string"])] = {"decision": decision, "cui": cui,
                                             "reason": row["reason"]}
    return out


def check_adjudication(adjudication: dict, umls_dir: Path, cfg: dict,
                       domain: pl.DataFrame) -> list[tuple[str, str, str, str]]:
    """(kind, string, cui, problem) for each adjudicated CUI that is not in MRCONSO, has no
    semantic type of the string's kind, or lies outside the kind's domain. Reported, never
    corrected. Project ids (PRJ:) are not UMLS CUIs and are not checked here."""
    wanted = {a["cui"] for a in adjudication.values() if _CUI.fullmatch(a["cui"])}
    known = set(pl.scan_parquet(umls_dir / "mrconso.parquet")
                .filter(pl.col("CUI").is_in(list(wanted))).select("CUI").unique()
                .collect()["CUI"])
    typed = set(pl.read_parquet(umls_dir / "mrsty.parquet", columns=["CUI", "TUI"])
                .filter(pl.col("CUI").is_in(list(wanted)))
                .join(_kind_tuis(cfg), on="TUI").select("CUI", "kind").iter_rows())
    inside = set(domain.filter(pl.col("CUI").is_in(list(wanted))).iter_rows())
    problems = []
    for (kind, string), a in sorted(adjudication.items()):
        cui = a["cui"]
        if not _CUI.fullmatch(cui):
            continue
        problem = ("not in MRCONSO" if cui not in known
                   else "no semantic type of the kind" if (cui, kind) not in typed
                   else "outside the domain" if (cui, kind) not in inside else None)
        if problem:
            problems.append((kind, string, cui, problem))
    return problems


def scope(table: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """kind, rank, string, mentions, studies, band for the `scope` head of each kind. Band "80" is
    reviewed in full, band "90" only where the link is not a single exact CUI, band "95" not."""
    cuts = [cfg["review_full"], cfg["review_partial"], cfg["scope"]]
    parts = []
    for kind in vocab.KINDS:
        part = (table.filter(pl.col("kind") == kind)
                .sort(["mentions", "string"], descending=[True, False]))
        levels = vocab.curve(part["mentions"].to_list(), cuts)
        n80, n90, n95 = (levels[c] for c in cuts)
        rank = pl.col("rank")
        parts.append(part.head(n95).with_row_index("rank", offset=1).with_columns(
            pl.when(rank <= n80).then(pl.lit("80")).when(rank <= n90).then(pl.lit("90"))
            .otherwise(pl.lit("95")).alias("band")))
    return pl.concat(parts).select("kind", "rank", "string", "mentions", "studies", "band")


def _kind_tuis(cfg: dict) -> pl.DataFrame:
    return pl.DataFrame([(kind, tui) for kind, labels in cfg["kinds"].items()
                         for label in labels for tui in cfg["labels"][label]],
                        schema={"kind": pl.String, "TUI": pl.String}, orient="row")


def _descendants(edges: pl.DataFrame, roots: Collection[str]) -> set[str]:
    """The roots and every CUI below them along (parent, child) edges."""
    children: dict[str, list[str]] = {}
    for parent, child in edges.iter_rows():
        children.setdefault(parent, []).append(child)
    seen, stack = set(roots), list(roots)
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def domain(umls_dir: Path, cfg: dict) -> pl.DataFrame:
    """CUI, kind: the CUIs a string of the kind may link to (D56, option C). Every CUI on the
    kind's label table that has a name outside the code-system sources, unless it lies below one
    of the kind's negative roots and below none of its positive roots. "Below" follows FMA parts
    and isa, and every source's CHD rows; MRREL CUI1 is the whole or parent (T39)."""
    dcfg = cfg["domain"]
    on_table = (pl.read_parquet(umls_dir / "mrsty.parquet", columns=["CUI", "TUI"])
                .join(_kind_tuis(cfg), on="TUI").select("CUI", "kind").unique())
    named = (pl.scan_parquet(umls_dir / "mrconso.parquet")
             .filter(~pl.col("SAB").is_in(dcfg["code_sabs"])).select("CUI").unique()
             .join(on_table.select("CUI").unique().lazy(), on="CUI").collect(engine="streaming"))
    on_table = on_table.join(named, on="CUI")
    rel = pl.scan_parquet(umls_dir / "mrrel.parquet")
    edges = pl.concat([
        rel.filter(pl.col("REL") == "CHD").select("CUI1", "CUI2"),
        rel.filter(pl.col("SAB") == "FMA", pl.col("RELA").is_in(dcfg["relas"]))
        .select("CUI1", "CUI2")]).unique().collect()
    parts = []
    for kind in vocab.KINDS:
        roots = dcfg[kind.lower()]
        dropped = _descendants(edges, roots["negative"]) - _descendants(edges, roots["positive"])
        parts.append(on_table.filter(pl.col("kind") == kind, ~pl.col("CUI").is_in(list(dropped))))
    return pl.concat(parts).sort("kind", "CUI")


def domain_sizes(umls_dir: Path, cfg: dict, domain: pl.DataFrame) -> dict[str, dict[str, int]]:
    """Per kind, how many CUIs the label table holds and how many of them the domain keeps."""
    table = (pl.read_parquet(umls_dir / "mrsty.parquet", columns=["CUI", "TUI"])
             .join(_kind_tuis(cfg), on="TUI").select("CUI", "kind").unique())
    return {kind: {"label_table": table.filter(pl.col("kind") == kind).height,
                   "domain": domain.filter(pl.col("kind") == kind).height}
            for kind in vocab.KINDS}


def exact_matches(queries: pl.DataFrame, umls_dir: Path, cfg: dict,
                  domain: pl.DataFrame) -> tuple[pl.DataFrame, set]:
    """(kind, query, rank, cui, n_candidates) for each CUI of the kind's `domain` with an MRCONSO
    name equal to `query`, best first: a preferred name, then more sources; and the (kind, query)
    pairs whose CUIs all lie outside the label table (a procedure, say)."""
    empty = pl.DataFrame(schema={"kind": pl.String, "query": pl.String, "rank": pl.Int64,
                                 "cui": pl.String, "n_candidates": pl.UInt32})
    wanted = queries.select("kind", "query").unique()
    conso = pl.scan_parquet(umls_dir / "mrconso.parquet")
    hits = (conso.select("CUI", _NORM.alias("query"), "TS", "ISPREF")
            .join(wanted.lazy(), on="query").collect(engine="streaming"))
    if hits.is_empty():
        return empty, set()
    cuis = hits["CUI"].unique().to_list()
    on_table = (pl.scan_parquet(umls_dir / "mrsty.parquet").filter(pl.col("CUI").is_in(cuis))
                .select("CUI", "TUI").collect().join(_kind_tuis(cfg), on="TUI"))
    kept = hits.join(domain, on=["CUI", "kind"])
    outside = (set(hits.select("kind", "query").unique().iter_rows())
               - set(hits.filter(pl.col("CUI").is_in(on_table["CUI"].unique().to_list()))
                     .select("kind", "query").unique().iter_rows()))
    if kept.is_empty():
        return empty, outside
    sabs = (conso.filter(pl.col("CUI").is_in(kept["CUI"].unique().to_list()))
            .group_by("CUI").agg(pl.col("SAB").n_unique().alias("n_sabs")).collect())
    ranked = (kept.group_by("kind", "query", "CUI")
              .agg(((pl.col("TS") == "P") & (pl.col("ISPREF") == "Y")).any().alias("preferred"))
              .join(sabs, on="CUI")
              .sort(["kind", "query", "preferred", "n_sabs", "CUI"],
                    descending=[False, False, True, True, False])
              .with_columns((pl.int_range(pl.len()).over("kind", "query") + 1).alias("rank"),
                            pl.len().over("kind", "query").alias("n_candidates")))
    ranked = ranked.select("kind", "query", "rank", pl.col("CUI").alias("cui"), "n_candidates")
    return ranked, outside


def describe(cuis: Collection[str], umls_dir: Path) -> pl.DataFrame:
    """cui, name (the preferred name of MSH, else NCI, FMA, HPO, else any source),
    semantic_types, n_sabs, sabs, tuis. UMLS names: for files under data/ only (T38)."""
    rows = (pl.scan_parquet(umls_dir / "mrconso.parquet").filter(pl.col("CUI").is_in(list(cuis)))
            .select("CUI", "STR", "SAB", "TS", "ISPREF", "STT").collect())
    order = {sab: i for i, sab in enumerate(_SAB_ORDER)}
    names = (rows.with_columns(
                 (~((pl.col("TS") == "P") & (pl.col("ISPREF") == "Y") & (pl.col("STT") == "PF")))
                 .alias("not_pref"),
                 pl.col("SAB").replace_strict(order, default=len(order),
                                              return_dtype=pl.Int64).alias("o"))
             .sort("not_pref", "o", "SAB", "STR").group_by("CUI", maintain_order=True).first()
             .select("CUI", pl.col("STR").alias("name")))
    sabs = rows.group_by("CUI").agg(pl.col("SAB").n_unique().alias("n_sabs"),
                                    pl.col("SAB").unique().sort().str.join(",").alias("sabs"))
    sty = (pl.scan_parquet(umls_dir / "mrsty.parquet").filter(pl.col("CUI").is_in(list(cuis)))
           .collect().group_by("CUI").agg(pl.concat_str("TUI", "STY", separator=" ").unique()
                                          .sort().str.join("; ").alias("semantic_types"),
                                          pl.col("TUI").unique().sort().str.join(",")
                                          .alias("tuis")))
    return (names.join(sabs, on="CUI").join(sty, on="CUI", how="left")
            .select(pl.col("CUI").alias("cui"), "name", "semantic_types", "n_sabs", "sabs",
                    "tuis"))


def _rule_links(heads: pl.DataFrame, exact: pl.DataFrame, outside: set, overrides: dict,
                cfg: dict) -> list[dict]:
    """The rules before SapBERT for each head string; `method` None sends `query` on to it."""
    ranked: dict[tuple[str, str], list[str]] = {}
    for kind, query, cui in exact.sort("rank").select("kind", "query", "cui").iter_rows():
        ranked.setdefault((kind, query), []).append(cui)
    normal, descriptive = set(cfg["normal"]), set(cfg["descriptive"])
    rows = []
    for kind, string in heads.select("kind", "string").iter_rows():
        method, cuis, query = None, [], string
        override = overrides.get(kind, {}).get(string)
        if string in normal:
            method = NORMAL
        elif override == SKIP:
            method = SKIP
        elif override:
            method, cuis = OVERRIDE, [override]
        elif kind == "Observation" and string in descriptive:
            method = DESCRIPTIVE
        elif (kind, string) in ranked:
            method, cuis = EXACT, ranked[(kind, string)]
        elif kind == "Anatomy" and string in cfg["adjectives"]:
            query = cfg["adjectives"][string]
            if (kind, query) in ranked:
                method, cuis = ADJECTIVE, ranked[(kind, query)]
        elif (kind, string) in outside:
            method = OUT_OF_SCOPE
        rows.append({"kind": kind, "string": string, "method": method, "query": query,
                     "matched": query if cuis and method in (EXACT, ADJECTIVE) else None,
                     "cosine": None, "cuis": cuis})
    return rows


def dictionary(umls_dir: Path, cfg: dict, domain: pl.DataFrame) -> pl.DataFrame:
    """name, Observation, Anatomy: each normalised MRCONSO name of a CUI on the label table, sorted,
    and whether it names a CUI of the kind's domain. The names stay those of the whole label
    table, so the embedding cache outlives a change of domain."""
    cuis = (pl.scan_parquet(umls_dir / "mrsty.parquet").select("CUI", "TUI")
            .join(_kind_tuis(cfg).lazy(), on="TUI").select("CUI").unique())
    names = (pl.scan_parquet(umls_dir / "mrconso.parquet").select("CUI", _NORM.alias("name"))
             .join(cuis, on="CUI").filter(pl.col("name") != "")
             .join(domain.lazy(), on="CUI", how="left")
             .group_by("name").agg([(pl.col("kind") == kind).any().alias(kind)
                                    for kind in vocab.KINDS])
             .collect(engine="streaming"))
    return names.sort("name")


def _candidates(rows: list[dict], umls_dir: Path, cfg: dict, domain: pl.DataFrame,
                encoder_factory: Callable[[], object], cache_dir: Path) -> pl.DataFrame:
    """kind, string, rank, cui, cosine, exact for each row left to the selection step: its exact
    CUIs first, then the domain CUIs of the `top_k` dictionary names nearest its query whose
    cosine reaches `floor` (D56), best first, each CUI once."""
    ccfg = cfg["candidates"]
    schema = {"kind": pl.String, "string": pl.String, "rank": pl.Int64, "cui": pl.String,
              "cosine": pl.Float64, "exact": pl.Boolean}
    if not rows:
        return pl.DataFrame(schema=schema)
    encoder = encoder_factory()
    names = dictionary(umls_dir, cfg, domain)
    vectors = sapbert.name_vectors(names["name"].to_list(), encoder,
                                   cache_dir / encoder.revision[:12])
    near: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for kind in vocab.KINDS:
        queries = sorted({r["query"] for r in rows if r["kind"] == kind})
        if not queries:
            continue
        idx, score = sapbert.nearest(encoder.encode(queries), vectors, names[kind].to_numpy(),
                                     k=ccfg["top_k"])
        for query, found, cosine in zip(queries, idx, score, strict=True):
            near[(kind, query)] = [(names["name"][int(i)], float(c))
                                   for i, c in zip(found, cosine, strict=True)
                                   if c >= ccfg["floor"]]
    matched = pl.DataFrame(sorted({(k, n) for (k, _), found in near.items() for n, _ in found}),
                           schema={"kind": pl.String, "query": pl.String}, orient="row")
    exact, _ = exact_matches(matched, umls_dir, cfg, domain)
    by_name: dict[tuple[str, str], list[str]] = {}
    for kind, name, cui in exact.sort("rank").select("kind", "query", "cui").iter_rows():
        by_name.setdefault((kind, name), []).append(cui)
    out = []
    for r in rows:
        seen = dict.fromkeys(r["cuis"], (1.0, True))
        for name, cosine in near.get((r["kind"], r["query"]), []):
            for cui in by_name.get((r["kind"], name), []):
                seen.setdefault(cui, (cosine, False))
        out += [(r["kind"], r["string"], rank, cui, cosine, is_exact)
                for rank, (cui, (cosine, is_exact)) in enumerate(seen.items(), start=1)]
    return pl.DataFrame(out, schema=schema, orient="row")


def _lateral(text: str | None, sides: set[str]) -> bool:
    return bool(text) and bool(sides & set(text.lower().split()))


def _unlateralise(links: pl.DataFrame, umls_dir: Path, cfg: dict,
                  domain: pl.DataFrame) -> pl.DataFrame:
    """Rule 7: a CUI whose matched or preferred name carries a side moves to its unlateralised
    parent in the kind's domain (MRREL PAR: CUI2 is a parent of CUI1, T39). With several
    such parents, the one named like the matched name without the side; else the CUI stays and
    `lateral_unresolved` is set."""
    sides = set(cfg["laterality"])
    linked = links.filter(pl.col("cui").is_not_null())
    names = dict(describe(linked["cui"].unique().to_list(), umls_dir)
                 .select("cui", "name").iter_rows()) if linked.height else {}
    lateral = {(k, c) for k, c, m in linked.select("kind", "cui", "matched").iter_rows()
               if _lateral(m, sides) or _lateral(names.get(c), sides)}
    if not lateral:
        return links.with_columns(pl.lit(None, pl.String).alias("lateral_from"),
                                  pl.lit(False).alias("lateral_unresolved"))
    parents = (pl.scan_parquet(umls_dir / "mrrel.parquet")
               .filter(pl.col("REL") == "PAR", pl.col("CUI1").is_in([c for _, c in lateral]))
               .select(pl.col("CUI1").alias("cui"), pl.col("CUI2").alias("parent")).unique()
               .collect())
    parent_cuis = parents["parent"].unique().to_list()
    pnames = dict(describe(parent_cuis, umls_dir).select("cui", "name").iter_rows())
    allowed = set(domain.filter(pl.col("CUI").is_in(parent_cuis)).iter_rows())

    def unlateralised(kind: str, cui: str, matched: str | None) -> str | None:
        options = [p for p in parents.filter(pl.col("cui") == cui)["parent"]
                   if (p, kind) in allowed and not _lateral(pnames.get(p), sides)]
        bare = " ".join(w for w in (matched or "").split() if w not in sides)
        named = [p for p in options if vocab.normalize(pnames.get(p, "")) == bare]
        return options[0] if len(options) == 1 else named[0] if len(named) == 1 else None

    out = []   # (cui, lateral_from, lateral_unresolved)
    for kind, cui, matched in links.select("kind", "cui", "matched").iter_rows():
        if (kind, cui) not in lateral:
            out.append((cui, None, False))
            continue
        new = unlateralised(kind, cui, matched)
        out.append((new, cui, False) if new else (cui, None, True))
    return links.with_columns(
        pl.Series("cui", [o[0] for o in out], dtype=pl.String),
        pl.Series("lateral_from", [o[1] for o in out], dtype=pl.String),
        pl.Series("lateral_unresolved", [o[2] for o in out], dtype=pl.Boolean))


def link_heads(heads: pl.DataFrame, umls_dir: Path, cfg: dict, overrides: dict,
               domain: pl.DataFrame, encoder_factory: Callable[[], object],
               cache_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, set]:
    """(links, candidates, outside). `links` holds every head string with the rule that decided
    it, or `select` and the route (exact, adjective, sapbert) that gathered its candidates for
    the selection step; `outside`, the (kind, string) pairs whose exact MRCONSO names are all
    outside the label table."""
    queries = pl.concat([
        heads.select("kind", pl.col("string").alias("query")),
        pl.DataFrame([("Anatomy", noun) for noun in set(cfg["adjectives"].values())],
                     schema={"kind": pl.String, "query": pl.String}, orient="row")])
    exact, outside = exact_matches(queries, umls_dir, cfg, domain)
    outside &= set(heads.select("kind", "string").iter_rows())
    rows = _rule_links(heads, exact, outside, overrides, cfg)
    for r in rows:
        r["selected"] = r["method"] in (EXACT, ADJECTIVE, None)
    cands = _candidates([r for r in rows if r["selected"]], umls_dir, cfg, domain,
                        encoder_factory, cache_dir)
    top = {(k, s): (c, cos) for k, s, c, cos in cands.filter(pl.col("rank") == 1)
           .select("kind", "string", "cui", "cosine").iter_rows()}
    counts = {(k, s): n for k, s, n in cands.group_by("kind", "string").len().iter_rows()}
    links = pl.DataFrame([{
        "kind": r["kind"], "string": r["string"],
        "method": SELECT if r["selected"] else r["method"],
        "route": (r["method"] or SAPBERT) if r["selected"] else None, "query": r["query"],
        "rule_cui": r["cuis"][0] if r["method"] == OVERRIDE else None,
        "n_candidates": counts.get((r["kind"], r["string"]), 0),
        "argmax_cui": top.get((r["kind"], r["string"]), (None, None))[0],
        "argmax_cosine": top.get((r["kind"], r["string"]), (None, None))[1],
    } for r in rows], schema={"kind": pl.String, "string": pl.String, "method": pl.String,
                              "route": pl.String, "query": pl.String, "rule_cui": pl.String,
                              "n_candidates": pl.Int64, "argmax_cui": pl.String,
                              "argmax_cosine": pl.Float64})
    return heads.join(links, on=["kind", "string"], how="left"), cands, outside


_RULE_DECISION = {NORMAL: NORMAL, SKIP: SKIP, OVERRIDE: CUI, DESCRIPTIVE: DESCRIPTIVE,
                  OUT_OF_SCOPE: OUT_OF_SCOPE}


def _says(choice: dict | None) -> str | None:
    if choice is None:
        return None
    return f"{choice['decision']} {choice['cui']}".strip() + f": {choice['reason']}"


def decide(links: pl.DataFrame, choices: dict[str, dict], adjudication: dict, umls_dir: Path,
           cfg: dict, domain: pl.DataFrame) -> pl.DataFrame:
    """The final link of every head string (D56): an adjudication wins; then the rule; a string
    of the selection step takes the decision both LLMs agree on, else `none` (`llm_disagree`), or
    stays `pending` until both have answered. Adds decision, cui, llm_<provider>, agree, then
    rule 7 (lateral_from, lateral_unresolved). `choices`: {provider: {(kind, string): choice}},
    a choice being {decision, cui, reason}; a choice whose decision is `rejected` never agrees."""
    providers = sorted(choices)
    rows = []
    for kind, string, method, rule_cui in links.select("kind", "string", "method",
                                                        "rule_cui").iter_rows():
        key = (kind, string)
        said = [choices[p].get(key) for p in providers]
        agree = None
        if key in adjudication:
            adj = adjudication[key]
            method, decision, cui = ADJUDICATED, adj["decision"], adj["cui"] or None
        elif method != SELECT:
            decision, cui = _RULE_DECISION[method], rule_cui
        elif any(c is None for c in said):
            method, decision, cui = PENDING, PENDING, None
        else:
            first = said[0]
            agree = all(c["decision"] == first["decision"] and c["cui"] == first["cui"]
                        for c in said) and first["decision"] != REJECTED
            method = LLM if agree else LLM_DISAGREE
            decision = first["decision"] if agree else NONE
            cui = first["cui"] if agree and decision == CUI else None
        rows.append((method, decision, cui, *[_says(c) for c in said], agree))
    schema = {"method": pl.String, "decision": pl.String, "cui": pl.String,
              **{f"llm_{p}": pl.String for p in providers}, "agree": pl.Boolean}
    decided = pl.DataFrame(rows, schema=schema, orient="row")
    out = links.drop("method").hstack(decided)
    # Rule 7 reads the chosen CUI's own preferred name: a side word in the pool string says
    # nothing about the CUI (an unsided lobe chosen for "bilateral lower lobe" stays).
    out = _unlateralise(out.with_columns(pl.lit(None, pl.String).alias("matched")), umls_dir,
                        cfg, domain)
    return out.drop("matched")


def located_pairs(mentions: pl.DataFrame, strings: Collection[str]) -> pl.DataFrame:
    """One row per (Observation mention of `strings`, Anatomy it is located_at): study_id, section,
    entity, obs, anat. Both sides are D52 strings; heads on the drop lists are left out."""
    kept = ~pl.col("dropped_head")
    obs = (mentions.filter(pl.col("kind") == "Observation", kept,
                           pl.col("joined").is_in(list(strings)))
           .select("study_id", "section", "entity", pl.col("joined").alias("obs"), "located_at")
           .explode("located_at", empty_as_null=True).drop_nulls("located_at"))
    anat = (mentions.filter(pl.col("kind") == "Anatomy", kept)
            .select("study_id", "section", pl.col("entity").alias("located_at"),
                    pl.col("joined").alias("anat")))
    return obs.join(anat, on=["study_id", "section", "located_at"]).drop("located_at")


def top_pairs(occurrences: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """obs, anat, n, share: per Observation string the most frequent pairs until they cover
    `coverage` of its located mentions, at most `max_pairs`."""
    n, total = pl.col("n"), pl.col("n").sum().over("obs")
    counts = (occurrences.group_by("obs", "anat").agg(pl.len().alias("n"))
              .sort(["obs", "n", "anat"], descending=[False, True, False]))
    return (counts.with_columns((n / total).alias("share"),
                                ((n.cum_sum().over("obs") - n) / total).alias("before"),
                                pl.int_range(pl.len()).over("obs").alias("i"))
            .filter(pl.col("before") < cfg["coverage"], pl.col("i") < cfg["max_pairs"])
            .select("obs", "anat", "n", "share"))


def pair_links(top: pl.DataFrame, umls_dir: Path, cfg: dict, domain: pl.DataFrame) -> pl.DataFrame:
    """top_pairs plus phrase, cui, n_candidates: the first of "anatomy observation" and
    "observation anatomy" that is an exact MRCONSO name of an Observation CUI of the domain."""
    phrases = [(f"{a} {o}", f"{o} {a}") for o, a in top.select("obs", "anat").iter_rows()]
    queries = pl.DataFrame([("Observation", p) for pair in phrases for p in pair],
                           schema={"kind": pl.String, "query": pl.String}, orient="row")
    exact, _ = exact_matches(queries, umls_dir, cfg, domain)
    best = {q: (c, n) for q, c, n in exact.filter(pl.col("rank") == 1)
            .select("query", "cui", "n_candidates").iter_rows()}
    chosen = [next((p for p in pair if p in best), None) for pair in phrases]
    return top.with_columns(
        pl.Series("phrase", chosen, dtype=pl.String),
        pl.Series("cui", [best[p][0] if p else None for p in chosen], dtype=pl.String),
        pl.Series("n_candidates", [best[p][1] if p else 0 for p in chosen], dtype=pl.Int64))


def descriptive_pairs(pairs: pl.DataFrame, cfg: dict, heads: Collection[str]) -> pl.DataFrame:
    """obs, anat, n, string: the pairs of a descriptive word (`descriptive`, linked only through
    pairs) that no exact name links, each as the phrase "obs anat" for the selection step (D59
    check); a phrase that is a head string of its own is left to that string."""
    return (pairs.filter(pl.col("obs").is_in(cfg["descriptive"]), pl.col("cui").is_null())
            .with_columns(pl.concat_str("obs", "anat", separator=" ").alias("string"))
            .filter(~pl.col("string").is_in(list(heads))).select("obs", "anat", "n", "string"))


def pair_rows(descriptive: pl.DataFrame) -> list[dict]:
    """The descriptive pairs as rows of the selection step (no exact CUI: the exact names were
    tried in both orders by `pair_links`)."""
    return [{"kind": "Observation", "string": s, "query": s, "cuis": [], "method": None}
            for s in descriptive["string"]]


def candidates_for(rows: list[dict], umls_dir: Path, cfg: dict, domain: pl.DataFrame,
                   encoder_factory: Callable[[], object], cache_dir: Path) -> pl.DataFrame:
    """The candidates of rows outside the head (the descriptive pairs), as `link_heads` builds
    them for head strings."""
    return _candidates(rows, umls_dir, cfg, domain, encoder_factory, cache_dir)


def decide_pairs(pairs: pl.DataFrame, descriptive: pl.DataFrame, choices: dict[str, dict],
                 adjudication: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(pairs with `via`: exact, llm or adjudicated; the descriptive pairs with both LLMs'
    answers). A descriptive pair takes the CUI both LLMs agree on, or its adjudication."""
    providers = sorted(choices)
    picked, review = {}, []
    for obs, anat, n, string in descriptive.iter_rows():
        key = ("Observation", string)
        said = [choices[p].get(key) for p in providers]
        agree = None if any(c is None for c in said) else (
            all(c["decision"] == said[0]["decision"] and c["cui"] == said[0]["cui"]
                for c in said) and said[0]["decision"] != REJECTED)
        if key in adjudication:
            adj = adjudication[key]
            via, cui = ADJUDICATED, adj["cui"] if adj["decision"] == CUI else None
        else:
            via = LLM if agree else None
            cui = said[0]["cui"] if agree and said[0]["decision"] == CUI else None
        if cui:
            picked[(obs, anat)] = (cui, via, string)
        review.append((obs, anat, n, string, *[_says(c) for c in said], agree, via, cui))
    cuis, vias, phrases = [], [], []
    for obs, anat, cui, phrase in pairs.select("obs", "anat", "cui", "phrase").iter_rows():
        via = EXACT if cui else None
        if (obs, anat) in picked:
            cui, via, phrase = picked[(obs, anat)]
        cuis.append(cui)
        vias.append(via)
        phrases.append(phrase)
    out = pairs.with_columns(pl.Series("cui", cuis, dtype=pl.String),
                             pl.Series("via", vias, dtype=pl.String),
                             pl.Series("phrase", phrases, dtype=pl.String))
    schema = {"obs": pl.String, "anat": pl.String, "n": pl.Int64, "string": pl.String,
              **{f"llm_{p}": pl.String for p in providers}, "agree": pl.Boolean,
              "via": pl.String, "cui": pl.String}
    return out, pl.DataFrame(review, schema=schema, orient="row")


def pair_share(mentions: pl.DataFrame, occurrences: pl.DataFrame, pairs: pl.DataFrame,
               strings: Sequence[str]) -> dict[str, dict[str, int]]:
    """For each Observation string: its mentions, those located_at an Anatomy, and those with a
    located pair that has a pair link."""
    key = ["study_id", "section", "entity"]
    linked = pairs.filter(pl.col("cui").is_not_null()).select("obs", "anat")
    out = {}
    for s in strings:
        total = mentions.filter(pl.col("kind") == "Observation", ~pl.col("dropped_head"),
                                pl.col("joined") == s).height
        located = occurrences.filter(pl.col("obs") == s)
        hit = located.join(linked, on=["obs", "anat"])
        out[s] = {"mentions": total, "located": located.select(key).unique().height,
                  "pair_linked": hit.select(key).unique().height}
    return out


_CUI_IN_SAYS = re.compile(r"^cui (C\d{7}):")


def flags(links: pl.DataFrame, info: pl.DataFrame, cfg: dict, skipped: Collection[str],
          mismatched: Collection[tuple[str, str]]) -> list[str]:
    """Per row, the automatic flags (D56) on the final CUI and on each LLM's CUI, `;`-joined:
    thin_sources (named only by `flags.thin_sources`), off_domain_word (a word of
    `flags.off_domain_words` in the name), type_mismatch (no semantic type of the string's kind),
    skip_listed (a string on a skip list that has a CUI), chexpert_mismatch, lateral_unresolved."""
    fcfg = cfg["flags"]
    thin, words = set(fcfg["thin_sources"]), set(fcfg["off_domain_words"])
    kind_tuis = {k: set(t) for k, t in _kind_tuis(cfg).group_by("kind").agg("TUI").iter_rows()}
    meta = {c: (n, set(s.split(",")), set((t or "").split(",")))
            for c, n, s, t in info.select("cui", "name", "sabs", "tuis").iter_rows()}
    said_cols = [c for c in links.columns if c.startswith("llm_")]
    out = []
    for row in links.iter_rows(named=True):
        cuis = {row["cui"], *(m.group(1) for c in said_cols
                              if row[c] and (m := _CUI_IN_SAYS.match(row[c])))} - {None}
        found = set()
        for cui in cuis & set(meta):
            name, sabs, tuis = meta[cui]
            found |= {"thin_sources"} if sabs <= thin else set()
            found |= {"off_domain_word"} if words & set(vocab.normalize(name).split()) else set()
            found |= {"type_mismatch"} if not tuis & kind_tuis[row["kind"]] else set()
        found |= {"skip_listed"} if row["cui"] and row["string"] in skipped else set()
        found |= {"chexpert_mismatch"} if (row["kind"], row["string"]) in mismatched else set()
        found |= {"lateral_unresolved"} if row["lateral_unresolved"] else set()
        out.append(";".join(sorted(found)))
    return out


def review_rows(links: pl.DataFrame, info: pl.DataFrame, row_flags: Sequence[str]) -> pl.DataFrame:
    """Every head string for adjudication, with the final CUI's UMLS names (data/ only, T38):
    the LLMs' disagreements first, then flagged rows, then the rest by kind and mentions."""
    said = [c for c in links.columns if c.startswith("llm_")]
    order = (pl.when(pl.col("method") == LLM_DISAGREE).then(0)
             .when(pl.col("flags") != "").then(1).otherwise(2))
    return (links.with_columns(pl.Series("flags", row_flags, dtype=pl.String))
            .join(info.select("cui", "name", "semantic_types", "sabs"), on="cui", how="left")
            .sort([order, "kind", "mentions"], descending=[False, False, True])
            .select("kind", "rank", "string", "mentions", "studies", "band", "method", "route",
                    "decision", "cui", "name", "semantic_types", "sabs", *said, "agree", "flags",
                    "n_candidates", "argmax_cui", "argmax_cosine", "lateral_from"))


def coverage(links: pl.DataFrame) -> dict[str, float]:
    """Share of the head's mentions that are decided (all but none and pending) and that carry a
    CUI, per kind and overall (gate, D56)."""
    out = {}
    for name, part in [("all", links), *links.group_by("kind")]:
        label = name if isinstance(name, str) else name[0]
        total = part["mentions"].sum()
        out[label] = {
            "decided": part.filter(~pl.col("decision").is_in(UNDECIDED))["mentions"].sum() / total,
            "linked": part.filter(pl.col("cui").is_not_null())["mentions"].sum() / total}
    return out


def agreement(links: pl.DataFrame) -> dict[str, dict[str, int]]:
    """Per kind, the strings both LLMs answered and those they agree on."""
    both = links.filter(pl.col("agree").is_not_null())
    return {k: {"answered": n, "agree": a} for k, n, a in
            both.group_by("kind").agg(pl.len(), pl.col("agree").sum()).sort("kind").iter_rows()}


def mention_counts(links: pl.DataFrame) -> dict:
    """Strings and mentions per kind and method."""
    return {f"{k}/{m}": {"strings": s, "mentions": n} for k, m, s, n in
            links.group_by("kind", "method").agg(pl.len(), pl.col("mentions").sum())
            .sort("kind", "method").iter_rows()}
