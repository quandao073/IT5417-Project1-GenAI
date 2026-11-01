"""Vocabulary strings -> CUIs (TIP V2-3, D56; docs/v2-kg-design.md, "Dựng KG offline" step 3).

Every string of the 95% head of the pool vocabulary (D52) gets one single link, from the first
rule that fires: the normal group, a hand override (a CUI or `skip`), a descriptive word on its
own, an exact MRCONSO name among the candidates of its kind, an anatomy adjective's noun, a name
whose CUIs all lie outside the label table (`out_of_scope`), then the nearest SapBERT name, kept
with its cosine. Candidates are the CUIs of the kind's domain (`domain`). A CUI named with a side
moves to its unlateralised parent (design, schema decision 3).

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
_CUI = re.compile(r"C\d{7}")
_SAB_ORDER = ["MSH", "NCI", "FMA", "HPO"]   # whose preferred name to show
_NORM = pl.col("STR").str.to_lowercase().str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()


class LinkerError(RuntimeError):
    """An input the linker cannot trust, such as an override that is neither a CUI nor skip."""


def load_overrides(path: Path) -> dict[str, dict[str, str]]:
    """{kind: {string: CUI or "skip"}} from configs/kg/linker_overrides.yaml."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    kinds = {kind.lower(): kind for kind in vocab.KINDS}
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
    return out


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
    semantic_types, n_sabs. UMLS names: for files under data/ only (T38)."""
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
    sabs = rows.group_by("CUI").agg(pl.col("SAB").n_unique().alias("n_sabs"))
    sty = (pl.scan_parquet(umls_dir / "mrsty.parquet").filter(pl.col("CUI").is_in(list(cuis)))
           .collect().group_by("CUI").agg(pl.concat_str("TUI", "STY", separator=" ").unique()
                                          .sort().str.join("; ").alias("semantic_types")))
    return (names.join(sabs, on="CUI").join(sty, on="CUI", how="left")
            .select(pl.col("CUI").alias("cui"), "name", "semantic_types", "n_sabs"))


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


def _sapbert_links(rows: list[dict], umls_dir: Path, cfg: dict, domain: pl.DataFrame,
                   encoder_factory: Callable[[], object], cache_dir: Path) -> None:
    """The last rule in place: the nearest dictionary name of the row's kind, its best exact CUI."""
    pending = [r for r in rows if r["method"] is None]
    if not pending:
        return
    encoder = encoder_factory()
    names = dictionary(umls_dir, cfg, domain)
    vectors = sapbert.name_vectors(names["name"].to_list(), encoder,
                                   cache_dir / encoder.revision[:12])
    for kind in vocab.KINDS:
        group = [r for r in pending if r["kind"] == kind]
        if not group:
            continue
        queries = sorted({r["query"] for r in group})
        idx, score = sapbert.nearest(encoder.encode(queries), vectors, names[kind].to_numpy())
        best = {q: (names["name"][int(i)], float(s)) for q, i, s in zip(queries, idx, score,
                                                                         strict=True)}
        for r in group:
            r["method"], (r["matched"], r["cosine"]) = SAPBERT, best[r["query"]]
    matched = pl.DataFrame([(r["kind"], r["matched"]) for r in pending],
                           schema={"kind": pl.String, "query": pl.String}, orient="row")
    exact, _ = exact_matches(matched, umls_dir, cfg, domain)
    ranked: dict[tuple[str, str], list[str]] = {}
    for kind, query, cui in exact.sort("rank").select("kind", "query", "cui").iter_rows():
        ranked.setdefault((kind, query), []).append(cui)
    for r in pending:
        r["cuis"] = ranked.get((r["kind"], r["matched"]), [])


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
               cache_dir: Path) -> tuple[pl.DataFrame, set]:
    """The single link of every head string, and the (kind, string) pairs whose exact MRCONSO
    names are all outside the label table."""
    queries = pl.concat([
        heads.select("kind", pl.col("string").alias("query")),
        pl.DataFrame([("Anatomy", noun) for noun in set(cfg["adjectives"].values())],
                     schema={"kind": pl.String, "query": pl.String}, orient="row")])
    exact, outside = exact_matches(queries, umls_dir, cfg, domain)
    outside &= set(heads.select("kind", "string").iter_rows())
    rows = _rule_links(heads, exact, outside, overrides, cfg)
    _sapbert_links(rows, umls_dir, cfg, domain, encoder_factory, cache_dir)
    links = pl.DataFrame({
        "kind": [r["kind"] for r in rows], "string": [r["string"] for r in rows],
        "method": [r["method"] for r in rows], "query": [r["query"] for r in rows],
        "matched": [r["matched"] for r in rows],
        "cosine": pl.Series([r["cosine"] for r in rows], dtype=pl.Float64),
        "cui": pl.Series([r["cuis"][0] if r["cuis"] else None for r in rows], dtype=pl.String),
        "n_candidates": pl.Series([len(r["cuis"]) for r in rows], dtype=pl.Int64),
        "alt1": pl.Series([r["cuis"][1] if len(r["cuis"]) > 1 else None for r in rows],
                          dtype=pl.String),
        "alt2": pl.Series([r["cuis"][2] if len(r["cuis"]) > 2 else None for r in rows],
                          dtype=pl.String),
    })
    links = _unlateralise(links, umls_dir, cfg, domain)
    return heads.join(links, on=["kind", "string"], how="left"), outside


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


def needs_review(links: pl.DataFrame) -> pl.Expr:
    """Band 80 in full; band 90 when the link is not one exact CUI (D56); in any band, a side
    left on the CUI (rule 7) and an adjective whose noun went to SapBERT (rule 5)."""
    single_exact = (pl.col("method") == EXACT) & (pl.col("n_candidates") == 1)
    flagged = pl.col("lateral_unresolved") | (
        (pl.col("method") == SAPBERT) & (pl.col("query") != pl.col("string")))
    return (pl.col("band") == "80") | ((pl.col("band") == "90") & ~single_exact) | flagged


def review_rows(links: pl.DataFrame, umls_dir: Path) -> pl.DataFrame:
    """The rows to review by hand, with UMLS names (for data/ only, T38) and empty decision."""
    rows = links.filter(needs_review(links))
    cuis = {c for col in ("cui", "alt1", "alt2", "lateral_from") for c in rows[col].drop_nulls()}
    info = describe(cuis, umls_dir) if cuis else pl.DataFrame(
        schema={"cui": pl.String, "name": pl.String, "semantic_types": pl.String,
                "n_sabs": pl.UInt32})
    name = dict(info.select("cui", "name").iter_rows())
    rows = rows.join(info, on="cui", how="left")
    return rows.with_columns(
        pl.col("alt1").replace_strict(name, default=None).alias("alt1_name"),
        pl.col("alt2").replace_strict(name, default=None).alias("alt2_name"),
        pl.lit("").alias("decision"), pl.lit("").alias("note"),
    ).select("kind", "rank", "string", "mentions", "studies", "band", "method", "query", "matched",
             "cosine", "cui", "name", "semantic_types", "n_sabs", "n_candidates", "lateral_from",
             "lateral_unresolved", "alt1", "alt1_name", "alt2", "alt2_name", "decision", "note")


def mention_counts(links: pl.DataFrame) -> dict:
    """Strings and mentions per kind and method."""
    return {f"{k}/{m}": {"strings": s, "mentions": n} for k, m, s, n in
            links.group_by("kind", "method").agg(pl.len(), pl.col("mentions").sum())
            .sort("kind", "method").iter_rows()}
