"""Knowledge part of the V2 KG as tables (TIP V2-4, D57, D58; docs/v2-kg-design.md, "Dựng KG
offline" steps 4-5): concept nodes, IS_A, PART_OF, HAS_STY, the CheXpert edges, and SUGGESTS and
TYPICALLY_AT weighted on the pool. Neo4j loads these tables in V2-6.

The core is the 14 labels and every CUI V2-3 linked. It grows inside the candidate domain of
V2-3 (`linker.domain`): anatomy along FMA parts and isa, findings and diseases along IS_A of any
source, never into a sided concept (design, schema decision 3). MRREL rows read CUI1 as the whole
or parent and CUI2 as the part or child (T39).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from pathlib import Path

import polars as pl

from src.kg import linker, vocab

ISA, PART_OF, HAS_STY = "IS_A", "PART_OF", "HAS_STY"
SUGGESTS, TYPICALLY_AT = "SUGGESTS", "TYPICALLY_AT"
EDGE_SCHEMA = {"src": pl.String, "rel": pl.String, "dst": pl.String, "source": pl.String,
               "n": pl.Int64, "n_src": pl.Int64, "p": pl.Float64, "review": pl.String}
KEEP, DROP = "keep", "drop"   # adjudicated SUGGESTS; a dropped edge stays, unused at query time


def _steps(edges: Iterable[tuple[str, str]], start: Collection[str], depth: int,
           allowed: Collection[str], stop: Collection[str] = ()) -> set[str]:
    """CUIs reached from `start` in at most `depth` steps along (from, to) edges, through
    `allowed` CUIs only and never into `stop`; `start` not included."""
    nxt: dict[str, list[str]] = {}
    for a, b in edges:
        nxt.setdefault(a, []).append(b)
    seen, frontier = set(start), set(start)
    for _ in range(depth):
        frontier = {b for a in frontier for b in nxt.get(a, ())
                    if b in allowed and b not in stop and b not in seen}
        seen |= frontier
    return seen - set(start)


def core(links: pl.DataFrame, pairs: pl.DataFrame, mapping: dict, gcfg: dict) -> pl.DataFrame:
    """cui, kind of the core: every CUI V2-3 linked (pair links are Observation), the CUIs of
    the 14 labels, PRJ ids, their `is_a` and `children`, the extra CheXpert children and the
    parent of the other devices (Observation)."""
    rows = set(links.filter(pl.col("cui").is_not_null()).select("cui", "kind").iter_rows())
    rows |= {(c, "Observation") for c in pairs["cui"].drop_nulls()}
    rows |= {(c, "Observation") for cuis in gcfg["chexpert_children"].values() for c in cuis}
    rows.add((gcfg["devices_parent"], "Observation"))
    for entry in mapping.values():
        rows |= {(c, "Observation") for c in [entry["id"], entry.get("is_a"),
                                               *entry.get("children", [])] if c}
    return pl.DataFrame(sorted(rows), schema={"cui": pl.String, "kind": pl.String}, orient="row")


def _sided(cuis: Collection[str], umls_dir: Path, sides: Collection[str]) -> set[str]:
    names = linker.describe(list(cuis), umls_dir).select("cui", "name") if cuis else None
    if names is None:
        return set()
    return {c for c, n in names.iter_rows() if set(sides) & set(vocab.normalize(n).split())}


def expand(core_cuis: pl.DataFrame, umls_dir: Path, lcfg: dict, gcfg: dict,
           domain: pl.DataFrame) -> pl.DataFrame:
    """cui, kind, in_core for the core and its growth (D58): anatomy along FMA parts and isa up
    and down; findings and diseases along IS_A, down `finding_down` steps, up `finding_up` steps
    stopping at `stop_up`; inside the domain, never a sided concept."""
    rel = pl.scan_parquet(umls_dir / "mrrel.parquet")
    fma = (rel.filter(pl.col("SAB") == "FMA", pl.col("RELA").is_in(lcfg["domain"]["relas"]))
           .select("CUI1", "CUI2").unique().collect().rows())
    isa = rel.filter(pl.col("REL") == "CHD").select("CUI1", "CUI2").unique().collect().rows()
    sty = pl.read_parquet(umls_dir / "mrsty.parquet", columns=["CUI", "TUI"])
    disease_tuis = [t for label in ("Finding", "Disease") for t in lcfg["labels"][label]]
    findings = set(sty.filter(pl.col("TUI").is_in(disease_tuis))["CUI"])
    anatomy_in = set(domain.filter(pl.col("kind") == "Anatomy")["CUI"])
    obs_in = set(domain.filter(pl.col("kind") == "Observation")["CUI"]) & findings
    anat0 = set(core_cuis.filter(pl.col("kind") == "Anatomy")["cui"])
    obs0 = set(core_cuis.filter(pl.col("kind") == "Observation")["cui"]) & findings
    depth = gcfg["anatomy_depth"]
    anat = (_steps(fma, anat0, depth, anatomy_in)
            | _steps([(b, a) for a, b in fma], anat0, depth, anatomy_in))
    obs = (_steps(isa, obs0, gcfg["finding_down"], obs_in)
           | _steps([(b, a) for a, b in isa], obs0, gcfg["finding_up"], obs_in, gcfg["stop_up"]))
    grown = [(c, "Anatomy") for c in anat] + [(c, "Observation") for c in obs]
    sided = _sided({c for c, _ in grown}, umls_dir, lcfg["laterality"])
    have = set(core_cuis.select("cui", "kind").iter_rows())
    rows = [(c, k, True) for c, k in have]
    rows += [(c, k, False) for c, k in grown if c not in sided and (c, k) not in have]
    return pl.DataFrame(rows, schema={"cui": pl.String, "kind": pl.String, "in_core": pl.Boolean},
                        orient="row").unique(["cui", "kind"]).sort("kind", "cui")


def nodes_table(grown: pl.DataFrame, mapping: dict, umls_dir: Path, lcfg: dict, gcfg: dict,
                pool_counts: pl.DataFrame) -> pl.DataFrame:
    """cui, name, labels, sty, chexpert_label, scorable, n_pool, in_core (one row per CUI). Labels
    follow the semantic-type table (design); a PRJ id takes `prj_labels`; the 14 label ids are
    the scorable concepts."""
    by_cui = grown.group_by("cui").agg(pl.col("in_core").any())
    cuis = [c for c in by_cui["cui"] if not c.startswith("PRJ:")]
    tui_label = {t: label for label, tuis in lcfg["labels"].items() for t in tuis}
    sty = (pl.read_parquet(umls_dir / "mrsty.parquet", columns=["CUI", "TUI"])
           .filter(pl.col("CUI").is_in(cuis)).unique().group_by("CUI")
           .agg(pl.col("TUI").sort().str.join(",").alias("sty"),
                pl.col("TUI").replace_strict(tui_label, default=None).drop_nulls().unique()
                .sort().str.join(",").alias("labels")))
    names = linker.describe(cuis, umls_dir).select("cui", "name")
    label_of = {entry["id"]: label for label, entry in mapping.items()}
    prj = pl.DataFrame([(c, label_of.get(c, c), "", gcfg["prj_labels"][c])
                        for c in by_cui["cui"] if c.startswith("PRJ:")],
                       schema=["cui", "name", "sty", "labels"], orient="row")
    out = pl.concat([names.join(sty.rename({"CUI": "cui"}), on="cui", how="left"), prj])
    return (by_cui.join(out, on="cui", how="left").join(pool_counts, on="cui", how="left")
            .with_columns(pl.col("cui").replace_strict(label_of, default=None)
                          .alias("chexpert_label"),
                          pl.col("cui").is_in(list(label_of)).alias("scorable"),
                          pl.col("n_pool").fill_null(0))
            .select("cui", "name", "labels", "sty", "chexpert_label", "scorable", "n_pool",
                    "in_core").sort("cui"))


def device_groups(links: pl.DataFrame, nodes: pl.DataFrame, umls_dir: Path, lcfg: dict,
                  device_phrases: Collection[str], parent: str,
                  children: Collection[str]) -> dict[str, list[str]]:
    """{"support_devices": [...], "other": [...]} (D58): a core device CUI goes under
    PRJ:support_devices when a pool string linked to it, or one of its MRCONSO names, holds a
    CheXpert labeler `support_devices` mention phrase (word for word, as the labeler matches
    them in report text), or when it is one of the label's `children`; the other core devices go
    under C0025080."""
    devices = set(pl.read_parquet(umls_dir / "mrsty.parquet", columns=["CUI", "TUI"])
                  .filter(pl.col("TUI").is_in(lcfg["labels"]["Device"]))["CUI"])
    # the parent of the other devices is never one of them (an IS_A loop otherwise)
    core_devices = set(nodes.filter(pl.col("in_core"))["cui"]) & devices - {parent}
    phrases = [p.split() for p in device_phrases]

    def says(text: str) -> bool:   # the labeler finds its phrases inside the text
        words = text.split()
        return any(words[i:i + len(p)] == p for p in phrases
                   for i in range(len(words) - len(p) + 1))

    by_string = {c for s, c in links.filter(pl.col("cui").is_in(list(core_devices)))
                 .select("string", "cui").iter_rows() if says(s)}
    by_name = {c for c, n in pl.scan_parquet(umls_dir / "mrconso.parquet")
               .filter(pl.col("CUI").is_in(list(core_devices)))
               .select("CUI", linker._NORM.alias("n")).unique().collect().iter_rows() if says(n)}
    support = (by_string | by_name | set(children)) & core_devices
    return {"support_devices": sorted(support), "other": sorted(core_devices - support)}


def umls_edges(nodes: pl.DataFrame, umls_dir: Path, lcfg: dict) -> pl.DataFrame:
    """IS_A (child -> parent, any source; `source` lists the SABs), PART_OF (part -> whole, FMA)
    between nodes, and HAS_STY (concept -> TUI)."""
    cuis = list(set(nodes["cui"]))
    rel = (pl.scan_parquet(umls_dir / "mrrel.parquet")
           .filter(pl.col("CUI1").is_in(cuis), pl.col("CUI2").is_in(cuis),
                   pl.col("CUI1") != pl.col("CUI2")))
    isa = (rel.filter(pl.col("REL") == "CHD")
           .group_by(pl.col("CUI2").alias("src"), pl.col("CUI1").alias("dst"))
           .agg(pl.col("SAB").unique().sort().str.join(",").alias("source"))
           .with_columns(pl.lit(ISA).alias("rel")).collect())
    parts = (rel.filter(pl.col("SAB") == "FMA", pl.col("RELA").is_in(
                ["regional_part_of", "constitutional_part_of"]))
             .select(pl.col("CUI2").alias("src"), pl.col("CUI1").alias("dst")).unique()
             .with_columns(pl.lit(PART_OF).alias("rel"), pl.lit("FMA").alias("source")).collect())
    sty = (pl.read_parquet(umls_dir / "mrsty.parquet", columns=["CUI", "TUI"])
           .filter(pl.col("CUI").is_in(cuis)).unique()
           .select(pl.col("CUI").alias("src"), pl.lit(HAS_STY).alias("rel"),
                   pl.col("TUI").alias("dst"), pl.lit("MRSTY").alias("source")))
    return pl.concat([_edges(isa), _edges(parts), _edges(sty)])


def semantic_types(umls_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(SemanticType nodes tui, name, stn; their IS_A edges from STN)."""
    st = (pl.read_parquet(umls_dir / "mrsty.parquet", columns=["TUI", "STY", "STN"]).unique()
          .rename({"TUI": "tui", "STY": "name", "STN": "stn"}).sort("tui"))
    by_stn = dict(st.select("stn", "tui").iter_rows())
    rows = [(t, ISA, by_stn[s.rsplit(".", 1)[0]], "STN") for t, s in st.select("tui", "stn")
            .iter_rows() if "." in s and s.rsplit(".", 1)[0] in by_stn]
    return st, _edges(pl.DataFrame(rows, schema=["src", "rel", "dst", "source"], orient="row"))


def chexpert_edges(mapping: dict, gcfg: dict, links: pl.DataFrame,
                   groups: dict[str, list[str]]) -> tuple[pl.DataFrame, list[str]]:
    """IS_A with source chexpert (D57, D58): the PRJ ids' is_a and children, the CheXpert
    hierarchy, extra children, generic strings to their label, and the device groups. Returns
    the edges and the generic strings that have no Observation CUI."""
    ids = {label: entry["id"] for label, entry in mapping.items()}
    rows = []
    for entry in mapping.values():
        if entry.get("is_a"):
            rows.append((entry["id"], entry["is_a"]))
        rows += [(child, entry["id"]) for child in entry.get("children", [])]
    rows += [(ids[child], ids[parent]) for child, parent in gcfg["chexpert_hierarchy"].items()]
    rows += [(c, ids[label]) for label, cuis in gcfg["chexpert_children"].items() for c in cuis]
    obs = dict(links.filter(pl.col("kind") == "Observation", pl.col("cui").is_not_null())
               .select("string", "cui").iter_rows())
    missing = []
    for label, strings in gcfg["generic_is_a"].items():
        for s in strings:
            if s in obs:
                rows.append((obs[s], ids[label]))
            else:
                missing.append(s)
    rows += [(c, ids[label]) for label, cuis in gcfg.get("phrase_is_a", {}).items() for c in cuis]
    rows += [(c, ids["Support Devices"]) for c in groups["support_devices"]]
    rows += [(c, gcfg["devices_parent"]) for c in groups["other"]]
    rows = [(a, ISA, b, "chexpert") for a, b in dict.fromkeys(rows) if a != b]
    return _edges(pl.DataFrame(rows, schema=["src", "rel", "dst", "source"], orient="row")), missing


def manual_edges(gcfg: dict) -> pl.DataFrame:
    """IS_A and SUGGESTS edges written by hand in configs/kg.yaml (graph.manual_edges), source
    `manual`. A SUGGESTS edge carries equal weights p = 1/k and no pool count, so queries keep it
    whatever `min_n` says (the `source` is how they tell)."""
    cfg = gcfg.get("manual_edges", {})
    rows = [(child, ISA, parent, "manual", None, None, None)
            for child, parent in cfg.get("is_a", [])]
    for disease, findings in cfg.get("suggests", {}).items():
        rows += [(f, SUGGESTS, disease, "manual", None, None, round(1 / len(findings), 4))
                 for f in findings]
    return _edges(pl.DataFrame(rows, schema=["src", "rel", "dst", "source", "n", "n_src", "p"],
                               orient="row"))


def _edges(df: pl.DataFrame) -> pl.DataFrame:
    for name, dtype in EDGE_SCHEMA.items():
        if name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype).alias(name))
    return df.select([pl.col(n).cast(t) for n, t in EDGE_SCHEMA.items()])


def mention_cuis(mentions: pl.DataFrame, links: pl.DataFrame,
                 pairs: pl.DataFrame) -> pl.DataFrame:
    """study_id, section, entity, kind, polarity, cui, located_at, suggests per pool mention with
    a CUI: an Observation located at an Anatomy whose (string, string) pair has a pair link takes
    it, any other mention its string's V2-3 link (D58)."""
    kept = mentions.filter(~pl.col("dropped_head")).select(
        "study_id", "section", "entity", "kind", "polarity", "joined", "located_at", "suggests")
    single = kept.join(links.filter(pl.col("cui").is_not_null())
                       .select("kind", pl.col("string").alias("joined"), "cui"),
                       on=["kind", "joined"], how="left")
    anat = kept.filter(pl.col("kind") == "Anatomy").select(
        "study_id", "section", pl.col("entity").alias("target"), pl.col("joined").alias("anat"))
    paired = (kept.filter(pl.col("kind") == "Observation")
              .select("study_id", "section", "entity", pl.col("joined").alias("obs"),
                      pl.col("located_at").alias("target"))
              .explode("target", empty_as_null=True).drop_nulls("target")
              .join(anat, on=["study_id", "section", "target"])
              .join(pairs.filter(pl.col("cui").is_not_null())
                    .select("obs", "anat", pl.col("cui").alias("pair_cui")), on=["obs", "anat"])
              .group_by("study_id", "section", "entity").agg(pl.col("pair_cui").sort().first()))
    return (single.join(paired, on=["study_id", "section", "entity"], how="left")
            .with_columns(pl.coalesce("pair_cui", "cui").alias("cui"))
            .filter(pl.col("cui").is_not_null())
            .select("study_id", "section", "entity", "kind", "polarity", "cui", "located_at",
                    "suggests"))


def weighted(mcuis: pl.DataFrame, column: str, target_kind: str, rel: str,
             min_n: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Edges src -> dst from a present Observation mention to the mentions of `column`
    (located_at or suggests): n = reports with the pair, n_src = reports with src present,
    p = n / n_src. Returns the edges kept (n >= min_n) and every pair before the cut."""
    src = mcuis.filter(pl.col("kind") == "Observation", pl.col("polarity") == "present")
    n_src = src.group_by(pl.col("cui").alias("src")).agg(
        pl.col("study_id").n_unique().alias("n_src"))
    dst = mcuis.filter(pl.col("kind") == target_kind).select(
        "study_id", "section", pl.col("entity").alias("target"), pl.col("cui").alias("dst"))
    pairs = (src.select("study_id", "section", pl.col("cui").alias("src"),
                        pl.col(column).alias("target"))
             .explode("target", empty_as_null=True).drop_nulls("target")
             .join(dst, on=["study_id", "section", "target"])
             .filter(pl.col("src") != pl.col("dst"))
             .group_by("src", "dst").agg(pl.col("study_id").n_unique().alias("n"))
             .join(n_src, on="src")
             .with_columns((pl.col("n") / pl.col("n_src")).alias("p"),
                           pl.lit(rel).alias("rel"), pl.lit("pool").alias("source")))
    order = dict(by=["n", "src", "dst"], descending=[True, False, False])
    every = _edges(pairs).sort(**order)
    return every.filter(pl.col("n") >= min_n), every


def n_pool(mcuis: pl.DataFrame) -> pl.DataFrame:
    """cui, n_pool: pool reports with a mention of the concept, any polarity."""
    return mcuis.group_by("cui").agg(pl.col("study_id").n_unique().alias("n_pool"))


def apply_review(edges: pl.DataFrame, review: pl.DataFrame) -> pl.DataFrame:
    """SUGGESTS edges take the adjudicated `decision` (keep or drop) of suggests_review.csv."""
    decided = (review.filter(pl.col("decision").is_in([KEEP, DROP]))
               .select("src", "dst", pl.col("decision").alias("decided"),
                       pl.lit(SUGGESTS).alias("rel")))
    return (edges.join(decided, on=["src", "rel", "dst"], how="left")
            .with_columns(pl.coalesce("decided", "review").alias("review")).drop("decided"))


def aliases(nodes: pl.DataFrame, umls_dir: Path, concepts: dict, mapping: dict,
            vi_review: pl.DataFrame, manual: dict | None = None) -> pl.DataFrame:
    """cui, text, norm, lang, source: the English MRCONSO names of every node, the en and vi forms
    of configs/concepts.yaml for the 14 labels, the Vietnamese aliases kept on review, and the
    hand-written ones of configs/kg/aliases_manual.yaml ({lang: {cui: [forms]}})."""
    cuis = [c for c in nodes["cui"] if not c.startswith("PRJ:")]
    mrconso = (pl.scan_parquet(umls_dir / "mrconso.parquet").filter(pl.col("CUI").is_in(cuis))
               .select(pl.col("CUI").alias("cui"), pl.col("STR").alias("text")).unique()
               .with_columns(pl.lit("en").alias("lang"), pl.lit("mrconso").alias("source"))
               .collect())
    ids = {label: entry["id"] for label, entry in mapping.items()}
    rows = [(ids[label], text, lang, "concepts_yaml") for label, entry in concepts.items()
            if label in ids for lang in ("en", "vi") for text in entry.get(lang, [])]
    yaml_rows = pl.DataFrame(rows, schema=["cui", "text", "lang", "source"], orient="row")
    kept = (vi_review.filter(pl.col("decision") == KEEP, pl.col("alias") != "")
            .select("cui", pl.col("alias").alias("text"), pl.lit("vi").alias("lang"),
                    pl.lit("llm").alias("source")))
    hand = pl.DataFrame([(cui, text, lang, "manual") for lang, by_cui in (manual or {}).items()
                         if lang in ("en", "vi") for cui, forms in by_cui.items()
                         for text in forms],
                        schema=["cui", "text", "lang", "source"], orient="row")
    out = pl.concat([mrconso, yaml_rows, kept, hand])
    return (out.with_columns(pl.col("text").str.to_lowercase().str.replace_all(r"[^\w]+", " ")
                             .str.strip_chars().alias("norm"))
            .unique(["cui", "norm", "lang"], keep="first", maintain_order=True)
            .select("cui", "text", "norm", "lang", "source").sort("cui", "lang", "norm"))


def scorable_reach(nodes: pl.DataFrame, edges: pl.DataFrame) -> pl.DataFrame:
    """cui, reaches (D59): a concept reaches the scorable ones when it is scorable; or IS_A* leads
    from it to a reaching concept; or it is a Disease with a SUGGESTS, not dropped on review,
    from a reaching concept; or it is an Anatomy whose own or a part's (PART_OF*) TYPICALLY_AT
    comes from a reaching concept. Grown to a fixed point."""
    def pairs(rel: str, query: bool = False) -> list[tuple[str, str]]:
        part = edges.filter(pl.col("rel") == rel)
        if query:
            part = part.filter(pl.col("review").fill_null(KEEP) != DROP)
        return part.select("src", "dst").rows()

    isa, part = pairs(ISA), pairs(PART_OF)               # (child, parent), (part, whole)
    suggests, at = pairs(SUGGESTS, query=True), pairs(TYPICALLY_AT)
    everything = set(nodes["cui"])
    reach = set(nodes.filter(pl.col("scorable"))["cui"])
    while True:
        grown = set(reach)
        grown |= _steps([(b, a) for a, b in isa], grown, len(everything), everything)
        grown |= {d for s, d in suggests if s in grown}
        targets = {d for s, d in at if s in grown}
        grown |= targets | _steps(part, targets, len(everything), everything)
        if grown == reach:
            break
        reach = grown
    return nodes.select("cui").with_columns(pl.col("cui").is_in(list(reach)).alias("reaches"))
