"""Load the KG tables and the top-N PREDICTED edges into Neo4j (docs/v2-kg-design.md, "Dựng KG
offline" step 6; WORKLOG D65).

Concept, Alias, SemanticType and the knowledge edges come from data/kg/graph/*.parquet in
`UNWIND` batches. Each Concept carries its `quality` (src/kg/quality.py). `PREDICTED` joins an
Image to a scorable Concept for that concept's `top_n` best images of the score matrix only, for
the demo and the explanation: scoring reads the matrix in RAM, never this graph. The Image nodes
are those of the top-N lists. A `KGBuild` node records what was loaded; the API reads it.

Loading wipes the database first: one graph, one build.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import polars as pl

from src.embeddings.validate import Check

CONCEPT_LABELS = ("Finding", "Disease", "Anatomy", "Device")   # the only secondary labels
KNOWLEDGE_EDGES = {"IS_A": "source", "PART_OF": "source", "SUGGESTS": "weights",
                   "TYPICALLY_AT": "weights"}
SCHEMA = [
    "CREATE CONSTRAINT concept_cui IF NOT EXISTS FOR (c:Concept) REQUIRE c.cui IS UNIQUE",
    "CREATE CONSTRAINT image_id IF NOT EXISTS FOR (i:Image) REQUIRE i.image_id IS UNIQUE",
    "CREATE CONSTRAINT semantic_tui IF NOT EXISTS FOR (s:SemanticType) REQUIRE s.tui IS UNIQUE",
    "CREATE INDEX alias_norm IF NOT EXISTS FOR (a:Alias) ON (a.norm)",
    "CREATE FULLTEXT INDEX alias_text IF NOT EXISTS FOR (a:Alias) ON EACH [a.text]",
]


def sha256_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def build_ids(graph_dir: Path, matrix_dir: Path) -> tuple[str, str]:
    """(kg_build_id, matrix_version): short hashes of the KG tables with their quality table,
    and of the score matrix, so a response names exactly the data it was computed from."""
    kg = sha256_files([graph_dir / f for f in ("nodes.parquet", "edges.parquet",
                                               "aliases.parquet", "semantic_types.parquet",
                                               "quality.json")])
    matrix = sha256_files([matrix_dir / "corpus_scores.parquet", matrix_dir / "probe.npz"])
    return f"kg_{kg[:12]}", f"{matrix_dir.name}@{matrix[:12]}"


def batches(rows: list[dict], size: int) -> Iterator[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def _run(session, query: str, rows: list[dict], size: int, **params) -> None:
    for chunk in batches(rows, size):
        session.run(query, rows=chunk, **params).consume()


def reset(session) -> None:
    session.run("CALL () { MATCH (n) DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS").consume()
    for statement in SCHEMA:
        session.run(statement).consume()


def load_concepts(session, nodes: pl.DataFrame, quality: dict, size: int) -> None:
    rows = []
    for r in nodes.iter_rows(named=True):
        q = quality.get(r["cui"], {})
        rows.append({
            "cui": r["cui"], "name": r["name"], "sty": r["sty"] or "",
            "chexpert_label": r["chexpert_label"], "scorable": bool(r["scorable"]),
            "n_pool": int(r["n_pool"]), "in_core": bool(r["in_core"]),
            "column": q.get("column"), "quality": q.get("auroc_lenient"),
            "quality_low": (q.get("ci95") or [None, None])[0],
            "quality_high": (q.get("ci95") or [None, None])[1],
            "weak": q.get("weak"), "borderline": q.get("borderline"),
            "labels": [label for label in (r["labels"] or "").split(",") if label]})
    _run(session, "UNWIND $rows AS r CREATE (c:Concept {cui: r.cui, name: r.name, sty: r.sty, "
         "chexpert_label: r.chexpert_label, scorable: r.scorable, n_pool: r.n_pool, "
         "in_core: r.in_core, `column`: r.column, quality: r.quality, "
         "quality_low: r.quality_low, quality_high: r.quality_high, weak: r.weak, "
         "borderline: r.borderline})", rows, size)
    for label in CONCEPT_LABELS:
        cuis = [r["cui"] for r in rows if label in r["labels"]]
        for chunk in batches([{"cui": c} for c in cuis], size):
            session.run(f"UNWIND $rows AS r MATCH (c:Concept {{cui: r.cui}}) SET c:{label}",
                        rows=chunk).consume()


def load_semantic_types(session, types: pl.DataFrame, size: int) -> None:
    _run(session, "UNWIND $rows AS r CREATE (:SemanticType {tui: r.tui, name: r.name, "
         "stn: r.stn})", types.to_dicts(), size)


def load_aliases(session, aliases: pl.DataFrame, size: int) -> None:
    _run(session, "UNWIND $rows AS r MATCH (c:Concept {cui: r.cui}) CREATE (c)-[:HAS_ALIAS]->"
         "(:Alias {text: r.text, norm: r.norm, lang: r.lang, source: r.source})",
         aliases.to_dicts(), size)


def load_edges(session, edges: pl.DataFrame, size: int) -> None:
    """IS_A and PART_OF between concepts, SUGGESTS and TYPICALLY_AT with their pool weights,
    HAS_STY to a semantic type, IS_A between semantic types (source STN)."""
    sty = edges.filter(pl.col("rel") == "HAS_STY")
    _run(session, "UNWIND $rows AS r MATCH (c:Concept {cui: r.src}), (s:SemanticType "
         "{tui: r.dst}) CREATE (c)-[:HAS_STY]->(s)", sty.to_dicts(), size)
    tree = edges.filter(pl.col("rel") == "IS_A", pl.col("source") == "STN")
    _run(session, "UNWIND $rows AS r MATCH (a:SemanticType {tui: r.src}), (b:SemanticType "
         "{tui: r.dst}) CREATE (a)-[:IS_A]->(b)", tree.to_dicts(), size)
    rest = edges.filter(pl.col("rel") != "HAS_STY", ~((pl.col("rel") == "IS_A")
                                                      & (pl.col("source") == "STN")))
    for rel, kind in KNOWLEDGE_EDGES.items():
        part = rest.filter(pl.col("rel") == rel)
        props = ("{source: r.source}" if kind == "source" else
                 "{n: r.n, n_src: r.n_src, p: r.p, review: r.review}")
        _run(session, f"UNWIND $rows AS r MATCH (a:Concept {{cui: r.src}}), (b:Concept "
             f"{{cui: r.dst}}) CREATE (a)-[:{rel} {props}]->(b)", part.to_dicts(), size)


def top_n_images(scores: np.ndarray, n: int) -> np.ndarray:
    """[n, k] row indices of the n best rows of each of the k columns, best first (ties broken
    by row index, so a rebuild gives the same edges)."""
    rows = np.arange(scores.shape[0])
    return np.stack([np.lexsort((rows, -scores[:, j]))[:n] for j in range(scores.shape[1])],
                    axis=1)


def load_predicted(session, scores: pl.DataFrame, concept_columns: dict[str, str],
                   images: pl.DataFrame, top_n: int, build: dict, size: int) -> dict:
    """Image nodes of the top-N lists and a PREDICTED edge per (concept, top-N image)."""
    values = scores.select(list(concept_columns.values())).to_numpy().astype(np.float64)
    order = top_n_images(values, top_n)
    meta = images.select("image_id", "study_id", "view").unique("image_id")
    by_image = {r["image_id"]: r for r in meta.to_dicts()}
    image_ids, study_ids = scores["image_id"].to_list(), scores["study_id"].to_list()
    edges, used = [], {}
    for j, cui in enumerate(concept_columns):
        for rank, row in enumerate(order[:, j].tolist(), start=1):
            image = image_ids[row]
            used[image] = study_ids[row]
            edges.append({"cui": cui, "image_id": image, "logit": float(values[row, j]),
                          "rank": rank})
    _run(session, "UNWIND $rows AS r CREATE (:Image {image_id: r.image_id, study_id: r.study_id, "
         "view: r.view})",
         [{"image_id": i, "study_id": s, "view": by_image[i]["view"]} for i, s in used.items()],
         size)
    _run(session, "UNWIND $rows AS r MATCH (c:Concept {cui: r.cui}), (i:Image "
         "{image_id: r.image_id}) CREATE (i)-[:PREDICTED {logit: r.logit, rank: r.rank, "
         "kg_build_id: $kg, matrix_version: $mv}]->(c)", edges, size,
         kg=build["kg_build_id"], mv=build["matrix_version"])
    return {"images": len(used), "predicted": len(edges)}


def load_all(session, *, nodes: pl.DataFrame, quality: dict, types: pl.DataFrame,
             aliases: pl.DataFrame, edges: pl.DataFrame, scores: pl.DataFrame,
             images: pl.DataFrame, concept_columns: dict[str, str], top_n: int, batch: int,
             build: dict) -> dict:
    """Wipe, load every table and the top-N PREDICTED edges; the counts of what is in the graph.
    `build` names the data ({kg_build_id, matrix_version}); the KGBuild node is written last."""
    reset(session)
    load_concepts(session, nodes, quality, batch)
    load_semantic_types(session, types, batch)
    load_aliases(session, aliases, batch)
    load_edges(session, edges, batch)
    loaded = load_predicted(session, scores, concept_columns, images, top_n, build, batch)
    manifest = {**build, "top_n": top_n, "scorable_concepts": len(concept_columns), **loaded,
                **counts(session)}
    write_build(session, manifest)
    return manifest


def counts(session) -> dict:
    nodes = {r["label"]: r["n"] for r in session.run(
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY label")}
    rels = {r["rel"]: r["n"] for r in session.run(
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY rel")}
    return {"nodes": nodes, "edges": rels}


def write_build(session, manifest: dict) -> None:
    session.run("CREATE (b:KGBuild) SET b = $props", props={
        k: v if isinstance(v, str | int | float | bool) else json.dumps(v)
        for k, v in manifest.items()}).consume()


def read_build(session) -> dict | None:
    row = session.run("MATCH (b:KGBuild) RETURN properties(b) AS b").single()
    return None if row is None else row["b"]


def validate(session, manifest: dict, matrix_columns: Sequence[str], label_ids: dict[str, str],
             matrix_build_id: str, expected_build_id: str) -> list[Check]:
    """The four checks of the design: one matrix column per scorable concept; one node per
    CheXpert label; the matrix' build_id matches the encoder in use; PREDICTED edges equal
    N x scorable concepts."""
    def columns() -> Check:
        rows = list(session.run("MATCH (c:Concept {scorable: true}) RETURN c.cui AS cui, "
                                "c.`column` AS col"))
        wanted = [r["col"] for r in rows]
        bad = [r["cui"] for r in rows if wanted.count(r["col"]) != 1
               or r["col"] not in matrix_columns]
        extra = sorted(set(matrix_columns) - set(wanted))
        return Check("kg_columns", not bad and not extra and len(rows) == len(matrix_columns),
                     f"{len(rows)} scorable concepts, {len(matrix_columns)} columns, "
                     f"{len(bad)} without exactly one, {len(extra)} columns unused")

    def labels() -> Check:
        found = {r["cui"]: r["n"] for r in session.run(
            "UNWIND $ids AS id OPTIONAL MATCH (c:Concept {cui: id}) RETURN id AS cui, "
            "count(c) AS n", ids=list(label_ids.values()))}
        bad = [i for i in label_ids.values() if found.get(i) != 1]
        return Check("kg_labels", len(set(label_ids.values())) == len(label_ids) and not bad,
                     f"{len(label_ids)} labels, {len(bad)} without exactly one node")

    def build() -> Check:
        return Check("kg_matrix_build", matrix_build_id == expected_build_id,
                     f"matrix {matrix_build_id} vs encoder build {expected_build_id}")

    def predicted() -> Check:
        got = session.run("MATCH ()-[r:PREDICTED]->() RETURN count(r) AS n").single()["n"]
        scorable = session.run("MATCH (c:Concept {scorable: true}) RETURN count(c) AS n"
                               ).single()["n"]
        want = manifest["top_n"] * scorable
        return Check("kg_predicted", got == want,
                     f"{got} edges, top_n {manifest['top_n']} x {scorable} scorable = {want}")

    out = []
    for check in (columns, labels, build, predicted):
        try:
            out.append(check())
        except Exception as exc:  # noqa: BLE001 - a failing check is a FAIL line
            out.append(Check(check.__name__, False, f"{type(exc).__name__}: {exc}"))
    return out
