"""Loader pieces that need no database (src/kg/neo4j_load.py, src/kg/quality.py)."""

from __future__ import annotations

import numpy as np

from src.kg import neo4j_load as nl
from src.kg import quality


def test_top_n_images_breaks_ties_by_row_and_goes_best_first():
    scores = np.array([[0.5, 1.0], [0.9, 1.0], [0.9, 0.2], [0.1, 1.0]])
    got = nl.top_n_images(scores, 2)
    # column 0: rows 1 and 2 tie at 0.9, row 1 first; column 1: rows 0 and 1 tie at 1.0
    assert got.tolist() == [[1, 0], [2, 1]]


def test_batches_cover_every_row_once():
    rows = [{"i": i} for i in range(7)]
    assert [len(b) for b in nl.batches(rows, 3)] == [3, 3, 1]


def test_build_ids_change_with_the_data(tmp_path):
    graph, matrix = tmp_path / "graph", tmp_path / "matrix_v1"
    graph.mkdir()
    matrix.mkdir()
    for name in ("nodes.parquet", "edges.parquet", "aliases.parquet", "semantic_types.parquet",
                 "quality.json"):
        (graph / name).write_bytes(name.encode())
    (matrix / "corpus_scores.parquet").write_bytes(b"scores")
    (matrix / "probe.npz").write_bytes(b"probe")
    first = nl.build_ids(graph, matrix)
    assert first[0].startswith("kg_") and first[1].startswith("matrix_v1@")
    assert nl.build_ids(graph, matrix) == first
    (graph / "quality.json").write_bytes(b"changed")
    assert nl.build_ids(graph, matrix)[0] != first[0]
    assert nl.build_ids(graph, matrix)[1] == first[1]


def test_quality_table_marks_weak_and_borderline_concepts():
    rng = np.random.default_rng(0)
    statuses = np.array(["PRESENT"] * 60 + ["ABSENT"] * 60, dtype=object)[:, None]
    logits = np.concatenate([rng.normal(1, 1, 60), rng.normal(-1, 1, 60)])[:, None]
    base = quality.base_columns(logits, statuses, ["Edema"], n_boot=50, seed=1)
    assert 0.8 < base["Edema"]["auroc_lenient"] < 1.0 and base["Edema"]["n_present"] == 60
    table = quality.table(
        base, {"Edema": "C0034063"},
        {"C9": {"auroc_lenient": 0.66, "ci95": [0.63, 0.69], "n_present": 80}}, gate=0.65)
    assert table["C0034063"]["weak"] is False and table["C0034063"]["column"] == "Edema"
    assert (table["C9"]["weak"], table["C9"]["borderline"]) == (False, True)
