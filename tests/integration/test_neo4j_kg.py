"""The loader and the four KG checks against a real Neo4j (tests/integration/conftest.py)."""

from __future__ import annotations

import pytest

from src.kg import neo4j_load as nl
from tests.kg import tiny_kg

pytestmark = pytest.mark.integration


def test_the_loader_writes_every_table_and_exactly_top_n_predicted_per_scorable(kg_neo4j):
    driver, manifest = kg_neo4j
    scorable = len(tiny_kg.COLUMNS)
    assert manifest["nodes"]["Concept"] == len(tiny_kg.NAMES)
    assert manifest["nodes"]["Alias"] == tiny_kg.aliases().height
    assert manifest["edges"]["PREDICTED"] == scorable * 3          # top_n = 3
    assert manifest["edges"]["SUGGESTS"] == 4 and manifest["edges"]["TYPICALLY_AT"] == 2
    assert manifest["edges"]["HAS_ALIAS"] == tiny_kg.aliases().height
    with driver.session() as session:
        row = session.run("MATCH (c:Concept {cui: 'C4'}) RETURN c.weak AS weak, "
                          "c.quality AS q, labels(c) AS labels, c.`column` AS col").single()
        assert (row["weak"], row["q"], row["col"]) == (True, 0.63, "Pneumonia")
        assert set(row["labels"]) == {"Concept", "Disease"}
        # the best three images of effusion are the even ones the fixture boosted
        best = [r["id"] for r in session.run(
            "MATCH (i:Image)-[r:PREDICTED]->(:Concept {cui: 'C1'}) "
            "RETURN i.image_id AS id ORDER BY r.rank")]
        assert best[0] in {"i0", "i2", "i4", "i6"} and len(best) == 3
        assert nl.read_build(session)["kg_build_id"] == "kg_test"


def test_the_four_checks_pass_on_a_consistent_load_and_fail_on_a_wrong_build(kg_neo4j):
    driver, manifest = kg_neo4j
    label_ids = {label: cui for cui, label in tiny_kg.LABEL_OF.items()}
    columns = list(tiny_kg.COLUMNS.values())
    with driver.session() as session:
        checks = nl.validate(session, manifest, columns, label_ids, "b1", "b1")
        assert [c.name for c in checks] == ["kg_columns", "kg_labels", "kg_matrix_build",
                                            "kg_predicted"]
        assert all(c.ok for c in checks), [c.detail for c in checks]
        bad = nl.validate(session, {**manifest, "top_n": 4}, columns[:-1], label_ids, "b1", "b2")
        assert [c.ok for c in bad] == [False, True, False, False]


def test_a_write_in_a_read_transaction_is_refused_by_the_server(kg_neo4j):
    driver, _ = kg_neo4j
    from neo4j.exceptions import ClientError

    with driver.session(default_access_mode="READ") as session:
        assert session.run("MATCH (c:Concept) RETURN count(c) AS n").single()["n"] > 0
        with pytest.raises(ClientError, match="read access mode"):
            session.run("CREATE (:Probe)").consume()
