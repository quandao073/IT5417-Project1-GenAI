"""`Neo4jStore` against a real Neo4j holds the same rows as the in-memory FakeStore that the
unit tests of lookup and expansion use (tests/integration/conftest.py loads the tiny KG)."""

from __future__ import annotations

import pytest

from src.common.config import load_config
from src.kg import expand, lookup
from src.kg.store import Neo4jStore, StoreError
from tests.kg.fake_store import FakeStore

pytestmark = pytest.mark.integration

QCFG = load_config("kg")["query"]
EXPAND_CFG = {**QCFG["expand"], "weak_or_children": ["C10"]}


@pytest.fixture
def real(kg_neo4j):
    return Neo4jStore(kg_neo4j[0], timeout_s=5)


def test_every_template_returns_what_the_fake_store_returns(real):
    fake = FakeStore()
    assert real.build()["kg_build_id"] == "kg_test"
    assert real.aliases_exact("giảm thể tích phổi") and sorted(
        r["cui"] for r in real.aliases_exact("giảm thể tích phổi")) == ["C12", "C13"]
    cuis = list(tiny_cuis())
    assert real.concepts(cuis) == fake.concepts(cuis)
    for cui in ("C7", "C5", "C1", "C10", "C6"):
        assert real.scorable_below(cui, 4) == fake.scorable_below(cui, 4), cui
        assert real.scorable_above(cui, 6) == fake.scorable_above(cui, 6), cui
    assert real.suggesting("C6", 2, 20) == fake.suggesting("C6", 2, 20)
    assert real.anatomy_findings("C8", 5, 20, 5) == fake.anatomy_findings("C8", 5, 20, 5)


def tiny_cuis():
    from tests.kg import tiny_kg

    return tiny_kg.NAMES


def test_the_fulltext_index_finds_an_alias_by_one_of_its_words(real):
    rows = real.aliases_fulltext("pleural", 10)
    assert {r["cui"] for r in rows} >= {"C1", "C7", "C9"}
    assert real.aliases_fulltext("???", 10) == []         # nothing left after the escape


def test_lookup_and_expansion_give_the_same_intent_on_both_stores(real):
    fake = FakeStore()
    for term, cui in (("suy tim", "C6"), ("pleural disease", "C7"), ("lobar pneumonia", "C5")):
        a = lookup.lookup_concept(real, term, QCFG)
        b = lookup.lookup_concept(fake, term, QCFG)
        assert [c.cui for c in a.candidates] == [c.cui for c in b.candidates] and a.candidates
        assert a.candidates[0].cui == cui
        for polarity in ("PRESENT", "ABSENT"):
            assert (expand.expand_concepts(real, cui, polarity, EXPAND_CFG, term).to_dict()
                    == expand.expand_concepts(fake, cui, polarity, EXPAND_CFG, term).to_dict())


def test_a_dead_connection_is_a_store_error_not_a_hang():
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver("bolt://127.0.0.1:1", auth=("neo4j", "x"),
                                  connection_timeout=1, max_transaction_retry_time=0)
    try:
        with pytest.raises(StoreError, match="connection"):
            Neo4jStore(driver, timeout_s=1).build()
    finally:
        driver.close()
