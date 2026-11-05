"""Read access to the loaded Neo4j graph (docs/v2-kg-design.md, "Template Cypher"; WORKLOG D65).

Every query is a template in this file with parameters; the language model never writes Cypher.
Each runs in a READ transaction with a timeout, and the variable-length bounds, which Cypher
cannot take as parameters, are integers passed through `int()`. A failure raises `StoreError`,
which the pipeline turns into the 14-label route (N8).
"""

from __future__ import annotations

import re

_LUCENE_SPECIAL = re.compile(r"[^\w\s]", re.UNICODE)


class StoreError(RuntimeError):
    """Neo4j is unreachable, timed out or refused a query."""


class Neo4jStore:
    def __init__(self, driver, timeout_s: float):
        self._driver = driver
        self._timeout = timeout_s

    def _read(self, query: str, **params) -> list[dict]:
        from neo4j import Query
        from neo4j.exceptions import Neo4jError

        try:
            with self._driver.session(default_access_mode="READ") as session:
                return [r.data() for r in session.run(Query(query, timeout=self._timeout),
                                                      **params)]
        except Exception as exc:  # noqa: BLE001 - every failure is the same fallback
            kind = "query" if isinstance(exc, Neo4jError) else "connection"
            raise StoreError(f"neo4j {kind}: {type(exc).__name__}: {str(exc)[:120]}") from exc

    def build(self) -> dict | None:
        rows = self._read("MATCH (b:KGBuild) RETURN properties(b) AS b")
        return rows[0]["b"] if rows else None

    def aliases_exact(self, norm: str) -> list[dict]:
        return self._read("MATCH (a:Alias {norm: $norm})<-[:HAS_ALIAS]-(c:Concept) "
                          "RETURN c.cui AS cui, a.text AS text, a.lang AS lang", norm=norm)

    def aliases_fulltext(self, text: str, limit: int) -> list[dict]:
        query = " ".join(_LUCENE_SPECIAL.sub(" ", text).split())
        if not query:
            return []
        return self._read(
            "CALL db.index.fulltext.queryNodes('alias_text', $q) YIELD node, score "
            "MATCH (c:Concept)-[:HAS_ALIAS]->(node) "
            "RETURN c.cui AS cui, node.text AS text, node.lang AS lang, score "
            "ORDER BY score DESC LIMIT $limit", q=query, limit=int(limit))

    def concepts(self, cuis: list[str]) -> dict[str, dict]:
        rows = self._read(
            "MATCH (c:Concept) WHERE c.cui IN $cuis OPTIONAL MATCH (c)-[:IS_A]->(p:Concept) "
            "RETURN c.cui AS cui, c.name AS name, "
            "[l IN labels(c) WHERE l <> 'Concept'][0] AS label, c.scorable AS scorable, "
            " c.`column` AS `column`, c.quality AS quality, "
            "c.weak AS weak, c.borderline AS borderline, c.n_pool AS n_pool, "
            "collect(DISTINCT p.name)[0..3] AS parents", cuis=list(cuis))
        return {r["cui"]: r for r in rows}

    def scorable_below(self, cui: str, depth: int) -> list[dict]:
        """The scorable concepts at or below `cui` along IS_A, nearest first."""
        return self._read(
            f"MATCH path = (c:Concept {{cui: $cui}})<-[:IS_A*0..{int(depth)}]-"
            "(x:Concept {scorable: true}) RETURN x.cui AS cui, length(path) AS hops, "
            "[n IN nodes(path) | n.name] AS via ORDER BY hops, cui", cui=cui)

    def scorable_above(self, cui: str, depth: int) -> list[dict]:
        """The nearest scorable ancestor along IS_A (a list of at most one row)."""
        return self._read(
            f"MATCH path = (c:Concept {{cui: $cui}})-[:IS_A*1..{int(depth)}]->"
            "(a:Concept {scorable: true}) RETURN a.cui AS cui, length(path) AS hops, "
            "[n IN nodes(path) | n.name] AS via ORDER BY hops, cui LIMIT 1", cui=cui)

    def suggesting(self, cui: str, depth: int, min_n: int) -> list[dict]:
        """Scorable findings with a SUGGESTS edge, kept on review, into `cui` or a concept up to
        `depth` IS_A steps below it; the strongest edge of each finding."""
        return self._read(
            f"MATCH (d:Concept {{cui: $cui}})<-[:IS_A*0..{int(depth)}]-(t:Concept)"
            "<-[s:SUGGESTS]-(f:Concept {scorable: true}) "
            "WHERE s.n >= $min_n AND coalesce(s.review, 'keep') <> 'drop' "
            "WITH f, s, t ORDER BY s.p DESC "
            "RETURN f.cui AS cui, collect(s.p)[0] AS p, collect(s.n)[0] AS n, "
            "collect(t.name)[0] AS target ORDER BY p DESC, cui", cui=cui, min_n=int(min_n))

    def anatomy_findings(self, cui: str, depth: int, min_n: int, top: int) -> list[dict]:
        """Scorable findings usually seen in the region `cui` or its parts (PART_OF), by the
        strongest TYPICALLY_AT of each, with the region of that edge."""
        return self._read(
            f"MATCH (a:Concept {{cui: $cui}})<-[:PART_OF*0..{int(depth)}]-(r:Anatomy)"
            "<-[t:TYPICALLY_AT]-(f:Concept {scorable: true}) WHERE t.n >= $min_n "
            "WITH f, t, r ORDER BY t.p DESC "
            "RETURN f.cui AS cui, collect(t.p)[0] AS p, collect(t.n)[0] AS n, "
            "collect(r.name)[0] AS region ORDER BY p DESC, cui LIMIT $top",
            cui=cui, min_n=int(min_n), top=int(top))
