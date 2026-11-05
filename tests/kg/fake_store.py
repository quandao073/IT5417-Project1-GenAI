"""An in-memory stand-in for `Neo4jStore` over the tiny KG (tests/kg/tiny_kg.py): the same
methods and the same rows, so the lookup and expansion logic is tested without a database. A
contract test (tests/integration/test_kg_store.py) holds it equal to the real store."""

from __future__ import annotations

from tests.kg import tiny_kg


class FakeStore:
    def __init__(self):
        nodes = tiny_kg.nodes().to_dicts()
        self.q = tiny_kg.quality()
        self.nodes = {n["cui"]: n for n in nodes}
        self.edges = tiny_kg.edges().to_dicts()
        self.aliases = tiny_kg.aliases().to_dicts()

    def build(self):
        return {"kg_build_id": "kg_test", "matrix_version": "matrix_test@0"}

    def aliases_exact(self, norm):
        return [{"cui": a["cui"], "text": a["text"], "lang": a["lang"]}
                for a in self.aliases if a["norm"] == norm]

    def aliases_fulltext(self, text, limit):
        words = set(text.split())
        rows = []
        for a in self.aliases:
            shared = len(words & set(a["norm"].split()))
            if shared:
                rows.append({"cui": a["cui"], "text": a["text"], "lang": a["lang"],
                             "score": float(shared)})
        return sorted(rows, key=lambda r: -r["score"])[:limit]

    def concepts(self, cuis):
        out = {}
        for cui in cuis:
            n = self.nodes.get(cui)
            if n is None:
                continue
            q = self.q.get(cui, {})
            parents = [self.nodes[e["dst"]]["name"] for e in self.edges
                       if e["rel"] == "IS_A" and e["src"] == cui and e["dst"] in self.nodes]
            out[cui] = {"cui": cui, "name": n["name"], "label": n["labels"],
                        "scorable": n["scorable"], "column": q.get("column"),
                        "quality": q.get("auroc_lenient"), "weak": q.get("weak"),
                        "borderline": q.get("borderline"), "n_pool": n["n_pool"],
                        "parents": parents[:3]}
        return out

    def _below(self, cui, depth, rel="IS_A"):
        """(cui, hops, names from the start down) of every concept up to `depth` steps below."""
        frontier, out = [(cui, 0, [self.nodes[cui]["name"]])], []
        while frontier:
            c, h, via = frontier.pop(0)
            out.append((c, h, via))
            if h < depth:
                for e in self.edges:
                    if e["rel"] == rel and e["dst"] == c and e["src"] in self.nodes:
                        frontier.append((e["src"], h + 1, [*via, self.nodes[e["src"]]["name"]]))
        return out

    def scorable_below(self, cui, depth):
        rows = [{"cui": c, "hops": h, "via": via} for c, h, via in self._below(cui, depth)
                if self.nodes[c]["scorable"]]
        return sorted(rows, key=lambda r: (r["hops"], r["cui"]))

    def scorable_above(self, cui, depth):
        frontier, rows = [(cui, 0, [self.nodes[cui]["name"]])], []
        while frontier:
            c, h, via = frontier.pop(0)
            if h and self.nodes[c]["scorable"]:
                rows.append({"cui": c, "hops": h, "via": via})
            if h < depth:
                for e in self.edges:
                    if e["rel"] == "IS_A" and e["src"] == c and e["dst"] in self.nodes:
                        frontier.append((e["dst"], h + 1, [*via, self.nodes[e["dst"]]["name"]]))
        return sorted(rows, key=lambda r: (r["hops"], r["cui"]))[:1]

    def suggesting(self, cui, depth, min_n):
        best = {}
        for t, _, _ in self._below(cui, depth):
            for e in self.edges:
                kept = (e["rel"] == "SUGGESTS" and e["dst"] == t
                        and (e["source"] == "manual" or e["n"] >= min_n)
                        and (e["review"] or "keep") != "drop"
                        and self.nodes[e["src"]]["scorable"])
                if kept and (e["src"] not in best or best[e["src"]]["p"] < e["p"]):
                    best[e["src"]] = {"cui": e["src"], "p": e["p"], "n": e["n"],
                                      "target": self.nodes[t]["name"]}
        return sorted(best.values(), key=lambda r: (-r["p"], r["cui"]))

    def anatomy_findings(self, cui, depth, min_n, top):
        best = {}
        for r, _, _ in self._below(cui, depth, rel="PART_OF"):
            if self.nodes[r]["labels"] != "Anatomy":
                continue
            for e in self.edges:
                kept = (e["rel"] == "TYPICALLY_AT" and e["dst"] == r and e["n"] >= min_n
                        and self.nodes[e["src"]]["scorable"])
                if kept and (e["src"] not in best or best[e["src"]]["p"] < e["p"]):
                    best[e["src"]] = {"cui": e["src"], "p": e["p"], "n": e["n"],
                                      "region": self.nodes[r]["name"]}
        return sorted(best.values(), key=lambda r: (-r["p"], r["cui"]))[:top]
