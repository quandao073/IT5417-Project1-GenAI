"""Read-only: the core Disease concepts with pool mentions that `expand_concepts` cannot route to
any score column, most mentioned first. Those listed in kg.yaml `graph.round_two_columns` are put
apart, as candidates for the next round of extra columns; the rest wait for adjudication (WORKLOG
D68). Needs the loaded Neo4j. Writes data/kg/graph/diseases_without_route.json (names are UMLS
content, so the file stays under data/; the screen shows codes and counts only).

    uv run python scripts/kg_disease_routes.py
"""
import json
from collections import Counter
from pathlib import Path

import polars as pl
from neo4j import GraphDatabase

from src.common.config import Settings, load_config
from src.kg import expand
from src.kg.store import Neo4jStore

settings = Settings()
driver = GraphDatabase.driver(settings.neo4j_uri, auth=(
    settings.neo4j_user, settings.neo4j_password.get_secret_value()))
store = Neo4jStore(driver, 20)
cfg = load_config("kg")
round_two = set(cfg["graph"].get("round_two_columns", []))
nodes = pl.read_parquet("data/kg/graph/nodes.parquet")
diseases = (nodes.filter(pl.col("labels").str.contains("Disease"), pl.col("in_core"),
                         pl.col("n_pool") > 0).sort("n_pool", descending=True))
kinds, groups = Counter(), {"round_two_columns": [], "pending": []}
for row in diseases.iter_rows(named=True):
    intent = expand.expand_concepts(store, row["cui"], "PRESENT", cfg["query"]["expand"],
                                    row["name"])
    kinds[intent.kind] += 1
    if intent.kind == "none":
        group = "round_two_columns" if row["cui"] in round_two else "pending"
        groups[group].append({"cui": row["cui"], "name": row["name"], "n_pool": row["n_pool"]})
driver.close()
out = {"note": {"round_two_columns": "cannot be scored; candidate columns of the next round",
                "pending": "no route yet; waits for adjudication"},
       "routes": dict(kinds), **groups}
Path("data/kg/graph/diseases_without_route.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{diseases.height} core diseases with mentions; routes {dict(kinds)}")
for name, rows in groups.items():
    print(f"{name}: {len(rows)}; first ten {[(r['cui'], r['n_pool']) for r in rows[:10]]}")
