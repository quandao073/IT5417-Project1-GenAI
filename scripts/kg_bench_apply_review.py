"""Apply the chat's gold reviews (data/eval/heldout_gold_review.json, then
data/eval/heldout_new_gold_review.json) to the KG benchmark.

    uv run python scripts/kg_bench_apply_review.py [--edits]

general_rule `weak_not_expansion_target`, to every question of both splits (D55): a weak column
stays in `expected_scorable` (or `exclude`) only when a PRESENT (ABSENT) term asks for it
directly or it is that term's nearest scorable ancestor and the narrower rule is the one that
applies (no strong scorable concept below the term, else it is scored wider); weak flags follow
the current data/kg/graph/quality.json; `weak_quality` leaves `expect_warnings` when no weak
column is left in the question's gold. `rule_fix` of the second review: in a question whose gold
expects `narrower`, a weak column is that term's ancestor by the gold's own reading and stays,
whatever route the graph takes; one the rule removed earlier comes back from the authored gold
(data/eval/kg_queries_v1.authored_gold.jsonl) with `weak_quality`. With --edits, the per-question
edits of both reviews are then applied. The file is rewritten in place; running it twice changes
nothing more.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from src.common.config import Settings, load_config
from src.evaluation import kg_bench as kb
from src.kg import expand, runtime

FILE = Path("benchmarks/kg_queries_v1.jsonl")
REVIEW = Path("data/eval/heldout_gold_review.json")
REVIEW_NEW = Path("data/eval/heldout_new_gold_review.json")
AUTHORED = Path("data/eval/kg_queries_v1.authored_gold.jsonl")
# the review names the warning by meaning; this is the system's code for "cannot be scored"
WARNING_ALIASES = {"<mã cảnh báo 'không chấm được' của hệ thống>": "no_path"}


def allowed_weak(row: dict, polarity: str, weak: set[str], scorable: set[str], above,
                 below) -> set[str]:
    """Weak columns a term of `polarity` may bring: its own, or its nearest scorable ancestor
    when the narrower rule is the one that applies (no strong scorable concept below it)."""
    out = set()
    for t in row["terms"]:
        if not t["cui"] or t["polarity"] != polarity:
            continue
        if t["cui"] in scorable:
            out.add(t["cui"])
        elif not below(t["cui"]) - weak:
            out |= above(t["cui"])
    return out & weak


def apply_rule(row: dict, weak: set[str], scorable: set[str], above, below) -> list[str]:
    """Changes made to `row`, described."""
    changes = []
    keep = allowed_weak(row, "PRESENT", weak, scorable, above, below)
    narrower = "narrower" in row["expect_warnings"]     # rule_fix: the gold's reading decides
    if narrower:
        keep = set(weak)
    gold = []
    for c in row["expected_scorable"]:
        if c["cui"] in weak and c["cui"] not in keep:
            changes.append(f"expected_scorable -{c['cui']}")
            continue
        flag = c["cui"] in weak
        if c["weak"] != flag:
            changes.append(f"weak {c['cui']} {c['weak']}->{flag}")
        gold.append({"cui": c["cui"], "weak": flag})
    row["expected_scorable"] = gold
    keep_ex = set(weak) if narrower else allowed_weak(row, "ABSENT", weak, scorable, above,
                                                      below)
    excl = [c for c in row["exclude"] if c not in weak or c in keep_ex]
    changes += [f"exclude -{c}" for c in row["exclude"] if c not in excl]
    row["exclude"] = excl
    left = {c["cui"] for c in gold} | set(excl)
    if "weak_quality" in row["expect_warnings"] and not left & weak:
        row["expect_warnings"] = [w for w in row["expect_warnings"] if w != "weak_quality"]
        changes.append("expect_warnings -weak_quality")
    return changes


def restore(row: dict, authored: dict, weak: set[str]) -> list[str]:
    """rule_fix: weak columns of the authored gold that a `narrower` question lost."""
    if "narrower" not in row["expect_warnings"]:
        return []
    changes = []
    have = {c["cui"] for c in row["expected_scorable"]}
    for c in authored["expected_scorable"]:
        if c["cui"] in weak and c["cui"] not in have:
            row["expected_scorable"].append({"cui": c["cui"], "weak": True})
            changes.append(f"expected_scorable +{c['cui']}")
    for c in authored["exclude"]:
        if c in weak and c not in row["exclude"]:
            row["exclude"].append(c)
            changes.append(f"exclude +{c}")
    if changes and "weak_quality" not in row["expect_warnings"]:
        row["expect_warnings"] = sorted([*row["expect_warnings"], "weak_quality"])
        changes.append("expect_warnings +weak_quality")
    return changes


def apply_chat_edit(row: dict, edit: dict, weak: set[str]) -> list[str]:
    """An edit of the second review: set_terms, set_expected_scorable, set_expect_warnings,
    add_expected_scorable, add_expect_warnings (weak flags from the current quality)."""
    changes = []

    def warnings(codes):
        return sorted({WARNING_ALIASES.get(c, c) for c in codes})

    if "set_terms" in edit and row["terms"] != edit["set_terms"]:
        row["terms"] = edit["set_terms"]
        changes.append("set terms")
    if "set_expected_scorable" in edit:
        gold = [{"cui": c["cui"], "weak": c["cui"] in weak} for c in edit["set_expected_scorable"]]
        if row["expected_scorable"] != gold:
            row["expected_scorable"] = gold
            changes.append("set expected_scorable")
    for c in edit.get("add_expected_scorable", []):
        if c["cui"] not in {x["cui"] for x in row["expected_scorable"]}:
            row["expected_scorable"].append({"cui": c["cui"], "weak": c["cui"] in weak})
            changes.append(f"expected_scorable +{c['cui']}")
    if "set_expect_warnings" in edit and row["expect_warnings"] != warnings(
            edit["set_expect_warnings"]):
        row["expect_warnings"] = warnings(edit["set_expect_warnings"])
        changes.append("set expect_warnings")
    for c in warnings(edit.get("add_expect_warnings", [])):
        if c not in row["expect_warnings"]:
            row["expect_warnings"] = sorted([*row["expect_warnings"], c])
            changes.append(f"expect_warnings +{c}")
    return changes


def apply_edit(row: dict, edit: dict, weak: set[str]) -> list[str]:
    changes = []
    for key, value in edit.get("set", {}).items():
        if row.get(key) != value:
            row[key] = value
            changes.append(f"set {key}")
    for c in edit.get("add_expected_scorable", []):
        if c["cui"] not in {x["cui"] for x in row["expected_scorable"]}:
            row["expected_scorable"].append({"cui": c["cui"], "weak": c["cui"] in weak})
            changes.append(f"expected_scorable +{c['cui']}")
    return changes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edits", action="store_true", help="also apply heldout_edits")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    settings, kcfg = Settings(), load_config("kg")
    graph_dir = Path(settings.kg_graph_dir)
    quality = json.loads((graph_dir / "quality.json").read_text(encoding="utf-8"))
    weak = {c for c, q in quality.items() if q["weak"]}
    expected = json.loads((graph_dir / "neo4j_load.json").read_text(encoding="utf-8"))
    store, why = runtime._neo4j_store(settings, kcfg["neo4j"]["query_timeout_seconds"], expected)
    if store is None:
        print(f"STOP: Neo4j: {why}", file=sys.stderr)
        return 1
    ecfg = kcfg["query"]["expand"]
    ups: dict[str, set[str]] = {}
    downs: dict[str, set[str]] = {}

    def above(cui: str) -> set[str]:
        if cui not in ups:
            ups[cui] = {r["cui"] for r in store.scorable_above(cui, ecfg["up_depth"])}
        return ups[cui]

    def below(cui: str) -> set[str]:
        if cui not in downs:
            downs[cui] = {r["cui"] for r in store.scorable_below(cui, ecfg["down_depth"])}
        return downs[cui]

    rows = kb.read_jsonl(FILE)
    authored = {r["id"]: r for r in kb.read_jsonl(AUTHORED)}
    by_split = {"dev": 0, "heldout": 0}
    for row in rows:
        changes = apply_rule(row, weak, set(quality), above, below)
        if changes:
            by_split[row["split"]] += 1
            print(f"{row['split']:7} {row['id']}: {'; '.join(changes)}")
    for row in rows:
        changes = restore(row, authored[row["id"]], weak)
        if changes:
            print(f"restore {row['split']:7} {row['id']}: {'; '.join(changes)}")
    if args.edits:
        edits = {e["id"]: e for e in review["heldout_edits"]}
        for row in rows:
            if row["id"] in edits:
                changes = apply_edit(row, edits[row["id"]], weak)
                print(f"edit    {row['id']}: {'; '.join(changes) or 'already applied'}")
        second = json.loads(REVIEW_NEW.read_text(encoding="utf-8"))
        chat = {e["id"]: e for e in second["edits"]}
        for e in list(chat.values()):
            if "same_as" in e:
                chat[e["id"]] = {**chat[e["same_as"]], "id": e["id"]}
        resolved = set(chat) | {k for k in second["sonnet_disagree_resolved"] if k != "other"}
        for row in rows:
            changes = apply_chat_edit(row, chat[row["id"]], weak) if row["id"] in chat else []
            if row["id"] in resolved and row["gold_disputed"]:
                row["gold_disputed"] = False
                changes.append("gold_disputed -> false")
            if row["id"] in chat or changes:
                print(f"edit2   {row['id']}: {'; '.join(changes) or 'already applied'}")
    # the automatic check is recorded against the gold as it now stands, on the current graph
    names = set(json.loads((graph_dir / "quality.json").read_text(encoding="utf-8")))
    names |= set(pl.read_parquet(graph_dir / "nodes.parquet", columns=["cui"])["cui"])
    for row in rows:
        row["auto_check"] = kb.expand_gold(store, row, ecfg, expand)
        row["auto_check"]["unknown_cuis"] = sorted(
            ({t["cui"] for t in row["terms"] if t["cui"]}
             | {c["cui"] for c in row["expected_scorable"]} | set(row["exclude"])) - names)
        row["auto_check"]["weak_flags"] = kb.weak_problems(row, quality)
    print(f"auto_check: graph differs from gold in "
          f"{sum(not r['auto_check']['columns_match'] for r in rows)} of {len(rows)}")
    sha = kb.write_jsonl(FILE, rows)
    print(f"rule changed {by_split}; weak {sorted(weak)}; file sha256 {sha}; heldout lines "
          f"{kb.split_sha256(rows, 'heldout')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
