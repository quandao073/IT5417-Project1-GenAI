"""Commands of the KG evaluation (V2-7, V2-8 on dev): `kg-bench-build` writes the benchmark and
its gold; `eval-kg` runs the systems, pools, judges and scores (docs/kg-evaluation.md)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from src.common.config import Settings, default_config_dir, load_config
from src.evaluation import kg_bench as kb
from src.evaluation import kg_llm
from src.kg import llm_select


def _ecfg() -> dict:
    return load_config("eval_kg")


def _ledger(ecfg: dict) -> kg_llm.Ledger:
    work = Path(ecfg["benchmark"]["work_dir"])
    return kg_llm.Ledger(work / "cost_ledger.jsonl", ecfg["pricing"], ecfg["budget_usd"])


def _prompt(ecfg: dict, key: str) -> tuple[str, str]:
    return llm_select.load_prompt(default_config_dir() / ecfg["llm"][key])


def _do_not_reuse() -> list[str]:
    parser_q = [json.loads(line)["query"] for line in
                Path("benchmarks/parser_queries_v1.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    smoke = [ln.strip() for ln in Path("docs/v2-6-smoke.txt").read_text(encoding="utf-8")
             .splitlines() if ln.strip() and not ln.startswith("#")]
    return parser_q + smoke


def _manual_cuis(gcfg: dict) -> tuple[list[str], list[str]]:
    """(every end of a hand-written edge, the diseases those edges route)."""
    manual = gcfg.get("manual_edges", {})
    ends = {c for pair in manual.get("is_a", []) for c in pair}
    routed = {child for child, _ in manual.get("is_a", [])} | set(manual.get("suggests", {}))
    for disease, findings in manual.get("suggests", {}).items():
        ends |= {disease, *findings}
    return sorted(ends), sorted(routed)


def _catalog_text(rows: list[dict]) -> str:
    return "\n".join(f"{r['cui']}|{r['kind']}|{'column' if r['scorable'] else '-'}|{r['name']}"
                     for r in rows)


def cmd_kg_bench_build(args: argparse.Namespace) -> int:
    """Author the questions and gold group by group, check them, have them reviewed, split."""
    from src.kg import expand, runtime

    sys.stdout.reconfigure(encoding="utf-8")
    ecfg, kcfg = _ecfg(), load_config("kg")
    bcfg = ecfg["benchmark"]
    work = Path(bcfg["work_dir"])
    key = llm_select.api_key("ANTHROPIC_API_KEY")
    if not key:
        print("STOP: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 1
    graph_dir = Path(Settings().kg_graph_dir)
    nodes = pl.read_parquet(graph_dir / "nodes.parquet")
    quality = json.loads((graph_dir / "quality.json").read_text(encoding="utf-8"))
    ends, routed = _manual_cuis(kcfg["graph"])
    columns, catalog = kb.catalog(nodes, quality, ends)
    known = {r["cui"] for r in catalog} | {c["cui"] for c in columns}
    reuse = _do_not_reuse()
    used = {kb.normalize(q) for q in reuse}
    ledger = _ledger(ecfg)
    author = kg_llm.CachedCaller("author", ecfg["llm"]["author"], key,
                                 work / "llm_cache_author.jsonl", ledger)
    reviewer = kg_llm.CachedCaller("reviewer", ecfg["llm"]["reviewer"], key,
                                   work / "llm_cache_reviewer.jsonl", ledger)
    a_text, a_sha = _prompt(ecfg, "author_prompt")
    r_text, r_sha = _prompt(ecfg, "review_prompt")
    context = {"score_columns": columns, "hand_routed_diseases": routed,
               "warning_codes": kb.WARNING_CODES, "do_not_reuse": reuse}
    catalog_text = _catalog_text(catalog)
    catalog_sha = kg_llm.sha256_text(catalog_text)

    rows, log = [], {}
    groups = list(bcfg["groups"].items())
    if args.groups:
        groups = [g for g in groups if g[0] in args.groups]
    for group, description in groups:
        n_en, n_vi = kb.lang_counts(list(bcfg["groups"]).index(group), bcfg["per_group"])
        problems: list[str] = []
        questions: list[dict] = []
        for attempt in range(args.attempts):
            payload = {"group": group, "description": description, "n_en": n_en, "n_vi": n_vi,
                       "attempt": attempt, "previous_problems": problems,
                       "catalog_sha256": catalog_sha}
            user = json.dumps({**payload, **context}, ensure_ascii=False) + (
                "\n\ncatalog (CUI|kind|column?|name):\n" + catalog_text)
            answer = author.call("author", a_text, a_sha, payload, user, kb.AUTHOR_SCHEMA)
            questions = answer["questions"]
            problems = kb.structure_problems(questions, n_en, n_vi, known, used)
            print(f"{group} attempt {attempt}: {len(questions)} questions, problems {problems}; "
                  f"spent {ledger.total:.2f} USD")
            if not problems:
                break
        log[group] = {"attempts": attempt + 1, "problems": problems}
        used |= {kb.normalize(q["question"]) for q in questions}
        rows += kb.assign_ids(group, questions)

    # (a) automatic check: every CUI is in the graph; the graph's expansion against the gold
    settings = Settings()
    expected = json.loads((graph_dir / "neo4j_load.json").read_text(encoding="utf-8"))
    store, why = runtime._neo4j_store(settings, kcfg["neo4j"]["query_timeout_seconds"],
                                      expected)
    if store is None:
        print(f"STOP: Neo4j: {why}", file=sys.stderr)
        return 1
    names = dict(nodes.select("cui", "name").iter_rows())
    for r in rows:
        r["auto_check"] = kb.expand_gold(store, r, kcfg["query"]["expand"], expand)
        r["auto_check"]["unknown_cuis"] = sorted(
            ({t["cui"] for t in r["terms"] if t["cui"]} | {c["cui"] for c in r["expected_scorable"]}
             | set(r["exclude"])) - set(names))
        r["auto_check"]["weak_flags"] = kb.weak_problems(r, quality)

    # (b) independent review, group by group
    for group in dict.fromkeys(r["group"] for r in rows):
        members = [r for r in rows if r["group"] == group]
        shown = [{"index": i, "question": r["question"], "query_en": r["query_en"],
                  "terms": [{**t, "name": names.get(t["cui"], "")} for t in r["terms"]],
                  "expected_scorable": [{**c, "name": names.get(c["cui"], c["cui"])}
                                        for c in r["expected_scorable"]],
                  "exclude": [{"cui": c, "name": names.get(c, c)} for c in r["exclude"]],
                  "views": r["views"], "expect_warnings": r["expect_warnings"]}
                 for i, r in enumerate(members)]
        payload = {"group": group, "questions": shown}
        user = json.dumps({**payload, "score_columns": columns}, ensure_ascii=False)
        answer = reviewer.call("review", r_text, r_sha, payload, user, kb.REVIEW_SCHEMA)
        verdicts = {v["index"]: v for v in answer["reviews"]}
        for i, r in enumerate(members):
            v = verdicts.get(i, {"verdict": "disagree", "reason": "not reviewed"})
            r["review"] = {"model": reviewer.model, "verdict": v["verdict"],
                           "reason": v["reason"]}
            r["gold_disputed"] = v["verdict"] != "agree"
        print(f"review {group}: {sum(r['gold_disputed'] for r in members)} disputed; "
              f"spent {ledger.total:.2f} USD")

    assignment = kb.split(rows, bcfg["dev_per_group"], bcfg["seed"])
    for r in rows:
        r["split"] = assignment[r["id"]]
        r["gold_by"] = {"author": author.model, "author_prompt_sha256": a_sha,
                        "reviewer_prompt_sha256": r_sha}
    # The author's note and the reviewer's reason quote concept names from the catalog (UMLS
    # content): they stay under data/, the committed file keeps codes and question text only.
    kb.write_jsonl(work / "bench_notes.jsonl", [
        {"id": r["id"], "note": r.pop("note"), "review_reason": r["review"].pop("reason")}
        for r in rows])
    out = Path(args.out or bcfg["file"])
    file_sha = kb.write_jsonl(out, rows)
    summary = {"file": str(out), "sha256": file_sha, "questions": len(rows),
               "dev": sum(r["split"] == "dev" for r in rows),
               "heldout_sha256": kb.split_sha256(rows, "heldout"),
               "dev_sha256": kb.split_sha256(rows, "dev"),
               "disputed": sum(r["gold_disputed"] for r in rows),
               "auto_mismatch": sum(not r["auto_check"]["columns_match"] for r in rows),
               "authoring": log, "spent_usd": round(ledger.total, 4)}
    (work / "bench_build.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def register(sub) -> None:
    b = sub.add_parser("kg-bench-build", help="V2-7: write the KG benchmark and its gold")
    b.add_argument("--out", default=None)
    b.add_argument("--groups", nargs="+", default=None, help="only these groups (a trial)")
    b.add_argument("--attempts", type=int, default=2, help="author retries on rule breaks")
    b.set_defaults(func=cmd_kg_bench_build)
