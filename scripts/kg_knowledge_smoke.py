"""The smoke questions through the KG search, each page explained twice from the same search
result: prompt v4 with the knowledge items (explain.knowledge true) and prompt v3 without them
(the layer-4 ablation). The order of the two alternates per question. Writes one JSON line per
(question, condition) and a summary: grounded explanations, dropped sentences per part, valid
[K#] citations, first-token time (N2) p50 / p95, and whether the rebuilt path edges walk the
concepts of intent.path. Real tables, matrix, Neo4j (read only) and the language model.

The chooser cache is a copy under data/kg/knowledge/cache/, so the shared data/kg/cache/ is not
written:

    uv run python scripts/kg_knowledge_smoke.py --out data/kg/knowledge/smoke/run1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.common.config import Settings, default_config_dir, load_config
from src.kg import runtime
from src.kg.llm_query import _SENTENCE, Explainer, build_facts

HAIKU = load_config("models")["llm"]["prices"]["claude-haiku-5-5"]


def _patched_models(cache: Path):
    """load_config with the chooser cache moved to `cache`."""
    original = runtime.load_config

    def load(name: str):
        cfg = original(name)
        if name == "models":
            cfg["kg_llm"]["select"]["cache_file"] = str(cache)
        return cfg
    return load


def _sentences(text: str) -> int:
    return len([s for s in _SENTENCE.split(text.strip()) if s])


def _path_check(base, intent) -> bool:
    """True when every concept of the rebuilt edges is named in intent.path (or is the intent's
    concept) and every column other than the concept itself has an edge."""
    edges = base.path_edges(intent)
    text = " | ".join(intent.path) + " | " + intent.name
    if intent.kind == "disease":
        edges = [e for e in edges if e[0] == "suggests"]
    elif intent.kind == "anatomy":
        edges = [e for e in edges if e[0] == "typically_at"]
    named = all(base.fallback.get(c, c) in text for e in edges for c in e[1:3])
    reached = {e[1] for e in edges} | {e[2] for e in edges}
    covered = all(c.cui in reached for c in intent.columns if c.cui != intent.cui)
    return named and covered


def _record(question, condition, result, got, items) -> dict:
    parts = {"match": [d for d in got.dropped if d.get("part") != "knowledge"],
             "knowledge": [d for d in got.dropped if d.get("part") == "knowledge"]}
    return {"question": question, "condition": condition, "status": got.status,
            "first_token_ms": got.first_token_ms, "total_ms": round(got.total_ms, 1),
            "error": got.error, "items": len(items or []),
            "kept_match": _sentences(got.summary) + len(got.images),
            "dropped_match": len(parts["match"]),
            "kept_knowledge": _sentences(got.knowledge),
            "dropped_knowledge": len(parts["knowledge"]),
            "citations": got.citations,
            "bad_citations": sum(d["why"].startswith("citation not in") or d["why"] == "no citation"
                                 for d in parts["knowledge"]),
            "summary": got.summary, "images": got.images, "knowledge": got.knowledge,
            "knowledge_items": items and [{"id": k["id"], "text": k["text"]} for k in items],
            "dropped": got.dropped, **_search(result)}


def _search(result) -> dict:
    return {"results": len(result.rows), "degraded": result.degraded,
            "warnings": result.warnings_text(),
            "intents": [[i.name, i.kind, len(i.columns)] for i in result.intents]}


def _q(values, q):
    values = [v for v in values if v is not None]
    return round(float(np.percentile(values, q)), 1) if values else None


def summarise(records: list[dict]) -> dict:
    out = {}
    for condition in ("knowledge", "v3"):
        rows = [r for r in records if r["condition"] == condition]
        explained = [r for r in rows if r["results"]]
        n = len(explained)
        kept_m = sum(r["kept_match"] for r in explained)
        drop_m = sum(r["dropped_match"] for r in explained)
        kept_k = sum(r["kept_knowledge"] for r in explained)
        drop_k = sum(r["dropped_knowledge"] for r in explained)
        out[condition] = {
            "pages": n, "ok": sum(r["status"] == "ok" for r in explained),
            "grounded": sum(r["status"] in ("ok", "partial") for r in explained),
            "failed": sum(r["status"] == "failed" for r in explained),
            "match_sentences": kept_m + drop_m, "match_dropped": drop_m,
            "knowledge_sentences": kept_k + drop_k, "knowledge_dropped": drop_k,
            "pages_with_knowledge": sum(r["kept_knowledge"] > 0 for r in explained),
            "citations_valid": sum(r["citations"] for r in explained),
            "citations_bad": sum(r["bad_citations"] for r in explained),
            "items_mean": round(sum(r["items"] for r in explained) / max(n, 1), 1),
            "first_token_ms_p50": _q([r["first_token_ms"] for r in explained], 50),
            "first_token_ms_p95": _q([r["first_token_ms"] for r in explained], 95),
            "total_ms_p50": _q([r["total_ms"] for r in explained], 50),
            "total_ms_p95": _q([r["total_ms"] for r in explained], 95)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="docs/v2-6-smoke.txt")
    ap.add_argument("--out", required=True, help="prefix: <out>.jsonl and <out>.summary.json")
    ap.add_argument("--matrix", default="data/indexes/image_scores/"
                    "20261024_chexpert25k_cxrclip_ft6_lr1e-04_v1_ext1_probe1")
    ap.add_argument("--max-items", type=int, default=None, help="override knowledge_max_items")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = out.parent.parent / "cache" / "chooser.jsonl"
    if not cache.is_file() and Path("data/kg/cache/chooser.jsonl").is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile("data/kg/cache/chooser.jsonl", cache)
    runtime.load_config = _patched_models(cache)

    settings = Settings()
    matrix = Path(args.matrix)
    meta = json.loads((matrix / "probe.json").read_text(encoding="utf-8"))
    images = pl.read_parquet(Path(settings.canonical_dir) / "images.parquet",
                             columns=["image_id", "projection"])
    search, notes = runtime.build_kg_search(
        settings.model_copy(update={"kg_matrix_dir": matrix}), {"build_id": meta["build_id"]},
        images)
    print(json.dumps({"notes": notes, "versions": search.versions}, ensure_ascii=False))
    v4 = search.explainer
    if v4 is None or v4.knowledge is None:
        print("STOP: no explainer with knowledge (key, or explain.knowledge false)")
        return 1
    if args.max_items:
        v4.knowledge.max_items = args.max_items
    v3_text = (default_config_dir() / load_config("models")["kg_llm"]["explain"]["prompt_file"]
               ).read_text(encoding="utf-8")
    v3 = Explainer(v4.client, v4.guard, v3_text, v4.timeout_s, v4.index)
    top = search.cfg["explain"]["top_results"]

    questions = [ln.strip() for ln in Path(args.file).read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
    records, path_ok, path_n = [], 0, 0
    for i, question in enumerate(questions):
        result = search.search(question, top_k=10)
        for it in result.intents:
            if it.columns:
                path_n += 1
                path_ok += _path_check(v4.knowledge, it)
        rows = result.rows[:top]
        probs = (search.matrix.probabilities([r["row"] for r in rows], result.intents)
                 if rows else [])
        names = v4.index.names(result.intents)
        items = v4.knowledge.items(result.intents, result.language)
        order = [("knowledge", v4, items), ("v3", v3, None)]
        for condition, explainer, given in (order if i % 2 == 0 else order[::-1]):
            if not rows:
                records.append({"question": question, "condition": condition,
                                "status": "no_results", **_search(result)})
                continue
            facts = build_facts(question, result.language, result.intents, rows, probs, names,
                                given)
            got = explainer.explain(facts, result.intents, given)
            records.append(_record(question, condition, result, got, given))
        last = [r for r in records[-2:] if r.get("results")]
        print(f"{i + 1:2d} " + "  ".join(
            f"{r['condition']}:{r['status']} ft={r['first_token_ms'] or 0:.0f}"
            f" k={r.get('kept_knowledge', 0)}/{r.get('dropped_knowledge', 0)}"
            for r in last) + f"  {question}")

    with open(f"{out}.jsonl", "w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    usage = v4.client.usage
    summary = {"questions": len(questions), "max_items": v4.knowledge.max_items,
               "path_edges_match": f"{path_ok}/{path_n}", **summarise(records),
               "explain_usage": usage,
               "explain_usd": round((usage["input_tokens"] * HAIKU["input"]
                                     + usage["output_tokens"] * HAIKU["output"]) / 1e6, 4),
               "versions": search.versions}
    Path(f"{out}.summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "versions"}, indent=1,
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
