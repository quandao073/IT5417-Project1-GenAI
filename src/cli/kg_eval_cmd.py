"""Commands of the KG evaluation (V2-7, V2-8 on dev): `kg-bench-build` writes the benchmark and
its gold; `eval-kg` runs the systems, pools, judges and scores (docs/kg-evaluation.md)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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


def _components(ecfg: dict, ledger, key: str):
    """Everything the run phase needs, built in-process: the corpus vectors of the FT6 build,
    its text encoder, the B1 parser, the KG search, the B0 translator, the label matrix."""
    from src.api.dependencies import build_query_parser
    from src.embeddings import index
    from src.evaluation import kg_eval as ke
    from src.evaluation.vector_baseline import _all_vectors
    from src.kg import runtime
    from src.retrieval import label_filter as lf

    settings = Settings()
    models, rcfg = load_config("models"), load_config("retrieval")
    build = models["image_encoder"]["selected"]
    manifest = json.loads((Path(settings.indexes_dir) / "builds" / _build_id(settings, build)
                           / "manifest.json").read_text(encoding="utf-8"))
    vectors, image_ids = _all_vectors(Path(settings.indexes_dir) / "embeddings"
                                      / manifest["build_id"])
    images = pl.read_parquet(Path(settings.canonical_dir) / "images.parquet")
    study_of = dict(zip(images["image_id"], images["study_id"], strict=True))
    view_of = dict(zip(images["image_id"], images["projection"], strict=True))
    corpus = ke.Corpus(ke.unit_rows(vectors), image_ids, [study_of[i] for i in image_ids],
                       np.array([view_of[i] for i in image_ids]))
    encoder = index.build_encoder(manifest, models["image_encoder"]["candidates"], "cpu")
    b1_parser, status = build_query_parser(settings)
    if status != "ok":
        raise SystemExit(f"STOP: B1 parser language model is {status}")
    matrix_dir = (Path(settings.indexes_dir) / "image_scores"
                  / f"{manifest['build_id']}_ext1")
    search, notes = runtime.build_kg_search(
        settings.model_copy(update={"kg_matrix_dir": matrix_dir}),
        {"build_id": manifest["build_id"]},
        images.select("image_id", "projection"))
    if search.store is None or search.chooser is None:
        raise SystemExit(f"STOP: the KG search is degraded at startup: {notes}")
    translator = kg_llm.CachedCaller("translator", ecfg["llm"]["translator"], key,
                                     Path(ecfg["benchmark"]["work_dir"])
                                     / "llm_cache_translator.jsonl", ledger)
    labels_long = pl.read_parquet(Path(settings.canonical_dir) / "labels.parquet")
    label_matrix = lf.build_label_matrix(
        labels_long, corpus.study_ids, rcfg["label_resolution"]["source_priority"],
        seed=rcfg["results"]["label_order_seed"])
    mapping = yaml_labels()
    label_of_cui = {entry["id"]: label for label, entry in mapping.items()}
    return {"corpus": corpus, "encoder": encoder, "b1_parser": b1_parser, "search": search,
            "translator": translator, "label_matrix": label_matrix,
            "label_of_cui": label_of_cui, "labels_cfg": rcfg["constraints"],
            "concepts_cfg": load_config("concepts"), "manifest": manifest, "notes": notes}


def yaml_labels() -> dict:
    import yaml

    return yaml.safe_load((default_config_dir() / "kg" / "chexpert_cuis.yaml")
                          .read_text(encoding="utf-8"))["labels"]


def _build_id(settings: Settings, encoder: str) -> str:
    """The registered build of `encoder` (BUILD_ID when set)."""
    if settings.build_id:
        return settings.build_id
    builds = Path(settings.indexes_dir) / "builds"
    found = [p.name for p in builds.iterdir() if (p / "manifest.json").is_file()
             and json.loads((p / "manifest.json").read_text(encoding="utf-8"))["encoder"]
             == encoder]
    if len(found) != 1:
        raise SystemExit(f"STOP: {len(found)} registered builds of {encoder}: {found}")
    return found[0]


def cmd_eval_kg(args: argparse.Namespace) -> int:
    """V2-7 on one split: run (systems + B2 ablations), judge (pool, cross judge, human
    sample), score (layers 2-4, cost, latency, ablations, error analysis)."""
    from src.evaluation import kg_eval as ke
    from src.evaluation import kg_eval_run as kr

    sys.stdout.reconfigure(encoding="utf-8")
    if args.split == "heldout" and not args.heldout_declared:
        print("STOP: heldout runs once and must be declared (--heldout-declared)",
              file=sys.stderr)
        return 1
    ecfg = _ecfg()
    bcfg, lcfg, vcfg = ecfg["benchmark"], ecfg["llm"], ecfg["eval"]
    work = Path(bcfg["work_dir"])
    key = llm_select.api_key("ANTHROPIC_API_KEY")
    if not key:
        print("STOP: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 1
    ledger = _ledger(ecfg)
    rows = [r for r in kb.read_jsonl(Path(bcfg["file"])) if r["split"] == args.split]
    rows = rows[:args.limit] if args.limit else rows
    phases = {"run", "judge", "score"} if args.phase == "all" else {args.phase}
    systems = args.systems or [*ke.SYSTEMS, "lookup_deterministic", "sapbert_off"]
    reports_df = pl.read_parquet(Path(Settings().canonical_dir) / "reports.parquet",
                                 columns=["study_id", "findings", "impression"])
    reports = {s: ke.report_text(f, i) for s, f, i in reports_df.iter_rows()}

    runner = None
    try:
        if "run" in phases:
            parts = _components(ecfg, ledger, key)
            t_text, t_sha = _prompt(ecfg, "translate_prompt")
            runner = kr.Runner(rows=rows, split=args.split, ecfg=ecfg, work=work,
                               ledger=ledger, t_text=t_text, t_sha=t_sha,
                               **{k: parts[k] for k in (
                                   "corpus", "encoder", "b1_parser", "search", "translator",
                                   "label_matrix", "label_of_cui", "labels_cfg",
                                   "concepts_cfg")})
            runner.runs.setdefault("_meta", {}).update(
                {"versions": parts["search"].versions, "notes": parts["notes"],
                 "build_id": parts["manifest"]["build_id"]})
            runner.run_all([s for s in systems if s != "_meta"], args.rerun)
            runner.run_transforms()
            print(f"run done; spent {ledger.total:.2f} USD")
        runs = json.loads((work / f"runs_{args.split}.json").read_text(encoding="utf-8"))
        meta = runs.pop("_meta", {})
        pooled = kr.pool(runs, rows, vcfg["top_k"])
        j_text, j_sha = _prompt(ecfg, "judge_prompt")
        judge = ke.Judge(llm_select.AnthropicChooser("judge", lcfg["judge"], key),
                         lcfg["judge"]["model"], j_text, j_sha, work / "judge_opus.jsonl",
                         ledger, "judge", lcfg["judge_batch"])
        cross = ke.Judge(llm_select.AnthropicChooser("cross", lcfg["cross_judge"], key),
                         lcfg["cross_judge"]["model"], j_text, j_sha,
                         work / "judge_sonnet.jsonl", ledger, "cross_judge",
                         lcfg["judge_batch"])
        sample = kr.cross_pairs(rows, pooled, vcfg["cross_judge_fraction"], bcfg["seed"])
        if "judge" in phases:
            n_pairs = sum(len(v) for v in pooled.values())
            print(f"pool: {n_pairs} pairs over {len(rows)} questions")
            judged = kr.judge_pool(judge, rows, pooled, reports)
            print(f"judged {judged} new pairs; spent {ledger.total:.2f} USD")
            by_q: dict[str, list[str]] = {}
            for qid, s in sample:
                by_q.setdefault(qid, []).append(s)
            by_id = {r["id"]: r for r in rows}
            for qid, studies in by_q.items():
                cross.grade(qid, by_id[qid]["question"], by_id[qid]["query_en"], studies,
                            reports)
            print(f"cross-judged {len(sample)} pairs; spent {ledger.total:.2f} USD")
            per = kr.write_human_sample(work / "human_sample.csv", work / "human_sample_key.csv",
                                        rows, judge, pooled, reports, vcfg["human_sample"],
                                        bcfg["seed"])
            print(f"human sample by judge grade: {per}")
        if "score" in phases:
            from src.evaluation import kg_eval_report as rep

            out = rep.score_all(rows, runs, meta, judge, cross, sample, pooled, ledger, ecfg,
                                kb.read_jsonl(Path(bcfg["file"])))
            path = work / f"metrics_{args.split}.json"
            path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
            (work / f"report_{args.split}.md").write_text(rep.markdown(out), encoding="utf-8")
            print(f"written: {path}, {work / f'report_{args.split}.md'}")
    except kg_llm.BudgetExceeded as exc:
        if runner is not None:
            runner.save()
        print(f"STOP (budget): {exc}", file=sys.stderr)
        return 2
    print(f"total spent this session: {ledger.total:.2f} USD; by step {ledger.by('step')}")
    return 0


def register(sub) -> None:
    e = sub.add_parser("eval-kg", help="V2-7: run, judge and score the KG benchmark on a split")
    e.add_argument("--split", choices=["dev", "heldout"], default="dev")
    e.add_argument("--heldout-declared", action="store_true",
                   help="heldout runs once (D24); required to run it")
    e.add_argument("--phase", choices=["run", "judge", "score", "all"], default="all")
    e.add_argument("--systems", nargs="+", default=None,
                   help="systems and search ablations to run (default: all)")
    e.add_argument("--rerun", action="store_true", help="run again systems already on disk")
    e.add_argument("--limit", type=int, default=None, help="only the first N questions (a trial)")
    e.set_defaults(func=cmd_eval_kg)

    b = sub.add_parser("kg-bench-build", help="V2-7: write the KG benchmark and its gold")
    b.add_argument("--out", default=None)
    b.add_argument("--groups", nargs="+", default=None, help="only these groups (a trial)")
    b.add_argument("--attempts", type=int, default=2, help="author retries on rule breaks")
    b.set_defaults(func=cmd_kg_bench_build)
