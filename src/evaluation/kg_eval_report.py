"""The score phase of `cxr eval-kg`: every number of docs/kg-evaluation.md from the runs, the
judgements and the cost ledger, as one JSON and a markdown report. Every table is given twice:
on all questions, and without the questions whose gold the reviewer disputed."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl

from src.common.config import Settings
from src.evaluation import kg_eval as ke
from src.evaluation import kg_eval_run as kr

MAIN_PAIRS = [("B2", "B1"), ("B1", "B0"), ("REF", "B2"), ("REF", "B1")]
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _view_of() -> dict[str, str]:
    images = pl.read_parquet(Path(Settings().canonical_dir) / "images.parquet",
                             columns=["study_id", "projection"])
    return dict(images.iter_rows())


def layer2_table(runs: dict, rows, systems, keep) -> dict:
    out = {}
    for system in systems:
        if system not in runs:
            continue
        per = {}
        for r in rows:
            run = runs[system].get(r["id"])
            if run is None or "intents" not in run or not keep(r):
                continue
            intents = [kr.intent_from_dict(d) for d in run["intents"]]
            per[r["id"]] = ke.layer2(r, intents, set(run.get("warnings", [])))
        table = {}
        for scope in ["all", *dict.fromkeys(r["group"] for r in rows)]:
            ids = [r["id"] for r in rows if r["id"] in per and (scope == "all"
                                                                 or r["group"] == scope)]
            if not ids:
                continue
            hits = sum(per[q]["cui_hits"] for q in ids)
            total = sum(per[q]["cui_total"] for q in ids)
            recalls = [per[q]["warnings_recall"] for q in ids
                       if per[q]["warnings_recall"] is not None]
            table[scope] = {
                "n": len(ids), "cui_accuracy": round(hits / total, 4) if total else None,
                "columns_exact": round(float(np.mean([per[q]["columns_exact"] for q in ids])), 4),
                "columns_f1": round(float(np.mean([per[q]["columns_f1"] for q in ids])), 4),
                "warnings_all": round(float(np.mean([per[q]["warnings_all"] for q in ids])), 4),
                "warnings_recall": round(float(np.mean(recalls)), 4) if recalls else None}
        out[system] = {"table": table, "per_question": per}
    return out


def layer4(runs: dict, rows) -> dict:
    statuses: dict[str, int] = {}
    kept = dropped = diagnostic = 0
    reasons: dict[str, int] = {}
    for r in rows:
        e = runs["B2"].get(r["id"], {}).get("explanation")
        if e is None:
            continue
        statuses[e["status"]] = statuses.get(e["status"], 0) + 1
        k = len([s for s in _SENTENCE.split((e.get("summary") or "").strip()) if s])
        k += len(e.get("images") or [])
        d = e.get("dropped") or []
        kept, dropped = kept + k, dropped + len(d)
        for item in d:
            why = item["why"].split(":")[0]
            reasons[why] = reasons.get(why, 0) + 1
            diagnostic += why.startswith("diagnostic")
    total = kept + dropped
    n = sum(statuses.values())
    return {"explanations": n, "status": statuses,
            "ok_share": round(statuses.get("ok", 0) / n, 4) if n else None,
            "sentences": total, "grounded_share": round(kept / total, 4) if total else None,
            "dropped_share": round(dropped / total, 4) if total else None,
            "diagnostic_dropped": diagnostic, "drop_reasons": reasons}


def cost_table(ledger, n_questions: int) -> dict:
    by_step: dict[str, dict] = {}
    for rec in ledger.records:
        cell = by_step.setdefault(rec["step"], {"usd": 0.0, "calls": 0})
        cell["usd"] += rec["usd"]
        cell["calls"] += rec.get("calls", 1)
    per_system = {"B0": ["b0_translate"], "B1": ["b1_parse"],
                  "B2 (search)": ["b2_parse", "b2_choose"], "B2 (explanation)": ["b2_explain"]}
    systems = {}
    for name, steps in per_system.items():
        usd = sum(by_step.get(s, {}).get("usd", 0.0) for s in steps)
        calls = sum(by_step.get(s, {}).get("calls", 0) for s in steps)
        systems[name] = {"usd_per_1000": round(usd / n_questions * 1000, 3),
                         "calls_per_question": round(calls / n_questions, 2)}
    return {"by_step": {k: {"usd": round(v["usd"], 4), "calls": v["calls"]}
                        for k, v in sorted(by_step.items())},
            "per_system": systems, "total_usd": round(ledger.total, 4)}


def latency_table(runs: dict) -> dict:
    out = {}
    for system in ("B0", "B1", "B2"):
        if system not in runs:
            continue
        steps: dict[str, list] = {}
        for run in runs[system].values():
            for k, v in run.get("ms", {}).items():
                steps.setdefault(k, []).append(v)
        out[system] = {k: ke.percentiles(v) for k, v in sorted(steps.items())}
    return out


def judge_stats(judge: ke.Judge, cross: ke.Judge, sample, pooled) -> dict:
    grades = [judge.get(q, s) for q, studies in pooled.items() for s in studies]
    done = [g for g in grades if g is not None]
    dist = {g: sum(1 for x in done if x["grade"] == g) for g in (0, 1, 2)}
    a, b = [], []
    for q, s in sample:
        x, y = judge.get(q, s), cross.get(q, s)
        if x is not None and y is not None:
            a.append(x["grade"])
            b.append(y["grade"])
    agree = sum(x == y for x, y in zip(a, b, strict=True))
    return {"pairs": len(grades), "judged": len(done), "grades": dist,
            "cross": {"pairs": len(a), "agreement": round(agree / len(a), 4) if a else None,
                      "kappa": ke.cohen_kappa(a, b),
                      "kappa_quadratic": ke.cohen_kappa(a, b, weighted=True),
                      "binary_relevant_agreement": round(
                          sum((x == 2) == (y == 2) for x, y in zip(a, b, strict=True)) / len(a),
                          4) if a else None}}


def error_analysis(rows, runs, per_q) -> list[dict]:
    out = []
    for r in rows:
        q = r["id"]
        if q not in per_q.get("B2", {}) or q not in per_q.get("B1", {}):
            continue
        b2, b1 = per_q["B2"][q]["ndcg10"], per_q["B1"][q]["ndcg10"]
        if b2 >= b1:
            continue
        run = runs["B2"][q]
        intents = [kr.intent_from_dict(d) for d in run["intents"]]
        out.append({
            "id": q, "group": r["group"], "lang": r["lang"], "question": r["question"],
            "ndcg_b2": round(b2, 3), "ndcg_b1": round(b1, 3),
            "step": kr.error_step(r, run), "gold_disputed": r["gold_disputed"],
            "gold_columns": sorted(f"{s}{c}" for s, c in ke.gold_columns(r)),
            "b2_columns": sorted(f"{s}{c}" for s, c in ke.signed_columns(intents)),
            "b2_kinds": [i.kind for i in intents],
            "b1_template": runs["B1"][q].get("text"),
            "graph_check_matched_gold": r["auto_check"]["columns_match"]})
    return out


def score_all(rows, runs, meta, judge, cross, sample, pooled, ledger, ecfg, all_rows) -> dict:
    vcfg = ecfg["eval"]
    view_of = _view_of()
    per_q = kr.score_runs(runs, rows, judge, view_of, vcfg["top_k"], vcfg["view_rule"], pooled)
    undisputed = lambda r: not r["gold_disputed"]          # noqa: E731
    ablations = [a for a in vcfg["ablations"] if a in runs]
    n_boot, seed = vcfg["bootstrap"], ecfg["benchmark"]["seed"]
    out = {"meta": meta, "split": rows[0]["split"] if rows else None, "questions": len(rows),
           "disputed": sum(r["gold_disputed"] for r in rows),
           "systems": sorted(per_q), "planned": list(ke.PLANNED),
           "judge": judge_stats(judge, cross, sample, pooled)}
    for name, keep in (("all", lambda r: True), ("undisputed", undisputed)):
        out[name] = {
            "layer3": kr.aggregate(per_q, rows, keep),
            "diffs": kr.diffs(per_q, rows, MAIN_PAIRS, n_boot, seed, keep),
            "ablation_diffs": kr.diffs(per_q, rows, [(a, "B2") for a in ablations], n_boot,
                                       seed, keep),
            "layer2": {s: v["table"] for s, v in layer2_table(
                runs, rows, ["B2", "lookup_deterministic", "sapbert_off"], keep).items()}}
    out["layer4"] = layer4(runs, rows)
    out["cost"] = cost_table(ledger, max(1, len(rows)))
    out["latency"] = latency_table(runs)
    out["errors"] = error_analysis(rows, runs, per_q)
    out["per_question"] = per_q
    out["graph_check"] = {"dev_mismatch": sum(not r["auto_check"]["columns_match"] for r in rows),
                          "all_mismatch": sum(not r["auto_check"]["columns_match"]
                                              for r in all_rows)}
    return out


# -- markdown ---------------------------------------------------------------------------------

def _f(x, nd=3) -> str:
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _ci(d: dict) -> str:
    if d.get("mean") is None:
        return "—"
    return (f"{d['mean']:+.3f} [{d['lo']:+.3f}; {d['hi']:+.3f}] "
            f"({d['wins']}/{d['losses']}/{d['ties']})")


def markdown(out: dict) -> str:
    lines = [f"# KG evaluation, split {out['split']} ({out['questions']} questions, "
             f"{out['disputed']} disputed)", ""]
    j = out["judge"]
    lines += [f"Judge: {j['judged']}/{j['pairs']} pairs, grades {j['grades']}; cross judge "
              f"{j['cross']}", ""]
    for scope in ("all", "undisputed"):
        block = out[scope]
        lines += [f"## Layer 3 ({scope})", "",
                  "| system | group | n | nDCG@10 | P@10 | P@10 (≥1) | excl-viol@10 | no result |",
                  "|---|---|---|---|---|---|---|---|"]
        for system, table in block["layer3"].items():
            for g, c in table.items():
                lines.append(f"| {system} | {g} | {c['n']} | {_f(c['ndcg10'])} | {_f(c['p10'])} "
                             f"| {_f(c['p10_lenient'])} | {_f(c['excl_viol10'])} | "
                             f"{_f(c['no_result'])} |")
        lines += ["", "Differences (mean [95% CI] (wins/losses/ties)):", "",
                  "| pair | nDCG@10 | P@10 |", "|---|---|---|"]
        for pair, d in {**block["diffs"], **block["ablation_diffs"]}.items():
            lines.append(f"| {pair} | {_ci(d['ndcg10'])} | {_ci(d['p10'])} |")
        lines += ["", f"## Layer 2 ({scope})", "",
                  "| system | group | n | CUI acc | columns exact | columns F1 | warnings all |"
                  " warnings recall |", "|---|---|---|---|---|---|---|---|"]
        for system, table in block["layer2"].items():
            for g, c in table.items():
                lines.append(f"| {system} | {g} | {c['n']} | {_f(c['cui_accuracy'])} | "
                             f"{_f(c['columns_exact'])} | {_f(c['columns_f1'])} | "
                             f"{_f(c['warnings_all'])} | {_f(c['warnings_recall'])} |")
        lines.append("")
    lines += ["## Layer 4", "", f"{out['layer4']}", "", "## Cost", "", f"{out['cost']}", "",
              "## Latency", "", f"{out['latency']}", "", "## Errors (B2 below B1)", "",
              "| id | step | nDCG B2 | nDCG B1 | gold | B2 | kinds | disputed |",
              "|---|---|---|---|---|---|---|---|"]
    for e in out["errors"]:
        lines.append(f"| {e['id']} | {e['step']} | {e['ndcg_b2']} | {e['ndcg_b1']} | "
                     f"{' '.join(e['gold_columns'])} | {' '.join(e['b2_columns'])} | "
                     f"{','.join(e['b2_kinds'])} | {e['gold_disputed']} |")
    return "\n".join(lines) + "\n"
