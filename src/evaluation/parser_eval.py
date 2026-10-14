"""Parser evaluation (spec 8): macro-F1 over concepts, statuses and views, exact
match, schema validity, per-language and per-group breakdowns, and the number of
studies each gold parse matches under every policy pair.

Pure Python metrics; an invalid parse is scored as an empty prediction. Classes
that occur only in predictions still count (F1 0): inventing a concept is an
error, not a free pass.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product
from statistics import median

from src.evaluation.parser_benchmark import GROUPS, BenchmarkQuery
from src.evaluation.vector_baseline import percentiles
from src.query.parser import QueryParser
from src.query.schema import ParseError, SearchQuery
from src.retrieval import label_filter as lf

POLICY_PAIRS = [f"{a}/{u}" for a, u in product(("exclude_present", "explicit_absence_required"),
                                                 ("EXCLUDE", "INCLUDE"))]


def concept_items(q: SearchQuery) -> set[str]:
    return {c.concept for c in q.include} | set(q.exclude)


def status_items(q: SearchQuery) -> set[tuple[str, str]]:
    return {(c.concept, c.status) for c in q.include} | {(c, "EXCLUDED") for c in q.exclude}


def view_items(q: SearchQuery) -> set[str]:
    return set(q.views)


def macro_f1(preds: Sequence[set], golds: Sequence[set], class_of: Callable) -> dict:
    counts: dict[str, list[int]] = {}
    for pred, gold in zip(preds, golds, strict=True):
        for item in pred | gold:
            tally = counts.setdefault(class_of(item), [0, 0, 0])
            if item in pred and item in gold:
                tally[0] += 1
            elif item in pred:
                tally[1] += 1
            else:
                tally[2] += 1
    per: dict[str, dict] = {}
    for cls, (tp, fp, fn) in sorted(counts.items()):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per[cls] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1,
                    "support": tp + fn}
    f1s = [v["f1"] for v in per.values()]
    return {"macro_f1": sum(f1s) / len(f1s) if f1s else None, "per_class": per}


_EMPTY = SearchQuery(language="en")


def evaluate(golds: Sequence[SearchQuery], preds: Sequence[SearchQuery | None]) -> dict:
    valid = [p is not None for p in preds]
    filled = [p if p is not None else _EMPTY for p in preds]
    concept = macro_f1([concept_items(p) for p in filled], [concept_items(g) for g in golds],
                       lambda item: item)
    status = macro_f1([status_items(p) for p in filled], [status_items(g) for g in golds],
                      lambda item: item[1])
    view = macro_f1([view_items(p) for p in filled], [view_items(g) for g in golds],
                    lambda item: item)
    exact = [status_items(p) == status_items(g) and view_items(p) == view_items(g)
             for p, g in zip(filled, golds, strict=True)]
    n = len(golds)
    return {
        "n": n,
        "schema_valid_rate": sum(valid) / n if n else None,
        "concept_macro_f1": concept["macro_f1"],
        "status_macro_f1": status["macro_f1"],
        "view_macro_f1": view["macro_f1"],
        "exact_match_rate": sum(exact) / n if n else None,
        "per_concept": concept["per_class"],
        "per_status": status["per_class"],
        "per_view": view["per_class"],
    }


@dataclass(frozen=True)
class Prediction:
    query: SearchQuery | None
    parser: str | None
    fallback_used: bool
    retried: bool
    ms: float
    error: str = ""


def run_parser(parser: QueryParser, rows: Sequence[BenchmarkQuery], mode: str) -> list[Prediction]:
    out: list[Prediction] = []
    for row in rows:
        started = time.perf_counter()
        try:
            result = parser.parse(row.query, mode=mode)
        except ParseError as exc:
            out.append(Prediction(None, None, False, False, _ms(started), str(exc)))
            continue
        out.append(Prediction(result.query, result.parser, result.fallback_used,
                              any(w.startswith("llm retried") for w in result.warnings),
                              _ms(started)))
    return out


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _subset(rows: Sequence[BenchmarkQuery], preds: Sequence[Prediction], keep) -> dict:
    idx = [i for i, r in enumerate(rows) if keep(r)]
    return evaluate([rows[i].gold for i in idx], [preds[i].query for i in idx])


def breakdowns(rows: Sequence[BenchmarkQuery], preds: Sequence[Prediction]) -> dict:
    languages = sorted({r.language for r in rows})
    groups = [g for g in GROUPS if any(g in r.groups for r in rows)]
    return {
        "by_language": {lang: _subset(rows, preds, lambda r, lang=lang: r.language == lang)
                        for lang in languages},
        "by_group": {g: _subset(rows, preds, lambda r, g=g: g in r.groups) for g in groups},
    }


def verdict(metrics: dict, thresholds: dict) -> dict:
    f1_min = thresholds["parser_macro_f1_min"]
    valid_min = thresholds["parser_schema_valid_rate_min"]
    out = {"schema_valid": metrics["schema_valid_rate"] >= valid_min}
    for key in ("concept_macro_f1", "status_macro_f1", "view_macro_f1"):
        value = metrics[key]
        out[key] = None if value is None else value >= f1_min
    out["passed"] = all(v for v in out.values() if v is not None)
    return out


def label_filter_counts(rows: Sequence[BenchmarkQuery], matrix: lf.LabelMatrix) -> dict:
    """How many studies each gold parse matches under every absence/uncertainty pair.
    Query-level evidence for WORKLOG D9."""
    per_query: dict[str, dict[str, int]] = {}
    for row in rows:
        if not row.gold.include and not row.gold.exclude:
            continue
        counts = {}
        for pair in POLICY_PAIRS:
            absence, uncertainty = pair.split("/")
            mask = lf.filter_studies_by_labels(matrix, row.gold.include, row.gold.exclude,
                                               absence_policy=absence,
                                               uncertainty_policy=uncertainty)
            counts[pair] = int(mask.sum())
        per_query[row.id] = counts
    summary: dict[str, dict] = {}
    for group in GROUPS:
        ids = [r.id for r in rows if group in r.groups and r.id in per_query]
        if not ids:
            continue
        summary[group] = {
            pair: {"n": len(ids),
                   "median": float(median(per_query[i][pair] for i in ids)),
                   "min": min(per_query[i][pair] for i in ids),
                   "zero": sum(per_query[i][pair] == 0 for i in ids)}
            for pair in POLICY_PAIRS}
    return {"per_query": per_query, "summary": summary, "corpus_studies": len(matrix.study_ids)}


def evaluate_benchmark(rows: Sequence[BenchmarkQuery], parser: QueryParser,
                       modes: Sequence[str], thresholds: dict) -> dict:
    report: dict = {"n": len(rows), "thresholds": thresholds, "parsers": {}}
    for mode in modes:
        preds = run_parser(parser, rows, mode)
        golds = [r.gold for r in rows]
        metrics = evaluate(golds, [p.query for p in preds])
        entry = {"metrics": metrics, **breakdowns(rows, preds),
                 "latency_ms": percentiles([p.ms for p in preds[1:]] or [0.0]),
                 "fallback_rate": sum(p.fallback_used for p in preds) / len(preds),
                 "retry_rate": sum(p.retried for p in preds) / len(preds),
                 "mismatches": [
                     {"id": r.id, "query": r.query,
                      "gold": {"include": [(c.concept, c.status) for c in r.gold.include],
                               "exclude": r.gold.exclude, "views": r.gold.views},
                      "pred": None if p.query is None else {
                          "include": [(c.concept, c.status) for c in p.query.include],
                          "exclude": p.query.exclude, "views": p.query.views},
                      "error": p.error}
                     for r, p in zip(rows, preds, strict=True)
                     if p.query is None or status_items(p.query) != status_items(r.gold)
                     or view_items(p.query) != view_items(r.gold)]}
        report["parsers"][mode] = entry
    judged = "auto" if "auto" in report["parsers"] else modes[-1]
    report["judged_parser"] = judged
    report["verdict"] = verdict(report["parsers"][judged]["metrics"], thresholds)
    return report
