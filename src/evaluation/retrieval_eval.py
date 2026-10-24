"""Retrieval evaluation (spec 11): the one search path scored against the gold.

The headline metric is binary nDCG@10; the graded nDCG@10 is a secondary number, because its
grades count label sources and so lean on the very reports the labels came from.
Relevance always comes from the benchmark gold, never from the parse. A study is relevant when
it passes the gold's label filter under one policy and its selected image is in the gold's
views; its grade adds how many label sources assert what the gold asks (spec 11.3), so an
explicit "no effusion" ranks above silence. Metrics are macro means over queries, None-aware
(a query without exclude has no violation rate). Pure: `search` is injected (the pipeline in
production, a fake in tests), so nothing here touches Qdrant or SQLite.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.data.labels import CHEXPERT_LABELS
from src.evaluation.parser_benchmark import GROUPS, BenchmarkQuery
from src.evaluation.vector_baseline import percentiles
from src.query.parser import QueryParser
from src.query.schema import SearchQuery
from src.retrieval import label_filter as lf
from src.retrieval import pipeline
from src.retrieval.evidence import wanted_polarity

DEFAULT_POLICY = ("exclude_present", "EXCLUDE")
NEGATION_POLICY = ("explicit_absence_required", "EXCLUDE")
AGREEMENT_STATUSES = ("PRESENT", "UNCERTAIN", "ABSENT")   # axis 2 of AgreementMatrix.counts
TOP_K = 10


@dataclass(frozen=True)
class AgreementMatrix:
    row_of: dict[str, int]
    col_of: dict[str, int]
    counts: np.ndarray        # int8 [n, 14, 3]: sources asserting PRESENT / UNCERTAIN / ABSENT


@dataclass(frozen=True)
class Relevance:
    relevant: frozenset[str]
    grade: dict[str, float]   # relevant studies only, each >= 1


def build_agreement_matrix(labels_long: pl.DataFrame,
                           study_ids: Sequence[str]) -> AgreementMatrix:
    row_of = {s: i for i, s in enumerate(study_ids)}
    col_of = {c: j for j, c in enumerate(CHEXPERT_LABELS)}
    axis_of = {s: k for k, s in enumerate(AGREEMENT_STATUSES)}
    # labels.parquet keys on (study, concept, source), so a row count is a source count.
    tally = (labels_long
             .filter(pl.col("study_id").is_in(list(row_of))
                     & pl.col("status").is_in(AGREEMENT_STATUSES))
             .group_by("study_id", "concept", "status").len())
    counts = np.zeros((len(row_of), len(col_of), len(axis_of)), dtype=np.int8)
    # Scattered by study id: polars guarantees no group order (the Phase 4 join lesson).
    counts[[row_of[s] for s in tally["study_id"].to_list()],
           [col_of[c] for c in tally["concept"].to_list()],
           [axis_of[s] for s in tally["status"].to_list()]] = tally["len"].to_numpy()
    return AgreementMatrix(row_of, col_of, counts)


def relevance_for(gold: SearchQuery, matrix: lf.LabelMatrix, agreement: AgreementMatrix, *,
                  views: Sequence[str], projection_of: Mapping[str, str], absence_policy: str,
                  uncertainty_policy: str) -> Relevance:
    mask = lf.filter_studies_by_labels(matrix, gold.include, gold.exclude,
                                       absence_policy=absence_policy,
                                       uncertainty_policy=uncertainty_policy)
    studies = matrix.study_ids[mask].tolist()
    if views:
        wanted = set(views)
        studies = [s for s in studies if projection_of[s] in wanted]
    # What a source must assert to agree: include PRESENT/UNCERTAIN as asked, exclude ABSENT.
    asked = ([(c.concept, 0 if c.status == "PRESENT" else 1) for c in gold.include]
             + [(concept, 2) for concept in gold.exclude])
    rows = np.array([agreement.row_of[s] for s in studies], dtype=np.intp)
    cols = np.array([agreement.col_of[c] for c, _ in asked], dtype=np.intp)
    axes = np.array([axis for _, axis in asked], dtype=np.intp)
    grades = 1.0 + agreement.counts[rows[:, None], cols, axes].mean(axis=1)
    return Relevance(frozenset(studies), dict(zip(studies, grades.tolist(), strict=True)))


def _dcg(gains: Sequence[float]) -> float:
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at(ranked: Sequence[str], grade: Mapping[str, float], k: int) -> float:
    """DCG of the top k over the DCG of the k best grades of the whole relevant set."""
    ideal = _dcg(heapq.nlargest(k, grade.values()))
    return _dcg([grade.get(s, 0.0) for s in ranked[:k]]) / ideal if ideal else 0.0


def query_metrics(ranked: Sequence[str], rel: Relevance, gold: SearchQuery,
                  matrix: lf.LabelMatrix, *, k: int = TOP_K) -> dict:
    """Precision divides by k even when fewer come back; the violation rate divides by what
    came back and is None when nothing did or the gold excludes nothing."""
    top = list(ranked[:k])
    hit = [s in rel.relevant for s in top]
    first = next((rank for rank, h in enumerate(hit, start=1) if h), None)
    violation = None
    if gold.exclude and top:
        status = matrix.status[np.ix_([matrix.row_of[s] for s in top],
                                      [matrix.col_of[c] for c in gold.exclude])]
        violation = float((status == lf.STATUS_CODE["PRESENT"]).any(axis=1).mean())
    return {f"ndcg_binary@{k}": ndcg_at(top, dict.fromkeys(rel.relevant, 1.0), k),
            f"ndcg_graded@{k}": ndcg_at(top, rel.grade, k),
            "p@5": sum(hit[:5]) / 5, f"p@{k}": sum(hit) / k,
            "mrr": 1 / first if first else 0.0,
            f"exclude_violation@{k}": violation, f"returned@{k}": len(top) / k}


def evidence_metrics(hits: Sequence[pipeline.Hit], query: SearchQuery) -> dict:
    """Coverage over (hit, include concept) pairs; polarity match over the quotes found."""
    wanted = wanted_polarity(query)
    pairs = covered = 0
    matched: list[bool] = []
    for hit in hits:
        for c in query.include:
            quotes = [e.polarity for e in hit.evidence if e.concept == c.concept]
            pairs += 1
            covered += bool(quotes)
            matched += [p == wanted[c.concept] for p in quotes]
    return {"evidence_coverage": covered / pairs if pairs else None,
            "evidence_polarity_match": sum(matched) / len(matched) if matched else None}


def aggregate(rows: Sequence[dict]) -> dict:
    """Macro mean of every key over the rows where it is not None, plus the row count."""
    out: dict = {}
    for key in dict.fromkeys(key for row in rows for key in row):
        values = [row[key] for row in rows if row.get(key) is not None]
        out[key] = sum(values) / len(values) if values else None
    return {**out, "n": len(rows)}


def _policy_name(policy: tuple[str, str]) -> str:
    return "/".join(policy)


def _query(parser: QueryParser, config: str,
           row: BenchmarkQuery) -> tuple[SearchQuery, bool | None]:
    """The query to run and whether its parse fell back (None for the gold, which is not parsed)."""
    if config == "gold":
        return parser.finalize(row.gold, row.query), None
    result = parser.parse(row.query, mode=config)           # "rule" or "auto"
    return result.query, result.fallback_used


def _breakdowns(rows: Sequence[BenchmarkQuery], per_query: Mapping[str, dict]) -> dict:
    languages = sorted({r.language for r in rows})
    groups = [g for g in GROUPS if any(g in r.groups for r in rows)]
    return {
        "by_group": {g: aggregate([per_query[r.id] for r in rows if g in r.groups])
                     for g in groups},
        "by_language": {lang: aggregate([per_query[r.id] for r in rows if r.language == lang])
                        for lang in languages},
    }


def evaluate_retrieval(rows: Sequence[BenchmarkQuery], *, state, parser: QueryParser,
                       matrix: lf.LabelMatrix, agreement: AgreementMatrix,
                       projection_of: Mapping[str, str], parsers: Sequence[str],
                       policies: Sequence[tuple[str, str]],
                       search=pipeline.run_search) -> dict:
    """`policies[0]` scores every row and fills `metrics`, the breakdowns and `per_query`;
    later policies score the negation rows only. `by_policy` reports every policy over the
    negation rows. A policy changes only the relevance, never the search, so each row is
    searched once per parser. Latency is search wall time only (no parse), first row dropped."""
    kept = [r for r in rows if r.gold.include or r.gold.exclude]
    negation = {r.id for r in kept if "negation" in r.groups}

    def scored(row: BenchmarkQuery, ranked: list[str], hits, query: SearchQuery,
               policy: tuple[str, str]) -> dict:
        absence, uncertainty = policy
        # Relevance is the gold's (spec 11.3); only the search runs the parse's views.
        rel = relevance_for(row.gold, matrix, agreement, views=row.gold.views,
                            projection_of=projection_of, absence_policy=absence,
                            uncertainty_policy=uncertainty)
        return {**query_metrics(ranked, rel, row.gold, matrix), **evidence_metrics(hits, query)}

    main, *others = policies
    report: dict = {"n": len(kept), "skipped_no_gold": len(rows) - len(kept),
                    "policies": [_policy_name(p) for p in policies], "parsers": {}}
    for config in parsers:
        per_query: dict[str, dict] = {}
        other: dict[str, list[dict]] = {_policy_name(p): [] for p in others}
        times: list[float] = []
        fallbacks: list[bool] = []
        for row in kept:
            query, used = _query(parser, config, row)
            if used is not None:
                fallbacks.append(used)
            started = time.perf_counter()
            outcome = search(state, query, top_k=TOP_K, views=query.views)
            times.append((time.perf_counter() - started) * 1000)
            # Not deduplicated: the corpus holds one image per study (WORKLOG Phase 1).
            ranked = [h.study_id for h in outcome.hits]
            per_query[row.id] = scored(row, ranked, outcome.hits, query, main)
            if row.id in negation:
                for policy in others:
                    other[_policy_name(policy)].append(
                        scored(row, ranked, outcome.hits, query, policy))
        by_policy = {_policy_name(main): aggregate([per_query[i] for i in per_query
                                                    if i in negation])}
        by_policy.update({name: aggregate(scores) for name, scores in other.items()})
        report["parsers"][config] = {
            "fallback_rate": sum(fallbacks) / len(fallbacks) if fallbacks else None,
            "metrics": aggregate(list(per_query.values())),
            **_breakdowns(kept, per_query), "by_policy": by_policy,
            "latency_ms": percentiles(times[1:]) if len(times) > 1 else None,
            "per_query": per_query}
    return report
