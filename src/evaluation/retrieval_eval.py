"""Retrieval evaluation (spec 11): the label, vector and hybrid modes scored against the gold.

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
from src.retrieval.hybrid import FusionParams

DEFAULT_POLICY = ("exclude_present", "EXCLUDE")
NEGATION_POLICY = ("explicit_absence_required", "EXCLUDE")
AGREEMENT_STATUSES = ("PRESENT", "UNCERTAIN", "ABSENT")   # axis 2 of AgreementMatrix.counts
TOP_K = 10
DEFAULT_FUSION = FusionParams(60, {"vector": 1.0, "report": 1.0}, 100, 100)
GRID = [FusionParams(rrf_k, {"vector": vector, "report": report, "image": image}, 100, 100, 100)
        for rrf_k in (10, 60) for vector in (0.0, 0.5, 1.0) for report in (0.5, 1.0)
        for image in (0.0, 0.5, 1.0)]
LEGS = ("vector", "report", "image")


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
    came back and is None when nothing did or the gold excludes nothing. HNRR (as in
    `template_bench`) is the share of what came back with every include AND every exclude
    PRESENT, the negation ignored; None unless the gold has both."""
    top = list(ranked[:k])
    hit = [s in rel.relevant for s in top]
    first = next((rank for rank, h in enumerate(hit, start=1) if h), None)
    violation = None
    if gold.exclude and top:
        status = matrix.status[np.ix_([matrix.row_of[s] for s in top],
                                      [matrix.col_of[c] for c in gold.exclude])]
        violation = float((status == lf.STATUS_CODE["PRESENT"]).any(axis=1).mean())
    hnrr = None
    if gold.include and gold.exclude and top:
        concepts = [c.concept for c in gold.include] + list(gold.exclude)
        status = matrix.status[np.ix_([matrix.row_of[s] for s in top],
                                      [matrix.col_of[c] for c in concepts])]
        hnrr = float((status == lf.STATUS_CODE["PRESENT"]).all(axis=1).mean())
    return {"p@5": sum(hit[:5]) / 5, f"p@{k}": sum(hit) / k,
            f"ndcg@{k}": ndcg_at(top, rel.grade, k),
            f"ndcg_binary@{k}": ndcg_at(top, dict.fromkeys(rel.relevant, 1.0), k),
            "mrr": 1 / first if first else 0.0,
            f"exclude_violation@{k}": violation, f"returned@{k}": len(top) / k,
            f"hnrr@{k}": hnrr}


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
                       modes: Sequence[str], policies: Sequence[tuple[str, str]],
                       fusion: FusionParams | None = None,
                       search=pipeline.run_search) -> dict:
    """`policies[0]` runs on every row and fills `metrics`, the breakdowns and `per_query`;
    later policies run on the negation rows only. `by_policy` reports every policy over the
    negation rows. Latency is search wall time only (no parse), first row dropped."""
    kept = [r for r in rows if r.gold.include or r.gold.exclude]
    negation = [r for r in kept if "negation" in r.groups]

    def scored(row: BenchmarkQuery, query: SearchQuery, mode: str,
               policy: tuple[str, str]) -> tuple[dict, float]:
        absence, uncertainty = policy
        # Relevance is the gold's (spec 11.3); only the search runs the parse's views.
        rel = relevance_for(row.gold, matrix, agreement, views=row.gold.views,
                            projection_of=projection_of, absence_policy=absence,
                            uncertainty_policy=uncertainty)
        started = time.perf_counter()
        outcome = search(state, query, mode, top_k=TOP_K, views=query.views,
                         absence_policy=absence, uncertainty_policy=uncertainty, fusion=fusion)
        ms = (time.perf_counter() - started) * 1000
        # Not deduplicated: the corpus holds one image per study (WORKLOG Phase 1).
        ranked = [h.study_id for h in outcome.hits]
        return {**query_metrics(ranked, rel, row.gold, matrix),
                **evidence_metrics(outcome.hits, query)}, ms

    main, *others = policies
    report: dict = {"n": len(kept), "skipped_no_gold": len(rows) - len(kept),
                    "policies": [_policy_name(p) for p in policies], "parsers": {}}
    for config in parsers:
        parsed = {r.id: _query(parser, config, r) for r in kept}
        queries = {i: query for i, (query, _) in parsed.items()}
        fallbacks = [used for _, used in parsed.values() if used is not None]
        entries: dict = {}
        for mode in modes:
            per_query: dict[str, dict] = {}
            times: list[float] = []
            for row in kept:
                per_query[row.id], ms = scored(row, queries[row.id], mode, main)
                times.append(ms)
            by_policy = {_policy_name(main): aggregate([per_query[r.id] for r in negation])}
            for policy in others:
                by_policy[_policy_name(policy)] = aggregate(
                    [scored(r, queries[r.id], mode, policy)[0] for r in negation])
            entries[mode] = {"metrics": aggregate(list(per_query.values())),
                             **_breakdowns(kept, per_query), "by_policy": by_policy,
                             "latency_ms": percentiles(times[1:]) if len(times) > 1 else None,
                             "per_query": per_query}
        report["parsers"][config] = {
            "fallback_rate": sum(fallbacks) / len(fallbacks) if fallbacks else None,
            "modes": entries}
    return report


def _weights(leg_weights: Mapping[str, float]) -> dict[str, float]:
    return {leg: leg_weights.get(leg, 0.0) for leg in LEGS}


def _is(config: Mapping, params: FusionParams) -> bool:
    return (config["rrf_k"] == params.rrf_k
            and _weights(config["leg_weights"]) == _weights(params.leg_weights))


def choose(configs: Sequence[dict], baseline: FusionParams, *,
           min_gain: float = 0.005) -> FusionParams:
    """The best mean ndcg@10 (first on ties), unless it beats the baseline by less than
    `min_gain`: 40 dev queries are too few to trust a smaller gain (spec 11.5)."""
    base = next((c["ndcg@10"] for c in configs if _is(c, baseline)), None)
    if base is None:
        raise ValueError(f"baseline {baseline.rrf_k}/{_weights(baseline.leg_weights)} "
                         "is not among the configs")
    best = max(configs, key=lambda c: c["ndcg@10"])
    if best["ndcg@10"] - base < min_gain:
        return baseline
    return FusionParams(best["rrf_k"], dict(best["leg_weights"]), baseline.vector_top_k,
                        baseline.report_top_k, baseline.image_top_k)


def _gold_run(rows: Sequence[BenchmarkQuery], mode: str, params: FusionParams,
                     **shared) -> dict:
    report = evaluate_retrieval(rows, parsers=["gold"], modes=[mode],
                                policies=[DEFAULT_POLICY], fusion=params, **shared)
    return report["parsers"]["gold"]["modes"][mode]


def grid_search(rows: Sequence[BenchmarkQuery], *, state, parser: QueryParser,
                matrix: lf.LabelMatrix, agreement: AgreementMatrix,
                projection_of: Mapping[str, str], baseline: FusionParams,
                search=pipeline.run_search) -> dict:
    """Hybrid on gold queries at the default policy, once per GRID entry; the chosen entry
    must beat `baseline` (the running fusion) by `min_gain`. Dev only: the heldout split
    never runs a grid (spec 11.5)."""
    shared = {"state": state, "parser": parser, "matrix": matrix, "agreement": agreement,
              "projection_of": projection_of, "search": search}
    configs = []
    for params in GRID:
        metrics = _gold_run(rows, "hybrid", params, **shared)["metrics"]
        configs.append({"rrf_k": params.rrf_k, "leg_weights": dict(params.leg_weights),
                        "ndcg@10": metrics["ndcg@10"], "p@10": metrics["p@10"]})
    chosen = choose(configs, baseline)
    return {"configs": configs, "chosen": next(c for c in configs if _is(c, chosen)),
            "baseline_ndcg": next(c["ndcg@10"] for c in configs if _is(c, baseline))}


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def image_vs_vector(report: dict, *, k: int = TOP_K) -> dict | None:
    """Gold-parser image against vector on the queries vector answered (returned > 0), so the
    image's wins on queries vector left empty do not flatter it; those are reported apart.
    `s2_passed` needs a better mean ndcg and, when any query excludes, fewer violations."""
    modes = report["parsers"].get("gold", {}).get("modes", {})
    if "image" not in modes or "vector" not in modes:
        return None
    image, vector = modes["image"]["per_query"], modes["vector"]["per_query"]
    ndcg, violation = f"ndcg@{k}", f"exclude_violation@{k}"
    common = [i for i, m in vector.items() if m[f"returned@{k}"] > 0]
    unanswered = [i for i in vector if i not in common]
    diffs = [image[i][ndcg] - vector[i][ndcg] for i in common]
    # Same ids on both sides, or the two rates are not comparable.
    scored = [i for i in common
              if vector[i][violation] is not None and image[i][violation] is not None]
    viol = {"image": _mean([image[i][violation] for i in scored]),
            "vector": _mean([vector[i][violation] for i in scored])}
    label = modes.get("label", {}).get("per_query")
    mean = {"image": _mean([image[i][ndcg] for i in common]),
            "vector": _mean([vector[i][ndcg] for i in common])}
    violation_ok = viol["vector"] is None or viol["image"] < viol["vector"]
    return {"n_common": len(common), ndcg: mean,
            "wins": sum(d > 1e-9 for d in diffs), "ties": sum(abs(d) <= 1e-9 for d in diffs),
            "losses": sum(d < -1e-9 for d in diffs), violation: viol,
            "vector_unanswered": {
                "n": len(unanswered),
                ndcg: {"image": _mean([image[i][ndcg] for i in unanswered]),
                       "label": _mean([label[i][ndcg] for i in unanswered]) if label else None}},
            "s2_passed": bool(common) and mean["image"] > mean["vector"] and violation_ok}


def mode_comparison(report: dict, new_mode: str, against: Sequence[str] = ("image", "vector"),
                    *, k: int = TOP_K) -> dict | None:
    """Gold-parser `new_mode` against each mode of `against` in the report, on the queries both
    answered (returned > 0): binary ndcg wins / ties / losses and means, and the HNRR means
    over the queries where both have one. None when `new_mode` was not run."""
    modes = report["parsers"].get("gold", {}).get("modes", {})
    if new_mode not in modes:
        return None
    new = modes[new_mode]["per_query"]
    ndcg, hnrr, returned = f"ndcg_binary@{k}", f"hnrr@{k}", f"returned@{k}"
    out = {}
    for other in against:
        if other not in modes:
            continue
        base = modes[other]["per_query"]
        common = [i for i in new if i in base and new[i][returned] > 0 and base[i][returned] > 0]
        diffs = [new[i][ndcg] - base[i][ndcg] for i in common]
        rated = [i for i in common if new[i][hnrr] is not None and base[i][hnrr] is not None]
        out[other] = {
            "n_common": len(common),
            ndcg: {new_mode: _mean([new[i][ndcg] for i in common]),
                   other: _mean([base[i][ndcg] for i in common])},
            "wins": sum(d > 1e-9 for d in diffs), "ties": sum(abs(d) <= 1e-9 for d in diffs),
            "losses": sum(d < -1e-9 for d in diffs),
            "n_hnrr": len(rated),
            hnrr: {new_mode: _mean([new[i][hnrr] for i in rated]),
                   other: _mean([base[i][hnrr] for i in rated])},
        }
    return out


def ablation(rows: Sequence[BenchmarkQuery], *, state, parser: QueryParser,
             matrix: lf.LabelMatrix, agreement: AgreementMatrix,
             projection_of: Mapping[str, str], fusion: FusionParams,
             search=pipeline.run_search) -> dict:
    """Which leg earns its place: the single-leg modes, hybrid with one leg only (the label
    filter stays on), the shipped-before fusion and the running one. Gold queries, default
    policy; dev only."""
    def with_weights(rrf_k: int, vector: float, report: float, image: float) -> FusionParams:
        return FusionParams(rrf_k, {"vector": vector, "report": report, "image": image},
                            fusion.vector_top_k, fusion.report_top_k, fusion.image_top_k)

    runs = {"label": ("label", fusion), "vector": ("vector", fusion), "image": ("image", fusion),
            "filter+vector": ("hybrid", with_weights(fusion.rrf_k, 1.0, 0.0, 0.0)),
            "filter+report": ("hybrid", with_weights(fusion.rrf_k, 0.0, 1.0, 0.0)),
            "filter+image": ("hybrid", with_weights(fusion.rrf_k, 0.0, 0.0, 1.0)),
            "hybrid_main": ("hybrid", with_weights(10, 0.5, 1.0, 0.0)),
            "hybrid": ("hybrid", fusion)}
    shared = {"state": state, "parser": parser, "matrix": matrix, "agreement": agreement,
              "projection_of": projection_of, "search": search}
    out: dict = {"n": None, "policy": _policy_name(DEFAULT_POLICY), "configs": {},
                 "per_query": {}}
    for name, (mode, params) in runs.items():
        entry = _gold_run(rows, mode, params, **shared)
        out["configs"][name] = entry["metrics"]
        out["per_query"][name] = {i: m["ndcg@10"] for i, m in entry["per_query"].items()}
        out["n"] = entry["metrics"]["n"]
    return out


def _check(value: float | None, threshold: float) -> dict:
    return {"value": value, "threshold": threshold,
            "passed": None if value is None else value >= threshold}


def verdict(report: dict, thresholds: dict) -> dict:
    """Spec 11.4, from the `gold` parser at the main policy. A check without a value (vector
    ndcg 0, no single-concept query) is None, and the verdict passes only when all three do."""
    modes = report["parsers"]["gold"]["modes"]
    hybrid = modes["hybrid"]
    vector_ndcg = modes["vector"]["metrics"]["ndcg@10"]
    gain = ((hybrid["metrics"]["ndcg@10"] - vector_ndcg) / vector_ndcg
            if vector_ndcg else None)
    out = {
        "single_p10": _check(hybrid["by_group"].get("single", {}).get("p@10"),
                             thresholds["simple_label_precision10_min"]),
        "constraint_satisfaction": _check(hybrid["metrics"]["p@10"],
                                          thresholds["constraint_satisfaction_min"]),
        "hybrid_gain_over_vector": _check(gain, thresholds["hybrid_ndcg10_relative_gain_min"]),
    }
    out["passed"] = all(check["passed"] is True for check in out.values())
    return out
