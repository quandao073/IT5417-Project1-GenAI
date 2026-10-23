"""Template benchmark for the image-only modes (V1.1 WP1; design spec 5.2, roadmap 7 WP1).

Queries are rendered by `templates.generate` from concept names, so the gold is known by
construction: a gallery image is relevant when its resolved labels pass the same truth
table as the label filter (`filter_studies_by_labels`) under one absence policy. Scores
come from an injected `scorer` ([Q, N], higher is better), so the same metrics run on the
joint text-image cosine, on the decomposed per-concept logits or on a fake. Pure numpy:
nothing here touches Qdrant, torch or the LLM.

Per query: P@k = relevant in the top k / k; hit@k = 1 if any relevant in the top k;
HNRR@k = fraction of the top k with every include AND every exclude PRESENT ("A and B
both present": the model read "no B" as "B"); exclude_violation@10 = fraction of the
top 10 with any exclude PRESENT. HNRR and the violation rate are None for a query
without an exclude. HNRR and the violation rate divide by the results returned (as the
retrieval evaluation does); P@k divides by k. Ties in score go to the lower gallery row.
A query with fewer than `min_relevant` relevant images under a policy is skipped for that
policy and counted in `skipped_small_relevant`, so no cell is a ratio over nothing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src.data.labels import CHEXPERT_LABELS
from src.evaluation import zero_shot as zs
from src.retrieval import label_filter as lf
from src.retrieval import templates as tp
from src.retrieval.templates import TemplateQuery

DEFAULT_POLICY = ("exclude_present", "EXCLUDE")
STRICT_POLICY = ("explicit_absence_required", "EXCLUDE")
POLICIES = {"default": DEFAULT_POLICY, "strict": STRICT_POLICY}
DEFAULT_SEED = 20260101
_COL = {c: j for j, c in enumerate(CHEXPERT_LABELS)}

Scorer = Callable[[Sequence[TemplateQuery]], np.ndarray]   # [Q, N] scores, higher = better


@dataclass(frozen=True)
class Gallery:
    study_ids: np.ndarray              # object [N]
    statuses: np.ndarray               # object [N, 14] status names, CHEXPERT_LABELS order
    vectors: np.ndarray | None         # float32 [N, D], aligned to study_ids
    projections: np.ndarray | None     # object [N], "AP" / "PA"


def policy_name(policy: tuple[str, str] | str) -> str:
    if isinstance(policy, str):
        return policy
    names = {value: name for name, value in POLICIES.items()}
    return names.get(tuple(policy), "/".join(policy))


def label_matrix_from_statuses(study_ids: Sequence[str], statuses: np.ndarray) -> lf.LabelMatrix:
    """A LabelMatrix over status names, so relevance reuses the label filter itself."""
    statuses = np.asarray(statuses, dtype=object)
    codes = np.zeros(statuses.shape, dtype=np.int8)
    known = np.zeros(statuses.shape, dtype=bool)
    for name, code in lf.STATUS_CODE.items():
        hit = statuses == name
        codes[hit] = code
        known |= hit
    if not known.all():
        raise ValueError(f"unknown status values {sorted(set(statuses[~known].tolist()))}")
    ids = np.asarray(list(study_ids), dtype=object)
    return lf.LabelMatrix(
        study_ids=ids,
        row_of={s: i for i, s in enumerate(ids.tolist())},
        concepts=tuple(CHEXPERT_LABELS),
        col_of=dict(_COL),
        status=codes,
        source=np.full(statuses.shape, None, dtype=object),
        order=np.arange(len(ids)),
    )


def _relevant(matrix: lf.LabelMatrix, query: TemplateQuery, policy: tuple[str, str]) -> np.ndarray:
    return lf.filter_studies_by_labels(matrix, query.gold.include, query.gold.exclude,
                                       absence_policy=policy[0], uncertainty_policy=policy[1])


def relevant_mask(statuses: np.ndarray, query: TemplateQuery,
                  policy: tuple[str, str]) -> np.ndarray:
    """Bool [N]: the same truth table as `filter_studies_by_labels` under `policy`."""
    matrix = label_matrix_from_statuses(np.arange(len(statuses)).astype(str), statuses)
    return _relevant(matrix, query, policy)


def hnrr_mask(statuses: np.ndarray, query: TemplateQuery) -> np.ndarray:
    """Bool [N]: every include PRESENT and every exclude PRESENT (the negation was ignored)."""
    statuses = np.asarray(statuses, dtype=object)
    mask = np.ones(len(statuses), dtype=bool)
    for concept in (*query.include, *query.exclude):
        mask &= statuses[:, _COL[concept]] == "PRESENT"
    return mask


def violation_mask(statuses: np.ndarray, query: TemplateQuery) -> np.ndarray:
    """Bool [N]: any exclude concept PRESENT, whatever the includes."""
    statuses = np.asarray(statuses, dtype=object)
    mask = np.zeros(len(statuses), dtype=bool)
    for concept in query.exclude:
        mask |= statuses[:, _COL[concept]] == "PRESENT"
    return mask


def metric_names(ks: Sequence[int]) -> list[str]:
    names = []
    for k in ks:
        names += [f"p@{k}", f"hit@{k}", f"hnrr@{k}"]
    return names + ["exclude_violation@10"]


def query_metrics(ranked_rows: np.ndarray, relevant: np.ndarray, hnrr: np.ndarray | None,
                  violation: np.ndarray | None, *, ks: Sequence[int] = (1, 3, 5, 10)) -> dict:
    """The per-query metrics over `ranked_rows` (gallery row indices, best first). `hnrr` and
    `violation` are None for a query without an exclude; their metrics are then None too."""
    ranked_rows = np.asarray(ranked_rows)
    out: dict = {}
    for k in ks:
        top = ranked_rows[:k]
        hits = relevant[top]
        out[f"p@{k}"] = float(hits.sum() / k)
        out[f"hit@{k}"] = float(hits.any())
        out[f"hnrr@{k}"] = None if hnrr is None else float(hnrr[top].mean())
    top10 = ranked_rows[:10]
    out["exclude_violation@10"] = None if violation is None else float(violation[top10].mean())
    out["n_relevant"] = int(relevant.sum())
    return out


def _mean_ci(values: Sequence[float], *, n_boot: int, rng: np.random.Generator) -> dict:
    """Mean over queries with a bootstrap CI95 over queries."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "ci95": None, "n": 0}
    reps = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    return {"mean": float(arr.mean()), "ci95": zs.ci95(reps), "n": int(arr.size)}


def evaluate(queries: Mapping[str, Sequence[TemplateQuery]], gallery: Gallery, scorer: Scorer, *,
             policies: Sequence[tuple[str, str]] = (DEFAULT_POLICY, STRICT_POLICY),
             ks: Sequence[int] = (1, 3, 5, 10), min_relevant: int = 5, n_boot: int = 1000,
             seed: int = DEFAULT_SEED) -> dict:
    """Score every query set with `scorer` and aggregate per set / kind / policy.

    Returns {"policies", "ks", "min_relevant", "sets": {set: {kind: {policy: {metric:
    {"mean", "ci95", "n"}}}}}, "skipped_small_relevant", "per_query": {set: {id: {"kind",
    "text", "policies": {policy: metrics | None}}}}}. A skipped (query, policy) is None in
    per_query. The bootstrap is re-seeded per (set, kind, policy), so a cell's CI does not
    depend on which other sets were evaluated.
    """
    matrix = label_matrix_from_statuses(gallery.study_ids, gallery.statuses)
    names = [policy_name(p) for p in policies]
    metrics = metric_names(ks)
    sets: dict = {}
    per_query: dict = {}
    skipped = 0
    for set_name, set_queries in queries.items():
        set_queries = list(set_queries)
        scores = np.asarray(scorer(set_queries))
        want = (len(set_queries), len(gallery.study_ids))
        if scores.shape != want:
            raise ValueError(f"scorer returned shape {scores.shape}, expected {want}")
        rows: dict[str, dict] = {}
        for q, row_scores in zip(set_queries, scores, strict=True):
            ranked = np.argsort(-row_scores, kind="stable")
            hnrr = hnrr_mask(gallery.statuses, q) if q.exclude else None
            violation = violation_mask(gallery.statuses, q) if q.exclude else None
            by_policy: dict = {}
            for policy, name in zip(policies, names, strict=True):
                relevant = _relevant(matrix, q, policy)
                if int(relevant.sum()) < min_relevant:
                    skipped += 1
                    by_policy[name] = None
                    continue
                by_policy[name] = query_metrics(ranked, relevant, hnrr, violation, ks=ks)
            rows[q.id] = {"kind": q.kind, "text": q.text, "policies": by_policy}
        per_query[set_name] = rows
        sets[set_name] = {}
        for kind in tp.KINDS:
            of_kind = [r for r in rows.values() if r["kind"] == kind]
            if not of_kind:
                continue
            sets[set_name][kind] = {}
            for name in names:
                rng = np.random.default_rng(seed)
                cells = [r["policies"][name] for r in of_kind if r["policies"][name] is not None]
                sets[set_name][kind][name] = {
                    m: _mean_ci([c[m] for c in cells if c[m] is not None], n_boot=n_boot, rng=rng)
                    for m in metrics
                }
    return {"policies": names, "ks": list(ks), "min_relevant": min_relevant, "n_boot": n_boot,
            "seed": seed, "sets": sets, "skipped_small_relevant": skipped,
            "per_query": per_query}


def paired_delta(a: Mapping, b: Mapping, *, set_name: str, kinds: Sequence[str],
                 policy: tuple[str, str] | str, metric: str, n_boot: int = 1000,
                 seed: int = DEFAULT_SEED) -> dict:
    """b - a of `metric`, paired by query id over `kinds` of one set and one policy, with a
    bootstrap CI95 over queries (K1: the lower bound must be > 0). Queries skipped or None
    in either report are dropped."""
    name = policy_name(policy)
    rows_a, rows_b = a["per_query"][set_name], b["per_query"][set_name]
    va, vb = [], []
    for qid, row in rows_a.items():
        if row["kind"] not in kinds:
            continue
        ma, mb = row["policies"].get(name), rows_b[qid]["policies"].get(name)
        if ma is None or mb is None or ma[metric] is None or mb[metric] is None:
            continue
        va.append(ma[metric])
        vb.append(mb[metric])
    if not va:
        return {"mean": None, "ci95": None, "n": 0, "a": None, "b": None}
    diff = np.asarray(vb, dtype=np.float64) - np.asarray(va, dtype=np.float64)
    reps = diff[np.random.default_rng(seed).integers(0, diff.size, size=(n_boot, diff.size))]
    return {"mean": float(diff.mean()), "ci95": zs.ci95(reps.mean(axis=1)), "n": int(diff.size),
            "a": float(np.mean(va)), "b": float(np.mean(vb))}


def query_sets(concepts_cfg: dict, training_cfg: dict, *,
               seed: int = DEFAULT_SEED) -> dict[str, list[TemplateQuery]]:
    """R10 (the paper's ten labels: single, pair, negation), S12 (the training concepts, same
    kinds), paraphrase (the paper's labels, `render_paraphrase`) and multi (60 three-concept
    queries over the training concepts, half 'a and b and c', half 'a and b and no c')."""
    paper = list(training_cfg["paper_concepts"])
    system = tp.training_concepts(training_cfg)
    core = ["single", "pair", "negation"]
    return {
        "R10": tp.generate(paper, core, concepts_cfg, seed=seed),
        "S12": tp.generate(system, core, concepts_cfg, seed=seed),
        "paraphrase": tp.generate(paper, ["paraphrase"], concepts_cfg, seed=seed),
        "multi": tp.generate(system, ["triple", "pair_negation"], concepts_cfg, seed=seed,
                             n_multi=60),
    }


def study_ids_sha256(study_ids: Sequence[str]) -> str:
    """Fingerprint of which studies form the gallery, independent of row order."""
    return hashlib.sha256("\n".join(sorted(str(s) for s in study_ids)).encode()).hexdigest()
