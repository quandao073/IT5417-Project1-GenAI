"""V2-7 evaluation of the KG search (docs/kg-evaluation.md): the systems, the judge, the metrics.

Systems, all in-process (no API, no Qdrant: cosine over the build's image vectors, which is what
Qdrant computes):

    B0   the question rendered in English by a small model, encoded by CXR-CLIP, nearest images
    B1   the B1 parser, the query template (`retrieval.pipeline.query_text`), CXR-CLIP, view filter
    B2   the KG search (`KGSearch.search`), the proposal of V2
    REF  reference, reads labels: the gold's CheXpert-label constraints filter the studies by
         their resolved labels, the gold English rendering ranks them; only a ceiling
    B3   the agent ablation of V2-9: not built yet, reported as skipped

The judge sees the question, its English rendering and the report of a study (findings,
impression): never the gold, the graph's concepts or which system returned the study. Its
answers are cached per (question, study, model, prompt sha256).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.kg.expand import ABSENT, NOISY_OR, PRESENT, WEIGHTED, WEIGHTED_OR

SYSTEMS = ("B0", "B1", "B2", "REF")
PLANNED = ("B3",)


# -- the corpus ------------------------------------------------------------------------------

@dataclass
class Corpus:
    """Unit image vectors of the build, one per study, with study ids and views."""
    vectors: np.ndarray            # [n, d] float32, unit rows
    image_ids: list[str]
    study_ids: list[str]
    views: np.ndarray              # str [n]

    def top(self, query: np.ndarray, k: int, views: Sequence[str] = (),
            allowed: np.ndarray | None = None) -> list[int]:
        scores = self.vectors @ (query / (np.linalg.norm(query) or 1.0)).astype(np.float32)
        mask = np.ones(len(scores), dtype=bool) if allowed is None else allowed.copy()
        if views:
            mask &= np.isin(self.views, list(views))
        scores = np.where(mask, scores, -np.inf)
        k = min(k, int(mask.sum()))
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        return idx[np.argsort(-scores[idx], kind="stable")].tolist()


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)


# -- the judge -------------------------------------------------------------------------------

JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["grades"],
                "properties": {"grades": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["id", "grade", "violates_exclusion", "reason"],
                    "properties": {"id": {"type": "string"},
                                   "grade": {"type": "integer", "enum": [0, 1, 2]},
                                   "violates_exclusion": {"type": "boolean"},
                                   "reason": {"type": "string"}}}}}}


def report_text(findings: str | None, impression: str | None) -> dict:
    return {"findings": (findings or "").strip(), "impression": (impression or "").strip()}


class Judge:
    """Grades (question, study) pairs in batches of one question's reports; every answer is
    cached on its own key, so a pair is judged once whatever batch it came in."""

    def __init__(self, chooser, model: str, system: str, prompt_sha: str, cache_path: Path,
                 ledger, step: str, batch: int):
        self.chooser, self.model, self.system, self.prompt_sha = chooser, model, system, prompt_sha
        self.path, self.ledger, self.step, self.batch = cache_path, ledger, step, batch
        self._lock = threading.Lock()
        self.answers: dict[str, dict] = {}
        if cache_path.is_file():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.answers[r["key"]] = r

    def key(self, qid: str, study: str) -> str:
        blob = json.dumps([qid, study, self.model, self.prompt_sha])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, qid: str, study: str) -> dict | None:
        return self.answers.get(self.key(qid, study))

    def grade(self, qid: str, question: str, query_en: str, studies: Sequence[str],
              reports: dict[str, dict]) -> int:
        """Judge the pairs of this question not yet cached; returns how many were judged."""
        todo = sorted(s for s in dict.fromkeys(studies) if self.get(qid, s) is None)
        done = 0
        for start in range(0, len(todo), self.batch):
            chunk = todo[start:start + self.batch]
            self.ledger.check()
            user = json.dumps({"question": question, "query_en": query_en,
                               "reports": [{"id": s, **reports[s]} for s in chunk]},
                              ensure_ascii=False)
            content, usage, _ = self.chooser.ask(self.system, user, JUDGE_SCHEMA)
            self.ledger.add(self.step, self.model, usage)
            got = {g["id"]: g for g in json.loads(content)["grades"]}
            with self._lock, open(self.path, "a", encoding="utf-8") as handle:
                for s in chunk:
                    if s not in got:
                        continue                  # not answered: judged again next time
                    r = {"key": self.key(qid, s), "qid": qid, "study": s, "model": self.model,
                         "prompt_sha256": self.prompt_sha, "grade": int(got[s]["grade"]),
                         "violates_exclusion": bool(got[s]["violates_exclusion"]),
                         "reason": got[s]["reason"],
                         "at": datetime.now(UTC).isoformat(timespec="seconds")}
                    self.answers[r["key"]] = r
                    handle.write(json.dumps(r, ensure_ascii=False) + "\n")
                    done += 1
        return done


# -- metrics ---------------------------------------------------------------------------------

def gains(studies: Sequence[str], judged: Callable[[str], dict | None],
          view_ok: Callable[[str], bool]) -> list[int]:
    """Grade of each returned study (0 when not judged or off the asked view)."""
    out = []
    for s in studies:
        j = judged(s)
        out.append(int(j["grade"]) if j is not None and view_ok(s) else 0)
    return out


def dcg(values: Sequence[float]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(values))


def ndcg(got: Sequence[int], ideal_pool: Sequence[int], k: int) -> float:
    ideal = dcg(sorted(ideal_pool, reverse=True)[:k])
    return dcg(list(got)[:k]) / ideal if ideal > 0 else 0.0


def query_metrics(studies: Sequence[str], judged, view_ok, ideal_pool: Sequence[int],
                  k: int, has_exclusion: bool) -> dict:
    g = gains(studies[:k], judged, view_ok)
    viol = [bool((judged(s) or {}).get("violates_exclusion")) for s in studies[:k]]
    return {"p10": sum(x == 2 for x in g) / k, "p10_lenient": sum(x >= 1 for x in g) / k,
            "ndcg10": ndcg(g, ideal_pool, k), "n_results": len(studies),
            "no_result": len(studies) == 0,
            "excl_viol10": (sum(viol) / k) if has_exclusion else None}


def bootstrap_mean_diff(a: Sequence[float], b: Sequence[float], n: int,
                        seed: int) -> dict:
    """Mean of a - b over paired queries, with a 95% percentile bootstrap interval."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": len(d),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}


def cohen_kappa(x: Sequence[int], y: Sequence[int], labels=(0, 1, 2),
                weighted: bool = False) -> float | None:
    """Cohen's kappa of two raters; quadratic weights when `weighted`."""
    if not x:
        return None
    k = len(labels)
    idx = {v: i for i, v in enumerate(labels)}
    obs = np.zeros((k, k))
    for a, b in zip(x, y, strict=True):
        obs[idx[a], idx[b]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    w = (np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
         if weighted else 1 - np.eye(k))
    denom = (w * exp).sum()
    return None if denom == 0 else float(1 - (w * obs).sum() / denom)


# -- layer 2: understanding the question (B2) -----------------------------------------------

def signed_columns(intents) -> set[tuple[str, str]]:
    return {("+" if i.polarity == PRESENT else "-", c.cui) for i in intents for c in i.columns}


def gold_columns(row: dict) -> set[tuple[str, str]]:
    return ({("+", c["cui"]) for c in row["expected_scorable"]}
            | {("-", c) for c in row["exclude"]})


def set_f1(got: set, want: set) -> float:
    if not got and not want:
        return 1.0
    tp = len(got & want)
    return 0.0 if tp == 0 else 2 * tp / (len(got) + len(want))


def layer2(row: dict, intents, warning_codes: set[str]) -> dict:
    """CUI choice (each gold concept chosen, with its polarity), the column set, warnings."""
    chosen = {(i.cui, i.polarity) for i in intents if i.cui}
    gold = [(t["cui"], t["polarity"]) for t in row["terms"] if t["cui"]]
    got, want = signed_columns(intents), gold_columns(row)
    expected = set(row["expect_warnings"])
    return {"cui_hits": sum(g in chosen for g in gold), "cui_total": len(gold),
            "columns_exact": got == want, "columns_f1": set_f1(got, want),
            "warnings_all": expected <= warning_codes,
            "warnings_recall": (len(expected & warning_codes) / len(expected)) if expected
            else None,
            "warnings_extra": sorted(warning_codes - expected)}


# -- ablations of B2 (V2-8), applied to the intents of a B2 search ---------------------------

KIND_OFF = {"no_narrower": "narrower", "no_wider": "wider", "no_anatomy": "anatomy",
            "no_disease": "disease"}


def transform(intents, name: str):
    """A copy of the intents with one kind of expansion turned off (its intents keep no
    column), SUGGESTS weights made equal, or noisy-OR marked to be combined by max."""
    out = copy.deepcopy(intents)
    for i in out:
        if name in KIND_OFF and i.kind == KIND_OFF[name]:
            i.columns = []
        elif name == "suggests_equal" and i.combine == WEIGHTED and i.columns:
            i.columns = [replace(c, weight=1.0 / len(i.columns)) for c in i.columns]
        elif name == "combine_max" and i.combine in (NOISY_OR, WEIGHTED_OR) and len(i.columns) > 1:
            i.combine = "max"
    return out


def score_with_max(matrix, intents) -> np.ndarray:
    """ScoreMatrix.score, except intents marked `max`: log of the largest probability of their
    columns (present), or log(1 - largest) (absent)."""
    plain = [i for i in intents if i.combine != "max"]
    total = matrix.score(plain)
    for i in intents:
        if i.combine != "max" or not i.columns:
            continue
        z = np.stack([matrix.z[matrix.index(c.column)] for c in i.columns]).astype(np.float64)
        p = (1.0 / (1.0 + np.exp(-z))).max(axis=0)
        part = np.log(np.clip(p, 1e-12, 1.0)) if i.polarity != ABSENT else np.log(
            np.clip(1.0 - p, 1e-12, 1.0))
        total = total + part.astype(np.float32)
    return total


def needs_text(intents) -> bool:
    """The search's rule for its text route (D73): nothing wanted has a column, and something
    was wanted or nothing at all has one."""
    wanted = [i for i in intents if i.polarity == PRESENT]
    return not any(i.columns for i in wanted) and (
        bool(wanted) or not any(i.columns for i in intents))


def text_top(fallback, text_vec: np.ndarray, k: int, views: Sequence[str]) -> list[int]:
    scores = fallback.vectors @ (text_vec / (np.linalg.norm(text_vec) or 1.0)).astype(np.float32)
    if views:
        scores = np.where(np.isin(fallback.views, list(views)), scores, -np.inf)
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")].tolist()


def rescore(matrix, intents, k: int, views: Sequence[str], fallback=None,
            fallback_vec: np.ndarray | None = None) -> list[str]:
    """Study ids of the top k for these intents. When the search would take its text route,
    that route with `fallback_vec` (no fallback given: no results)."""
    if needs_text(intents) and fallback is not None and fallback_vec is not None:
        return [matrix.study_ids[r] for r in text_top(fallback, fallback_vec, k, views)]
    if not any(i.columns for i in intents):
        return []
    scores = score_with_max(matrix, intents)
    return [matrix.study_ids[r] for r in matrix.top_k(scores, k, tuple(views))]


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def fuse(matrix, intents, vectors: np.ndarray, text_vec: np.ndarray, w: float, k: int,
         views: Sequence[str]) -> list[str] | None:
    """z(KG score) + w z(cosine with the text), each over the whole corpus for this question,
    top k after the view filter; None when the KG scores nothing (the run keeps its studies)."""
    if not any(i.columns for i in intents):
        return None
    kg = zscore(score_with_max(matrix, intents))
    cos = vectors @ (text_vec / (np.linalg.norm(text_vec) or 1.0)).astype(np.float32)
    total = (kg + w * zscore(cos)).astype(np.float32)
    return [matrix.study_ids[r] for r in matrix.top_k(total, k, tuple(views))]


def seeded_sample(items: Sequence, fraction: float, seed: int) -> list:
    rng = random.Random(seed)
    n = max(1, round(len(items) * fraction)) if items else 0
    return rng.sample(list(items), n)


def percentiles(values: Sequence[float]) -> dict:
    v = [x for x in values if x is not None]
    if not v:
        return {"p50": None, "p95": None, "n": 0}
    return {"p50": round(float(np.percentile(v, 50)), 1),
            "p95": round(float(np.percentile(v, 95)), 1), "n": len(v)}
