"""`score_images` (docs/v2-kg-design.md, "Truy vấn online" step 4): rank images from the score
matrix held in RAM. It never reads Neo4j (N4); the graph only chose the columns and weights.

The matrix is kept column-major ([columns, images], float32), so one concept's logits are one
contiguous row. log sigma(z) and log sigma(-z) of a column are computed when a query first needs
them (columns in parallel: numpy releases the GIL) and kept, so a column costs one pass over the
images once. An intent has a probability p and contributes

    single / noisy-OR   log p         or, negated, log(1 - p)   (noisy-OR via sums of log sigma(-z))
    weighted            sum w log sigma(z)   or, negated, log(sum w sigma(-z))

and an image's score is the sum over intents. A view filter masks rows before the top-k.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from src.kg.expand import ABSENT, NOISY_OR, SINGLE, WEIGHTED, WEIGHTED_OR, Column, Intent

_TINY = np.float32(-1e-30)


class ScoreMatrix:
    def __init__(self, columns: list[str], values: np.ndarray, study_ids: list[str],
                 image_ids: list[str], views: np.ndarray):
        if values.shape != (len(columns), len(image_ids)):
            raise ValueError(f"values {values.shape} do not fit {len(columns)} columns x "
                             f"{len(image_ids)} images")
        self.columns, self.z = list(columns), np.ascontiguousarray(values, dtype=np.float32)
        self.study_ids, self.image_ids, self.views = study_ids, image_ids, views
        self._index = {c: j for j, c in enumerate(self.columns)}
        self._lsp: dict[int, np.ndarray] = {}   # log sigma(z), per column, filled on demand
        self._lsn: dict[int, np.ndarray] = {}   # log sigma(-z)
        self._pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))

    @classmethod
    def load(cls, matrix_dir: Path, images: pl.DataFrame) -> ScoreMatrix:
        """`corpus_scores.parquet` of a score matrix and the projection (AP, PA) of each image."""
        scores = pl.read_parquet(matrix_dir / "corpus_scores.parquet")
        columns = [c for c in scores.columns if c not in ("study_id", "image_id")]
        views = (scores.select("image_id").join(images.select("image_id", "projection"),
                                                on="image_id", how="left")["projection"]
                 .fill_null("").to_numpy().astype("U2"))
        return cls(columns, scores.select(columns).to_numpy().T, scores["study_id"].to_list(),
                   scores["image_id"].to_list(), views)

    @classmethod
    def random(cls, rows: int, columns: list[str], seed: int = 1) -> ScoreMatrix:
        """A matrix of normal logits, for the benchmark."""
        rng = np.random.default_rng(seed)
        values = rng.normal(-1.0, 2.0, (len(columns), rows)).astype(np.float32)
        views = np.where(rng.random(rows) < 0.2, "PA", "AP").astype("U2")
        ids = [str(i) for i in range(rows)]
        return cls(columns, values, ids, ids, views)

    def index(self, column: str) -> int:
        return self._index[column]

    def _fill(self, store: dict[int, np.ndarray], js: list[int], sign: int) -> None:
        missing = [j for j in dict.fromkeys(js) if j not in store]

        def one(j: int) -> np.ndarray:
            z = np.clip(self.z[j] * np.float32(sign), -30, 30)
            return -np.log1p(np.exp(-z))                 # log sigma(sign * z)

        for j, value in zip(missing, self._pool.map(one, missing), strict=True):
            store[j] = value

    def preload(self) -> float:
        """log sigma(z) and log sigma(-z) of every column, now: the first query then costs what
        the next ones do (N4). Memory: two more matrices the size of `z`. Returns the ms spent."""
        started = time.perf_counter()
        every = list(range(len(self.columns)))
        self._fill(self._lsp, every, 1)
        self._fill(self._lsn, every, -1)
        return (time.perf_counter() - started) * 1000

    def _sum(self, store: dict[int, np.ndarray], columns: list[Column],
             weights: bool = False) -> np.ndarray:
        js = [self._index[c.column] for c in columns]
        total = np.zeros(self.z.shape[1], dtype=np.float32)
        for j, c in zip(js, columns, strict=True):
            total += store[j] * np.float32(c.weight if weights else 1.0)
        return total

    def contribution(self, intent: Intent) -> np.ndarray | None:
        """The intent's term of every image's score, or None for an intent with no columns."""
        if not intent.columns:
            return None
        js = [self._index[c.column] for c in intent.columns]
        absent = intent.polarity == ABSENT
        if intent.combine == SINGLE and len(js) == 1 and not absent:
            self._fill(self._lsp, js, 1)
            return self._lsp[js[0]]
        if intent.combine == WEIGHTED:
            self._fill(self._lsn if absent else self._lsp, js, -1 if absent else 1)
            if not absent:
                return self._sum(self._lsp, intent.columns, weights=True)
            out = None                                    # log sum w sigma(-z)
            for j, c in zip(js, intent.columns, strict=True):
                term = self._lsn[j] + np.float32(np.log(c.weight))
                out = term if out is None else np.logaddexp(out, term)
            return out
        if intent.combine == WEIGHTED_OR:                 # sum of log(1 - w sigma(z))
            self._fill(self._lsp, js, 1)
            s = np.zeros(self.z.shape[1], dtype=np.float32)
            for j, c in zip(js, intent.columns, strict=True):
                s += np.log1p(-np.float32(c.weight) * np.exp(self._lsp[j]))
            s = np.minimum(s, _TINY)
            return s if absent else np.log(-np.expm1(s))
        self._fill(self._lsn, js, -1)                     # noisy-OR: sum of log sigma(-z)
        s = np.minimum(self._sum(self._lsn, intent.columns), _TINY)
        return s if absent else np.log(-np.expm1(s))

    def score(self, intents: list[Intent]) -> np.ndarray:
        total = np.zeros(self.z.shape[1], dtype=np.float32)
        for intent in intents:
            part = self.contribution(intent)
            if part is not None:
                total += part
        return total

    def top_k(self, scores: np.ndarray, k: int, views: tuple[str, ...] = ()) -> np.ndarray:
        """Rows of the k best images, best first (ties by row), among the given views."""
        if views:
            scores = np.where(np.isin(self.views, list(views)), scores, -np.inf)
        k = min(k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.isfinite(scores[top])]
        return top[np.lexsort((top, -scores[top]))]

    def probabilities(self, rows: np.ndarray, intents: list[Intent]) -> list[dict]:
        """Per result row and intent: the probability of the intent and of each of its columns
        (the numbers the explanation may quote)."""
        out = []
        for row in rows:
            entry = {}
            for i, intent in enumerate(intents):
                if not intent.columns:
                    continue
                p = [float(1 / (1 + np.exp(-self.z[self._index[c.column], row])))
                     for c in intent.columns]
                if intent.combine == WEIGHTED:
                    whole = sum(c.weight * v for c, v in zip(intent.columns, p, strict=True))
                elif intent.combine == WEIGHTED_OR:
                    whole = 1 - float(np.prod([1 - c.weight * v
                                               for c, v in zip(intent.columns, p, strict=True)]))
                else:
                    whole = 1 - float(np.prod([1 - v for v in p]))
                entry[i] = {"intent": whole,
                            "columns": {c.cui: v for c, v in zip(intent.columns, p,
                                                                  strict=True)}}
            out.append(entry)
        return out


def _intent(cui: str, polarity: str, combine: str, columns: list[tuple[str, float]]) -> Intent:
    cols = [Column(cui=c, column=c, weight=w, quality=None, weak=False, borderline=False)
            for c, w in columns]
    return Intent(cui, cui, cui, polarity, "benchmark", combine, cols)


def benchmark(rows: int, columns: list[str], k: int = 10, repeats: int = 15) -> dict:
    """Wall time of score + top-k on a random matrix: a first query that touches every column
    (cold: log sigma computed) and the same query again (warm), for a typical mix of intents
    (4 intents, 8 columns) and a heavy one (every column)."""
    matrix = ScoreMatrix.random(rows, columns)
    cols = columns
    typical = [_intent("a", "PRESENT", SINGLE, [(cols[0], 1.0)]),
               _intent("b", "ABSENT", SINGLE, [(cols[1], 1.0)]),
               _intent("c", "PRESENT", NOISY_OR, [(c, 1.0) for c in cols[2:5]]),
               _intent("d", "PRESENT", WEIGHTED, [(cols[5], 0.6), (cols[6], 0.3),
                                                  (cols[7], 0.1)])]
    heavy = [_intent("h", "PRESENT", NOISY_OR, [(c, 1.0) for c in cols[:len(cols) // 2]]),
             _intent("g", "ABSENT", NOISY_OR, [(c, 1.0) for c in cols[len(cols) // 2:]])]

    def run(intents: list[Intent]) -> float:
        started = time.perf_counter()
        matrix.top_k(matrix.score(intents), k, ("PA",))
        return (time.perf_counter() - started) * 1000

    out = {"rows": rows, "columns": len(columns), "k": k,
           "matrix_mb": round(matrix.z.nbytes / 1e6, 1)}
    for name, intents in (("typical", typical), ("heavy", heavy)):
        cold = run(intents)
        warm = sorted(run(intents) for _ in range(repeats))
        out[name] = {"cold_ms": round(cold, 1), "warm_p50_ms": round(warm[len(warm) // 2], 1),
                     "warm_p95_ms": round(warm[int(0.95 * (len(warm) - 1))], 1),
                     "columns_touched": len({c.column for i in intents for c in i.columns})}
    # what the API does at startup: preload, so the first query of every kind is already warm
    matrix = ScoreMatrix.random(rows, columns)
    out["preload_ms"] = round(matrix.preload(), 1)
    out["preloaded_mb"] = round((matrix.z.nbytes + sum(a.nbytes for a in matrix._lsp.values())
                                 + sum(a.nbytes for a in matrix._lsn.values())) / 1e6, 1)
    for name, intents in (("typical", typical), ("heavy", heavy)):
        out[name]["first_after_preload_ms"] = round(run(intents), 1)
    return out
