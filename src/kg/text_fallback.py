"""The B1 route inside the KG search, for a question the KG did not understand (WORKLOG D73).

When no wanted concept is left after lookup and expansion, the KG search ranks by text instead:
the caption B1 would encode (`render(include, exclude)` of the 14 labels the parser found), or,
when the parser found none, the English of the question as B0 encodes it (src/kg/translate.py;
never the parser's `semantic_query`, which can be `""`: T44); its cosine with every corpus image,
the view filter. The page carries a warning and the trace says which text was used.

`fuse` is the score fusion of D74, locked before heldout (D75): z(KG score) + w z(cosine with the
English), each z over the whole corpus for the question, the view filter after.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.kg.translate import usable
from src.query.schema import SearchQuery
from src.retrieval.templates import render


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def fuse(kg_scores: np.ndarray, cosine: np.ndarray, w: float) -> np.ndarray:
    return (zscore(kg_scores) + w * zscore(cosine)).astype(np.float32)


class TextFallback:
    def __init__(self, encoder, vectors: np.ndarray, views: np.ndarray, concepts_cfg: dict):
        """`vectors` [n, d]: one per matrix row, in matrix row order; `views` [n]."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)
        self.views, self.encoder, self.concepts_cfg = np.asarray(views), encoder, concepts_cfg

    def text(self, query: SearchQuery, english: str | None, question: str) -> str:
        """The caption of the labels, else the English, else the question as written: never
        blank."""
        include = [c.concept for c in query.include]
        if include or query.exclude:
            try:
                return render(include, list(query.exclude), self.concepts_cfg)
            except ValueError:          # "A and no A": the words of the question say it
                pass
        return usable(english) or question.strip()

    def cosine(self, text: str) -> np.ndarray:
        q = np.asarray(self.encoder.encode_texts([text]), dtype=np.float32)[0]
        return self.vectors @ (q / (np.linalg.norm(q) or 1.0))

    def top(self, text: str, k: int, views: Sequence[str] = ()) -> list[tuple[int, float]]:
        """(row, cosine) of the k best rows, best first."""
        scores = self.cosine(text)
        if views:
            scores = np.where(np.isin(self.views, list(views)), scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(int(i), float(scores[i])) for i in idx]
