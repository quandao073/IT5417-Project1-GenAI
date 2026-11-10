"""The B1 route inside the KG search, for a question the KG did not understand (WORKLOG D73).

When no wanted concept is left after lookup and expansion, the KG search ranks by text instead:
the caption B1 would encode (`render(include, exclude)` of the 14 labels the parser found, or the
parser's free-text `semantic_query` when there are none), its cosine with every corpus image,
the view filter. The page carries a warning and the trace says which text was used.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.query.schema import SearchQuery
from src.retrieval.templates import render


class TextFallback:
    def __init__(self, encoder, vectors: np.ndarray, views: np.ndarray, concepts_cfg: dict):
        """`vectors` [n, d]: one per matrix row, in matrix row order; `views` [n]."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)
        self.views, self.encoder, self.concepts_cfg = np.asarray(views), encoder, concepts_cfg

    def text(self, query: SearchQuery) -> str:
        include = [c.concept for c in query.include]
        if include or query.exclude:
            try:
                return render(include, list(query.exclude), self.concepts_cfg)
            except ValueError:          # "A and no A": the free text says what was asked
                pass
        return query.semantic_query

    def top(self, text: str, k: int, views: Sequence[str] = ()) -> list[tuple[int, float]]:
        """(row, cosine) of the k best rows, best first."""
        q = np.asarray(self.encoder.encode_texts([text]), dtype=np.float32)[0]
        scores = self.vectors @ (q / (np.linalg.norm(q) or 1.0))
        if views:
            scores = np.where(np.isin(self.views, list(views)), scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(int(i), float(scores[i])) for i in idx]
