"""Vector leg of retrieval (plan 11.2, tool contract 12.1).

Encodes the semantic query with the pinned encoder and asks Qdrant for the
nearest image vectors. The only constraint it knows is the AP/PA view facet
(WORKLOG G3); label constraints belong to the hybrid leg, because `vector` is
the baseline hybrid has to beat. Hybrid narrows the search to its label-allowed set by
passing `point_ids` (Qdrant point ids, WORKLOG D7).
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageCandidate:
    image_id: str
    study_id: str
    score: float   # cosine similarity as returned by Qdrant; a ranking signal, not a probability
    rank: int      # 1-based


def points_filter(views: Sequence[str] | None, point_ids: Sequence[str] | None):
    """View facet first, then an id allow-list; None when neither is given.

    An empty `point_ids` list is an allow-list that allows nothing, not "no restriction"."""
    from qdrant_client import models

    must = []
    if views:
        must.append(models.FieldCondition(key="projection",
                                          match=models.MatchAny(any=list(views))))
    if point_ids is not None:
        must.append(models.HasIdCondition(has_id=list(point_ids)))
    return models.Filter(must=must) if must else None


def views_filter(views: Sequence[str] | None):
    return points_filter(views, None)


def search_image_embeddings(
    semantic_query: str,
    views: Sequence[str] | None,
    candidate_k: int,
    *,
    encoder,
    client,
    collection: str,
    lock: threading.Lock | None = None,
    point_ids: Sequence[str] | None = None,
) -> list[ImageCandidate]:
    if point_ids is not None and len(point_ids) == 0:
        return []   # an empty allowed set has no candidates; never search unrestricted
    # torch inference is not thread-safe by default; the API serves sync
    # endpoints from a threadpool, so callers hand us one lock per process.
    if lock is None:
        vector = encoder.encode_texts([semantic_query])[0]
    else:
        with lock:
            vector = encoder.encode_texts([semantic_query])[0]
    response = client.query_points(
        collection, query=vector.tolist(), limit=candidate_k,
        query_filter=points_filter(views, point_ids), with_payload=["image_id", "study_id"],
    )
    return [
        ImageCandidate(image_id=p.payload["image_id"], study_id=p.payload["study_id"],
                       score=float(p.score), rank=rank)
        for rank, p in enumerate(response.points, start=1)
    ]
