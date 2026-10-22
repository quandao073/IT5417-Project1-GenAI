"""Live check of the trained image probe on the real corpus: the API state loads it, mode
`image` ranks with it, and the eighth `validate-indexes` check passes.

Run with `uv run pytest -m integration` while `docker compose up -d qdrant` is up, after
`cxr train-probe` has written data/indexes/image_probe/. Skipped when the probe is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.dependencies import build_state
from src.common.config import Settings
from src.retrieval.image_concepts import validate_image_probe
from src.retrieval.pipeline import run_search
from tests.integration.test_vector_index import qdrant  # noqa: F401 - pytest fixture

pytestmark = pytest.mark.integration
PROBE = Path("data/indexes/image_probe/probe.json")


def test_image_mode_ranks_the_real_corpus_with_the_probe(qdrant):  # noqa: F811 - the fixture
    if not PROBE.is_file():
        pytest.skip(f"{PROBE} not found; run cxr train-probe")
    settings = Settings(llm_disabled=True)
    state = build_state(settings)
    try:
        assert state.ready, state.not_ready_reason
        assert state.image_status == "ok", state.image_reason

        query = state.parser.parse("Find edema with no pleural effusion.").query
        outcome = run_search(state, query, "image", top_k=10, views=[],
                             absence_policy="exclude_present", uncertainty_policy="EXCLUDE")
        assert len(outcome.hits) == 10
        assert [h.image_rank for h in outcome.hits] == list(range(1, 11))
        scores = [h.score for h in outcome.hits]
        assert scores == sorted(scores, reverse=True)

        check = validate_image_probe(
            settings.image_probe_dir, build_id=state.manifest["build_id"],
            revision=state.manifest["embedding_model"]["revision"],
            study_ids=state.labels.study_ids.tolist())
        assert check.ok, check.detail
    finally:
        if state.report_db is not None:
            state.report_db.close()
