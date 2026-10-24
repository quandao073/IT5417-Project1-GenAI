"""GET /api/v1/studies/{study_id} (spec 5.1): image, view, the 14 resolved labels and the report.

Study ids contain a slash (`patient00001/study1`); the route takes the rest of the path, so both
the raw id and the percent-encoded one the UI sends resolve. The report comes from the canonical
`reports.parquet` (the FTS index keeps only `search_text`), one row read per request.
"""

from __future__ import annotations

import logging
from typing import Annotated

import polars as pl
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import AppState, require_ready
from src.api.routes.search import projection_of
from src.api.schemas.search import LabelAssertion
from src.api.schemas.studies import StudyDetail, StudyReport
from src.data.labels import CHEXPERT_LABELS
from src.retrieval.label_filter import resolved_labels

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/v1/studies/{study_id:path}", response_model=StudyDetail)
def study(study_id: str, state: Annotated[AppState, Depends(require_ready)]) -> StudyDetail:
    image_id = state.study_image.get(study_id)
    if image_id is None:
        raise HTTPException(status_code=404, detail={"reason": f"unknown study {study_id}"})
    return StudyDetail(
        study_id=study_id, image_id=image_id, projection=projection_of(state, image_id),
        labels=[LabelAssertion(**a)
                for a in resolved_labels(state.labels, study_id, CHEXPERT_LABELS)],
        report=_report(state, study_id), build_id=state.manifest["build_id"],
    )


def _report(state: AppState, study_id: str) -> StudyReport | None:
    path = state.settings.canonical_dir / "reports.parquet"
    try:
        rows = (pl.scan_parquet(path).filter(pl.col("study_id") == study_id)
                .select("impression", "findings", "search_text").collect())
    except Exception as exc:  # noqa: BLE001 - a missing report never fails the detail panel
        logger.warning("study %s: report unavailable: %s", study_id, exc)
        return None
    return StudyReport(**rows.row(0, named=True)) if rows.height else None
