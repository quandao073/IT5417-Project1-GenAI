"""GET /api/v1/studies/{study_id}/report - a study's report text, to compare with a result.

Display only (WORKLOG D25): no search route and no evaluation calls it, so the report never
shapes a rank. It answers only when configs/demo.yaml sets `show_report: true`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import AppState, require_ready
from src.retrieval.report_search import ReportSearchError, fetch_texts

router = APIRouter()


@router.get("/api/v1/studies/{study_id:path}/report")    # a study id holds a slash
def study_report(study_id: str, state: Annotated[AppState, Depends(require_ready)]) -> dict:
    if not state.show_report:
        raise HTTPException(status_code=403, detail={
            "status": "disabled", "reason": "configs/demo.yaml: show_report is false"})
    if state.report_status != "ok":
        raise HTTPException(status_code=503, detail={
            "status": "report_index_unavailable", "reason": state.report_reason})
    try:
        texts = fetch_texts(state.report_db, state.report_lock, [study_id])
    except ReportSearchError as exc:
        raise HTTPException(status_code=503, detail={
            "status": "report_index_unavailable", "reason": str(exc)}) from exc
    if study_id not in texts:
        raise HTTPException(status_code=404, detail="unknown study_id or no report")
    return {"study_id": study_id, "text": texts[study_id], "display_only": True}
