"""GET /api/v1/health - 200 with the build when ready, 503 with the reason otherwise."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.dependencies import AppState, get_state
from src.api.schemas.search import HealthResponse

router = APIRouter()


@router.get("/api/v1/health", response_model=HealthResponse,
            responses={503: {"description": "index, collection or encoder disagree"}})
def health(state: Annotated[AppState, Depends(get_state)]):
    if not state.ready:
        return JSONResponse(status_code=503,
                            content={"status": "not_ready", "reason": state.not_ready_reason})
    model = state.manifest["embedding_model"]
    return HealthResponse(
        status="ready", build_id=state.manifest["build_id"], model_id=model["model_id"],
        revision=model["revision"], dimension=model["dimension"],
        vectors=state.manifest["counts"]["vectors"],
        studies=state.manifest["counts"]["studies"], qdrant="ok",
        report_index=state.report_status,
        report_studies=(int(state.report_meta["studies"]) if state.report_status == "ok"
                        else None),
        llm=state.llm_status, llm_model=state.settings.llm_model,
    )
