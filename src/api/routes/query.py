"""POST /api/v1/query/parse - the parsed query, flat, with how it was obtained."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import AppState, require_ready
from src.api.schemas.search import ParsedQuery, ParseRequest

router = APIRouter()


@router.post("/api/v1/query/parse", response_model=ParsedQuery)
def parse_query(body: ParseRequest,
                state: Annotated[AppState, Depends(require_ready)]) -> ParsedQuery:
    return ParsedQuery.from_result(state.parser.parse(body.query))
