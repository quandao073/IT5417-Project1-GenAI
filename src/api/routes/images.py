"""GET /api/v1/images/{image_id} - serve the original JPEG by id, never by path."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from src.api.dependencies import AppState, require_ready

router = APIRouter()


@router.get("/api/v1/images/{image_id}")
def image(image_id: str, state: Annotated[AppState, Depends(require_ready)]):
    record = state.images.get(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown image_id")
    path = state.settings.image_root / record.path_to_image
    if not path.is_file():
        raise HTTPException(status_code=404, detail="image file missing on disk")
    # Images are immutable (their checksum is part of the canonical table).
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})
