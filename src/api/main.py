"""FastAPI entry point. `uvicorn src.api.main:app` serves the Phase 3 vector baseline."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.dependencies import AppState, build_state
from src.api.routes import health, images, search
from src.common.config import Settings


def create_app(state: AppState | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # An injected state is attached below, so TestClient(app) works without `with`.
        if state is None:
            app.state.cxr = build_state(Settings())
        yield

    app = FastAPI(title="CXR semantic retrieval", version="0.1.0", lifespan=lifespan)
    if state is not None:
        app.state.cxr = state
    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(images.router)
    return app


app = create_app()
