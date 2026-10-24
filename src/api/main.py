"""FastAPI entry point. `uvicorn src.api.main:app` serves the Phase 3 vector baseline."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import AppState, build_state
from src.api.routes import health, images, query, search, studies
from src.common.config import Settings


def create_app(state: AppState | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # An injected state is attached below, so TestClient(app) works without `with`.
        if state is None:
            app.state.cxr = build_state(Settings(), warm_up=True)
        yield

    app = FastAPI(title="CXR semantic retrieval", version="0.1.0", lifespan=lifespan)
    if state is not None:
        app.state.cxr = state
    origins = (state.settings if state is not None else Settings()).cors_origins
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])
    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(query.router)
    app.include_router(images.router)
    app.include_router(studies.router)
    return app


app = create_app()
