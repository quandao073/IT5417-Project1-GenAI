"""FastAPI entry point. `uvicorn src.api.main:app` serves the Phase 3 vector baseline."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import AppState, build_state
from src.api.routes import health, images, kg, query, search, studies
from src.common.config import Settings, load_config


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
    app.add_middleware(CORSMiddleware, allow_origins=load_config("demo")["cors_origins"],
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(kg.router)
    app.include_router(query.router)
    app.include_router(images.router)
    app.include_router(studies.router)
    return app


app = create_app()
