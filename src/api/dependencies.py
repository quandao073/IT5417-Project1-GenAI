"""Everything the API needs at request time, built once at startup.

If the manifest, the Qdrant collection and the encoder do not agree, the app
still starts but refuses to serve (plan section 16): /health answers 503 with
the reason, which is how the Docker healthcheck notices a stale index.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

import polars as pl
from fastapi import HTTPException, Request

from src.common.config import Settings, load_config
from src.embeddings import index


@dataclass(frozen=True)
class ImageRecord:
    path_to_image: str
    study_id: str


@dataclass
class AppState:
    settings: Settings
    manifest: dict | None = None
    encoder: object | None = None
    client: object | None = None
    collection: str | None = None
    images: dict[str, ImageRecord] = field(default_factory=dict)
    ready: bool = False
    not_ready_reason: str | None = None
    encode_lock: threading.Lock = field(default_factory=threading.Lock)


def _default_client(url: str):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, timeout=30)


def _default_encoder(name: str, candidate: dict, device: str):
    from src.embeddings.base import load_encoder

    return load_encoder(name, candidate, device)


def build_state(settings: Settings, *, encoder_loader=None, client_factory=None) -> AppState:
    state = AppState(settings=settings)
    encoder_loader = encoder_loader or _default_encoder
    client_factory = client_factory or _default_client
    try:
        manifest_path = settings.indexes_dir / index.MANIFEST_FILE
        if not manifest_path.is_file():
            raise RuntimeError(f"{manifest_path} not found: run cxr index")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if settings.build_id and settings.build_id != manifest["build_id"]:
            raise RuntimeError(
                f"BUILD_ID {settings.build_id} != manifest {manifest['build_id']}")
        collection = index.collection_name(settings.qdrant_collection_prefix, manifest["build_id"])

        client = client_factory(settings.qdrant_url)
        if not client.collection_exists(collection):
            raise RuntimeError(f"collection {collection} does not exist in Qdrant")
        n = client.count(collection).count
        if n != manifest["counts"]["vectors"]:
            raise RuntimeError(
                f"collection has {n} vectors, manifest says {manifest['counts']['vectors']}")

        images = pl.read_parquet(settings.canonical_dir / "images.parquet")
        rows = images.select("image_id", "path_to_image", "study_id").iter_rows(named=True)
        records = {r["image_id"]: ImageRecord(r["path_to_image"], r["study_id"]) for r in rows}
        if len(records) != manifest["counts"]["images"]:
            raise RuntimeError(f"images.parquet has {len(records)} rows, manifest says "
                               f"{manifest['counts']['images']}")

        cfg = load_config("models")["image_encoder"]
        encoder = encoder_loader(cfg["selected"], cfg["candidates"][cfg["selected"]],
                                 settings.embedding_device)
        if encoder.revision != manifest["embedding_model"]["revision"]:
            raise RuntimeError(f"encoder revision {encoder.revision} != manifest "
                               f"{manifest['embedding_model']['revision']}")

        state.manifest, state.client, state.collection = manifest, client, collection
        state.images, state.encoder, state.ready = records, encoder, True
    except Exception as exc:  # noqa: BLE001 - every failure is a readiness reason
        state.ready = False
        state.not_ready_reason = str(exc)
    return state


def get_state(request: Request) -> AppState:
    return request.app.state.cxr


def require_ready(request: Request) -> AppState:
    state = get_state(request)
    if not state.ready:
        raise HTTPException(status_code=503,
                            detail={"status": "not_ready", "reason": state.not_ready_reason})
    return state
