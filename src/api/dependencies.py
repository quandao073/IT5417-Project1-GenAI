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
from src.query import llm_parser as llm
from src.query.parser import QueryParser
from src.query.rule_parser import RuleParser
from src.retrieval.label_filter import LabelMatrix, build_label_matrix


@dataclass(frozen=True)
class ImageRecord:
    path_to_image: str
    study_id: str
    projection: str


@dataclass
class AppState:
    settings: Settings
    manifest: dict | None = None
    encoder: object | None = None
    client: object | None = None
    collection: str | None = None
    images: dict[str, ImageRecord] = field(default_factory=dict)
    labels: LabelMatrix | None = None
    study_image: dict[str, str] = field(default_factory=dict)
    parser: QueryParser | None = None
    llm_status: str = "disabled"
    ready: bool = False
    not_ready_reason: str | None = None
    encode_lock: threading.Lock = field(default_factory=threading.Lock)


def _default_client(url: str):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, timeout=30)


def _default_encoder(name: str, candidate: dict, device: str):
    from src.embeddings.base import load_encoder

    return load_encoder(name, candidate, device)


def build_query_parser(settings: Settings, *, llm_factory=None,
                       warm_up: bool = False) -> tuple[QueryParser, str]:
    """The parser and the LLM status (ok | disabled | unreachable | missing_model).
    The LLM never blocks readiness: any failure leaves a rule-only parser."""
    models_cfg = load_config("models")
    llm_cfg = models_cfg["llm"]
    concepts = load_config("concepts")
    template = models_cfg["image_encoder"]["text_prompt_template"]
    rule = RuleParser(concepts)
    if settings.llm_disabled:
        return QueryParser(rule, template=template), "disabled"
    common = {"num_predict": llm_cfg["num_predict"], "seed": llm_cfg["seed"],
              "keep_alive": llm_cfg["keep_alive"]}
    if llm_factory is not None:
        client = llm_factory(settings.llm_base_url, settings.llm_model,
                             settings.llm_timeout_seconds, **common)
    else:
        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        try:
            client = llm.make_client(settings.llm_provider, settings.llm_base_url,
                                     settings.llm_model, settings.llm_timeout_seconds,
                                     api_key=key, **common,
                                     options=llm.request_options(llm_cfg, settings.llm_model))
        except ValueError as exc:   # a missing key is a slip, not a reason to refuse search
            warning = f"llm misconfigured: {exc}; rule parser only"
            return QueryParser(rule, template=template, startup_warning=warning), "unreachable"
    status = client.probe()
    if status != "ok":
        return QueryParser(rule, template=template,
                           startup_warning=f"llm {status} at startup; rule parser only"), status
    parser = QueryParser(rule, template=template, llm_client=client,
                         system_prompt=llm.load_prompt(llm_cfg["prompt_file"], concepts),
                         max_retries=llm_cfg["max_retries"], cache_size=llm_cfg["cache_size"])
    if warm_up:
        # The first Ollama call after install took 72 s; after a fresh server start on
        # 2026-10-14 it took 5.3 s (WORKLOG T16). Either can exceed the 20 s timeout or
        # delay the first user query, so pay it off the request path.
        threading.Thread(target=parser.parse, args=("cardiomegaly",), daemon=True).start()
    return parser, status


def build_state(settings: Settings, *, encoder_loader=None, client_factory=None,
                llm_factory=None, warm_up: bool = False) -> AppState:
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
        rows = images.select("image_id", "path_to_image", "study_id",
                             "projection").iter_rows(named=True)
        records = {r["image_id"]: ImageRecord(r["path_to_image"], r["study_id"], r["projection"])
                   for r in rows}
        if len(records) != manifest["counts"]["images"]:
            raise RuntimeError(f"images.parquet has {len(records)} rows, manifest says "
                               f"{manifest['counts']['images']}")

        studies = pl.read_parquet(settings.canonical_dir / "studies.parquet")
        if studies.height != manifest["counts"]["studies"]:
            raise RuntimeError(f"studies.parquet has {studies.height} rows, manifest says "
                               f"{manifest['counts']['studies']}")
        study_image = dict(zip(studies["study_id"], studies["selected_image_id"], strict=True))
        retrieval_cfg = load_config("retrieval")
        labels = build_label_matrix(
            pl.read_parquet(settings.canonical_dir / "labels.parquet"),
            studies["study_id"].to_list(),
            retrieval_cfg["label_resolution"]["source_priority"],
            seed=retrieval_cfg["results"]["label_order_seed"])
        parser, llm_status = build_query_parser(settings, llm_factory=llm_factory,
                                                warm_up=warm_up)

        cfg = load_config("models")["image_encoder"]
        encoder = encoder_loader(cfg["selected"], cfg["candidates"][cfg["selected"]],
                                 settings.embedding_device)
        if encoder.revision != manifest["embedding_model"]["revision"]:
            raise RuntimeError(f"encoder revision {encoder.revision} != manifest "
                               f"{manifest['embedding_model']['revision']}")

        state.manifest, state.client, state.collection = manifest, client, collection
        state.images, state.encoder, state.ready = records, encoder, True
        state.labels, state.study_image = labels, study_image
        state.parser, state.llm_status = parser, llm_status
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
