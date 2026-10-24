"""Everything the API needs at request time, built once at startup.

If the manifest, the Qdrant collection and the encoder do not agree, the app
still starts but refuses to serve (plan section 16): /health answers 503 with
the reason, which is how the Docker healthcheck notices a stale index.
The report index is the exception: without it the API stays ready and serves
search without evidence (spec 9.3). The labels are not loaded: search never reads them.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Literal

import polars as pl
from fastapi import HTTPException, Request

from src.common.config import Settings, load_config
from src.embeddings import index
from src.query import llm_parser as llm
from src.query.parser import QueryParser
from src.query.rule_parser import RuleParser
from src.retrieval.evidence import EvidenceMatcher
from src.retrieval.report_index import ReportIndexError, open_report_index, read_meta

logger = logging.getLogger(__name__)


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
    concepts_cfg: dict = field(default_factory=dict)   # what `templates.render` phrases
    parser: QueryParser | None = None
    llm_status: str = "disabled"
    ready: bool = False
    not_ready_reason: str | None = None
    encode_lock: threading.Lock = field(default_factory=threading.Lock)
    # One read-only connection shared by the threadpool, serialised by report_lock.
    report_db: sqlite3.Connection | None = None
    report_lock: threading.Lock = field(default_factory=threading.Lock)
    report_status: Literal["ok", "disabled"] = "disabled"
    report_reason: str | None = None
    report_meta: dict = field(default_factory=dict)
    report_min_token_chars: int = 3
    evidence: EvidenceMatcher | None = None


def _open_report_index(state: AppState, studies: int) -> None:
    """Attach the report index; any problem disables evidence, never readiness."""
    try:
        con = open_report_index(state.settings.fts_db_path)
    except ReportIndexError as exc:
        state.report_reason = str(exc)
    else:
        meta = read_meta(con)
        if meta.get("studies") == str(studies):
            state.report_db, state.report_meta, state.report_status = con, meta, "ok"
            return
        con.close()
        state.report_reason = (f"report index has {meta.get('studies')} studies, "
                               f"manifest says {studies}; run cxr build-report-index")
    logger.warning("report index disabled: %s", state.report_reason)


def _default_client(url: str):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, timeout=30)


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
    client_factory = client_factory or _default_client
    try:
        cfg = load_config("models")["image_encoder"]
        # BUILD_ID picks a registered build; without it, the selected encoder's only build.
        manifest = index.resolve_build(
            settings.indexes_dir, build_id=settings.build_id,
            encoder=None if settings.build_id else cfg["selected"])
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
        _open_report_index(state, manifest["counts"]["studies"])
        concepts_cfg = load_config("concepts")
        parser, llm_status = build_query_parser(settings, llm_factory=llm_factory,
                                                warm_up=warm_up)

        encoder = index.build_encoder(manifest, cfg["candidates"], settings.embedding_device,
                                      loader=encoder_loader)

        state.manifest, state.client, state.collection = manifest, client, collection
        state.images, state.encoder, state.ready = records, encoder, True
        state.concepts_cfg = concepts_cfg
        state.parser, state.llm_status = parser, llm_status
        state.report_min_token_chars = (
            load_config("retrieval")["report_search"]["min_token_chars"])
        state.evidence = EvidenceMatcher(concepts_cfg)
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
