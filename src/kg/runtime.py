"""Build the KG search from the files and services of a deployment (WORKLOG D66).

Missing tables or a score matrix of another encoder build is an error (the KG endpoint then answers
503 and the 14-label search keeps working). A Neo4j that is down, or loaded from other tables,
only leaves `store` None: the search then runs its 14-label route. A missing Anthropic key leaves
the rule parser and no chooser or explainer. Every version that shaped an answer is collected in
`versions`, for the response and the log (N9).
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import polars as pl

from src.common.config import Settings, default_config_dir, load_config
from src.kg import knowledge, lookup
from src.kg.guard import Guard, GuardedClient
from src.kg.llm_query import ConceptChooser, Explainer, GroundingIndex, sha256
from src.kg.pipeline import KGSearch
from src.kg.scoring import ScoreMatrix
from src.kg.store import Neo4jStore, StoreError
from src.query import llm_parser as llm
from src.query.parser import QueryParser
from src.query.rule_parser import RuleParser


def _text(relative: str) -> str:
    return (default_config_dir() / relative).read_text(encoding="utf-8")


def _neo4j_store(settings: Settings, timeout_s: float, expected: dict | None):
    """(store or None, reason): None when Neo4j cannot be reached or holds other tables."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
        connection_timeout=2, max_transaction_retry_time=0)
    store = Neo4jStore(driver, timeout_s)
    try:
        build = store.build()
    except StoreError as exc:
        driver.close()
        return None, str(exc)
    if build is None:
        driver.close()
        return None, "Neo4j holds no KGBuild: run cxr kg-load-neo4j"
    if expected and build["kg_build_id"] != expected["kg_build_id"]:
        driver.close()
        return None, (f"Neo4j holds {build['kg_build_id']}, the tables are "
                      f"{expected['kg_build_id']}: run cxr kg-load-neo4j")
    return store, ""


def build_kg_search(settings: Settings, manifest: dict, images: pl.DataFrame, *,
                    warm_up: bool = False) -> tuple[KGSearch, list[str]]:
    """(the search, notes on what is degraded at startup)."""
    cfg, models, concepts = load_config("kg"), load_config("models"), load_config("concepts")
    cfg["query"]["modifiers"] = lookup.modifier_words(cfg, _yaml("linker_overrides.yaml"))
    graph_dir = Path(settings.kg_graph_dir)
    matrix_dir = (Path(settings.kg_matrix_dir) if settings.kg_matrix_dir else
                  Path(settings.indexes_dir) / "image_scores" / f"{manifest['build_id']}_ext1")
    quality = json.loads((graph_dir / "quality.json").read_text(encoding="utf-8"))
    meta = json.loads((matrix_dir / "probe.json").read_text(encoding="utf-8"))
    if meta["build_id"] != manifest["build_id"]:
        raise RuntimeError(f"score matrix is of build {meta['build_id']}, the API serves "
                           f"{manifest['build_id']}")
    matrix = ScoreMatrix.load(matrix_dir, images)
    matrix.preload()                      # the first query costs what the next ones do (N4)
    nodes = pl.read_parquet(graph_dir / "nodes.parquet", columns=["cui", "name", "labels",
                                                                  "in_core"])
    label_ids = {label: entry["id"] for label, entry in _label_mapping().items()}
    loaded = graph_dir / "neo4j_load.json"
    expected = json.loads(loaded.read_text(encoding="utf-8")) if loaded.is_file() else None
    notes: list[str] = []
    store, why = _neo4j_store(settings, cfg["neo4j"]["query_timeout_seconds"], expected)
    if store is None:
        notes.append(f"neo4j: {why}")

    template = models["image_encoder"]["text_prompt_template"]
    rule = RuleParser(concepts)
    versions = {
        "kg_build_id": (expected or {}).get("kg_build_id"),
        "matrix_version": (expected or {}).get("matrix_version",
                                               f"{matrix_dir.name}@{_matrix_hash(matrix_dir)}"),
        "build_id": meta["build_id"], "encoder": meta["encoder"]["name"],
        "encoder_revision": meta["encoder"]["revision"]}
    key = settings.anthropic_api_key
    if settings.llm_disabled or key is None:
        notes.append("language model disabled" if settings.llm_disabled
                     else "no ANTHROPIC_API_KEY: rule parser only")
        parser = QueryParser(rule, template=template, with_terms=True)
        chooser = explainer = None
    else:
        kl, breaker = models["kg_llm"], models["kg_llm"]["breaker"]
        guard = Guard(breaker["failures"], breaker["cooldown_seconds"])

        def client(role: str, tool: tuple[str, str] | None = None):
            c = kl[role]
            return llm.AnthropicClient(
                c["model"], c["timeout_seconds"], api_key=key.get_secret_value(),
                max_tokens=c.get("max_tokens", 2048), max_retries=0, tool=tool,
                connect_timeout_s=kl["connect_timeout_seconds"],
                options=llm.request_options(models["llm"], c["model"]))

        pclient = GuardedClient(client("parser"), guard, kl["parser"]["timeout_seconds"])
        system = llm.load_prompt(models["llm"]["kg_prompt_file"], concepts,
                                 tuple(cfg["query"]["ambiguous_forms"]))
        parser = QueryParser(rule, template=template, llm_client=pclient, system_prompt=system,
                             max_retries=0, cache_size=models["llm"]["cache_size"],
                             with_terms=True)
        choose_text = _text(kl["select"]["prompt_file"])
        chooser = ConceptChooser(client("select", ("choose_concept", "Choose the concept that "
                                                   "matches the phrase.")), guard, choose_text,
                                 kl["select"]["timeout_seconds"], gloss=_gloss(),
                                 cache_file=Path(kl["select"]["cache_file"]))
        index = GroundingIndex.build(
            nodes, pl.read_parquet(graph_dir / "aliases.parquet",
                                   columns=["cui", "norm", "lang", "source"]), cfg["explain"])
        base = None
        if cfg["explain"]["knowledge"]:
            definitions = graph_dir.parent / "knowledge" / "definitions.parquet"
            base = knowledge.KnowledgeBase.load(
                definitions, pl.read_parquet(graph_dir / "edges.parquet"),
                pl.read_parquet(graph_dir / "aliases.parquet", columns=["cui", "text", "lang",
                                                                        "source"]),
                nodes, cfg["query"]["expand"], cfg["explain"]["knowledge_max_items"],
                {cui: {"en": label, "vi": concepts["concepts"][label]["vi"][0]}
                 for label, cui in label_ids.items()})
            versions["definitions"] = (f"{len(base.glosses)}@{_file_hash(definitions)}"
                                       if definitions.is_file() else None)
            if not definitions.is_file():
                notes.append(f"knowledge: {definitions} missing, path edges only")
        explain_text = _text(kl["explain"]["knowledge_prompt_file" if base else "prompt_file"])
        explainer = Explainer(client("explain", ("write_explanation", "Write the summary and "
                                                 "one line per image.")), guard, explain_text,
                              kl["explain"]["timeout_seconds"], index, base)
        versions |= {"parser_model": kl["parser"]["model"], "parser_prompt_sha256":
                     parser.prompt_sha256, "select_model": kl["select"]["model"],
                     "select_prompt_sha256": chooser.prompt_sha256,
                     "explain_model": kl["explain"]["model"],
                     "explain_prompt_sha256": explainer.prompt_sha256}
    semantic = None
    if cfg["query"]["lookup"]["sapbert"]["enabled"]:
        try:
            semantic = _alias_index(cfg, graph_dir)
            semantic.search("cardiomegaly", 1)   # the slow first encode, paid before any request
        except Exception as exc:  # noqa: BLE001 - no model, no network: lookup without the stage
            notes.append(f"sapbert: {type(exc).__name__}: {exc}"[:200])
    if semantic is not None:
        versions["lookup_sapbert"] = cfg["linker"]["sapbert"]["revision"]
    search = KGSearch(cfg=cfg, matrix=matrix, parser=parser, store=store, quality=quality,
                      label_ids=label_ids, names=dict(nodes.select("cui", "name").iter_rows()),
                      chooser=chooser, explainer=explainer, versions=versions, semantic=semantic)
    if warm_up and parser.llm is not None:
        # a TLS handshake and the prompt cache are paid off the request path
        threading.Thread(target=parser.parse, args=("cardiomegaly",), daemon=True).start()
    return search, notes


def _label_mapping() -> dict:
    import yaml

    return yaml.safe_load((default_config_dir() / "kg" / "chexpert_cuis.yaml")
                          .read_text(encoding="utf-8"))["labels"]


def _alias_index(cfg: dict, graph_dir: Path):
    from src.kg import sapbert

    aliases = pl.read_parquet(graph_dir / "aliases.parquet", columns=["cui", "text", "norm"])
    return sapbert.AliasIndex(sapbert.SapBert(cfg["linker"]["sapbert"]), aliases["cui"],
                              aliases["text"], aliases["norm"],
                              graph_dir.parent / "sapbert_alias")


def _yaml(name: str) -> dict:
    import yaml

    return yaml.safe_load((default_config_dir() / "kg" / name).read_text(encoding="utf-8"))


def _gloss() -> dict[str, str]:
    manual = _yaml("aliases_manual.yaml")
    return {cui: " ".join(text.split()) for cui, text in manual.get("gloss", {}).items()}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _matrix_hash(matrix_dir: Path) -> str:
    return sha256(str(np.load(matrix_dir / "probe.npz")["b"].tolist()))[:12]
