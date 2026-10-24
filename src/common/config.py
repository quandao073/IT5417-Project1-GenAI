"""YAML config loading, the build config hash, and runtime settings.

`config_hash` goes into build_manifest.json so a built index can be told apart
from one produced under different settings. It hashes the *parsed* configs, not
the files, so editing a comment does not invalidate an index that is still
correct. Only the configs that actually shape a build are included: logging and
evaluation cannot change what lands in Qdrant.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# What shapes the vectors in Qdrant: the corpus definition and the image encoder.
# retrieval.yaml, concepts.yaml and models.yaml:llm are query-time parameters that
# Phase 4-5 tune freely; hashing them would fail validate-indexes on every edit.
# A `name:section` selector hashes one top-level section of a config.
VECTOR_INDEX_CONFIGS: tuple[str, ...] = ("data", "models:image_encoder")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def default_config_dir() -> Path:
    """Locate configs/: an explicit override, the working directory, then the repo."""
    if override := os.getenv("CONFIG_DIR"):
        return Path(override)
    local = Path.cwd() / "configs"
    return local if local.is_dir() else _REPO_ROOT / "configs"


def load_config(name: str, config_dir: Path | None = None) -> dict:
    """Load `configs/<name>.yaml`. `name` is a bare name, never a path."""
    if name != Path(name).name or name in {"", ".", ".."}:
        raise ValueError(f"Config name must be a bare name, got {name!r}")
    path = (config_dir or default_config_dir()) / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _select(selector: str, config_dir: Path | None) -> object:
    """`data` -> the whole parsed file; `models:image_encoder` -> that section."""
    name, _, section = selector.partition(":")
    parsed = load_config(name, config_dir)
    if not section:
        return parsed
    if not isinstance(parsed, dict) or section not in parsed:
        raise KeyError(f"{name}.yaml has no top-level section {section!r}")
    return parsed[section]


def config_hash(
    names: Sequence[str] = VECTOR_INDEX_CONFIGS,
    config_dir: Path | None = None,
) -> str:
    """sha256 over the parsed contents of the named configs or config sections."""
    parsed = {selector: _select(selector, config_dir) for selector in sorted(names)}
    canonical = json.dumps(parsed, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Settings(BaseSettings):
    """Runtime settings. Defaults suit a local run; .env overrides them in Docker."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_prefix: str = "cxr_images"

    image_root: Path = Path("data/raw/CheXpert-v1.0-small")
    canonical_dir: Path = Path("data/canonical")
    fts_db_path: Path = Path("data/indexes/report_search.sqlite")
    indexes_dir: Path = Path("data/indexes")
    image_probe_dir: Path = Path("data/indexes/image_probe")
    # The image modes (WORKLOG D31, D34): the second collection's manifest, written by
    # `cxr index --role image`, and the decomposed scores of `cxr build-image-scores`.
    image_build_manifest: Path = Path("data/indexes/build_manifest_image.json")
    image_scores_dir: Path = Path("data/indexes/image_scores")
    # Blank in .env.example: the API then reads it from build_manifest.json.
    build_id: str | None = None

    # OpenAI by default (WORKLOG D22, replacing the local qwen 3B of D5). `openai` is
    # any OpenAI-compatible API, so Gemini works too through its OpenAI endpoint.
    # `ollama` is still available: LLM_PROVIDER=ollama, LLM_BASE_URL=http://localhost:11434,
    # LLM_MODEL=qwen2.5:3b-instruct-q4_K_M. Without a key the parser runs rule-only.
    llm_provider: Literal["ollama", "openai"] = "openai"
    llm_base_url: str = "https://api.openai.com/v1"
    # A dated snapshot, never an alias, so a parse is reproducible; options per model
    # live in models.yaml:llm.hosted_request.
    llm_model: str = "gpt-5-mini-2025-08-07"
    # SecretStr keeps the key out of repr() and therefore out of logs and tracebacks.
    # Read from LLM_API_KEY, else the conventional OPENAI_API_KEY.
    llm_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("llm_api_key", "openai_api_key"))
    llm_timeout_seconds: int = 20
    llm_disabled: bool = False

    api_host: str = "0.0.0.0"
    # Origins the web UI is served from (Next.js dev and the compose `web` service publish :3000).
    # The browser calls the API directly (web/lib/api.ts), so without CORS every call is blocked.
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    api_port: int = 8000
    log_level: str = "INFO"
    disclaimer_text: str = "Research/education POC - not for diagnosis"

    embedding_model: str | None = None
    embedding_batch_size: int = 64
    embedding_device: str = "cpu"

    @field_validator("build_id", "embedding_model", mode="after")
    @classmethod
    def _blank_is_unset(cls, value: str | None) -> str | None:
        return value or None

    @field_validator("llm_api_key", mode="after")
    @classmethod
    def _blank_key_is_unset(cls, value: SecretStr | None) -> SecretStr | None:
        return value if value and value.get_secret_value() else None
