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

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BUILD_CONFIGS: tuple[str, ...] = ("data", "models", "retrieval", "concepts")

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


def config_hash(
    names: Sequence[str] = BUILD_CONFIGS,
    config_dir: Path | None = None,
) -> str:
    """sha256 over the parsed contents of the named configs."""
    parsed = {name: load_config(name, config_dir) for name in sorted(names)}
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
    # Blank in .env.example: the API then reads it from build_manifest.json.
    build_id: str | None = None

    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b-instruct-q4_K_M"
    llm_timeout_seconds: int = 20
    llm_disabled: bool = False

    api_host: str = "0.0.0.0"
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
