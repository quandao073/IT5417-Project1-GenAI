"""Thin wrapper over `src.embeddings.cxrclip` for training: the converted checkpoint, the
tokenizer and the config, built as the adapter builds them so the export loads back (H3)."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer, BertConfig, SwinConfig

from src.embeddings.cxrclip import CXRClipModule, build_model, load_state

CONFIG_FILES = ["config.json", "vocab.txt", "tokenizer*", "special_tokens_map.json"]


def resolve_config_dir(spec: str | Path) -> Path:
    """A local directory as is; '<repo_id>@<commit sha>' through the pinned HF snapshot."""
    path = Path(spec)
    if path.is_dir():
        return path
    repo, _, revision = str(spec).partition("@")
    if not revision:
        raise ValueError(f"{spec!r} is neither a directory nor '<repo_id>@<commit sha>'")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=CONFIG_FILES))


def load_base(checkpoint: Path, swin_cfg_dir: Path, bert_cfg_dir: Path
              ) -> tuple[CXRClipModule, object, dict]:
    """(model with the checkpoint's weights, tokenizer, the checkpoint's plain config)."""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    config = payload["config"]
    model = build_model(config["model"], SwinConfig.from_pretrained(swin_cfg_dir),
                        BertConfig.from_pretrained(bert_cfg_dir))
    load_state(model, payload["model"])
    # eval: deterministic until the loop calls .train(); dropout and drop-path are off.
    return model.eval(), AutoTokenizer.from_pretrained(bert_cfg_dir), config
