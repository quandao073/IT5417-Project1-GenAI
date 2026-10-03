"""BiomedCLIP (PubMedBERT + ViT-B/16), an open_clip checkpoint on HF."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.embeddings.base import dir_bytes, l2_normalize, load_rgb

_FILES = [
    "open_clip_config.json",
    "open_clip_pytorch_model.bin",
    "tokenizer*",
    "vocab.txt",
    "special_tokens_map.json",
]


class BiomedCLIPEncoder:
    name = "biomedclip"

    def __init__(self, candidate: dict, device: str = "cpu"):
        import open_clip
        import torch
        from huggingface_hub import snapshot_download

        self.model_id = candidate["model_id"]
        self.revision = candidate["revision"]
        self.license = candidate["license"]
        self.device = device
        self._torch = torch

        # open_clip 2.24's hf-hub: loader ignores revisions, so fetch the pinned
        # snapshot ourselves and load from local files.
        local = Path(
            snapshot_download(self.model_id, revision=self.revision, allow_patterns=_FILES)
        )
        cfg = json.loads((local / "open_clip_config.json").read_text(encoding="utf-8"))
        arch = f"biomedclip-{self.revision[:12]}"
        # Private open_clip 2.24 internal; re-check on any version bump.
        open_clip.factory._MODEL_CONFIGS[arch] = cfg["model_cfg"]
        self._model = open_clip.create_model(
            arch, pretrained=str(local / "open_clip_pytorch_model.bin"), device=device
        ).eval()

        size = cfg["model_cfg"]["vision_cfg"]["image_size"]
        mean, std = tuple(cfg["preprocess_cfg"]["mean"]), tuple(cfg["preprocess_cfg"]["std"])
        self._transform = open_clip.image_transform(size, is_train=False, mean=mean, std=std)
        context = cfg["model_cfg"]["text_cfg"]["context_length"]
        self._tokenizer = open_clip.tokenizer.HFTokenizer(str(local), context_length=context)

        self.weights_bytes = dir_bytes(local)
        self.preprocessing = {
            "image_size": size,
            "resize": "open_clip image_transform: shortest side, bicubic, center crop",
            "mean": list(mean),
            "std": list(std),
            "context_length": context,
            "tokenizer": f"{self.model_id}@{self.revision}",
            "text_arch_config_unpinned": cfg["model_cfg"]["text_cfg"]["hf_model_name"],
        }
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode_texts(["chest x-ray"]).shape[1])
        return self._dim

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        batch = self._torch.stack([self._transform(load_rgb(p)) for p in paths]).to(self.device)
        with self._torch.inference_mode():
            return l2_normalize(self._model.encode_image(batch).float().cpu().numpy())

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        tokens = self._tokenizer(list(texts)).to(self.device)
        with self._torch.inference_mode():
            return l2_normalize(self._model.encode_text(tokens).float().cpu().numpy())
