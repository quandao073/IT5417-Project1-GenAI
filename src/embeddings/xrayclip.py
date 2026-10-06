"""XrayCLIP ViT-B/16 (Stanford AIMI, released with CheXagent). A transformers
CLIPModel at 512 px, CC-BY-NC-4.0. Its training data is not documented on the
model card; see docs/model-decision.md before reading much into its scores."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.embeddings.base import dir_bytes, l2_normalize, load_rgb


class XrayCLIPEncoder:
    name = "xrayclip"

    def __init__(self, candidate: dict, device: str = "cpu"):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import CLIPModel, CLIPProcessor

        self.model_id = candidate["model_id"]
        self.revision = candidate["revision"]
        self.license = candidate["license"]
        self.device = device
        self._torch = torch

        local = Path(snapshot_download(self.model_id, revision=self.revision))
        self._model = CLIPModel.from_pretrained(local).to(device).eval()
        self._processor = CLIPProcessor.from_pretrained(local)

        image_cfg = self._processor.image_processor
        self.weights_bytes = dir_bytes(local)
        self.preprocessing = {
            "size": image_cfg.size,
            "crop_size": image_cfg.crop_size,
            "do_center_crop": image_cfg.do_center_crop,
            "resample": int(image_cfg.resample),
            "mean": list(image_cfg.image_mean),
            "std": list(image_cfg.image_std),
            "max_text_length": self._model.config.text_config.max_position_embeddings,
        }
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode_texts(["chest x-ray"]).shape[1])
        return self._dim

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        inputs = self._processor(images=[load_rgb(p) for p in paths], return_tensors="pt")
        with self._torch.inference_mode():
            features = self._model.get_image_features(
                pixel_values=inputs["pixel_values"].to(self.device)
            )
        return l2_normalize(features.float().cpu().numpy())

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self._processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True,
            max_length=self.preprocessing["max_text_length"],
        )
        with self._torch.inference_mode():
            features = self._model.get_text_features(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
            )
        return l2_normalize(features.float().cpu().numpy())
