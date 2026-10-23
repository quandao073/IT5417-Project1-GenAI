"""CXR-CLIP (Kakao Brain, 2023): Swin-Tiny + Bio_ClinicalBERT pre-trained on MIMIC-CXR.

The image-mode encoder of V1.1 (WORKLOG D30). The architecture is rebuilt here from the
authors' code (cxrclip/model/clip.py, read 2026-10-23), not from the README, because two
details decide the vectors: the image feature is token 0 of Swin's last hidden state (not
the pooler) and the text feature is the last attended token, i.e. [SEP] ("eos" pooling),
not [CLS]. Both towers are then projected linearly to 512 and L2-normalised.

The published checkpoint is a `torch.save` tar whose config is an omegaconf object, so the
main venv cannot unpickle it. `convert_checkpoint` (run once, with omegaconf available) writes
a plain file `{"model": state_dict, "config": plain dict, "source": {...}}` that `torch.load`
reads with `weights_only=True`; the candidate pins the sha256 of that file and records the
tar's. A fine-tuned export (training/lora.py) has the same layout, so one adapter loads both.

The module keeps the authors' attribute nesting (`image_encoder.image_encoder`,
`text_projection.projection`, ...) so a state dict round-trips without key remapping; the
only checkpoint key dropped is BERT's `position_ids`, a non-persistent buffer since
transformers 4.31.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.embeddings.base import l2_normalize, load_rgb
from src.embeddings.chexzero import sha256_of, verify_sha256

CONFIG_SECTIONS = ("model", "tokenizer", "transform", "base")
POSITION_IDS = "text_encoder.text_encoder.embeddings.position_ids"
MEAN = STD = (0.5, 0.5, 0.5)
DEFAULT_IMAGE_SIZE = 224
DEFAULT_TEXT_MAX_LENGTH = 256


class ImageTower(nn.Module):
    """`HuggingfaceImageEncoder` of the original repo: a Swin model, last hidden state out."""

    def __init__(self, swin_config):
        super().__init__()
        from transformers import SwinModel

        self.image_encoder = SwinModel(swin_config)
        self.out_dim = int(swin_config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(pixel_values=pixel_values)["last_hidden_state"]


class TextTower(nn.Module):
    """`HuggingfaceTextEncoder`: a BERT model (pooler included, unused), last hidden state out."""

    def __init__(self, bert_config):
        super().__init__()
        from transformers import BertModel

        self.text_encoder = BertModel(bert_config)
        self.out_dim = int(bert_config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids=input_ids,
                                 attention_mask=attention_mask)["last_hidden_state"]


class ProjectionHead(nn.Module):
    """`LinearProjectionHead`."""

    def __init__(self, embedding_dim: int, projection_dim: int):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class CXRClipModule(nn.Module):
    def __init__(self, model_cfg: dict, swin_config, bert_config):
        super().__init__()
        if model_cfg.get("text_encoder", {}).get("pooling", "eos") != "eos":
            raise ValueError("only 'eos' text pooling is implemented (the published checkpoint)")
        head = model_cfg.get("projection_head")
        if not head or head.get("name", "linear") != "linear":
            raise ValueError("only a linear projection head is implemented")
        self.image_encoder = ImageTower(swin_config)
        self.text_encoder = TextTower(bert_config)
        self.image_projection = ProjectionHead(self.image_encoder.out_dim, int(head["proj_dim"]))
        self.text_projection = ProjectionHead(self.text_encoder.out_dim, int(head["proj_dim"]))
        temperature = float(model_cfg.get("temperature", 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Projected, un-normalised image embedding: token 0 of Swin's last hidden state."""
        return self.image_projection(self.image_encoder(pixel_values)[:, 0])

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Projected, un-normalised text embedding: the last attended token ([SEP])."""
        hidden = self.text_encoder(input_ids, attention_mask)
        last = attention_mask.sum(dim=-1) - 1
        return self.text_projection(hidden[torch.arange(hidden.shape[0]), last])


def build_model(model_cfg: dict, swin_config, bert_config) -> CXRClipModule:
    return CXRClipModule(model_cfg, swin_config, bert_config)


def load_state(module: CXRClipModule, state_dict: dict) -> None:
    """Strict load of a checkpoint or a fine-tuned export; drops BERT's position_ids buffer."""
    state = {k: v for k, v in state_dict.items() if k != POSITION_IDS}
    module.load_state_dict(state, strict=True)


def make_transform(size: int = DEFAULT_IMAGE_SIZE):
    """The authors' test-time path: Resize(short side), CenterCrop, ToTensor, Normalize(0.5)."""
    from torchvision import transforms

    pipeline = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    def transform(image: Image.Image) -> torch.Tensor:
        return pipeline(image.convert("RGB"))

    return transform


def plain_config(config: dict) -> dict:
    """The checkpoint config reduced to the sections the adapter reads, as plain dicts."""
    return {k: config[k] for k in CONFIG_SECTIONS if k in config}


def convert_checkpoint(tar: Path, out: Path) -> dict:
    """Rewrite the authors' tar (omegaconf config, optimizer state) as a plain weights file.

    Needs omegaconf at call time only: `uv run --with omegaconf python -c ...`. Returns the
    summary that goes next to the candidate in configs/models.yaml.
    """
    from omegaconf import DictConfig, ListConfig, OmegaConf

    def to_plain(node):
        # The tar's config is a dict of DictConfig/ListConfig nodes, nested at any depth.
        if isinstance(node, DictConfig | ListConfig):
            node = OmegaConf.to_container(node, resolve=True)
        if isinstance(node, dict):
            return {str(k): to_plain(v) for k, v in node.items()}
        if isinstance(node, list | tuple):
            return [to_plain(v) for v in node]
        return node

    checkpoint = torch.load(tar, map_location="cpu", weights_only=False)
    config = to_plain(checkpoint["config"])
    payload = {
        "model": {k: v.contiguous() for k, v in checkpoint["model"].items()},
        "config": plain_config(config),
        "source": {"file": Path(tar).name, "sha256": sha256_of(tar),
                   "epoch": int(checkpoint.get("epoch", -1))},
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    torch.save(payload, part)
    part.replace(out)
    return {"out": str(out), "revision": sha256_of(out),
            "source_sha256": payload["source"]["sha256"], "n_tensors": len(payload["model"]),
            "n_params": int(sum(v.numel() for v in payload["model"].values()))}


def _split_pin(pin: str) -> tuple[str, str]:
    repo, _, revision = pin.partition("@")
    if not revision:
        raise ValueError(f"{pin!r} must be '<repo_id>@<commit sha>'")
    return repo, revision


class CXRClipEncoder:
    name = "cxrclip"

    def __init__(self, candidate: dict, device: str = "cpu"):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, BertConfig, SwinConfig

        weights = Path(candidate["weights"])
        if not weights.is_file():
            raise FileNotFoundError(
                f"{weights} not found; download the checkpoint and run convert_checkpoint")
        verify_sha256(weights, candidate.get("revision"))
        self.model_id = candidate["model_id"]
        self.revision = candidate["revision"]
        self.license = candidate["license"]
        self.device = device

        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        swin_repo, swin_rev = _split_pin(candidate["swin_config"])
        text_repo, text_rev = _split_pin(candidate["text_config"])
        swin_dir = Path(snapshot_download(swin_repo, revision=swin_rev,
                                          allow_patterns=["config.json"]))
        text_dir = Path(snapshot_download(
            text_repo, revision=text_rev,
            allow_patterns=["config.json", "vocab.txt", "tokenizer*",
                            "special_tokens_map.json"]))
        self._model = build_model(config["model"], SwinConfig.from_pretrained(swin_dir),
                                  BertConfig.from_pretrained(text_dir))
        load_state(self._model, checkpoint["model"])
        self._model.to(device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(text_dir)
        self._transform = make_transform(int(config.get("base", {}).get("image_size",
                                                                         DEFAULT_IMAGE_SIZE)))
        self._max_length = int(config.get("base", {}).get("text_max_length",
                                                          DEFAULT_TEXT_MAX_LENGTH))
        self.weights_bytes = weights.stat().st_size
        self.preprocessing = {
            "image_size": int(config.get("base", {}).get("image_size", DEFAULT_IMAGE_SIZE)),
            "resize": "torchvision Resize(short side, bilinear) then CenterCrop",
            "mean": list(MEAN),
            "std": list(STD),
            "image_feature": "swin last_hidden_state token 0, linear projection",
            "text_pooling": "eos: last attended token ([SEP]), linear projection",
            "text_max_length": self._max_length,
            "tokenizer": f"{text_repo}@{text_rev}",
            "do_lower_case": bool(getattr(self._tokenizer, "do_lower_case", False)),
            "swin_config": candidate["swin_config"],
            "text_config": candidate["text_config"],
            "checkpoint": weights.name,
            "source_sha256": checkpoint.get("source", {}).get("sha256"),
            "finetune": checkpoint.get("finetune"),
        }
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode_texts(["chest x-ray"]).shape[1])
        return self._dim

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        batch = torch.stack([self._transform(load_rgb(p)) for p in paths]).to(self.device)
        with torch.inference_mode():
            return l2_normalize(self._model.encode_image(batch).float().cpu().numpy())

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        tokens = self._tokenizer(list(texts), padding="longest", truncation=True,
                                 max_length=self._max_length, return_tensors="pt")
        with torch.inference_mode():
            out = self._model.encode_text(tokens["input_ids"].to(self.device),
                                          tokens["attention_mask"].to(self.device))
            return l2_normalize(out.float().cpu().numpy())


def file_sha256(path: Path) -> str:
    """Bare hex digest (no 'sha256:' prefix), for manifests and notes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
