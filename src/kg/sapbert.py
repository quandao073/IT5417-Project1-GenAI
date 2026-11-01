"""SapBERT name embeddings for the V2-3 linker (D56): [CLS] of the last layer, L2-normalised.

The model is pinned to a Hugging Face commit (configs/kg.yaml: linker.sapbert). The dictionary,
every normalised MRCONSO name of a CUI on the design's label table, is embedded once and cached
under data/kg/sapbert/, keyed by the revision, `max_length` and the sha256 of the name list. The
cache holds UMLS names, so it stays under data/ (T38).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"


class SapBert:
    def __init__(self, cfg: dict, device: str = "cpu"):
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        path = snapshot_download(cfg["model_id"], revision=cfg["revision"],
                                 allow_patterns=["*.json", "vocab.txt", "model.safetensors"])
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, use_safetensors=True).eval().to(device)
        self.revision, self.device = cfg["revision"], device
        self.max_length, self.batch_size = cfg["max_length"], cfg["batch_size"]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """[n, 768] float32, unit rows. Texts are batched by length, so padding stays short."""
        import torch

        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                rows = order[start:start + self.batch_size]
                batch = self.tokenizer([texts[i] for i in rows], padding=True, truncation=True,
                                       max_length=self.max_length, return_tensors="pt")
                batch = batch.to(self.device)
                cls = self.model(**batch).last_hidden_state[:, 0]
                out[rows] = torch.nn.functional.normalize(cls, dim=1).cpu().numpy()
        return out


def cache_key(names: Sequence[str], revision: str, max_length: int) -> str:
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    return f"{revision}:{max_length}:{digest}"


def name_vectors(names: Sequence[str], encoder, cache_dir: Path) -> np.ndarray:
    """The encoder's vectors of `names`, float16, from the cache when its key matches."""
    key = cache_key(names, encoder.revision, encoder.max_length)
    meta_path, vec_path = cache_dir / META_FILE, cache_dir / VECTORS_FILE
    if (meta_path.is_file() and vec_path.is_file()
            and json.loads(meta_path.read_text(encoding="utf-8")).get("key") == key):
        return np.load(vec_path)
    vectors = encoder.encode(list(names)).astype(np.float16)
    cache_dir.mkdir(parents=True, exist_ok=True)
    part = cache_dir / (VECTORS_FILE + ".part")
    with open(part, "wb") as handle:
        np.save(handle, vectors)
    part.replace(vec_path)
    meta_path.write_text(json.dumps({"key": key, "names": len(names)}), encoding="utf-8")
    return vectors


def nearest(queries: np.ndarray, vectors: np.ndarray, allowed: np.ndarray, k: int = 1,
            chunk: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    """(rows, cosines), each [n, k] best first: the k best allowed dictionary rows for each unit
    query vector."""
    q = np.asarray(queries, dtype=np.float32)
    best = np.full((len(q), k), -np.inf, dtype=np.float32)
    arg = np.zeros((len(q), k), dtype=np.int64)
    for start in range(0, len(vectors), chunk):
        block = np.asarray(vectors[start:start + chunk], dtype=np.float32)
        scores = q @ block.T
        scores[:, ~allowed[start:start + chunk]] = -np.inf
        top = np.argpartition(-scores, min(k, scores.shape[1] - 1), axis=1)[:, :k]
        values = np.concatenate([best, np.take_along_axis(scores, top, axis=1)], axis=1)
        rows = np.concatenate([arg, start + top], axis=1)
        order = np.argsort(-values, axis=1, kind="stable")[:, :k]
        best = np.take_along_axis(values, order, axis=1)
        arg = np.take_along_axis(rows, order, axis=1)
    return arg, best
