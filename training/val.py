"""Pool-val scoring between epochs: P@5 per S12 kind with joint cosine (plan Task 12, D33).

Relevance is `template_bench.relevant_mask` under the default policy (`exclude_present`,
`EXCLUDE`): the label filter's truth table, the same one `template_bench.evaluate` scores.
Only P@5 is computed here (no bootstrap, no other k), because it runs after every epoch.
Queries with fewer than `min_relevant` relevant images are skipped; the mean is over the
kinds that still have a scored query.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch.nn.functional import normalize

from src.evaluation.template_bench import DEFAULT_POLICY, relevant_mask
from src.retrieval.templates import TemplateQuery, generate
from training.data import eval_transform

KINDS = ("single", "pair", "negation")
K = 5


def p_at_5_by_kind(scores: np.ndarray, queries: Sequence[TemplateQuery], statuses: np.ndarray,
                   *, min_relevant: int = 5) -> dict[str, float]:
    """Mean P@5 per kind from `scores` [Q, N] (higher is better) and the statuses of the N images.

    Returns {kind: P@5 or nan, "mean": mean over scored kinds, "n_<kind>": queries scored}."""
    statuses = np.asarray(statuses, dtype=object)
    per_kind: dict[str, list[float]] = {k: [] for k in KINDS}
    for q, query in enumerate(queries):
        relevant = relevant_mask(statuses, query, DEFAULT_POLICY)
        if int(relevant.sum()) < min_relevant:
            continue
        top = np.argsort(-scores[q], kind="stable")[:K]
        per_kind[query.kind].append(float(relevant[top].mean()))
    out: dict[str, float] = {k: float(np.mean(v)) if v else math.nan for k, v in per_kind.items()}
    scored = [out[k] for k in KINDS if per_kind[k]]
    out["mean"] = float(np.mean(scored)) if scored else math.nan
    out.update({f"n_{k}": len(v) for k, v in per_kind.items()})
    return out


def embed_images(model, images_uint8: np.ndarray, device: torch.device, *,
                 batch_size: int = 256, amp: bool = False) -> torch.Tensor:
    """L2-normalised image vectors [N, D] (float32, cpu) with the eval transform."""
    transform = eval_transform()
    was_training = model.training
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(images_uint8), batch_size):
            batch = torch.stack([transform(img) for img in images_uint8[start:start + batch_size]])
            with torch.autocast(device.type, dtype=_amp_dtype(device), enabled=amp):
                z = model.encode_image(batch.to(device))
            chunks.append(normalize(z.float(), dim=-1).cpu())
    model.train(was_training)
    return torch.cat(chunks)


def encode_texts(model, tokenizer, texts: Sequence[str], device: torch.device, *,
                 text_max_length: int, amp: bool = False) -> torch.Tensor:
    """L2-normalised text vectors [Q, D] (float32, cpu)."""
    was_training = model.training
    model.eval()
    tokens = tokenizer(list(texts), padding="longest", truncation=True,
                       max_length=text_max_length, return_tensors="pt")
    with torch.inference_mode(), torch.autocast(device.type, dtype=_amp_dtype(device),
                                                enabled=amp):
        z = model.encode_text(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
    model.train(was_training)
    return normalize(z.float(), dim=-1).cpu()


def _amp_dtype(device: torch.device) -> torch.dtype:
    """bf16 where the GPU has it (fp32's range, no overflow), else fp16; bf16 on CPU."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16


def val_p5(model, tokenizer, images_uint8: np.ndarray, statuses: np.ndarray,
           concepts: Sequence[str], concepts_cfg: dict, device: torch.device, *,
           text_max_length: int = 64, batch_size: int = 256, amp: bool = False,
           min_relevant: int = 5) -> dict[str, float]:
    """P@5 of the S12 single / pair / negation templates on the given images (pool-val), plus
    "text_cos": the mean cosine between distinct templates."""
    queries = generate(concepts, KINDS, concepts_cfg)
    image_vecs = embed_images(model, images_uint8, device, batch_size=batch_size, amp=amp)
    text_vecs = encode_texts(model, tokenizer, [q.text for q in queries], device,
                             text_max_length=text_max_length, amp=amp)
    scores = (text_vecs @ image_vecs.T).numpy()
    out = p_at_5_by_kind(scores, queries, statuses, min_relevant=min_relevant)
    # Collapse monitor: when the text tower maps every template to one point this nears 1
    # (FT1 of B14: 0.53 -> 0.94 in two epochs while P@5 fell below zero-shot).
    sims = text_vecs @ text_vecs.T
    off = ~torch.eye(len(sims), dtype=torch.bool)
    out["text_cos"] = float(sims[off].mean())
    return out
