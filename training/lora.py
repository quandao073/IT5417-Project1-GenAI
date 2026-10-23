"""LoRA on the attention projections of both towers, and the merged export (H3, D30, D34).

`apply_lora` wraps every `query`/`key`/`value` Linear of Swin and BERT with peft; everything
else (projections, logit_scale) stays frozen. `export_merged` folds the adapters into the
Linear weights of a copy and saves `{"model", "config", "finetune"}` with the checkpoint's
original key names, so `src.embeddings.cxrclip.load_state` reads it unchanged.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from src.embeddings.cxrclip import CXRClipModule, file_sha256


def apply_lora(module: CXRClipModule, *, rank: int, alpha: int, dropout: float,
               targets: Sequence[str] = ("query", "key", "value")) -> PeftModel:
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                        target_modules=list(targets), bias="none")
    return get_peft_model(module, config)


def unwrap(peft_model: PeftModel) -> CXRClipModule:
    """The CXRClipModule inside the peft wrapper, LoRA layers injected in place."""
    return peft_model.base_model.model


def trainable_fraction(model: torch.nn.Module) -> float:
    total = sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / total


def export_merged(peft_model: PeftModel, ckpt_config: dict, out: Path, meta: dict) -> str:
    """Merge LoRA into a copy, save the plain state dict; returns the file's sha256."""
    merged = copy.deepcopy(peft_model).merge_and_unload()
    state = {k: v.detach().cpu().contiguous() for k, v in merged.state_dict().items()}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    torch.save({"model": state, "config": ckpt_config, "finetune": meta}, part)
    part.replace(out)
    return file_sha256(out)
