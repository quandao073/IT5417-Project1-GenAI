"""Per-epoch pool-val scoring (plan Task 12): P@5 per S12 kind under the default policy, the
label filter's truth table, queries with fewer than five relevant images skipped."""

from __future__ import annotations

import math

import numpy as np
import torch

from src.common.config import load_config
from src.retrieval.templates import generate
from training.tests.conftest import status_row, tiny_model, write_tiny_checkpoint
from training.val import p_at_5_by_kind, val_p5

CFG = load_config("concepts")
CONCEPTS = ["Cardiomegaly", "Edema", "Pleural Effusion"]


def _statuses() -> np.ndarray:
    rows = []
    for k in range(16):
        s = {"Edema": "PRESENT" if k < 12 else "ABSENT"}
        if k < 6:
            s["Cardiomegaly"] = "PRESENT"
        elif k < 12:
            s["Cardiomegaly"] = "ABSENT"
        if k < 3:
            s["Pleural Effusion"] = "PRESENT"
        rows.append(status_row(s))
    return np.stack(rows)


def test_p_at_5_by_kind_matches_hand_count():
    queries = generate(CONCEPTS, ["single", "pair", "negation"], CFG)
    statuses = _statuses()
    top5 = {
        (("Edema",), ()): [0, 1, 2, 3, 12],                    # 4 relevant of 5 -> 0.8
        (("Cardiomegaly",), ()): [0, 1, 2, 3, 4],              # 1.0
        (("Cardiomegaly", "Edema"), ()): [0, 1, 6, 7, 8],      # rows 6-8 lack cardiomegaly -> 0.4
        (("Edema",), ("Cardiomegaly",)): [6, 7, 8, 9, 10],     # 1.0
        (("Edema",), ("Pleural Effusion",)): [0, 1, 2, 3, 4],  # rows 0-2 have effusion -> 0.4
    }
    scores = np.tile(-np.arange(16, dtype=np.float32), (len(queries), 1))
    for q, query in enumerate(queries):
        rows = top5.get((query.include, query.exclude))
        if rows is not None:
            scores[q, rows] = 100.0 - np.arange(5)
    out = p_at_5_by_kind(scores, queries, statuses, min_relevant=5)
    assert math.isclose(out["single"], 0.9)
    assert math.isclose(out["pair"], 0.4)
    assert math.isclose(out["negation"], 0.7)
    assert math.isclose(out["mean"], (0.9 + 0.4 + 0.7) / 3)
    # Pleural Effusion singles (3 relevant), the other pairs and the other negations are skipped.
    assert (out["n_single"], out["n_pair"], out["n_negation"]) == (2, 1, 2)


def test_p_at_5_mean_ignores_kinds_without_a_scoreable_query():
    queries = generate(["Edema"], ["single", "pair", "negation"], CFG)
    assert [q.kind for q in queries] == ["single"]
    statuses = _statuses()
    scores = np.tile(-np.arange(16, dtype=np.float32), (1, 1))
    out = p_at_5_by_kind(scores, queries, statuses, min_relevant=5)
    assert out["single"] == 1.0 and math.isnan(out["pair"]) and math.isnan(out["negation"])
    assert out["mean"] == 1.0


def test_val_p5_runs_the_tiny_model_end_to_end(tmp_path):
    from transformers import AutoTokenizer

    _, _, bert_dir, _ = write_tiny_checkpoint(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(bert_dir)
    model = tiny_model()
    images = np.random.default_rng(0).integers(0, 255, size=(16, 32, 32), dtype=np.uint8)
    out = val_p5(model, tokenizer, images, _statuses(), CONCEPTS, CFG, torch.device("cpu"),
                 text_max_length=16, batch_size=5)
    assert set(out) >= {"single", "pair", "negation", "mean", "n_single", "n_pair", "n_negation"}
    assert (out["n_single"], out["n_pair"], out["n_negation"]) == (2, 1, 2)
    for kind in ("single", "pair", "negation", "mean"):
        assert 0.0 <= out[kind] <= 1.0
    # Collapse monitor: mean cosine between distinct template texts (FT1 reached 0.94).
    assert -1.0 <= out["text_cos"] < 1.0
    again = val_p5(model, tokenizer, images, _statuses(), CONCEPTS, CFG, torch.device("cpu"),
                   text_max_length=16, batch_size=7)
    assert again == out
