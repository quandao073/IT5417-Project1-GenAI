"""The CXR-RETRIEVE loss (plan Task 10; roadmap section 7 WP4): CLIP when nothing is shared,
a repel term only for conflicting pairs, never NaN."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as fn

from training.captions import CODE_ABSENT, CODE_PRESENT, CODE_UNCERTAIN
from training.loss import clip_loss, image_label_loss, image_targets, label_aware_loss

SCALE = torch.tensor(1 / 0.07)


def _unit(x: torch.Tensor) -> torch.Tensor:
    return fn.normalize(x, dim=-1)


def _batch(seed: int, b: int = 6, d: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return _unit(torch.randn(b, d, generator=g)), _unit(torch.randn(b, d, generator=g))


def test_clip_loss_is_symmetric_cross_entropy():
    zi, zt = _batch(0)
    logits = SCALE * zi @ zt.T
    labels = torch.arange(6)
    expected = 0.5 * (fn.cross_entropy(logits, labels) + fn.cross_entropy(logits.T, labels))
    assert torch.allclose(clip_loss(zi, zt, SCALE), expected, atol=1e-6)


def test_matches_clip_loss_when_no_shared_labels():
    zi, zt = _batch(1)
    y = torch.eye(6, dtype=torch.int8)          # every caption affirms its own concept
    assert torch.allclose(label_aware_loss(zi, zt, y, SCALE, lam=0.3), clip_loss(zi, zt, SCALE),
                          atol=1e-6)


def test_repel_positive_for_conflicting_pair_with_positive_cosine():
    g = torch.Generator().manual_seed(2)
    base = _unit(torch.randn(1, 8, generator=g))
    # Two images that look alike, captioned "a" and "no a": the pair conflicts (K < 0).
    zi = _unit(base + 0.05 * torch.randn(2, 8, generator=g))
    zt = _unit(base + 0.05 * torch.randn(2, 8, generator=g))
    y = torch.tensor([[1, 0], [-1, 0]], dtype=torch.int8)
    logits = SCALE * zi @ zt.T
    conflict = (y.float() @ y.float().T) < 0
    repel = logits[conflict].clamp(min=0).mean()
    assert repel > 0
    with_repel = label_aware_loss(zi, zt, y, SCALE, lam=0.3)
    without = label_aware_loss(zi, zt, y, SCALE, lam=0.0)
    assert with_repel > without
    assert torch.allclose(with_repel - without, 0.3 * repel, atol=1e-6)


def test_no_nan_when_a_row_shares_nothing():
    zi, zt = _batch(3)
    y = torch.zeros(6, 3, dtype=torch.int8)     # the impression of an all-NOT_MENTIONED study
    y[0, 0] = 1
    y[1, 1] = 1
    loss = label_aware_loss(zi, zt, y, SCALE, lam=0.3)
    assert math.isfinite(loss.item())


def test_lam_zero_drops_repel():
    zi, zt = _batch(4)
    y = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 1], [1, 1, 0], [0, 0, 1]],
                     dtype=torch.int8)
    logits = SCALE * zi @ zt.T
    k = y.float() @ y.float().T
    repel = logits[k < 0].clamp(min=0).mean()
    assert torch.allclose(label_aware_loss(zi, zt, y, SCALE, lam=0.3)
                          - label_aware_loss(zi, zt, y, SCALE, lam=0.0), 0.3 * repel, atol=1e-6)


def test_shared_labels_spread_the_target():
    # Rows 0 and 1 both affirm concept 0: each is a positive for the other, so the loss is
    # strictly below the one that treats them as negatives of each other.
    zi, zt = _batch(5)
    shared = torch.eye(6, dtype=torch.int8)
    shared[1] = shared[0]
    assert label_aware_loss(zi, zt, shared, SCALE, lam=0.0) != clip_loss(zi, zt, SCALE)
    loss = label_aware_loss(zi, zt, shared, SCALE, lam=0.0)
    assert loss.requires_grad is False and math.isfinite(loss.item())


# --- image-label targets (FT6; WORKLOG "FT1 kẹt ở ln 128") -------------------------------------
P, A, U = CODE_PRESENT, CODE_ABSENT, CODE_UNCERTAIN


def _codes(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int8)


def test_image_targets_match_clip_when_each_image_has_only_its_caption_concept():
    zi, zt = _batch(6)
    y = torch.eye(6, dtype=torch.int8)
    codes = torch.eye(6, dtype=torch.int8) * P
    assert torch.allclose(image_label_loss(zi, zt, y, SCALE, lam=0.3, codes=codes),
                          clip_loss(zi, zt, SCALE), atol=1e-6)


def test_a_caption_the_image_satisfies_is_a_positive_even_if_drawn_elsewhere():
    # Image 0 has concepts 0 and 1; its caption this time is "0". Caption 1 ("1") comes from
    # image 1, which has only concept 1. The paper's targets call (image 0, caption 1) a negative.
    y = torch.tensor([[1, 0], [0, 1]], dtype=torch.int8)
    codes = _codes([[P, P], [0, P]])
    relevant, conflict = image_targets(y, codes)
    assert relevant.tolist() == [[True, True], [False, True]]
    assert not conflict.any()


def test_negated_concept_must_be_absent_or_not_mentioned():
    # Caption 0 is "a and no b"; four images that all have a PRESENT and b in each status.
    y = torch.tensor([[1, -1]] * 4, dtype=torch.int8)
    codes = _codes([[P, A], [P, 0], [P, U], [P, P]])
    relevant, conflict = image_targets(y, codes)
    assert relevant[:, 0].tolist() == [True, True, False, False]
    assert conflict[:, 0].tolist() == [False, False, False, True]


def test_affirmed_concept_absent_in_the_image_conflicts_not_mentioned_does_not():
    y = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int8)
    codes = _codes([[P, 0], [A, P], [0, P]])
    relevant, conflict = image_targets(y, codes)
    assert not relevant[1, 0] and conflict[1, 0]          # "a" for an image with a ABSENT
    assert not relevant[2, 0] and not conflict[2, 0]      # "a" for an image that never mentions a
    assert relevant.diagonal().all()


def test_image_targets_repel_only_conflicts_and_never_nan():
    zi, zt = _batch(7)
    y = torch.tensor([[1, 0, 0], [-1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, -1, 1]],
                     dtype=torch.int8)
    codes = _codes([[P, 0, A], [A, P, 0], [U, P, 0], [0, 0, P], [P, P, 0], [0, A, P]])
    _, conflict = image_targets(y, codes)
    assert conflict.any()
    logits = SCALE * zi @ zt.T
    repel = logits[conflict].clamp(min=0).mean()
    with_repel = image_label_loss(zi, zt, y, SCALE, lam=0.3, codes=codes)
    without = image_label_loss(zi, zt, y, SCALE, lam=0.0, codes=codes)
    assert math.isfinite(with_repel.item())
    assert torch.allclose(with_repel - without, 0.3 * repel, atol=1e-6)
