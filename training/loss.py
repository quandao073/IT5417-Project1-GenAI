"""The label-aware contrastive loss of CXR-RETRIEVE (roadmap section 7 WP4, WORKLOG D30).

`y` is the caption's label vector: +1 for an affirmed concept, -1 for a negated one, 0
otherwise. `K = y @ y.T` is positive where two captions agree on a concept (both affirm or
both negate it) and negative where they contradict each other ("a" against "no a"). The
attract term is CLIP's symmetric cross-entropy with every agreeing pair as a soft positive;
the repel term pushes contradicting pairs below zero logit. With one-hot, distinct rows the
loss is exactly CLIP's.
"""

from __future__ import annotations

import torch

from training.captions import CODE_ABSENT, CODE_PRESENT, CODE_UNCERTAIN


def clip_loss(zi: torch.Tensor, zt: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric cross-entropy over the diagonal; zi, zt [B, D] L2-normalised."""
    logits = logit_scale * zi @ zt.T
    labels = torch.arange(zi.shape[0], device=zi.device)
    return 0.5 * (torch.nn.functional.cross_entropy(logits, labels)
                  + torch.nn.functional.cross_entropy(logits.T, labels))


def label_aware_loss(zi: torch.Tensor, zt: torch.Tensor, y: torch.Tensor,
                     logit_scale: torch.Tensor, lam: float = 0.3) -> torch.Tensor:
    # zi, zt: [B, D] đã L2-norm; y: [B, L] ∈ {-1, 0, 1} lấy từ câu
    y = y.float()
    K = y @ y.T
    K = K - torch.diag(K.diag()) + torch.diag(K.diag().clamp(min=1))
    T = K.clamp(min=0)
    T = T / T.sum(1, keepdim=True)
    logits = logit_scale * zi @ zt.T
    attract = -0.5 * ((T * logits.log_softmax(1)).sum(1).mean()
                      + (T * logits.T.log_softmax(1)).sum(1).mean())
    conflict = K < 0
    repel = logits[conflict].clamp(min=0).mean() if conflict.any() else logits.new_zeros(())
    return attract + lam * repel


def image_targets(y: torch.Tensor, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(relevant, conflict), bool [B images, B captions], from each image's own labels.

    Caption j is relevant to image i under the bench's default policy: every concept it affirms
    is PRESENT in image i, none it negates is PRESENT or UNCERTAIN there. It conflicts when image
    i has an affirmed concept ABSENT or a negated one PRESENT. The paper's K = y yT compares
    captions only, so an image is pushed away from a caption naming its other findings (33.8 %
    of the negatives on the real pool). The diagonal is always relevant.
    """
    affirmed, negated = (y > 0).float(), (y < 0).float()
    present = (codes == CODE_PRESENT).float()
    blocked = present + (codes == CODE_UNCERTAIN).float()
    absent = (codes == CODE_ABSENT).float()
    relevant = (((1 - present) @ affirmed.T) == 0) & ((blocked @ negated.T) == 0)
    relevant |= torch.eye(len(y), dtype=torch.bool, device=y.device)
    conflict = (((absent @ affirmed.T) > 0) | ((present @ negated.T) > 0)) & ~relevant
    return relevant, conflict


def image_label_loss(zi: torch.Tensor, zt: torch.Tensor, y: torch.Tensor,
                     logit_scale: torch.Tensor, lam: float = 0.3, *,
                     codes: torch.Tensor) -> torch.Tensor:
    """`label_aware_loss` with `image_targets`: every caption an image satisfies is a soft
    positive (image -> text), every image a caption describes is one (text -> image), and the
    repel term acts on true conflicts only. codes: [B, L] int8 of `captions.status_codes`."""
    relevant, conflict = image_targets(y, codes)
    r = relevant.float()
    logits = logit_scale * zi @ zt.T
    attract = -0.5 * ((r / r.sum(1, keepdim=True) * logits.log_softmax(1)).sum(1).mean()
                      + (r.T / r.T.sum(1, keepdim=True) * logits.T.log_softmax(1)).sum(1).mean())
    repel = logits[conflict].clamp(min=0).mean() if conflict.any() else logits.new_zeros(())
    return attract + lam * repel
