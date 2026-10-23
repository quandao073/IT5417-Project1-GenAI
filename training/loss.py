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
