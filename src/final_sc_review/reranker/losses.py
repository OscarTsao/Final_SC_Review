"""Hybrid losses for reranker training."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F


def _split_by_group(tensor: torch.Tensor, group_sizes: Sequence[int]) -> List[torch.Tensor]:
    return list(torch.split(tensor, list(group_sizes)))


def pointwise_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with logits."""
    return F.binary_cross_entropy_with_logits(logits, labels)


def pairwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    max_pairs_per_group: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Pairwise logistic loss over pos/neg pairs within each group."""
    losses = []
    for group_logits, group_labels in zip(_split_by_group(logits, group_sizes), _split_by_group(labels, group_sizes)):
        pos = group_logits[group_labels > 0.5]
        neg = group_logits[group_labels <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        if max_pairs_per_group is not None:
            # Sample pairs if needed
            pos_idx = torch.randint(
                len(pos),
                (max_pairs_per_group,),
                generator=rng,
                device=pos.device,
            )
            neg_idx = torch.randint(
                len(neg),
                (max_pairs_per_group,),
                generator=rng,
                device=neg.device,
            )
            diffs = pos[pos_idx] - neg[neg_idx]
            losses.append(F.softplus(-diffs).mean())
        else:
            diffs = pos[:, None] - neg[None, :]
            losses.append(F.softplus(-diffs).mean())
    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


def listwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    temperature: float = 1.0,
) -> torch.Tensor:
    """ListNet-style KL loss within each group."""
    losses = []
    for group_logits, group_labels in zip(_split_by_group(logits, group_sizes), _split_by_group(labels, group_sizes)):
        if torch.all(group_labels <= 0.0):
            continue
        target = F.softmax(group_labels / max(temperature, 1e-6), dim=0)
        pred = F.softmax(group_logits, dim=0)
        losses.append(F.kl_div(pred.log(), target, reduction="batchmean"))
    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


def hybrid_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    w_list: float,
    w_pair: float,
    w_point: float,
    temperature: float,
    max_pairs_per_group: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Weighted hybrid loss = listwise + pairwise + pointwise."""
    loss_list = listwise_loss(logits, labels, group_sizes, temperature)
    loss_pair = pairwise_loss(logits, labels, group_sizes, max_pairs_per_group, rng)
    loss_point = pointwise_loss(logits, labels)
    return w_list * loss_list + w_pair * loss_pair + w_point * loss_point
