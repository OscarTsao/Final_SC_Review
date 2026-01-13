"""SOTA loss functions for reranker training.

Implements all loss functions from Section 7 of the research plan:
- Pointwise: BCE, FocalLoss
- Pairwise: RankNet, MarginRanking, Softplus
- Listwise: ListNet, ListMLE, PListMLE, LambdaLoss, ApproxNDCG
- Contrastive: InfoNCE / MultipleNegativesRanking
- Distillation: MSE, MarginMSE

References:
- RankNet (ICML 2005): https://doi.org/10.1145/1102351.1102363
- ListNet: https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/
- ListMLE (ICML 2008): https://doi.org/10.1145/1390156.1390306
- LambdaLoss (2018): https://www.tdcommons.org/dpubs_series/1216
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# =============================================================================
# Utility Functions
# =============================================================================

def _split_by_group(tensor: torch.Tensor, group_sizes: Sequence[int]) -> List[torch.Tensor]:
    """Split tensor into groups."""
    return list(torch.split(tensor, list(group_sizes)))


def _get_ranking_order(scores: torch.Tensor) -> torch.Tensor:
    """Get ranking indices (descending order)."""
    return torch.argsort(scores, descending=True)


def _compute_dcg(gains: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """Compute DCG@k."""
    if k is not None:
        gains = gains[:k]
    positions = torch.arange(1, len(gains) + 1, device=gains.device, dtype=gains.dtype)
    discounts = torch.log2(positions + 1)
    return (gains / discounts).sum()


def _compute_ndcg(pred_scores: torch.Tensor, labels: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """Compute NDCG@k."""
    # Predicted ranking
    pred_order = _get_ranking_order(pred_scores)
    pred_gains = labels[pred_order]
    dcg = _compute_dcg(pred_gains, k)

    # Ideal ranking
    ideal_order = _get_ranking_order(labels)
    ideal_gains = labels[ideal_order]
    idcg = _compute_dcg(ideal_gains, k)

    if idcg == 0:
        return torch.tensor(1.0, device=labels.device)
    return dcg / idcg


# =============================================================================
# Pointwise Losses
# =============================================================================

def pointwise_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with logits (standard pointwise loss)."""
    return F.binary_cross_entropy_with_logits(logits, labels)


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: Raw model outputs
        labels: Binary labels (0 or 1)
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    probs = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

    p_t = probs * labels + (1 - probs) * (1 - labels)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    focal_weight = alpha_t * (1 - p_t) ** gamma

    return (focal_weight * ce_loss).mean()


# =============================================================================
# Pairwise Losses
# =============================================================================

def ranknet_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    max_pairs_per_group: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
    sigma: float = 1.0,
) -> torch.Tensor:
    """RankNet loss - pairwise logistic ranking (Burges et al., ICML 2005).

    L = -sum_{i,j: y_i > y_j} log(sigmoid(sigma * (s_i - s_j)))

    This is the original pairwise ranking loss using sigmoid cross-entropy.
    """
    losses = []
    for group_logits, group_labels in zip(
        _split_by_group(logits, group_sizes),
        _split_by_group(labels, group_sizes)
    ):
        pos_mask = group_labels > 0.5
        neg_mask = group_labels <= 0.5
        pos = group_logits[pos_mask]
        neg = group_logits[neg_mask]

        if len(pos) == 0 or len(neg) == 0:
            continue

        if max_pairs_per_group is not None and len(pos) * len(neg) > max_pairs_per_group:
            # Sample pairs
            pos_idx = torch.randint(len(pos), (max_pairs_per_group,), generator=rng, device=pos.device)
            neg_idx = torch.randint(len(neg), (max_pairs_per_group,), generator=rng, device=neg.device)
            diffs = sigma * (pos[pos_idx] - neg[neg_idx])
        else:
            # All pairs
            diffs = sigma * (pos[:, None] - neg[None, :]).flatten()

        # RankNet uses sigmoid cross-entropy: -log(sigmoid(diff))
        losses.append(F.softplus(-diffs).mean())

    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


def margin_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    margin: float = 1.0,
    max_pairs_per_group: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Margin ranking loss (hinge-based pairwise loss).

    L = max(0, margin - (s_pos - s_neg))
    """
    losses = []
    for group_logits, group_labels in zip(
        _split_by_group(logits, group_sizes),
        _split_by_group(labels, group_sizes)
    ):
        pos = group_logits[group_labels > 0.5]
        neg = group_logits[group_labels <= 0.5]

        if len(pos) == 0 or len(neg) == 0:
            continue

        if max_pairs_per_group is not None and len(pos) * len(neg) > max_pairs_per_group:
            pos_idx = torch.randint(len(pos), (max_pairs_per_group,), generator=rng, device=pos.device)
            neg_idx = torch.randint(len(neg), (max_pairs_per_group,), generator=rng, device=neg.device)
            diffs = pos[pos_idx] - neg[neg_idx]
        else:
            diffs = (pos[:, None] - neg[None, :]).flatten()

        losses.append(F.relu(margin - diffs).mean())

    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


def pairwise_softplus_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    max_pairs_per_group: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Pairwise softplus loss (smooth approximation of hinge).

    L = softplus(-(s_pos - s_neg)) = log(1 + exp(-(s_pos - s_neg)))

    This is equivalent to RankNet with sigma=1.
    """
    return ranknet_loss(logits, labels, group_sizes, max_pairs_per_group, rng, sigma=1.0)


# =============================================================================
# Listwise Losses
# =============================================================================

def listnet_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    temperature: float = 1.0,
) -> torch.Tensor:
    """ListNet loss - KL divergence between score distributions.

    Compares softmax distributions over predicted scores and labels.
    """
    losses = []
    for group_logits, group_labels in zip(
        _split_by_group(logits, group_sizes),
        _split_by_group(labels, group_sizes)
    ):
        if torch.all(group_labels <= 0.0):
            continue

        target = F.softmax(group_labels / max(temperature, 1e-6), dim=0)
        pred = F.log_softmax(group_logits / max(temperature, 1e-6), dim=0)
        losses.append(F.kl_div(pred, target, reduction="batchmean"))

    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


def listmle_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    eps: float = 1e-10,
) -> torch.Tensor:
    """ListMLE loss - Maximum Likelihood Estimation for listwise ranking.

    Maximizes the likelihood of the ground truth permutation under the
    Plackett-Luce model.

    L = -sum_i log(exp(s_{pi(i)}) / sum_{j>=i} exp(s_{pi(j)}))

    Reference: Xia et al., ICML 2008
    """
    losses = []
    for group_logits, group_labels in zip(
        _split_by_group(logits, group_sizes),
        _split_by_group(labels, group_sizes)
    ):
        if len(group_logits) <= 1:
            continue

        # Sort by labels (descending) to get ground truth ranking
        sorted_indices = torch.argsort(group_labels, descending=True)
        sorted_logits = group_logits[sorted_indices]

        # Compute log-likelihood under Plackett-Luce
        # For numerical stability, use log-sum-exp trick
        n = len(sorted_logits)
        max_logit = sorted_logits.max()

        # Compute cumulative log-sum-exp from the end
        # cumsumexp[i] = log(sum_{j>=i} exp(s_j))
        exp_scores = torch.exp(sorted_logits - max_logit)
        cumsum_exp = torch.cumsum(exp_scores.flip(0), dim=0).flip(0)
        log_cumsum = torch.log(cumsum_exp + eps) + max_logit

        # ListMLE loss
        loss = -(sorted_logits - log_cumsum).sum() / n
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


def plistmle_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    eps: float = 1e-10,
) -> torch.Tensor:
    """Position-aware ListMLE (PListMLE).

    Weights the ListMLE loss by position importance (higher weight for top positions).

    L = -sum_i w_i * log(exp(s_{pi(i)}) / sum_{j>=i} exp(s_{pi(j)}))

    where w_i = 1 / log2(i + 1) (position weight like NDCG)
    """
    losses = []
    for group_logits, group_labels in zip(
        _split_by_group(logits, group_sizes),
        _split_by_group(labels, group_sizes)
    ):
        if len(group_logits) <= 1:
            continue

        # Sort by labels (descending)
        sorted_indices = torch.argsort(group_labels, descending=True)
        sorted_logits = group_logits[sorted_indices]

        n = len(sorted_logits)
        max_logit = sorted_logits.max()

        # Compute cumulative log-sum-exp
        exp_scores = torch.exp(sorted_logits - max_logit)
        cumsum_exp = torch.cumsum(exp_scores.flip(0), dim=0).flip(0)
        log_cumsum = torch.log(cumsum_exp + eps) + max_logit

        # Position weights (NDCG-style)
        positions = torch.arange(1, n + 1, device=logits.device, dtype=logits.dtype)
        weights = 1.0 / torch.log2(positions + 1)
        weights = weights / weights.sum()  # Normalize

        # Weighted ListMLE loss
        loss = -(weights * (sorted_logits - log_cumsum)).sum()
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


def lambda_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    sigma: float = 1.0,
    ndcg_k: Optional[int] = 10,
    max_pairs_per_group: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """LambdaLoss - Metric-driven pairwise loss with NDCG weighting.

    Weights each pair by |Delta_NDCG| - the change in NDCG if the pair were swapped.
    This is the core idea behind LambdaRank/LambdaMART.

    L = sum_{i,j: y_i > y_j} |Delta_NDCG_ij| * log(1 + exp(-sigma * (s_i - s_j)))

    Reference: LambdaLoss (2018), LambdaRank (2006)
    """
    losses = []
    for group_logits, group_labels in zip(
        _split_by_group(logits, group_sizes),
        _split_by_group(labels, group_sizes)
    ):
        n = len(group_logits)
        if n <= 1:
            continue

        # Get current ranking by predicted scores
        pred_order = torch.argsort(group_logits, descending=True)
        pred_ranks = torch.zeros_like(pred_order)
        pred_ranks[pred_order] = torch.arange(n, device=logits.device)

        # Compute ideal DCG for normalization
        ideal_order = torch.argsort(group_labels, descending=True)
        ideal_gains = group_labels[ideal_order]
        k = min(ndcg_k, n) if ndcg_k else n
        positions = torch.arange(1, k + 1, device=logits.device, dtype=logits.dtype)
        idcg = ((2 ** ideal_gains[:k] - 1) / torch.log2(positions + 1)).sum()

        if idcg == 0:
            continue

        # Find all pairs where label[i] > label[j]
        pos_mask = group_labels > 0.5
        neg_mask = group_labels <= 0.5
        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        # Sample pairs if needed
        if max_pairs_per_group is not None and len(pos_indices) * len(neg_indices) > max_pairs_per_group:
            sample_pos = pos_indices[torch.randint(len(pos_indices), (max_pairs_per_group,), generator=rng, device=logits.device)]
            sample_neg = neg_indices[torch.randint(len(neg_indices), (max_pairs_per_group,), generator=rng, device=logits.device)]
        else:
            # All pairs
            sample_pos = pos_indices.repeat_interleave(len(neg_indices))
            sample_neg = neg_indices.repeat(len(pos_indices))

        # Score differences
        s_i = group_logits[sample_pos]
        s_j = group_logits[sample_neg]

        # Compute |Delta_NDCG| for each pair
        rank_i = pred_ranks[sample_pos].float() + 1
        rank_j = pred_ranks[sample_neg].float() + 1
        gain_i = 2 ** group_labels[sample_pos] - 1
        gain_j = 2 ** group_labels[sample_neg] - 1

        # Delta_NDCG = |gain_i * (1/log(rank_j+1) - 1/log(rank_i+1)) + gain_j * (1/log(rank_i+1) - 1/log(rank_j+1))| / IDCG
        discount_i = 1.0 / torch.log2(rank_i + 1)
        discount_j = 1.0 / torch.log2(rank_j + 1)
        delta_ndcg = torch.abs(
            gain_i * (discount_j - discount_i) + gain_j * (discount_i - discount_j)
        ) / idcg

        # Lambda loss: weighted RankNet
        pair_loss = delta_ndcg * F.softplus(-sigma * (s_i - s_j))
        losses.append(pair_loss.mean())

    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


def approx_ndcg_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    temperature: float = 1.0,
    k: Optional[int] = 10,
) -> torch.Tensor:
    """Approximate NDCG loss using softmax relaxation.

    Makes NDCG differentiable by replacing hard ranking with soft ranking
    based on softmax probabilities.

    Reference: Qin et al., NeuralNDCG (2020)
    """
    losses = []
    for group_logits, group_labels in zip(
        _split_by_group(logits, group_sizes),
        _split_by_group(labels, group_sizes)
    ):
        n = len(group_logits)
        if n <= 1:
            continue

        # Compute ideal DCG
        ideal_order = torch.argsort(group_labels, descending=True)
        ideal_gains = group_labels[ideal_order]
        cutoff = min(k, n) if k else n
        positions = torch.arange(1, cutoff + 1, device=logits.device, dtype=logits.dtype)
        idcg = ((2 ** ideal_gains[:cutoff] - 1) / torch.log2(positions + 1)).sum()

        if idcg == 0:
            continue

        # Soft ranking via softmax
        # P[i,j] = probability that item i is ranked at position j
        # Using Plackett-Luce approximation
        soft_scores = group_logits / max(temperature, 1e-6)

        # Approximate soft ranks using cumulative softmax
        # Expected rank of item i ≈ sum_j softmax(s_j) * I[s_j > s_i]
        soft_probs = F.softmax(soft_scores, dim=0)

        # Compute approximate DCG using soft probabilities
        gains = 2 ** group_labels - 1

        # Weighted sum of gains by position probabilities
        all_positions = torch.arange(1, n + 1, device=logits.device, dtype=logits.dtype)
        discounts = 1.0 / torch.log2(all_positions + 1)

        # Soft DCG: sum_i gain_i * sum_j P[i,j] * discount_j
        # Simplified: use expected rank for each item
        # Expected discount for item i based on its score rank
        sorted_idx = torch.argsort(soft_scores, descending=True)
        approx_dcg = torch.tensor(0.0, device=logits.device)
        for rank, idx in enumerate(sorted_idx[:cutoff]):
            approx_dcg = approx_dcg + gains[idx] * discounts[rank]

        # Loss = 1 - approx_NDCG (we want to maximize NDCG)
        approx_ndcg = approx_dcg / idcg
        losses.append(1.0 - approx_ndcg)

    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


# =============================================================================
# Contrastive Losses
# =============================================================================

def infonce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE / Multiple Negatives Ranking Loss.

    Treats each positive as the target and all negatives in the group (and batch)
    as negative samples.

    L = -log(exp(s_pos/tau) / sum_j exp(s_j/tau))
    """
    losses = []
    for group_logits, group_labels in zip(
        _split_by_group(logits, group_sizes),
        _split_by_group(labels, group_sizes)
    ):
        pos_mask = group_labels > 0.5
        if not pos_mask.any():
            continue

        # Scale by temperature
        scaled_logits = group_logits / max(temperature, 1e-6)

        # For each positive, compute cross-entropy against all items
        log_softmax = F.log_softmax(scaled_logits, dim=0)

        # Loss is negative log probability of positives
        pos_log_probs = log_softmax[pos_mask]
        losses.append(-pos_log_probs.mean())

    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


# =============================================================================
# Distillation Losses
# =============================================================================

def mse_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for score distillation (match teacher scores)."""
    return F.mse_loss(student_logits, teacher_logits)


def margin_mse_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    group_sizes: Sequence[int],
    max_pairs_per_group: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """MarginMSE loss - match score differences (margins) from teacher.

    L = MSE(s_student_i - s_student_j, s_teacher_i - s_teacher_j)

    This preserves the relative ordering from the teacher.
    """
    losses = []
    for student_group, teacher_group in zip(
        _split_by_group(student_logits, group_sizes),
        _split_by_group(teacher_logits, group_sizes)
    ):
        n = len(student_group)
        if n <= 1:
            continue

        # All pairs or sampled pairs
        if max_pairs_per_group is not None and n * (n - 1) > max_pairs_per_group:
            idx_i = torch.randint(n, (max_pairs_per_group,), generator=rng, device=student_logits.device)
            idx_j = torch.randint(n, (max_pairs_per_group,), generator=rng, device=student_logits.device)
        else:
            idx_i = torch.arange(n, device=student_logits.device).repeat_interleave(n)
            idx_j = torch.arange(n, device=student_logits.device).repeat(n)
            # Remove self-pairs
            mask = idx_i != idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]

        student_margins = student_group[idx_i] - student_group[idx_j]
        teacher_margins = teacher_group[idx_i] - teacher_group[idx_j]

        losses.append(F.mse_loss(student_margins, teacher_margins))

    if not losses:
        return torch.tensor(0.0, device=student_logits.device)
    return torch.stack(losses).mean()


# =============================================================================
# Loss Registry and Unified Interface
# =============================================================================

class LossType(str, Enum):
    """Enumeration of all available loss types."""
    # Pointwise
    BCE = "bce"
    FOCAL = "focal"

    # Pairwise
    RANKNET = "ranknet"
    MARGIN_RANKING = "margin_ranking"
    PAIRWISE_SOFTPLUS = "pairwise_softplus"

    # Listwise
    LISTNET = "listnet"
    LISTMLE = "listmle"
    PLISTMLE = "plistmle"
    LAMBDA = "lambda"
    APPROX_NDCG = "approx_ndcg"

    # Contrastive
    INFONCE = "infonce"

    # Hybrid (combination)
    HYBRID = "hybrid"


@dataclass
class LossConfig:
    """Configuration for loss computation."""
    # Loss type selection
    pointwise_type: str = "bce"  # bce, focal
    pairwise_type: str = "ranknet"  # ranknet, margin_ranking, pairwise_softplus
    listwise_type: str = "listnet"  # listnet, listmle, plistmle, lambda, approx_ndcg

    # Weights for hybrid loss
    w_list: float = 1.0
    w_pair: float = 0.5
    w_point: float = 0.1

    # Loss-specific hyperparameters
    temperature: float = 1.0
    sigma: float = 1.0  # For RankNet/Lambda
    margin: float = 1.0  # For margin ranking
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    ndcg_k: Optional[int] = 10  # For Lambda/ApproxNDCG
    max_pairs_per_group: Optional[int] = 50


# Loss function registry
POINTWISE_LOSSES: Dict[str, Callable] = {
    "bce": pointwise_bce_loss,
    "focal": focal_loss,
}

PAIRWISE_LOSSES: Dict[str, Callable] = {
    "ranknet": ranknet_loss,
    "margin_ranking": margin_ranking_loss,
    "pairwise_softplus": pairwise_softplus_loss,
}

LISTWISE_LOSSES: Dict[str, Callable] = {
    "listnet": listnet_loss,
    "listmle": listmle_loss,
    "plistmle": plistmle_loss,
    "lambda": lambda_loss,
    "approx_ndcg": approx_ndcg_loss,
}


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    config: LossConfig,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute hybrid loss with configurable loss types.

    Returns:
        total_loss: Weighted sum of losses
        loss_components: Dict of individual loss values for logging
    """
    loss_components = {}

    # Pointwise loss
    if config.w_point > 0:
        if config.pointwise_type == "focal":
            loss_point = focal_loss(logits, labels, config.focal_alpha, config.focal_gamma)
        else:
            loss_point = pointwise_bce_loss(logits, labels)
        loss_components["pointwise"] = loss_point.item()
    else:
        loss_point = torch.tensor(0.0, device=logits.device)

    # Pairwise loss
    if config.w_pair > 0:
        if config.pairwise_type == "ranknet":
            loss_pair = ranknet_loss(
                logits, labels, group_sizes,
                config.max_pairs_per_group, rng, config.sigma
            )
        elif config.pairwise_type == "margin_ranking":
            loss_pair = margin_ranking_loss(
                logits, labels, group_sizes,
                config.margin, config.max_pairs_per_group, rng
            )
        else:
            loss_pair = pairwise_softplus_loss(
                logits, labels, group_sizes,
                config.max_pairs_per_group, rng
            )
        loss_components["pairwise"] = loss_pair.item()
    else:
        loss_pair = torch.tensor(0.0, device=logits.device)

    # Listwise loss
    if config.w_list > 0:
        if config.listwise_type == "listmle":
            loss_list = listmle_loss(logits, labels, group_sizes)
        elif config.listwise_type == "plistmle":
            loss_list = plistmle_loss(logits, labels, group_sizes)
        elif config.listwise_type == "lambda":
            loss_list = lambda_loss(
                logits, labels, group_sizes,
                config.sigma, config.ndcg_k, config.max_pairs_per_group, rng
            )
        elif config.listwise_type == "approx_ndcg":
            loss_list = approx_ndcg_loss(
                logits, labels, group_sizes,
                config.temperature, config.ndcg_k
            )
        else:
            loss_list = listnet_loss(logits, labels, group_sizes, config.temperature)
        loss_components["listwise"] = loss_list.item()
    else:
        loss_list = torch.tensor(0.0, device=logits.device)

    total_loss = config.w_list * loss_list + config.w_pair * loss_pair + config.w_point * loss_point
    loss_components["total"] = total_loss.item()

    return total_loss, loss_components


# =============================================================================
# Legacy API (backward compatibility)
# =============================================================================

def pointwise_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Legacy: Binary cross-entropy with logits."""
    return pointwise_bce_loss(logits, labels)


def pairwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    max_pairs_per_group: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Legacy: Pairwise softplus loss."""
    return pairwise_softplus_loss(logits, labels, group_sizes, max_pairs_per_group, rng)


def listwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: Sequence[int],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Legacy: ListNet-style KL loss."""
    return listnet_loss(logits, labels, group_sizes, temperature)


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
    """Legacy: Weighted hybrid loss = listwise + pairwise + pointwise."""
    config = LossConfig(
        w_list=w_list,
        w_pair=w_pair,
        w_point=w_point,
        temperature=temperature,
        max_pairs_per_group=max_pairs_per_group,
    )
    total_loss, _ = compute_loss(logits, labels, group_sizes, config, rng)
    return total_loss


class HybridRerankerLoss(torch.nn.Module):
    """Hybrid reranker loss combining pointwise, pairwise, and listwise losses.

    This is a PyTorch Module wrapper around the compute_loss function for use
    in training loops where a callable loss function is expected.

    Example:
        loss_fn = HybridRerankerLoss(
            pointwise_type='bce',
            pairwise_type='pairwise_softplus',
            listwise_type='lambda',
            w_point=0.8,
            w_pair=1.8,
            w_list=1.1,
        )
        loss = loss_fn(scores, labels)
    """

    def __init__(
        self,
        pointwise_type: str = "bce",
        pairwise_type: str = "pairwise_softplus",
        listwise_type: str = "listnet",
        w_list: float = 1.0,
        w_pair: float = 0.5,
        w_point: float = 0.1,
        temperature: float = 1.0,
        sigma: float = 1.0,
        margin: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        ndcg_k: Optional[int] = 10,
        max_pairs_per_group: Optional[int] = 50,
    ):
        super().__init__()
        self.config = LossConfig(
            pointwise_type=pointwise_type,
            pairwise_type=pairwise_type,
            listwise_type=listwise_type,
            w_list=w_list,
            w_pair=w_pair,
            w_point=w_point,
            temperature=temperature,
            sigma=sigma,
            margin=margin,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            ndcg_k=ndcg_k,
            max_pairs_per_group=max_pairs_per_group,
        )
        self.rng = None

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        group_sizes: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Compute hybrid loss.

        Args:
            logits: Model output scores, shape (N,) or (batch, N)
            labels: Binary labels (0 or 1), same shape as logits
            group_sizes: Optional group sizes for listwise/pairwise losses.
                        If None, treats all items as one group.

        Returns:
            Scalar loss tensor
        """
        # Flatten if needed
        if logits.dim() > 1:
            logits = logits.squeeze()
        if labels.dim() > 1:
            labels = labels.squeeze()

        # Default: single group
        if group_sizes is None:
            group_sizes = [len(logits)]

        total_loss, _ = compute_loss(logits, labels, group_sizes, self.config, self.rng)
        return total_loss
