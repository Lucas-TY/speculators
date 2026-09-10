"""Metrics and loss functions for DFlash draft model."""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from speculators.losses import (
    LossConfig,
    compound_loss,
    dflash_loss_decay,
    dpace_loss_decay,
    dpard_loss_decay,
    kl_div_loss,
    tv_loss,
)
from speculators.models.metrics import compute_accuracy_multi_step

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


def compute_metrics(  # noqa: C901 - keep the two unary objectives and shared metrics together
    logits: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, num_anchors*block_size]
    block_size: int = 1,
    gamma: float = 4.0,
    loss_config: LossConfig | None = None,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = False,
    dpard_alpha: float = 0.5,
    dflash_loss_reduction: str = "token-mean",
    tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
) -> tuple[torch.Tensor, dict]:
    """Compute loss and accuracy metrics for draft model predictions.

    Args:
        logits: Model logits [1, T, V]
        targets: Target logits [1, T, V]
        loss_mask: Binary mask [1, T]
        block_size: Block size for per-position metrics
        gamma: Temperature for exponential decay in loss weighting
        loss_config: Mapping of ``{name: (loss_fn, weight)}``
        per_position_loss_weight: Weighting option for per-position block-drafting loss
        dpace_alpha: Smoothing constant for D-Pace loss weighting

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict contains:
            - loss: Scalar loss value
            - full_acc: Overall accuracy
            - position {i} acc: Accuracy at position i within blocks
            - eal: Expected Accepted Length (headline speculative-decoding metric)
    """
    if loss_config is None:
        loss_config = _DEFAULT_LOSS_CONFIG
    if dflash_loss_reduction not in {"token-mean", "sequence-mean-valid-anchor"}:
        raise ValueError("Unknown DFlash loss reduction")
    reduction_block_size = (
        block_size if dflash_loss_reduction == "sequence-mean-valid-anchor" else None
    )
    seq_len = logits.shape[1]
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)  # shape: [1, T]

    acceptance = None
    if per_position_loss_weight == "dpard" or reduction_block_size is not None:
        with torch.no_grad():
            # Full-vocabulary unary overlap; never a selector or hard-target proxy.
            acceptance = (1.0 - tv_loss_fn(logits, targets)).clamp(0.0, 1.0)

    if per_position_loss_weight == "dpard":
        if set(loss_config) != {"renyi_half"} or loss_config["renyi_half"][1] != 1.0:
            raise ValueError("D-PARD requires exactly unit-weight loss_fn=renyi_half")
        credit = dpard_loss_decay(
            acceptance,
            loss_mask,
            block_size,
            dpard_alpha,
            start_pos=0 if sample_from_anchor else 1,
        )

        def decay_fn(_pos, **_kwargs):
            return credit
    elif per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )

    loss, term_losses = compound_loss(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=decay_fn,
        reduction_block_size=reduction_block_size,
    )

    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)

    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )

    ones = torch.tensor(1.0, device=logits.device)
    metrics: dict[str, Any] = {}
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    # Start position: 0 if sample_from_anchor else 1 (skip anchor)
    start_pos = 0 if sample_from_anchor else 1
    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()

    # EAL = sum_k prod_{i<=k} acc_i over drafted positions
    eal = torch.zeros((), device=logits.device)
    cum = torch.ones((), device=logits.device)
    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]
        acc = correct_per_pos[pos] / total_per_pos[pos].clamp(min=1.0)
        cum = cum * acc
        eal = eal + cum
    metrics["eal_sum"] = eal
    metrics["eal_total"] = ones.clone()
    if acceptance is not None:
        with torch.no_grad():
            active = loss_mask.reshape(-1, block_size)[:, start_pos:] > 0
            block_valid = active.any(-1)
            overlap = acceptance.reshape(-1, block_size)[:, start_pos:]
            # Raw (unsmoothed) unary overlap proxy, including the bonus token.
            tau = 1.0 + torch.cumprod(overlap * active, -1).sum(-1)
            metrics["acceptance_sum"] = (acceptance * loss_mask).sum()
            metrics["acceptance_total"] = loss_mask.sum()
            metrics["tau_sum"] = (tau * block_valid).sum()
            metrics["tau_total"] = block_valid.sum()
    return loss, metrics
