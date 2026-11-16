"""Domain-agnostic loss resolution for classification, margin, DLR, regression, and custom objectives."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _margin_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-sample classification margin loss: max_{j != y} z_j - z_y.

    Negative when class ``y`` is predicted with margin; positive once fooled.
    """
    if logits.ndim != 2:
        raise ValueError(f"Margin loss expects 2D logits (B, C), got shape {tuple(logits.shape)}.")
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    correct_logit = logits[batch_idx, y]
    masked = logits.clone()
    masked[batch_idx, y] = float("-inf")
    max_other_logit = masked.amax(dim=1)
    return max_other_logit - correct_logit


def _dlr_loss(logits: torch.Tensor, y: torch.Tensor, targeted: bool = False) -> torch.Tensor:
    """Difference of Logits Ratio (DLR) loss (Croce & Hein, ICML 2020).

    Shift- and scale-invariant logit ratio designed to resist Softmax saturation
    and rescaling-based gradient obfuscation. Falls back to normalized margin
    when ``num_classes < 3``.
    """
    if logits.ndim != 2:
        raise ValueError(f"DLR loss expects 2D logits (B, C), got shape {tuple(logits.shape)}.")
    num_classes = logits.shape[1]
    if num_classes < 3:
        spread = (logits.amax(dim=1) - logits.amin(dim=1)).clamp(min=1e-12)
        return _margin_loss(logits, y) / spread

    z_sorted, idx_sorted = logits.sort(dim=1, descending=True)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    z_y = logits[batch_idx, y]
    z_other = torch.where(idx_sorted[:, 0] == y, z_sorted[:, 1], z_sorted[:, 0])

    if targeted:
        denom = (z_sorted[:, 0] - 0.5 * (z_sorted[:, 1] + z_sorted[:, 2])).clamp(min=1e-12)
    else:
        denom = (z_sorted[:, 0] - z_sorted[:, 2]).clamp(min=1e-12)
    return (z_other - z_y) / denom


def _mse_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample mean squared error for regression / signal reconstruction models."""
    diff = (preds - target).reshape(preds.shape[0], -1)
    if diff.is_complex():
        return (diff.real.pow(2) + diff.imag.pow(2)).mean(dim=1)
    return diff.pow(2).mean(dim=1)


def resolve_loss_fn(loss: str | LossFn | None, targeted: bool = False) -> LossFn:
    """Resolve a built-in loss name or custom callable into a per-sample loss function."""
    if callable(loss):
        return loss
    if loss is None or loss == "ce":
        return lambda logits, y: F.cross_entropy(logits, y, reduction="none")
    if loss == "margin":
        return _margin_loss
    if loss == "dlr":
        return lambda logits, y: _dlr_loss(logits, y, targeted=targeted)
    if loss == "mse":
        return _mse_loss
    raise ValueError(f"Unknown loss '{loss}'. Expected 'ce', 'margin', 'dlr', 'mse', or a callable.")


def outputs_to_objective(
    outputs: torch.Tensor,
    y: torch.Tensor,
    loss_fn: LossFn,
    targeted: bool = False,
) -> torch.Tensor:
    """Convert precomputed model outputs into per-sample scalar objective to MAXIMIZE."""
    raw_loss = loss_fn(outputs, y)

    if raw_loss.is_complex():
        raw_loss = raw_loss.real
    if raw_loss.ndim == 0:
        raw_loss = raw_loss.expand(outputs.shape[0])
    elif raw_loss.ndim > 1:
        raw_loss = raw_loss.reshape(outputs.shape[0], -1).mean(dim=1)

    return -raw_loss if targeted else raw_loss


def compute_attack_objective(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_adv: torch.Tensor,
    y: torch.Tensor,
    loss_fn: LossFn,
    targeted: bool = False,
) -> torch.Tensor:
    """Evaluate the per-sample scalar objective to MAXIMIZE during an attack.

    For untargeted attacks, the objective equals ``+loss(model(x_adv), y)``.
    For targeted attacks, the objective equals ``-loss(model(x_adv), y)``.
    """
    outputs = model_or_fn(x_adv)
    return outputs_to_objective(outputs, y, loss_fn, targeted=targeted)
