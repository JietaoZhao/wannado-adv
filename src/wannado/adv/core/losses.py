"""Domain-agnostic loss resolution for classification, margin, DLR, regression, and custom objectives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def unwrap_model_outputs(outputs: Any) -> torch.Tensor:
    """Extract raw tensor from standard tensors, HuggingFace ModelOutput (``.logits``), or tuple outputs."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits"):
        return unwrap_model_outputs(outputs.logits)
    if isinstance(outputs, (tuple, list)) and outputs:
        return unwrap_model_outputs(outputs[0])
    return torch.as_tensor(outputs)


def normalize_classification_logits(outputs: Any) -> torch.Tensor:
    """Normalize binary single-logit ``(B, 1)`` / ``(B,)`` or HF ``.logits`` into ``(B, C>=2)`` or ``(B, C>=2, H, W)``."""
    logits = unwrap_model_outputs(outputs)
    if logits.ndim == 1:
        z = logits.unsqueeze(1)
        return torch.cat([-0.5 * z, 0.5 * z], dim=1)
    if logits.ndim >= 2 and logits.shape[1] == 1:
        return torch.cat([-0.5 * logits, 0.5 * logits], dim=1)
    return logits


def predict_labels(outputs: Any) -> torch.Tensor:
    """Extract predicted integer class IDs uniformly for multi-class, single-logit binary, segmentation, or label-only oracles."""
    out = unwrap_model_outputs(outputs)
    if out.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
        return out.reshape(out.shape[0], -1)[:, 0].to(torch.long) if out.ndim >= 2 else out.reshape(-1).to(torch.long)
    norm_logits = normalize_classification_logits(out)
    return norm_logits.argmax(dim=1).to(torch.long)


def _flatten_dense_logits(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
    """Flatten ``(B, C)`` or ``(B, C, H, W)`` logits and targets cleanly for margin/DLR losses."""
    norm = normalize_classification_logits(logits)
    batch_size = norm.shape[0]
    if norm.ndim == 2:
        return norm, y.reshape(-1).to(torch.long), batch_size, False
    if norm.ndim >= 3:
        c = norm.shape[1]
        perm = norm.permute(0, *range(2, norm.ndim), 1).reshape(-1, c)
        return perm, y.reshape(-1).to(torch.long), batch_size, True
    raise ValueError(f"Expected classification logits with ndim >= 1, got shape {tuple(norm.shape)}.")


def _margin_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-sample classification margin loss: max_{j != y} z_j - z_y.

    Supports ``(B, C)`` multi-class, single-logit binary ``(B, 1)`` / ``(B,)``, HuggingFace ``.logits``,
    and dense spatial segmentation ``(B, C, H, W)``. Negative when predicted correctly; positive once fooled.
    """
    flat_logits, flat_y, batch_size, is_dense = _flatten_dense_logits(logits, y)
    batch_idx = torch.arange(flat_logits.shape[0], device=flat_logits.device)
    correct_logit = flat_logits[batch_idx, flat_y]
    masked = flat_logits.clone()
    masked[batch_idx, flat_y] = float("-inf")
    max_other_logit = masked.amax(dim=1)
    margin = max_other_logit - correct_logit
    return margin.reshape(batch_size, -1).mean(dim=1) if is_dense else margin


def _dlr_loss(logits: torch.Tensor, y: torch.Tensor, targeted: bool = False) -> torch.Tensor:
    """Difference of Logits Ratio (DLR) loss (Croce & Hein, ICML 2020).

    Shift- and scale-invariant logit ratio supporting multi-class ``(B, C)``, single-logit binary
    ``(B, 1)`` / ``(B,)``, HuggingFace ``.logits``, and dense segmentation ``(B, C, H, W)``.
    """
    flat_logits, flat_y, batch_size, is_dense = _flatten_dense_logits(logits, y)
    num_classes = flat_logits.shape[1]
    if num_classes < 3:
        spread = (flat_logits.amax(dim=1) - flat_logits.amin(dim=1)).clamp(min=1e-12)
        margin_raw = _margin_loss(flat_logits, flat_y)
        out = margin_raw / spread
        return out.reshape(batch_size, -1).mean(dim=1) if is_dense else out

    z_sorted, idx_sorted = flat_logits.sort(dim=1, descending=True)
    batch_idx = torch.arange(flat_logits.shape[0], device=flat_logits.device)
    z_y = flat_logits[batch_idx, flat_y]
    z_other = torch.where(idx_sorted[:, 0] == flat_y, z_sorted[:, 1], z_sorted[:, 0])

    if targeted:
        if num_classes >= 4:
            denom = (z_sorted[:, 0] - 0.5 * (z_sorted[:, 2] + z_sorted[:, 3])).clamp(min=1e-12)
        else:
            denom = (z_sorted[:, 0] - 0.5 * (z_sorted[:, 1] + z_sorted[:, 2])).clamp(min=1e-12)
    else:
        denom = (z_sorted[:, 0] - z_sorted[:, 2]).clamp(min=1e-12)
    dlr = (z_other - z_y) / denom
    return dlr.reshape(batch_size, -1).mean(dim=1) if is_dense else dlr


def _mse_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample mean squared error for regression / signal reconstruction models."""
    p = unwrap_model_outputs(preds)
    diff = (p - target).reshape(p.shape[0], -1)
    if diff.is_complex():
        return (diff.real.pow(2) + diff.imag.pow(2)).mean(dim=1)
    return diff.pow(2).mean(dim=1)


def resolve_loss_fn(loss: str | LossFn | None, targeted: bool = False) -> LossFn:
    """Resolve a built-in loss name or custom callable into a per-sample loss function."""
    if callable(loss):
        return loss
    if loss is None or loss == "ce":
        return lambda logits, y: F.cross_entropy(normalize_classification_logits(logits), y.to(torch.long), reduction="none")
    if loss == "margin":
        return _margin_loss
    if loss == "dlr":
        return lambda logits, y: _dlr_loss(logits, y, targeted=targeted)
    if loss == "mse":
        return _mse_loss
    raise ValueError(f"Unknown loss '{loss}'. Expected 'ce', 'margin', 'dlr', 'mse', or a callable.")


def outputs_to_objective(
    outputs: Any,
    y: torch.Tensor,
    loss_fn: LossFn,
    targeted: bool = False,
) -> torch.Tensor:
    """Convert precomputed model outputs (including HuggingFace ``.logits`` objects) into per-sample scalar objective."""
    out_t = unwrap_model_outputs(outputs)
    raw_loss = loss_fn(out_t, y)

    if raw_loss.is_complex():
        raw_loss = raw_loss.real
    if raw_loss.ndim == 0:
        raw_loss = raw_loss.expand(out_t.shape[0])
    elif raw_loss.ndim > 1:
        raw_loss = raw_loss.reshape(out_t.shape[0], -1).mean(dim=1)

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
