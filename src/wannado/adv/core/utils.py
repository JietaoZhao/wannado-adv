"""Lightweight execution utilities shared across single-file attacks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def eval_mode(model_or_fn: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[None]:
    """Temporarily set an nn.Module to eval() and restore its original mode on exit."""
    if not isinstance(model_or_fn, nn.Module):
        yield
        return
    was_training = model_or_fn.training
    model_or_fn.eval()
    try:
        yield
    finally:
        model_or_fn.train(was_training)


def make_generator(seed: int | None, device: torch.device | str) -> torch.Generator | None:
    """Create a seeded torch.Generator on the target device when seed is provided."""
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def update_best_by_objective(
    best_delta: torch.Tensor,
    best_obj: torch.Tensor,
    cand_delta: torch.Tensor,
    cand_obj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep per-sample perturbations that strictly improve the attack objective."""
    improve_mask = cand_obj > best_obj
    updated_obj = torch.where(improve_mask, cand_obj, best_obj)
    mask_view = improve_mask.view((cand_delta.shape[0],) + (1,) * (cand_delta.ndim - 1))
    updated_delta = torch.where(mask_view, cand_delta, best_delta)
    return updated_delta, updated_obj
