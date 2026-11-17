"""Lightweight execution utilities shared across single-file attacks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch import nn


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


class SklearnModelAdapter:
    """Wrap any scikit-learn / tree / tabular estimator into a PyTorch black-box victim callable.

    Automatically bridges CPU/GPU ``torch.Tensor`` batches to NumPy, extracts ``log(predict_proba)``,
    ``decision_function``, or hard ``predict`` labels, and returns normalized ``(B, C)`` PyTorch tensors
    compatible with ``square_attack``, ``simba``, ``boundary``, ``hopskipjump``, ``evaluate_suite``, and
    ``audit_robustness``.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        prefer_proba: bool = True,
        prob_clip: float = 1e-6,
    ) -> None:
        if not any(hasattr(estimator, m) for m in ("predict_proba", "decision_function", "predict")):
            raise TypeError(
                f"Expected a scikit-learn-style estimator with predict_proba/decision_function/predict, got {type(estimator).__name__}."
            )
        self.estimator = estimator
        self.prefer_proba = bool(prefer_proba)
        self.prob_clip = float(prob_clip)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().reshape(x.shape[0], -1).cpu().numpy()
        out_dtype = x.dtype if x.is_floating_point() else torch.float32
        if self.prefer_proba and hasattr(self.estimator, "predict_proba"):
            probs = np.asarray(self.estimator.predict_proba(x_np), dtype=np.float32)
            if probs.ndim == 1:
                probs = np.stack([1.0 - probs, probs], axis=1)
            elif probs.shape[1] == 1:
                probs = np.concatenate([1.0 - probs, probs], axis=1)
            logits = np.log(np.clip(probs, self.prob_clip, 1.0))
            return torch.from_numpy(logits).to(device=x.device, dtype=out_dtype)
        if hasattr(self.estimator, "decision_function"):
            dec = np.asarray(self.estimator.decision_function(x_np), dtype=np.float32)
            if dec.ndim == 1:
                dec = dec[:, None]
            return torch.from_numpy(dec).to(device=x.device, dtype=out_dtype)
        preds = np.asarray(self.estimator.predict(x_np), dtype=np.int64).reshape(-1)
        return torch.from_numpy(preds).to(device=x.device)


def wrap_sklearn_model(
    estimator: Any,
    *,
    prefer_proba: bool = True,
    prob_clip: float = 1e-6,
) -> SklearnModelAdapter:
    """Wrap a scikit-learn / XGBoost / LightGBM estimator for black-box ``wannado.adv`` attacks."""
    return SklearnModelAdapter(estimator, prefer_proba=prefer_proba, prob_clip=prob_clip)
