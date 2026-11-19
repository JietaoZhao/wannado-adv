"""Lightweight execution utilities shared across single-file attacks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch import nn


class NormalizedModelWrapper(nn.Module):
    """Differentiable input normalization wrapper ``model((x - mean) / std)`` for ``[0, 1]`` pixel attacks.

    Solves the #1 real-world vision pitfall (``torchvision`` / ``timm`` ImageNet/CIFAR normalization):
    attacks perturb raw ``[0, 1]`` inputs under exact ``eps=8/255`` and ``clip_bounds=(0, 1)``, while
    gradients flow seamlessly through channel-wise ``(x - mean) / std`` into the wrapped classifier.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        model_or_fn: Callable[[torch.Tensor], torch.Tensor],
        mean: tuple[float, ...] | list[float] | torch.Tensor = (0.485, 0.456, 0.406),
        std: tuple[float, ...] | list[float] | torch.Tensor = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self._base_model = model_or_fn
        if isinstance(model_or_fn, nn.Module):
            self.base_module: nn.Module | None = model_or_fn
            self.training = model_or_fn.training
        else:
            self.base_module = None
            self.training = False
        m_t = torch.as_tensor(mean, dtype=torch.float32)
        s_t = torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-12)
        self.register_buffer("mean", m_t)
        self.register_buffer("std", s_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mean.to(device=x.device, dtype=x.dtype)
        s = self.std.to(device=x.device, dtype=x.dtype)
        if m.ndim == 1 and x.ndim >= 3 and x.shape[1] == m.shape[0]:
            view = (1, m.shape[0]) + (1,) * (x.ndim - 2)
            m = m.view(view)
            s = s.view(view)
        elif m.ndim == 1 and x.ndim == 2 and x.shape[1] == m.shape[0]:
            m = m.view(1, -1)
            s = s.view(1, -1)
        return self._base_model((x - m) / s)


def infer_model_device(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    fallback: torch.device,
) -> torch.device:
    """Return the device of an ``nn.Module``'s parameters/buffers, falling back cleanly for callables."""
    if isinstance(model_or_fn, nn.Module):
        for p in model_or_fn.parameters():
            return p.device
        for b in model_or_fn.buffers():
            return b.device
    return fallback


def iter_train_batches(
    train_data: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    *,
    batch_size: int = 64,
    device: torch.device | str | None = None,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Stream mini-batches onto ``device`` one batch at a time (O(batch_size) VRAM) with optional Tensor shuffle."""
    dev = torch.device(device) if device is not None else None
    if isinstance(train_data, torch.Tensor):
        if y is None:
            raise ValueError("Target tensor `y` must be provided when `train_data` is a torch.Tensor.")
        x_t = train_data.detach()
        y_t = y.detach()
        n = x_t.shape[0]
        if n == 0:
            raise ValueError("Training tensor `train_data` is empty.")
        chunk = max(1, int(batch_size))
        if shuffle and n > 1:
            perm = torch.randperm(n, generator=generator, device=x_t.device)
            x_t = x_t[perm]
            y_t = y_t[perm]
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            xb = x_t[start:end]
            yb = y_t[start:end]
            if dev is not None:
                xb = xb.to(dev)
                yb = yb.to(dev)
            yield xb, yb
        return

    yielded = 0
    for xb, yb in iter_eval_batches(train_data, None, batch_size=None, device=dev):
        yielded += 1
        yield xb, yb
    if yielded == 0:
        raise ValueError(
            "Training iterable yielded zero batches. Pass a reusable PyTorch DataLoader or Tensor (x, y) "
            "rather than an exhausted single-pass generator."
        )


def squeeze_class_targets(y: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
    """Normalize ``(B,)`` or ``(B, 1)`` integer classification labels to 1D ``(B,)`` ``torch.long``."""
    if y.ndim == 2 and y.shape[1] == 1:
        return y.squeeze(1).to(torch.long)
    if batch_size is not None and y.ndim == 0 and batch_size == 1:
        return y.reshape(1).to(torch.long)
    return y.to(torch.long)


class DefendedModelWrapper(nn.Module):
    """Composable ``nn.Module`` wrapping ``model_or_fn(defense_fn(x))`` while preserving ``eval()`` isolation."""

    def __init__(
        self,
        model_or_fn: Callable[[torch.Tensor], torch.Tensor],
        defense_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self._base_model = model_or_fn
        self._defense_fn = defense_fn
        if isinstance(model_or_fn, nn.Module):
            self.base_module: nn.Module | None = model_or_fn
            self.training = model_or_fn.training
        else:
            self.base_module = None
            self.training = False
        if isinstance(defense_fn, nn.Module):
            self.defense_module: nn.Module | None = defense_fn
        else:
            self.defense_module = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._base_model(self._defense_fn(x))


def iter_eval_batches(
    x_or_loader: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    *,
    batch_size: int | None = None,
    max_samples: int | None = None,
    device: torch.device | str | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(x_batch, y_batch)`` chunks from either ``(x, y)`` tensors or a PyTorch ``DataLoader``/iterable."""
    seen = 0
    cap = None if (max_samples is None or max_samples <= 0) else int(max_samples)
    dev = torch.device(device) if device is not None else None

    if isinstance(x_or_loader, torch.Tensor):
        if y is None:
            raise ValueError("Target tensor `y` must be provided when `x` is a torch.Tensor.")
        x_t = x_or_loader.detach()
        y_t = y.detach()
        n = x_t.shape[0] if cap is None else min(x_t.shape[0], cap)
        chunk = n if (batch_size is None or batch_size <= 0) else int(batch_size)
        for start in range(0, n, max(chunk, 1)):
            end = min(start + chunk, n)
            xb = x_t[start:end]
            yb = y_t[start:end]
            if dev is not None:
                xb = xb.to(dev)
                yb = yb.to(dev)
            yield xb, yb
        return

    for item in x_or_loader:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError("DataLoader/iterable items must unpack as (x_batch, y_batch).")
        xb_raw, yb_raw = item[0], item[1]
        xb = xb_raw.detach() if isinstance(xb_raw, torch.Tensor) else torch.as_tensor(xb_raw)
        yb = yb_raw.detach() if isinstance(yb_raw, torch.Tensor) else torch.as_tensor(yb_raw)
        if cap is not None:
            rem = cap - seen
            if rem <= 0:
                break
            if xb.shape[0] > rem:
                xb = xb[:rem]
                yb = yb[:rem]
        if batch_size is not None and batch_size > 0 and xb.shape[0] > int(batch_size):
            sub = int(batch_size)
            for s in range(0, xb.shape[0], sub):
                e = min(s + sub, xb.shape[0])
                xs = xb[s:e].to(dev) if dev is not None else xb[s:e]
                ys = yb[s:e].to(dev) if dev is not None else yb[s:e]
                seen += xs.shape[0]
                yield xs, ys
        else:
            if dev is not None:
                xb = xb.to(dev)
                yb = yb.to(dev)
            seen += xb.shape[0]
            yield xb, yb
        if cap is not None and seen >= cap:
            break


def make_reiterable_eval_batches(
    x_or_loader: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    *,
    batch_size: int | None = None,
    max_samples: int | None = None,
) -> Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]:
    """Return a zero-arg factory yielding deterministic ``(x_batch, y_batch)`` chunks.

    Streams lazily in O(batch_size) memory for ``torch.Tensor``, ``Sequence``s, and
    non-shuffled ``DataLoader``s (``SequentialSampler``). One-shot iterators or shuffled
    loaders are cached once so per-sample alignment stays exact across multi-attack passes.
    """
    from collections.abc import Sequence as _Seq

    from torch.utils.data import DataLoader as _DL
    from torch.utils.data import SequentialSampler as _SeqSampler

    is_deterministic_reiterable = (
        isinstance(x_or_loader, (torch.Tensor, _Seq))
        or (isinstance(x_or_loader, _DL) and isinstance(getattr(x_or_loader, "sampler", None), _SeqSampler))
    )
    if is_deterministic_reiterable:
        return lambda: iter_eval_batches(x_or_loader, y, batch_size=batch_size, max_samples=max_samples)
    cached = list(iter_eval_batches(x_or_loader, y, batch_size=batch_size, max_samples=max_samples))
    return lambda: iter(cached)


def materialize_eval_tensors(
    x_or_loader: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    *,
    max_samples: int | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect a bounded pair of ``(x, y)`` tensors from tensors or a ``DataLoader``/iterable."""
    if isinstance(x_or_loader, torch.Tensor):
        if y is None:
            raise ValueError("Target tensor `y` must be provided when `x` is a torch.Tensor.")
        x_t = x_or_loader.detach()
        y_t = y.detach()
        if max_samples is not None and max_samples > 0:
            x_t = x_t[: int(max_samples)]
            y_t = y_t[: int(max_samples)]
        if device is not None:
            dev = torch.device(device)
            x_t = x_t.to(dev)
            y_t = y_t.to(dev)
        return x_t, y_t

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for xb, yb in iter_eval_batches(x_or_loader, max_samples=max_samples, device=device):
        xs.append(xb)
        ys.append(yb)
    if not xs:
        raise ValueError("Evaluation DataLoader/iterable yielded zero samples.")
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


@contextmanager
def eval_mode(model_or_fn: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[None]:
    """Temporarily set an nn.Module to eval() and losslessly restore every submodule's exact training flag."""
    if not isinstance(model_or_fn, nn.Module):
        yield
        return
    saved_states = [(mod, mod.training) for mod in model_or_fn.modules()]
    model_or_fn.eval()
    try:
        yield
    finally:
        for mod, was_training in saved_states:
            mod.training = was_training


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
        preds_arr = np.asarray(self.estimator.predict(x_np))
        if np.issubdtype(preds_arr.dtype, np.floating):
            return torch.from_numpy(preds_arr.astype(np.float32)).to(device=x.device, dtype=out_dtype)
        preds = preds_arr.astype(np.int64).reshape(-1)
        return torch.from_numpy(preds).to(device=x.device)


def wrap_normalized_model(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    mean: tuple[float, ...] | list[float] | torch.Tensor = (0.485, 0.456, 0.406),
    std: tuple[float, ...] | list[float] | torch.Tensor = (0.229, 0.224, 0.225),
) -> NormalizedModelWrapper:
    """Wrap a vision/tabular model with differentiable ``(x - mean) / std`` input normalization."""
    return NormalizedModelWrapper(model_or_fn, mean=mean, std=std)


def wrap_sklearn_model(
    estimator: Any,
    *,
    prefer_proba: bool = True,
    prob_clip: float = 1e-6,
) -> SklearnModelAdapter:
    """Wrap a scikit-learn / XGBoost / LightGBM estimator for black-box ``wannado.adv`` attacks."""
    return SklearnModelAdapter(estimator, prefer_proba=prefer_proba, prob_clip=prob_clip)
