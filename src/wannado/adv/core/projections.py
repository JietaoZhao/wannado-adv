"""Domain-agnostic norm projections and physics budget conversions.

Supports standard real-valued tensors (vision, tabular, DRL states) as well as
complex-valued baseband signals (torch.complex64/complex128) and wireless
Jamming-to-Signal Ratio (JSR in dB) budgets.
"""

from __future__ import annotations

import math
from typing import Any

import torch

SUPPORTED_NORMS = frozenset({"linf", "l2", "l1", "relative_l2", "jsr_db"})


def _real_dtype_for(x: torch.Tensor) -> torch.dtype:
    if x.is_complex():
        return torch.float32 if x.dtype == torch.complex64 else torch.float64
    return x.dtype if x.is_floating_point() else torch.float32


def coerce_clip_bounds(
    clip_bounds: tuple[Any, Any],
    x_ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert scalar, NumPy, list, or Tensor ``(low, high)`` bounds into broadcastable tensors."""
    if not isinstance(clip_bounds, (tuple, list)) or len(clip_bounds) != 2:
        raise ValueError(f"clip_bounds must be a 2-tuple (min, max), got {clip_bounds!r}.")
    real_dtype = _real_dtype_for(x_ref)
    low_raw, high_raw = clip_bounds
    lo_t = torch.as_tensor(low_raw, device=x_ref.device, dtype=real_dtype)
    hi_t = torch.as_tensor(high_raw, device=x_ref.device, dtype=real_dtype)

    def _align(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            return t
        if t.shape == x_ref[0].shape:
            return t.unsqueeze(0)
        if t.ndim == 1 and x_ref.ndim >= 2 and t.shape[0] == x_ref.shape[1]:
            return t.view((1, t.shape[0]) + (1,) * (x_ref.ndim - 2))
        if t.numel() == x_ref[0].numel():
            return t.reshape((1,) + x_ref[0].shape)
        return t

    lo_t = _align(lo_t)
    hi_t = _align(hi_t)
    if bool((lo_t > hi_t).any().item()):
        raise ValueError(f"Invalid clip_bounds: min > max detected (min={lo_t}, max={hi_t}).")
    return lo_t, hi_t


def clamp_to_bounds(x_val: torch.Tensor, clip_bounds: tuple[Any, Any] | None) -> torch.Tensor:
    """Clamp real-valued tensor to scalar OR per-column/element ``(low, high)`` bounds."""
    if clip_bounds is None:
        return x_val
    if x_val.is_complex():
        raise ValueError("clip_bounds is only valid for real-valued tensors.")
    lo_t, hi_t = coerce_clip_bounds(clip_bounds, x_val)
    return torch.minimum(torch.maximum(x_val, lo_t), hi_t)


def safe_l2_norm(
    x: torch.Tensor,
    *,
    dim: int | tuple[int, ...] | None = 1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Compute Euclidean L2 norm safely across CPU/CUDA/MPS and real/complex tensors.

    PyTorch MPS currently raises ``RuntimeError: norm ops are not supported for complex yet``
    on ``torch.linalg.vector_norm`` for ``complex64``; ``real^2 + imag^2`` matches L2 exactly.
    """
    if x.is_complex() and x.device.type == "mps":
        sq = x.real.pow(2) + x.imag.pow(2)
        summed = sq.sum() if dim is None else sq.sum(dim=dim, keepdim=keepdim)
        return summed.clamp(min=0.0).sqrt()
    return torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=keepdim)


def _batch_l2_norm(x: torch.Tensor) -> torch.Tensor:
    b = x.shape[0]
    return safe_l2_norm(x.reshape(b, -1), dim=1)


def resolve_eps_budget(x: torch.Tensor, eps: float | torch.Tensor, norm: str) -> torch.Tensor:
    """Resolve a scalar or per-sample perturbation budget into a broadcastable tensor.

    Args:
        x: Clean input tensor with batch dimension at axis 0.
        eps: Perturbation magnitude (scalar or ``(B,)`` tensor). For ``norm="jsr_db"``,
            this is the Jamming-to-Signal Ratio in decibels (may be negative, e.g. ``-10.0``).
            For all other norms, must be ``>= 0``.
        norm: Constraint type (``"linf"``, ``"l2"``, ``"l1"``, ``"relative_l2"``, ``"jsr_db"``).

    Returns:
        Real-valued tensor of shape ``(B, 1, ..., 1)`` representing the maximum
        allowed element-wise magnitude (for ``"linf"``), L1 radius (for ``"l1"``),
        or per-sample L2 radius (for ``"l2"``, ``"relative_l2"``, and ``"jsr_db"``).
    """
    if norm not in SUPPORTED_NORMS:
        raise ValueError(f"Unsupported norm '{norm}'. Choose from {sorted(SUPPORTED_NORMS)}.")
    if x.ndim < 1:
        raise ValueError(f"Input tensor must have at least 1 batch dimension, got shape {tuple(x.shape)}.")

    real_dtype = _real_dtype_for(x)
    view_shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    if isinstance(eps, torch.Tensor) and eps.ndim > 0:
        eps_t = eps.to(device=x.device, dtype=real_dtype).reshape(-1)
        if eps_t.numel() == 1:
            eps_t = eps_t.expand(x.shape[0])
        elif eps_t.numel() != x.shape[0]:
            raise ValueError(
                f"Per-sample eps tensor must match batch size {x.shape[0]}, got shape {tuple(eps.shape)}."
            )
        eps_t = eps_t.view(view_shape)
        if norm != "jsr_db" and bool((eps_t < 0).any().item()):
            raise ValueError(f"eps must be >= 0 for norm='{norm}', got min={float(eps_t.min().item())}.")
        if norm in ("linf", "l2", "l1"):
            return eps_t
        x_norm = _batch_l2_norm(x).to(real_dtype).view(view_shape)
        if norm == "relative_l2":
            return eps_t * x_norm
        scale = torch.where(
            torch.isinf(eps_t) & (eps_t < 0),
            torch.zeros_like(eps_t),
            torch.pow(10.0, eps_t.to(torch.float64) / 20.0).to(real_dtype),
        )
        return scale * x_norm

    eps_f = float(eps.item()) if isinstance(eps, torch.Tensor) else float(eps)
    if norm != "jsr_db" and eps_f < 0:
        raise ValueError(f"eps must be >= 0 for norm='{norm}', got {eps_f}.")

    if norm in ("linf", "l2", "l1"):
        return torch.full(view_shape, eps_f, device=x.device, dtype=real_dtype)

    x_norm = _batch_l2_norm(x).to(real_dtype)
    if norm == "relative_l2":
        return (eps_f * x_norm).view(view_shape)

    # JSR_dB = 10 * log10(P_jam / P_sig) = 20 * log10(||delta||_2 / ||x||_2)
    if math.isinf(eps_f) and eps_f < 0:
        scale_f = 0.0
    else:
        scale_f = math.pow(10.0, eps_f / 20.0)
    return (scale_f * x_norm).view(view_shape)


def step_direction(grad: torch.Tensor, norm: str) -> torch.Tensor:
    """Compute a unit-scaled ascent direction matching the target norm geometry."""
    if norm not in SUPPORTED_NORMS:
        raise ValueError(f"Unsupported norm '{norm}'. Choose from {sorted(SUPPORTED_NORMS)}.")

    if norm == "linf":
        # torch.sign raises on complex tensors; torch.sgn returns z / |z| (0 if |z| == 0).
        return torch.sgn(grad) if grad.is_complex() else torch.sign(grad)

    if norm == "l1":
        # Exact steepest L1 ascent aligns all weight onto the dominant magnitude coordinate.
        b = grad.shape[0]
        flat = grad.reshape(b, -1)
        max_idx = flat.abs().argmax(dim=1, keepdim=True)
        picked = torch.gather(flat, 1, max_idx)
        unit_val = torch.sgn(picked) if grad.is_complex() else torch.sign(picked)
        out = torch.zeros_like(flat)
        out.scatter_(1, max_idx, unit_val)
        return out.reshape_as(grad)

    view_shape = (grad.shape[0],) + (1,) * (grad.ndim - 1)
    g_norm = _batch_l2_norm(grad).to(_real_dtype_for(grad)).view(view_shape)
    return grad / g_norm.clamp(min=1e-12)


def adam_step_direction(update: torch.Tensor, norm: str) -> torch.Tensor:
    """Scale an adaptive Adam update vector to match the target constraint geometry."""
    if norm not in SUPPORTED_NORMS:
        raise ValueError(f"Unsupported norm '{norm}'. Choose from {sorted(SUPPORTED_NORMS)}.")

    if norm == "linf":
        if update.is_complex():
            mag = update.abs()
            return torch.where(mag > 1.0, update / mag.clamp(min=1e-12), update)
        return torch.clamp(update, -1.0, 1.0)

    if norm == "l1":
        b = update.shape[0]
        u_l1 = update.abs().reshape(b, -1).sum(dim=1).to(_real_dtype_for(update))
        view_shape = (b,) + (1,) * (update.ndim - 1)
        return update / u_l1.view(view_shape).clamp(min=1e-12)

    view_shape = (update.shape[0],) + (1,) * (update.ndim - 1)
    u_norm = _batch_l2_norm(update).to(_real_dtype_for(update)).view(view_shape)
    return update / u_norm.clamp(min=1e-12)


def _project_l1_box(
    delta: torch.Tensor,
    x: torch.Tensor,
    eps_budget: torch.Tensor,
    clip_bounds: tuple[Any, Any] | None,
) -> torch.Tensor:
    """Exact Euclidean projection onto the L1 ball intersected with optional box bounds."""
    b = delta.shape[0]
    z = delta.reshape(b, -1)
    eps_flat = eps_budget.reshape(b, 1)
    real_dtype = _real_dtype_for(delta)

    lo_b: torch.Tensor | None = None
    hi_b: torch.Tensor | None = None
    if clip_bounds is not None:
        lo_t, hi_t = coerce_clip_bounds(clip_bounds, x)
        x_box = torch.minimum(torch.maximum(x, lo_t), hi_t)
        excess_l1 = (x_box - x).abs().reshape(b, -1).sum(dim=1, keepdim=True).to(real_dtype)
        if bool((excess_l1 > eps_flat + 1e-4).any().item()):
            raise ValueError(
                f"Clean inputs lie outside clip_bounds by more than the L1 budget "
                f"(required L1 shift up to {float(excess_l1.max().item()):.4g} > budget "
                f"{float(eps_flat.min().item()):.4g}). Pass clip_bounds=None for standardized or unbounded inputs."
            )
        lo_b = (torch.broadcast_to(lo_t, x.shape) - x).reshape(b, -1)
        hi_b = (torch.broadcast_to(hi_t, x.shape) - x).reshape(b, -1)
        z_cand = torch.minimum(torch.maximum(z, lo_b), hi_b)
    else:
        z_cand = z

    cand_l1 = z_cand.abs().sum(dim=1, keepdim=True).to(real_dtype)
    needs_proj = cand_l1 > eps_flat
    if not needs_proj.any():
        return z_cand.reshape_as(delta)

    mag = z.abs().to(real_dtype)
    phase = torch.sgn(z) if z.is_complex() else torch.sign(z)
    lo_lam = torch.zeros_like(eps_flat)
    hi_lam = mag.amax(dim=1, keepdim=True).clamp(min=0.0)

    for _ in range(36):
        mid_lam = 0.5 * (lo_lam + hi_lam)
        shrunk_mag = (mag - mid_lam).clamp(min=0.0)
        cand = phase * shrunk_mag if not z.is_complex() else phase * shrunk_mag.to(z.dtype)
        if lo_b is not None and hi_b is not None:
            cand = torch.minimum(torch.maximum(cand, lo_b), hi_b)
        l1_val = cand.abs().sum(dim=1, keepdim=True).to(real_dtype)
        too_large = l1_val > eps_flat
        lo_lam = torch.where(too_large, mid_lam, lo_lam)
        hi_lam = torch.where(too_large, hi_lam, mid_lam)

    shrunk_final = (mag - hi_lam).clamp(min=0.0)
    proj = phase * shrunk_final if not z.is_complex() else phase * shrunk_final.to(z.dtype)
    if lo_b is not None and hi_b is not None:
        proj = torch.minimum(torch.maximum(proj, lo_b), hi_b)
    out = torch.where(needs_proj, proj, z_cand)
    return out.reshape_as(delta)


def project_perturbation(
    delta: torch.Tensor,
    x: torch.Tensor,
    eps_budget: torch.Tensor,
    norm: str,
    clip_bounds: tuple[Any, Any] | None = None,
    perturb_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project perturbation ``delta`` onto the norm ball, optional immutable-feature mask, and box bounds."""
    if clip_bounds is not None and x.is_complex():
        raise ValueError(
            "clip_bounds=(min, max) is only valid for real-valued tensors; "
            "for complex signals use norm constraints such as 'l2', 'relative_l2', or 'jsr_db'."
        )

    if perturb_mask is not None:
        mask = perturb_mask.to(device=delta.device, dtype=delta.dtype)
        delta = delta * mask

    if norm == "linf":
        if delta.is_complex():
            mag = delta.abs()
            scaled = delta * (eps_budget / mag.clamp(min=1e-12))
            delta = torch.where(mag > eps_budget, scaled, delta)
            delta = torch.where(eps_budget == 0, torch.zeros_like(delta), delta)
        else:
            delta = torch.clamp(delta, min=-eps_budget, max=eps_budget)
    elif norm == "l1":
        proj_l1 = _project_l1_box(delta, x, eps_budget, clip_bounds)
        return proj_l1 * perturb_mask.to(device=delta.device, dtype=delta.dtype) if perturb_mask is not None else proj_l1
    elif norm in ("l2", "relative_l2", "jsr_db"):
        d_norm = _batch_l2_norm(delta).to(_real_dtype_for(delta)).view_as(eps_budget)
        factor = torch.where(
            d_norm > eps_budget,
            eps_budget / d_norm.clamp(min=1e-12),
            torch.ones_like(d_norm),
        )
        delta = delta * factor
    else:
        raise ValueError(f"Unsupported norm '{norm}'.")

    if clip_bounds is not None:
        lo_t, hi_t = coerce_clip_bounds(clip_bounds, x)
        x_box = torch.minimum(torch.maximum(x, lo_t), hi_t)
        excess = x_box - x
        if norm == "linf":
            if bool((excess.abs() > eps_budget + 1e-5).any().item()):
                raise ValueError(
                    f"Clean inputs lie outside clip_bounds by more than the Linf budget "
                    f"(min={float(x.min().item()):.4g}, max={float(x.max().item()):.4g}). "
                    "Pass clip_bounds=None for standardized or unbounded inputs."
                )
        else:
            excess_l2 = _batch_l2_norm(excess).to(_real_dtype_for(x)).view_as(eps_budget)
            if bool((excess_l2 > eps_budget + 1e-4).any().item()):
                raise ValueError(
                    f"Clean inputs lie outside clip_bounds by more than the {norm} budget "
                    f"(required shift L2 up to {float(excess_l2.max().item()):.4g}). "
                    "Pass clip_bounds=None for standardized or unbounded inputs."
                )
        x_adv = torch.minimum(torch.maximum(x + delta, lo_t), hi_t)
        delta = x_adv - x
        if norm != "linf" and bool((excess != 0).any().item()):
            post_l2 = _batch_l2_norm(delta).to(_real_dtype_for(x)).view_as(eps_budget)
            if bool((post_l2 > eps_budget + 1e-5).any().item()):
                t_lo = torch.zeros_like(eps_budget)
                t_hi = torch.ones_like(eps_budget)
                diff_box = x_adv - x_box
                for _ in range(28):
                    t_mid = 0.5 * (t_lo + t_hi)
                    cand_delta = excess + t_mid * diff_box
                    cand_l2 = _batch_l2_norm(cand_delta).to(_real_dtype_for(x)).view_as(eps_budget)
                    too_big = cand_l2 > eps_budget
                    t_lo = torch.where(too_big, t_lo, t_mid)
                    t_hi = torch.where(too_big, t_mid, t_hi)
                delta = excess + t_lo * diff_box

    if perturb_mask is not None:
        delta = delta * perturb_mask.to(device=delta.device, dtype=delta.dtype)
    return delta


def random_init_perturbation(
    x: torch.Tensor,
    eps_budget: torch.Tensor,
    norm: str,
    clip_bounds: tuple[Any, Any] | None = None,
    generator: torch.Generator | None = None,
    perturb_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample a random starting perturbation strictly inside the feasible set."""
    real_dtype = _real_dtype_for(x)

    if norm == "linf":
        if x.is_complex():
            radius = torch.rand(x.shape, device=x.device, dtype=real_dtype, generator=generator)
            phase = (
                torch.rand(x.shape, device=x.device, dtype=real_dtype, generator=generator)
                * (2.0 * math.pi)
            )
            delta = torch.polar(radius, phase).to(x.dtype) * eps_budget
        else:
            delta = (
                torch.rand(x.shape, device=x.device, dtype=real_dtype, generator=generator) * 2.0 - 1.0
            ).to(x.dtype) * eps_budget
    elif norm in ("l1", "l2", "relative_l2", "jsr_db"):
        if norm == "l1":
            # Exponential(1) magnitudes give uniform samples on the L1 simplex once normalized.
            exp_mags = -torch.log(
                torch.rand(x.shape, device=x.device, dtype=real_dtype, generator=generator).clamp(min=1e-12)
            )
            if x.is_complex():
                phase = (
                    torch.rand(x.shape, device=x.device, dtype=real_dtype, generator=generator)
                    * (2.0 * math.pi)
                )
                raw = torch.polar(exp_mags, phase).to(x.dtype)
            else:
                signs = (
                    torch.randint(0, 2, x.shape, device=x.device, generator=generator, dtype=torch.int64).to(
                        real_dtype
                    )
                    * 2.0
                    - 1.0
                )
                raw = (exp_mags * signs).to(x.dtype)
            raw_norm = raw.abs().reshape(x.shape[0], -1).sum(dim=1).to(real_dtype).view_as(eps_budget)
        else:
            if x.is_complex():
                re = torch.randn(x.shape, device=x.device, dtype=real_dtype, generator=generator)
                im = torch.randn(x.shape, device=x.device, dtype=real_dtype, generator=generator)
                raw = torch.complex(re, im).to(x.dtype)
            else:
                raw = torch.randn(x.shape, device=x.device, dtype=real_dtype, generator=generator).to(x.dtype)
            raw_norm = _batch_l2_norm(raw).to(real_dtype).view_as(eps_budget)
        unit_vec = raw / raw_norm.clamp(min=1e-12)
        dims = x[0].numel() * (2 if x.is_complex() else 1)
        radii = torch.rand(eps_budget.shape, device=x.device, dtype=real_dtype, generator=generator).pow(
            1.0 / max(dims, 1)
        )
        delta = unit_vec * (eps_budget * radii)
    else:
        raise ValueError(f"Unsupported norm '{norm}'.")

    return project_perturbation(
        delta,
        x,
        eps_budget,
        norm=norm,
        clip_bounds=clip_bounds,
        perturb_mask=perturb_mask,
    )


def project_discrete_features(
    x_adv: torch.Tensor,
    x_clean: torch.Tensor | None = None,
    *,
    eps: float | None = None,
    norm: str = "linf",
    binary_indices: list[int] | tuple[int, ...] | None = None,
    integer_indices: list[int] | tuple[int, ...] | None = None,
    clip_bounds: tuple[Any, Any] | None = None,
) -> torch.Tensor:
    """Snap tabular/protocol discrete or binary feature columns to valid discrete levels.

    When ``x_clean`` and ``eps`` are provided, reverts any snapped column whose discrete jump
    would violate the allowed perturbation radius back to its clean value.
    """
    out = x_adv.clone()
    flat = out.reshape(out.shape[0], -1)
    if binary_indices:
        idx_b = list(binary_indices)
        flat[:, idx_b] = (flat[:, idx_b] >= 0.5).to(flat.dtype)
    if integer_indices:
        idx_i = list(integer_indices)
        flat[:, idx_i] = torch.round(flat[:, idx_i])
    out = flat.reshape_as(x_adv)
    out = clamp_to_bounds(out, clip_bounds)
    if x_clean is not None and eps is not None:
        within = within_eps_budget(out - x_clean, x_clean, eps=eps, norm=norm).view(
            (out.shape[0],) + (1,) * (out.ndim - 1)
        )
        out = torch.where(within, out, x_clean)
    return out


def compute_perturbation_norm(delta: torch.Tensor, norm: str, x_ref: torch.Tensor | None = None) -> torch.Tensor:
    """Compute per-sample perturbation magnitude under the requested norm."""
    if norm == "linf":
        return delta.abs().reshape(delta.shape[0], -1).amax(dim=1)
    if norm == "l1":
        return delta.abs().reshape(delta.shape[0], -1).sum(dim=1)
    if norm == "l0":
        return (delta.abs() > 1e-8).reshape(delta.shape[0], -1).sum(dim=1).to(_real_dtype_for(delta))

    l2 = _batch_l2_norm(delta)
    if norm == "l2":
        return l2
    if x_ref is None:
        raise ValueError(f"x_ref is required to compute norm='{norm}'.")

    x_l2 = _batch_l2_norm(x_ref).clamp(min=1e-12)
    ratio = l2 / x_l2
    if norm == "relative_l2":
        return ratio
    if norm == "jsr_db":
        return 20.0 * torch.log10(ratio.clamp(min=1e-12))
    raise ValueError(f"Unsupported norm '{norm}'.")


def within_eps_budget(
    delta: torch.Tensor,
    x_ref: torch.Tensor,
    eps: float | torch.Tensor,
    norm: str,
    atol: float = 1e-6,
) -> torch.Tensor:
    """Return a per-sample boolean mask ``(B,)`` checking ``delta`` against ``eps`` in true ``norm`` units."""
    batch_size = delta.shape[0]
    if norm == "l0":
        measured = compute_perturbation_norm(delta, norm="l0")
        budget = torch.as_tensor(eps, device=delta.device, dtype=measured.dtype).reshape(-1)
        if budget.numel() == 1:
            budget = budget.expand(batch_size)
        return measured <= budget + atol

    # resolve_eps_budget converts jsr_db and relative_l2 into linear L2 radii,
    # so we must measure linear L2 (not dB/ratio) to keep unit consistency.
    eps_budget = resolve_eps_budget(x_ref, eps=eps, norm=norm).reshape(batch_size)
    measure_norm = "l2" if norm in ("relative_l2", "jsr_db") else norm
    measured = compute_perturbation_norm(delta, norm=measure_norm, x_ref=x_ref)
    return measured <= eps_budget + atol
