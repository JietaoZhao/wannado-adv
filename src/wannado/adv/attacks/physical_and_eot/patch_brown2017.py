"""Adversarial Patch: Localized Universal / Masked Patch Attack (Brown et al., NIPS 2017 Workshop).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : A (Canonical Localized / Physical Patch & Burst Jamming Attack)
- Reference     : Brown, T. B., Mané, D., Roy, A., Abadi, M., & Gilmer, J.
                  (2017). "Adversarial Patch." NIPS 2017 Workshop on Machine
                  Learning and Computer Security.
- URL           : https://arxiv.org/abs/1712.09665
- Plain Intuition:
    Most adversarial attacks constrain tiny noise across the ENTIRE input.
    Brown et al. showed an attacker can instead replace a small localized region
    (like a printed sticker on an image, or a short localized time-frequency
    burst in an RF signal) with a single learned universal patch pattern ``P``.
    Applied anywhere inside the scene (averaged over random shifts/placements
    via EoT), that small unmasked patch reliably hijacks the classifier.
- Diff vs. EoT-PGD / UAP:
    Restricts perturbation support strictly to a spatial/temporal mask (``mask``
    or ``patch_fraction``), learning a single universal patch across a batch (or
    per-sample patch when ``universal=False``), optionally randomized across
    spatial positions via EoT shifts.
- Why Chosen as a Baseline:
    Standard physical-world CV sticker baseline and localized burst-jamming
    baseline for wireless packets.
- Extension Directions / Prompt Hooks:
    1. Supply custom circular/polygonal binary ``mask`` or contiguous wireless
       sub-carrier/time-slot burst mask.
    2. Enable random affine/channel transforms around patch placement.
===============================================================================
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import torch

from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    clamp_to_bounds,
    coerce_clip_bounds,
    eval_mode,
    make_generator,
    project_perturbation,
    resolve_eps_budget,
    resolve_loss_fn,
)
from wannado.adv.core.losses import LossFn
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="adv_patch",
    name="Adversarial Patch",
    family="physical_and_eot",
    tier="A",
    threat_model="white_box",
    requires_training=False,
    compute_cost="medium",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "timeseries", "tabular"),
    parent_id="eot_pgd",
    diff_from_parent="Confines perturbations strictly inside a localized binary mask / contiguous patch region, learning universal or per-sample patch patterns under optional random placement.",
    summary_plain="Learns a localized patch or burst pattern (masked to a sub-region) that hijacks predictions when stamped onto inputs.",
    when_to_use="Physical sticker/patch threats in vision, occlusion/patch-dropout defense audits, or time/frequency-burst jamming in wireless signals.",
    when_not_to_use="Imperceptible full-frame norm-ball evaluation (use apgd_ce/autoattack) or black-box hard-label access.",
    extension_hooks=(
        "Pass an explicit custom `mask` matching your physical sticker shape or RF burst window",
        "Enable `random_location=True` to optimize shift-robust universal physical patches",
    ),
    module_path="wannado.adv.attacks.physical_and_eot.patch_brown2017",
    citation=AttackCitation(
        key="brown2017adversarial",
        title="Adversarial Patch",
        authors="Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, and Justin Gilmer",
        venue="NIPS 2017 Workshop on Machine Learning and Computer Security",
        year=2017,
        url="https://arxiv.org/abs/1712.09665",
        bibtex=(
            "@inproceedings{brown2017adversarial,\n"
            "  title={Adversarial Patch},\n"
            "  author={Brown, Tom B. and Man{\\'e}, Dandelion and Roy, Aurko and Abadi, Mart{\\'i}n and Gilmer, Justin},\n"
            "  booktitle={NIPS 2017 Workshop on Machine Learning and Computer Security},\n"
            "  year={2017},\n"
            "  url={https://arxiv.org/abs/1712.09665},\n"
            "}"
        ),
        ieee='T. B. Brown, D. Mané, A. Roy, M. Abadi, and J. Gilmer, "Adversarial patch," in NIPS Workshop Mach. Learn. Comput. Secur., 2017.',
        apa="Brown, T. B., Mané, D., Roy, A., Abadi, M., & Gilmer, J. (2017). Adversarial patch. NIPS Workshop on Machine Learning and Computer Security.",
    ),
)


def _default_patch_mask(x: torch.Tensor, patch_fraction: float) -> torch.Tensor:
    """Create a centered localized float mask ``(1, ...)`` covering ``patch_fraction`` elements."""
    mask = torch.zeros((1, *x.shape[1:]), device=x.device, dtype=torch.float32)
    if x.ndim == 4:
        _, _, h, w = x.shape
        ph = max(1, min(h, round(h * math.sqrt(patch_fraction))))
        pw = max(1, min(w, round(w * math.sqrt(patch_fraction))))
        top = (h - ph) // 2
        left = (w - pw) // 2
        mask[:, :, top : top + ph, left : left + pw] = 1.0
        return mask

    # 2D / 3D / ND: contiguous center burst along the last axis
    last_len = x.shape[-1]
    span = max(1, min(last_len, round(last_len * patch_fraction)))
    start = (last_len - span) // 2
    slices = [slice(None)] * mask.ndim
    slices[-1] = slice(start, start + span)
    mask[tuple(slices)] = 1.0
    return mask


def _shift_mask_and_patch(
    mask: torch.Tensor,
    patch: torch.Tensor,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    """Randomly roll mask & patch across spatial dimensions for EoT shift invariance."""
    if patch.ndim == 4:
        _, _, h, w = patch.shape
        dy = int(torch.randint(0, h, (), device=patch.device, generator=generator).item())
        dx = int(torch.randint(0, w, (), device=patch.device, generator=generator).item())
        return (
            torch.roll(mask, shifts=(dy, dx), dims=(2, 3)),
            torch.roll(patch, shifts=(dy, dx), dims=(2, 3)),
            (dy, dx),
            (2, 3),
        )
    length = patch.shape[-1]
    shift = int(torch.randint(0, length, (), device=patch.device, generator=generator).item())
    return (
        torch.roll(mask, shifts=shift, dims=-1),
        torch.roll(patch, shifts=shift, dims=-1),
        (shift,),
        (-1,),
    )


def adv_patch(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float | None = None,
    patch_fraction: float = 0.25,
    mask: torch.Tensor | None = None,
    steps: int = 80,
    lr: float = 0.1,
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    universal: bool = True,
    random_location: bool = False,
    eot_samples: int = 2,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Learn a localized adversarial patch (universal or per-sample, real or complex64)."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if lr <= 0.0:
        raise ValueError(f"lr must be > 0, got {lr}.")
    if not 0.0 < patch_fraction <= 1.0:
        raise ValueError(f"patch_fraction must be in (0, 1], got {patch_fraction}.")
    if eot_samples < 1:
        raise ValueError(f"eot_samples must be >= 1, got {eot_samples}.")
    if clip_bounds is not None and x.is_complex():
        raise ValueError("clip_bounds is only supported for real-valued tensors.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    loss_fn = resolve_loss_fn(loss, targeted=targeted)
    generator = make_generator(seed, x_clean.device)

    eps_budget: torch.Tensor | None = None
    if eps is not None:
        eps_budget = resolve_eps_budget(x_clean, eps=eps, norm=norm)
        if (eps_budget == 0).all():
            return AttackResult(
                x_adv=x_clean.clone(),
                perturbation=torch.zeros_like(x_clean),
                eps=eps,
                norm=norm,
                steps_run=0,
                attack_id=META.id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                extra={"patch_fraction": patch_fraction, "universal": universal},
            )

    if mask is None:
        base_mask = _default_patch_mask(x_clean, patch_fraction=patch_fraction)
    else:
        base_mask = (mask.detach().to(device=x_clean.device, dtype=torch.float32) > 0).float()
        if base_mask.ndim == x_clean.ndim - 1:
            base_mask = base_mask.unsqueeze(0)

    patch_shape = (1, *x_clean.shape[1:]) if universal else x_clean.shape
    patch = torch.zeros(patch_shape, device=x_clean.device, dtype=x_clean.dtype)

    if clip_bounds is not None and eps_budget is None:
        lo_t, hi_t = coerce_clip_bounds(clip_bounds, x_clean)
        mid = 0.5 * (lo_t + hi_t)
        patch = patch + mid

    best_obj_mean = float("-inf")
    best_patch = patch.clone()
    best_mask = base_mask.clone()

    def _apply_patch(
        x_in: torch.Tensor, patch_t: torch.Tensor, mask_t: torch.Tensor
    ) -> torch.Tensor:
        m = mask_t.to(dtype=x_in.dtype)
        # In Brown et al., patch replaces masked region (or adds masked delta when eps is bounded).
        x_out = (x_in + m * patch_t) if eps_budget is not None else ((1.0 - m) * x_in + m * patch_t)
        return clamp_to_bounds(x_out, clip_bounds)

    mom = torch.zeros_like(patch)
    use_eot = random_location and universal
    eval_every = max(1, steps // 10)

    def _eval_obj(patch_t: torch.Tensor, mask_t: torch.Tensor) -> float:
        with torch.no_grad():
            x_p = _apply_patch(x_clean, patch_t, mask_t)
            logits_p = model_or_fn(x_p)
            raw_p = loss_fn(logits_p, y)
            if raw_p.is_complex():
                raw_p = raw_p.real
            return float((-raw_p if targeted else raw_p).mean().item())

    with eval_mode(model_or_fn):
        for step_idx in range(steps):
            if use_eot:
                grad_acc = torch.zeros_like(patch)
                for k_idx in range(eot_samples):
                    shifts: tuple[int, ...]
                    dims: tuple[int, ...]
                    if k_idx == 0 and step_idx % 4 == 0:
                        step_mask, step_patch_view, shifts, dims = base_mask, patch, (), ()
                    else:
                        step_mask, step_patch_view, shifts, dims = _shift_mask_and_patch(
                            base_mask, patch, generator
                        )
                    patch_var = step_patch_view.detach().requires_grad_(True)
                    x_patched = _apply_patch(x_clean, patch_var, step_mask)
                    # [AI-Prompt Hook]: Brown et al. (2017) maximizes expected patch fooling objective
                    # over random placements. Insert brightness/rotation/fading EoT transforms before forward.
                    logits = model_or_fn(x_patched)
                    obj_raw = loss_fn(logits, y)
                    if obj_raw.is_complex():
                        obj_raw = obj_raw.real
                    obj = (-obj_raw if targeted else obj_raw).mean()
                    (grad_p,) = torch.autograd.grad(obj, patch_var, only_inputs=True)
                    grad_masked = grad_p.detach() * step_mask.to(dtype=x_clean.dtype)
                    if shifts:
                        grad_masked = torch.roll(
                            grad_masked, shifts=tuple(-s for s in shifts), dims=dims
                        )
                    grad_acc = grad_acc + grad_masked / float(eot_samples)
                mom = 0.7 * mom + grad_acc
                grad_for_step = mom
            else:
                patch_var = patch.detach().requires_grad_(True)
                x_patched = _apply_patch(x_clean, patch_var, base_mask)
                logits = model_or_fn(x_patched)
                obj_raw = loss_fn(logits, y)
                if obj_raw.is_complex():
                    obj_raw = obj_raw.real
                obj = (-obj_raw if targeted else obj_raw).mean()
                val = float(obj.item())
                if val > best_obj_mean:
                    best_obj_mean = val
                    best_patch = patch.clone()
                (grad_p,) = torch.autograd.grad(obj, patch_var, only_inputs=True)
                grad_for_step = grad_p.detach() * base_mask.to(dtype=x_clean.dtype)

            step_dir = (
                torch.sgn(grad_for_step)
                if grad_for_step.is_complex()
                else torch.sign(grad_for_step)
            )
            step_scale = lr
            if eps_budget is not None and norm == "linf":
                step_scale = (
                    min(lr, float(eps_budget.min().item()) * 0.25)
                    if float(eps_budget.min().item()) > 0
                    else lr
                )
            updated_patch = (patch + step_scale * step_dir) * base_mask.to(dtype=x_clean.dtype)

            if eps_budget is not None:
                univ_budget = eps_budget.amin(dim=0, keepdim=True) if universal else eps_budget
                ref_x = x_clean[:1] if universal else x_clean
                updated_patch = project_perturbation(
                    updated_patch,
                    ref_x,
                    univ_budget,
                    norm=norm,
                    clip_bounds=None,
                )
            elif clip_bounds is not None:
                updated_patch = clamp_to_bounds(updated_patch, clip_bounds)

            patch = updated_patch

            if use_eot and (step_idx % eval_every == eval_every - 1 or step_idx == steps - 1):
                val = _eval_obj(patch, base_mask)
                if val > best_obj_mean:
                    best_obj_mean = val
                    best_patch = patch.clone()

        if not use_eot:
            val_final = _eval_obj(patch, base_mask)
            if val_final > best_obj_mean:
                best_obj_mean = val_final
                best_patch = patch.clone()

    x_adv = _apply_patch(x_clean, best_patch, best_mask)
    delta = x_adv - x_clean
    if eps_budget is not None:
        delta = project_perturbation(delta, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds)
        x_adv = x_clean + delta

    active_frac = float((best_mask > 0.5).float().mean().item())
    return AttackResult(
        x_adv=x_adv,
        perturbation=delta,
        eps=float(eps) if eps is not None else 0.0,
        norm=norm,
        steps_run=steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "patch": best_patch.detach(),
            "mask": best_mask.detach(),
            "patch_fraction_actual": active_frac,
            "universal": universal,
            "random_location": random_location,
        },
    )
