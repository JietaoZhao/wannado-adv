"""FAB: Fast Adaptive Boundary Attack (Croce & Hein, ICML 2020).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Current Minimum-Norm Geometric Boundary SOTA)
- Reference     : Croce, F., & Hein, M. (2020). "Minimally distorted Adversarial
                  Examples with a Fast Adaptive Boundary Attack." ICML 2020
                  (PMLR 119:2196-2205).
- URL           : https://proceedings.mlr.press/v119/croce20a.html
- Plain Intuition:
    DeepFool steps onto the closest decision boundary, which can drift away
    from ``x_clean`` along a curved surface. FAB fixes this by projecting BOTH
    the current point ``x_curr`` AND the original point ``x_clean`` onto the
    linearized boundary, taking a biased boundary-tracking step capped by
    ``alpha_max``, then executing a backward step ``beta = 0.9`` toward
    ``x_clean`` so the iterate tightly hugs the minimal-distortion boundary.
- Diff vs. DeepFool (2016):
    1. Biased step mixing projections from ``x_curr`` and ``x_clean`` (capped by ``alpha_max``).
    2. Backward contraction step ``beta = 0.9`` pulling iterates back toward ``x_clean``.
    3. Multi-restart exploration initialized near the best known distortion radius.
- Why Chosen as a Baseline:
    One of the 4 pillars of AutoAttack (ICML 2020); finds tighter minimal L2 and
    Linf adversarial examples than DeepFool and avoids gradient-scaling traps.
- Extension Directions (for further research):
    1. Use FAB-T (targeted FAB) across top candidate classes on many-class tasks.
    2. Pair with APGD-CE, APGD-DLR, and Square Attack -> full AutoAttack suite.
===============================================================================
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    clamp_to_bounds,
    coerce_clip_bounds,
    compute_perturbation_norm,
    eval_mode,
    normalize_classification_logits,
    predict_labels,
    random_init_perturbation,
    safe_l2_norm,
    step_direction,
)
from wannado.adv.core.projections import within_eps_budget
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="fab",
    name="Fast Adaptive Boundary Attack (FAB)",
    family="minimum_norm",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="medium",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    parent_id="deepfool",
    diff_from_parent="Improves DeepFool with biased forward projections toward x_clean (alpha_max), a backward contraction step (beta=0.9), and scaled restarts.",
    summary_plain="Geometric boundary SOTA that tracks the decision boundary while contracting back toward x_clean for minimal L2/Linf distortion.",
    when_to_use="When measuring exact minimal Linf/L2 robustness margins or building an AutoAttack-grade evaluation suite.",
    when_not_to_use="For non-classification regression tasks without logits (use PGD/Adam-PGD with custom loss).",
    extension_hooks=(
        "Combine with APGD-CE, APGD-DLR, and Square Attack -> AutoAttack",
        "Run targeted FAB over top-K runner-up classes on many-class tasks",
    ),
    module_path="wannado.adv.attacks.minimum_norm.fab_croce2020",
    citation=AttackCitation(
        key="croce2020minimally",
        title="Minimally distorted Adversarial Examples with a Fast Adaptive Boundary Attack",
        authors="Francesco Croce and Matthias Hein",
        venue="37th International Conference on Machine Learning (ICML)",
        year=2020,
        url="https://proceedings.mlr.press/v119/croce20a.html",
        bibtex=(
            "@inproceedings{croce2020minimally,\n"
            "  title={Minimally distorted Adversarial Examples with a Fast Adaptive Boundary Attack},\n"
            "  author={Croce, Francesco and Hein, Matthias},\n"
            "  booktitle={Proceedings of the 37th International Conference on Machine Learning},\n"
            "  pages={2196--2205},\n"
            "  year={2020},\n"
            "  volume={119},\n"
            "  series={Proceedings of Machine Learning Research},\n"
            "  publisher={PMLR},\n"
            "  url={https://proceedings.mlr.press/v119/croce20a.html},\n"
            "}"
        ),
        ieee='F. Croce and M. Hein, "Minimally distorted adversarial examples with a fast adaptive boundary attack," in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR, 2020, pp. 2196-2205.',
        apa="Croce, F., & Hein, M. (2020). Minimally distorted adversarial examples with a fast adaptive boundary attack. Proceedings of the 37th International Conference on Machine Learning (ICML), 2196-2205.",
    ),
)


def _norm_p_batch(x: torch.Tensor, norm: str) -> torch.Tensor:
    """Compute per-sample L_inf or L2 distortion norm as real tensor of shape ``(M,)``."""
    m = x.shape[0]
    flat = x.reshape(m, -1)
    if norm == "linf":
        return flat.abs().amax(dim=1)
    return safe_l2_norm(flat, dim=1)


def _project_hyperplane_box_batch(
    x_ref: torch.Tensor,
    w: torch.Tensor,
    h_val: torch.Tensor,
    norm: str,
    clip_bounds: tuple[Any, Any] | None,
) -> torch.Tensor:
    """Vectorized minimal-norm displacement ``d`` from ``x_ref`` to ``h_val + <w, d> = 0`` for batch ``(M, ...)``."""
    m = x_ref.shape[0]
    real_dtype = x_ref.real.dtype if x_ref.is_complex() else x_ref.dtype
    w_flat = w.reshape(m, -1)
    h_col = h_val.to(device=x_ref.device, dtype=real_dtype).view(m, 1)
    zero_out = torch.zeros_like(w_flat)

    if clip_bounds is None:
        sign_factor = torch.where(h_col >= 0.0, -torch.ones_like(h_col), torch.ones_like(h_col))
        if norm == "linf":
            w_l1 = w_flat.abs().sum(dim=1, keepdim=True).to(real_dtype)
            valid = (w_l1 > 1e-12) & (h_col.abs() > 1e-12)
            dir_linf = step_direction(w, norm="linf").reshape(m, -1)
            scale = (h_col.abs() / w_l1.clamp(min=1e-12)) * sign_factor
            d_flat = torch.where(valid, scale.to(w_flat.dtype) * dir_linf, zero_out)
            return d_flat.view_as(x_ref)
        w_l2_sq = w_flat.abs().pow(2).sum(dim=1, keepdim=True).to(real_dtype)
        valid = (w_l2_sq > 1e-12) & (h_col.abs() > 1e-12)
        scale = -h_col / w_l2_sq.clamp(min=1e-12)
        d_flat = torch.where(valid, scale.to(w_flat.dtype) * w_flat, zero_out)
        return d_flat.view_as(x_ref)

    # Real tensor with box constraints [low, high]: exact vectorized 1D monotonic bisection.
    low, high = coerce_clip_bounds(clip_bounds, x_ref)
    d_low = (low - x_ref).reshape(m, -1)
    d_high = (high - x_ref).reshape(m, -1)
    sign_h = torch.where(h_col >= 0.0, -torch.ones_like(h_col), torch.ones_like(h_col))
    target = h_col.abs()
    active = target > 1e-12

    if norm == "linf":
        direction = sign_h * torch.sign(w_flat)
        r_hi = torch.maximum(d_low.abs(), d_high.abs()).amax(dim=1, keepdim=True).clamp(min=1e-6)
        valid = active & (w_flat.abs().sum(dim=1, keepdim=True) > 1e-12)
    else:
        w_l2_sq = w_flat.pow(2).sum(dim=1, keepdim=True)
        valid = active & (w_l2_sq > 1e-12)
        direction = sign_h * w_flat
        r_max_box = torch.maximum(d_low.abs(), d_high.abs()).amax(dim=1, keepdim=True)
        abs_w = w_flat.abs()
        w_pos = torch.where(abs_w > 1e-12, abs_w, torch.full_like(abs_w, float("inf")))
        w_min_nz = w_pos.amin(dim=1, keepdim=True)
        w_min_nz = torch.where(torch.isfinite(w_min_nz), w_min_nz, torch.ones_like(w_min_nz))
        r_hi = torch.maximum(r_max_box / w_min_nz.clamp(min=1e-6), target / w_l2_sq.clamp(min=1e-12))

    cand_hi = torch.minimum(torch.maximum(r_hi * direction, d_low), d_high)
    max_achievable = (sign_h * (w_flat * cand_hi).sum(dim=1, keepdim=True))
    needs_bisect = valid & (max_achievable > target)

    r_lo = torch.zeros_like(r_hi)
    r_hi_work = r_hi.clone()
    if needs_bisect.any():
        for _ in range(30):
            r_mid = 0.5 * (r_lo + r_hi_work)
            cand_mid = torch.minimum(torch.maximum(r_mid * direction, d_low), d_high)
            val = sign_h * (w_flat * cand_mid).sum(dim=1, keepdim=True)
            too_small = val < target
            r_lo = torch.where(needs_bisect & too_small, r_mid, r_lo)
            r_hi_work = torch.where(needs_bisect & (~too_small), r_mid, r_hi_work)

    cand_final = torch.minimum(torch.maximum(r_hi_work * direction, d_low), d_high)
    chosen = torch.where(needs_bisect, cand_final, cand_hi)
    d_flat = torch.where(valid, chosen, zero_out)
    return d_flat.view_as(x_ref)


def _fab_batched(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_orig: torch.Tensor,
    y_orig: torch.Tensor,
    norm: str,
    steps: int,
    restarts: int,
    alpha_max: float,
    beta: float,
    eta: float,
    max_classes: int,
    targeted: bool,
    target_labels: torch.Tensor | None,
    clip_bounds: tuple[Any, Any] | None,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, int]:
    """Run vectorized Algorithm 1 of Croce & Hein (ICML 2020) across active batch ``(M, ...)``."""
    m = x_orig.shape[0]
    real_dtype = x_orig.real.dtype if x_orig.is_complex() else x_orig.dtype
    view_shape = (m,) + (1,) * (x_orig.ndim - 1)
    ar = torch.arange(m, device=x_orig.device)

    best_adv = x_orig.clone()
    best_dist = torch.full((m,), float("inf"), device=x_orig.device, dtype=real_dtype)
    total_steps = 0

    with torch.no_grad():
        logits_clean = normalize_classification_logits(model_or_fn(x_orig))
        num_classes = logits_clean.shape[1]
        if targeted:
            if target_labels is None:
                raise ValueError("target_labels must be provided when targeted=True.")
            cand_classes = target_labels.reshape(m, 1).to(torch.long)
        else:
            k_cand = max(1, min(num_classes - 1, max_classes))
            masked_clean = logits_clean.clone()
            masked_clean[ar, y_orig] = float("-inf")
            cand_classes = masked_clean.topk(k_cand, dim=1).indices.to(torch.long)

    num_cands = cand_classes.shape[1]
    proj_norm = "linf" if norm == "linf" else "l2"

    for r_idx in range(restarts):
        if r_idx == 0:
            x_curr = x_orig.clone()
        else:
            probe_radius = torch.where(
                torch.isfinite(best_dist),
                0.5 * best_dist,
                torch.full_like(best_dist, 0.1),
            ).view(view_shape)
            x_curr = x_orig + random_init_perturbation(
                x_orig, probe_radius, norm=proj_norm, clip_bounds=clip_bounds, generator=generator
            )

        for _ in range(steps):
            total_steps += 1
            x_in = x_curr.detach().requires_grad_(True)
            logits = normalize_classification_logits(model_or_fn(x_in))
            preds = predict_labels(logits)

            if targeted:
                assert target_labels is not None
                is_fooled = preds == target_labels
            else:
                is_fooled = preds != y_orig

            cur_dist = _norm_p_batch(x_curr - x_orig, norm=norm).to(real_dtype)
            improved = is_fooled & (cur_dist < best_dist)
            if improved.any():
                best_dist = torch.where(improved, cur_dist, best_dist)
                best_adv = torch.where(improved.view(view_shape), x_curr.detach(), best_adv)

            if targeted:
                assert target_labels is not None
                masked_logits = logits.detach().clone()
                masked_logits[ar, target_labels] = float("-inf")
                runner_up = masked_logits.argmax(dim=1)
                ref_cls = torch.where(preds != target_labels, preds, runner_up)
            else:
                ref_cls = y_orig

            (grad_orig,) = torch.autograd.grad(logits[ar, ref_cls].sum(), x_in, retain_graph=True)
            grad_orig = grad_orig.detach()

            min_hp_dist = torch.full((m,), float("inf"), device=x_orig.device, dtype=real_dtype)
            best_w = torch.zeros_like(x_orig)
            best_b = torch.zeros((m,), device=x_orig.device, dtype=real_dtype)

            for k_idx in range(num_cands):
                cls_k = cand_classes[:, k_idx]
                retain = k_idx < num_cands - 1
                (grad_k,) = torch.autograd.grad(logits[ar, cls_k].sum(), x_in, retain_graph=retain)
                w_k = grad_k.detach() - grad_orig
                b_k = (logits[ar, cls_k] - logits[ar, ref_cls]).detach().to(real_dtype)

                if norm == "linf":
                    dual_norm = w_k.abs().reshape(m, -1).sum(dim=1).to(real_dtype)
                else:
                    dual_norm = safe_l2_norm(w_k.reshape(m, -1), dim=1).to(real_dtype)

                valid_k = dual_norm > 1e-12
                hp_dist = torch.where(valid_k, b_k.abs() / dual_norm.clamp(min=1e-12), torch.full_like(b_k, float("inf")))
                better = valid_k & (hp_dist < min_hp_dist)
                if better.any():
                    min_hp_dist = torch.where(better, hp_dist, min_hp_dist)
                    best_w = torch.where(better.view(view_shape), w_k, best_w)
                    best_b = torch.where(better, b_k, best_b)

            valid_step = torch.isfinite(min_hp_dist)
            if not valid_step.any():
                break

            d_curr = _project_hyperplane_box_batch(x_curr, best_w, best_b, norm=norm, clip_bounds=clip_bounds)
            diff_clean = x_orig - x_curr
            if best_w.is_complex():
                inner_clean = (best_w.conj() * diff_clean).real.reshape(m, -1).sum(dim=1).to(real_dtype)
            else:
                inner_clean = (best_w * diff_clean).reshape(m, -1).sum(dim=1).to(real_dtype)
            h_clean = best_b + inner_clean
            d_orig = _project_hyperplane_box_batch(x_orig, best_w, h_clean, norm=norm, clip_bounds=clip_bounds)

            n1 = _norm_p_batch(d_curr, norm=norm).to(real_dtype)
            n2 = _norm_p_batch(d_orig, norm=norm).to(real_dtype)
            alpha = torch.clamp(n1 / (n1 + n2 + 1e-12), max=alpha_max).view(view_shape)

            x_forward = (1.0 - alpha) * (x_curr + eta * d_curr) + alpha * (x_orig + eta * d_orig)
            # Croce & Hein (ICML 2020) Alg. 1: already-fooled iterates contract directly toward x_orig.
            x_contract = (1.0 - beta) * x_orig + beta * x_curr
            x_next = torch.where(is_fooled.view(view_shape), x_contract, x_forward)
            x_next = clamp_to_bounds(x_next, clip_bounds)
            x_curr = torch.where(valid_step.view(view_shape) | is_fooled.view(view_shape), x_next, x_curr)

        with torch.no_grad():
            preds_end = predict_labels(model_or_fn(x_curr))
            if targeted:
                assert target_labels is not None
                is_fooled_end = preds_end == target_labels
            else:
                is_fooled_end = preds_end != y_orig
            cur_dist_end = _norm_p_batch(x_curr - x_orig, norm=norm).to(real_dtype)
            improved_end = is_fooled_end & (cur_dist_end < best_dist)
            if improved_end.any():
                best_dist = torch.where(improved_end, cur_dist_end, best_dist)
                best_adv = torch.where(improved_end.view(view_shape), x_curr.detach(), best_adv)

    return best_adv, total_steps


def fab(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    eps: float | None = None,
    steps: int = 25,
    restarts: int = 1,
    alpha_max: float = 0.1,
    beta: float = 0.9,
    eta: float = 1.05,
    max_classes: int = 10,
    norm: str = "l2",
    targeted: bool = False,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
    batch_size: int | None = None,
) -> AttackResult:
    """Run vectorized Fast Adaptive Boundary (FAB) attack for minimal L2 or Linf distortion.

    Args:
        model_or_fn: Differentiable ``nn.Module`` or callable mapping ``x -> (B, C)`` logits.
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Ground-truth labels (or target class when ``targeted=True``). Optional when untargeted.
        eps: Optional maximum budget cutoff. Perturbations exceeding ``eps_budget`` revert to ``x``.
        steps: FAB iterations per restart.
        restarts: Number of restarts (restart 0 starts at ``x``; subsequent restarts probe near best norm).
        alpha_max: Cap on biased projection mixing coefficient ``alpha`` (paper default ``0.1``).
        beta: Backward contraction factor toward ``x_clean`` (paper default ``0.9``).
        eta: Boundary overshoot factor (paper default ``1.05``).
        max_classes: Number of top alternative classes evaluated per step.
        norm: Distortion geometry: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        targeted: If ``True``, searches minimal perturbation reaching target label ``y``.
        clip_bounds: Optional valid range ``(min, max)`` for real-valued inputs.
        seed: Optional random seed for restarts ``> 1``.
        batch_size: Optional mini-batch chunk size to cap peak GPU/MPS memory on large evaluation sets.

    Returns:
        ``AttackResult`` containing minimal boundary adversarial examples and distortion stats.
    """
    if steps < 1 or restarts < 1:
        raise ValueError("steps and restarts must both be >= 1.")
    if clip_bounds is not None and x.is_complex():
        raise ValueError("clip_bounds is only supported for real-valued tensors.")
    if targeted and y is None:
        raise ValueError("y (target labels) must be provided when targeted=True.")

    if norm not in ("linf", "l2", "relative_l2", "jsr_db"):
        raise ValueError(
            f"fab supports norm in ('linf', 'l2', 'relative_l2', 'jsr_db'), got '{norm}'. "
            "For L1 minimum-norm/sparse attacks use 'ead', 'apgd_l1', or 'sparsefool'."
        )

    t0 = time.perf_counter()
    x_clean = x.detach()
    n_total = x_clean.shape[0]
    x_adv = x_clean.clone()
    total_steps = 0

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=x_clean.device)
        generator.manual_seed(seed)

    chunk = n_total if (batch_size is None or batch_size <= 0) else int(batch_size)
    with eval_mode(model_or_fn):
        for start in range(0, n_total, chunk):
            end = min(start + chunk, n_total)
            xb = x_clean[start:end]
            yb: torch.Tensor | None = None
            if y is not None:
                yb_raw = y[start:end].to(device=x_clean.device, dtype=torch.long)
                yb = yb_raw.squeeze(1) if yb_raw.ndim == 2 and yb_raw.shape[1] == 1 else yb_raw

            with torch.no_grad():
                clean_preds = predict_labels(model_or_fn(xb))

            if targeted:
                assert yb is not None
                active_mask = clean_preds != yb
                y_orig_b = clean_preds
                target_b: torch.Tensor | None = yb
            else:
                if yb is not None:
                    active_mask = clean_preds == yb
                    y_orig_b = yb
                else:
                    active_mask = torch.ones((xb.shape[0],), dtype=torch.bool, device=x_clean.device)
                    y_orig_b = clean_preds
                target_b = None

            if not active_mask.any():
                continue

            act_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
            adv_act, s_chunk = _fab_batched(
                model_or_fn=model_or_fn,
                x_orig=xb[act_idx],
                y_orig=y_orig_b[act_idx],
                norm=norm,
                steps=steps,
                restarts=restarts,
                alpha_max=alpha_max,
                beta=beta,
                eta=eta,
                max_classes=max_classes,
                targeted=targeted,
                target_labels=target_b[act_idx] if target_b is not None else None,
                clip_bounds=clip_bounds,
                generator=generator,
            )
            x_adv[start + act_idx] = adv_act
            total_steps = max(total_steps, s_chunk)

    delta = x_adv - x_clean
    if eps is not None:
        within_budget = within_eps_budget(delta, x_clean, eps=eps, norm=norm).view(
            (n_total,) + (1,) * (x_clean.ndim - 1)
        )
        x_adv = torch.where(within_budget, x_adv, x_clean)
        delta = x_adv - x_clean

    dist_norm = compute_perturbation_norm(delta, norm=norm, x_ref=x_clean)
    return AttackResult(
        x_adv=x_adv,
        perturbation=delta,
        eps=float(eps) if eps is not None else float(dist_norm.median().item()),
        norm=norm,
        steps_run=total_steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "mean_distortion": float(dist_norm.mean().item()),
            "median_distortion": float(dist_norm.median().item()),
            "alpha_max": alpha_max,
            "beta": beta,
            "eta": eta,
            "restarts": restarts,
            "batched": True,
        },
    )
