"""SparseFool: Sparse L0 Geometric Boundary Attack (Modas et al., CVPR 2019).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Fast Geometry-Based L0 Sparse Minimum-Norm Benchmark)
- Reference     : Modas, A., Moosavi-Dezfooli, S.-M., & Frossard, P. (2019).
                  "SparseFool: A Few Pixels Make a Big Difference." CVPR 2019
                  (pp. 9087-9096).
- URL           : https://arxiv.org/abs/1811.02248
- Plain Intuition:
    Solving exact L0 minimization (changing as few features/pixels/tones as
    possible) is NP-hard. SparseFool makes L0 fast using neural geometry:
    at each outer iteration it calls DeepFool to find a nearby boundary point
    ``x_B`` and computes the local tangent hyperplane ``w^T (z - x_B) >= 0``.
    Moving onto a hyperplane inside a box naturally creates a linear program
    whose greedy solution pushes ONLY a handful of highest-leverage coordinates
    all the way to their bounds until the hyperplane constraint is met.
- Diff vs. DeepFool (Moosavi-Dezfooli et al., 2016):
    1. Alternates DeepFool boundary localization with SparseFool's greedy L0
       hyperplane solver (Modas et al., 2019, Algorithm 1).
    2. Controls the sparsity vs. success trade-off via ``lambda_`` scaling.
- Why Chosen as a Baseline:
    Orders of magnitude faster than combinatorial L0 search while achieving
    state-of-the-art L0 sparsity across vision and wireless subcarrier jamming.
- Extension Directions / Prompt Hooks:
    1. Replace box bounds with domain-specific RF peak amplitude limits for
       sparse sub-carrier/tone wireless jamming.
    2. Initialize boundary search from a warm-started coordinate subset mask.
===============================================================================
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from wannado.adv.attacks.minimum_norm.deepfool_moosavi2016 import _deepfool_single_sample
from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    compute_perturbation_norm,
    eval_mode,
)
from wannado.adv.core.projections import within_eps_budget
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="sparsefool",
    name="SparseFool",
    family="minimum_norm",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="medium",
    supported_norms=("l0", "l1", "l2", "linf", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    parent_id="deepfool",
    diff_from_parent="Linearizes the DeepFool decision boundary and solves coordinate-wise greedy projection inside bounds to minimize L0 sparsity.",
    summary_plain="Geometry-driven L0 sparse attack that flips predictions while perturbing as few pixels, features, or frequency bins as possible.",
    when_to_use="When measuring L0 sparsity resilience (how few pixels or IQ tones need to be jammed to fool a model).",
    when_not_to_use="When optimizing dense small L2/Linf perturbations or without white-box gradients (use DeepFool, APGD, or Square Attack).",
    extension_hooks=(
        "Adjust `lambda_` (>1 for sparser L0, ~1 for smaller amplitude per feature)",
        "Group coordinates (e.g., RGB patches or OFDM sub-bands) inside `_linear_solver`",
    ),
    module_path="wannado.adv.attacks.minimum_norm.sparsefool_modas2019",
    citation=AttackCitation(
        key="modas2019sparsefool",
        title="SparseFool: A Few Pixels Make a Big Difference",
        authors="Apostolos Modas, Seyed-Mohsen Moosavi-Dezfooli, and Pascal Frossard",
        venue="IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
        year=2019,
        url="https://arxiv.org/abs/1811.02248",
        bibtex=(
            "@inproceedings{modas2019sparsefool,\n"
            "  title={SparseFool: A Few Pixels Make a Big Difference},\n"
            "  author={Modas, Apostolos and Moosavi-Dezfooli, Seyed-Mohsen and Frossard, Pascal},\n"
            "  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},\n"
            "  pages={9087--9096},\n"
            "  year={2019},\n"
            "  url={https://arxiv.org/abs/1811.02248},\n"
            "}"
        ),
        ieee='A. Modas, S.-M. Moosavi-Dezfooli, and P. Frossard, "SparseFool: A few pixels make a big difference," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 9087-9096.',
        apa="Modas, A., Moosavi-Dezfooli, S.-M., & Frossard, P. (2019). SparseFool: A few pixels make a big difference. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9087-9096.",
    ),
)


def _linear_solver(
    x_0: torch.Tensor,
    normal: torch.Tensor,
    boundary_point: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> torch.Tensor:
    """Solve SparseFool greedy hyperplane projection inside [lb, ub] (Modas et al., Algorithm 1)."""
    # For complex signals, optimize real and imag parts as joint 2D coordinates along the normal direction.
    if x_0.is_complex():
        x_flat = torch.cat([x_0.real.reshape(-1), x_0.imag.reshape(-1)])
        b_flat = torch.cat([boundary_point.real.reshape(-1), boundary_point.imag.reshape(-1)])
        w_flat = torch.cat([normal.real.reshape(-1), normal.imag.reshape(-1)])
        half = x_0.numel()
    else:
        x_flat = x_0.reshape(-1)
        b_flat = boundary_point.reshape(-1)
        w_flat = normal.reshape(-1)

    lb_flat = lb.reshape(-1)
    ub_flat = ub.reshape(-1)

    # Target linear inequality: <w, x_new - b_flat> >= 0, starting from delta_cur = 0 at x_flat.
    # Let v_flat solve <w, x_flat + r - b_flat> >= 0 <=> <w, r> >= <w, b_flat - x_flat>.
    f_val = float(torch.dot(w_flat, b_flat - x_flat).item())
    if f_val <= 0.0:
        return x_0.clone()

    # Allowed shift per coordinate in direction sign(w_i): ub - x_i if w_i > 0 else lb - x_i.
    max_shift = torch.where(w_flat >= 0.0, ub_flat - x_flat, lb_flat - x_flat)
    gain = w_flat * max_shift  # Non-negative projection contribution if coordinate goes to bound.
    scores = w_flat.abs()
    order = torch.argsort(scores, descending=True)

    r = torch.zeros_like(x_flat)
    accum = 0.0
    for idx in order.tolist():
        if float(scores[idx].item()) <= 1e-12:
            break
        g_i = float(gain[idx].item())
        if g_i <= 1e-12:
            continue
        need = f_val - accum
        if g_i >= need:
            # Overshoot slightly onto boundary side of hyperplane
            step_i = need / float(w_flat[idx].item())
            if step_i >= 0.0:
                step_i = min(step_i, float(max_shift[idx].item()))
            else:
                step_i = max(step_i, float(max_shift[idx].item()))
            r[idx] = step_i
            accum = f_val
            break
        r[idx] = max_shift[idx]
        accum += g_i

    x_sol_flat = x_flat + r
    if x_0.is_complex():
        return torch.complex(
            x_sol_flat[:half].reshape_as(x_0.real),
            x_sol_flat[half:].reshape_as(x_0.imag),
        ).to(x_0.dtype)
    return x_sol_flat.reshape_as(x_0)


def _sparsefool_single_sample(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_single: torch.Tensor,
    lambda_: float,
    max_iter: int,
    deepfool_steps: int,
    overshoot: float,
    max_classes: int,
    clip_bounds: tuple[float, float] | None,
) -> tuple[torch.Tensor, int]:
    """Run SparseFool coordinate alternation on a single sample ``(1, ...)``."""
    with torch.no_grad():
        orig_label = int(model_or_fn(x_single).argmax(dim=1)[0].item())

    x_cur = x_single.clone()
    total_steps = 0

    if clip_bounds is not None:
        lb_box = torch.full_like(x_single, float(clip_bounds[0]))
        ub_box = torch.full_like(x_single, float(clip_bounds[1]))
    else:
        # Unbounded fallback (wireless/tabular): cap single-step coordinate jump to signal scale.
        mag_tensor = x_single.abs() if x_single.is_complex() else x_single
        spread = float(mag_tensor.std(unbiased=False).item()) if mag_tensor.numel() > 1 else 0.0
        scale = max(spread, float(mag_tensor.abs().mean().item()), 1.0) * 3.0

    for _ in range(max_iter):
        total_steps += 1
        x_df, df_steps = _deepfool_single_sample(
            model_or_fn=model_or_fn,
            x_single=x_cur,
            norm="l2",
            steps=deepfool_steps,
            overshoot=overshoot,
            max_classes=max_classes,
            clip_bounds=clip_bounds,
        )
        total_steps += df_steps

        with torch.no_grad():
            df_pred = int(model_or_fn(x_df).argmax(dim=1)[0].item())
        if df_pred == orig_label:
            break

        # Evaluate hyperplane normal w at boundary point x_df between df_pred and orig_label.
        x_in = x_df.detach().requires_grad_(True)
        logits_df = model_or_fn(x_in)
        diff_logit = logits_df[0, df_pred] - logits_df[0, orig_label]
        (w,) = torch.autograd.grad(diff_logit, x_in, only_inputs=True)
        w = w.detach()

        # [AI-Prompt Hook]: Modas et al. (2019) Eq. (7) scales distance toward x_B by `lambda_`
        # to trade lower L0 sparsity (larger lambda_) vs lower L1 magnitude (lambda_ near 1).
        b_scaled = x_cur + lambda_ * (x_df - x_cur)

        if clip_bounds is not None:
            lb, ub = lb_box, ub_box
        elif x_single.is_complex():
            lb = torch.cat([x_cur.real.reshape(-1) - scale, x_cur.imag.reshape(-1) - scale])
            ub = torch.cat([x_cur.real.reshape(-1) + scale, x_cur.imag.reshape(-1) + scale])
        else:
            lb = x_cur - scale
            ub = x_cur + scale

        x_next = _linear_solver(x_cur, w, b_scaled, lb, ub)
        if clip_bounds is not None:
            x_next = torch.clamp(x_next, min=clip_bounds[0], max=clip_bounds[1])

        with torch.no_grad():
            pred_next = int(model_or_fn(x_next).argmax(dim=1)[0].item())
        x_cur = x_next
        if pred_next != orig_label:
            break

    # Fall back to last DeepFool point if greedy linear steps never crossed boundary inside max_iter.
    with torch.no_grad():
        needs_fallback = int(model_or_fn(x_cur).argmax(dim=1)[0].item()) == orig_label
    if needs_fallback:
        x_df, df_steps = _deepfool_single_sample(
            model_or_fn=model_or_fn,
            x_single=x_single,
            norm="l2",
            steps=deepfool_steps,
            overshoot=overshoot,
            max_classes=max_classes,
            clip_bounds=clip_bounds,
        )
        total_steps += df_steps
        with torch.no_grad():
            if int(model_or_fn(x_df).argmax(dim=1)[0].item()) != orig_label:
                x_cur = x_df

    return x_cur.detach(), total_steps


def sparsefool(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    eps: float | None = None,
    norm: str = "l0",
    lambda_: float = 3.0,
    max_iter: int = 12,
    deepfool_steps: int = 20,
    overshoot: float = 0.02,
    max_classes: int = 10,
    clip_bounds: tuple[float, float] | None = None,
) -> AttackResult:
    """Run SparseFool (Modas et al., CVPR 2019) to find sparse L0 adversarial examples."""
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}.")
    if lambda_ <= 0.0:
        raise ValueError(f"lambda_ must be > 0, got {lambda_}.")
    if clip_bounds is not None and x.is_complex():
        raise ValueError("clip_bounds is only supported for real-valued tensors.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    batch_size = x_clean.shape[0]
    x_adv = x_clean.clone()
    total_steps = 0

    with eval_mode(model_or_fn):
        with torch.no_grad():
            clean_preds = model_or_fn(x_clean).argmax(dim=1)

        for idx in range(batch_size):
            if y is not None and int(clean_preds[idx].item()) != int(y[idx].item()):
                continue
            adv_i, s_i = _sparsefool_single_sample(
                model_or_fn=model_or_fn,
                x_single=x_clean[idx : idx + 1],
                lambda_=lambda_,
                max_iter=max_iter,
                deepfool_steps=deepfool_steps,
                overshoot=overshoot,
                max_classes=max_classes,
                clip_bounds=clip_bounds,
            )
            x_adv[idx] = adv_i[0]
            total_steps += s_i

    delta = x_adv - x_clean
    if eps is not None:
        within = within_eps_budget(delta, x_clean, eps=eps, norm=norm).view(
            (batch_size,) + (1,) * (x_clean.ndim - 1)
        )
        x_adv = torch.where(within, x_adv, x_clean)
        delta = x_adv - x_clean

    l0_dist = compute_perturbation_norm(delta, norm="l0")
    l1_dist = compute_perturbation_norm(delta, norm="l1")
    return AttackResult(
        x_adv=x_adv,
        perturbation=delta,
        eps=float(eps) if eps is not None else float(l0_dist.median().item()),
        norm=norm,
        steps_run=total_steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "mean_l0": float(l0_dist.mean().item()),
            "median_l0": float(l0_dist.median().item()),
            "mean_l1": float(l1_dist.mean().item()),
            "lambda_": lambda_,
        },
    )
