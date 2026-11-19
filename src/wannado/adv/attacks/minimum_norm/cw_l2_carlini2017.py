"""C&W L2: Carlini & Wagner Optimization Attack (Carlini & Wagner, IEEE S&P 2017).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Gold-Standard Optimization-Based Minimum-L2 Attack)
- Reference     : Carlini, N., & Wagner, D. (2017). "Towards Evaluating the
                  Robustness of Neural Networks." 2017 IEEE Symposium on
                  Security and Privacy (SP), pp. 39-57.
- URL           : https://arxiv.org/abs/1608.04644 (DOI: 10.1109/SP.2017.49)
- Plain Intuition:
    Instead of constraining perturbation inside a hard ball, C&W solves a
    smooth joint optimization:
        min ||delta||_2^2 + c * f_CW(x + delta)
    where ``f_CW`` is a hinge-like logit margin loss (0 once misclassified by
    margin ``confidence``), and an outer binary search tunes ``c`` to find the
    smallest possible L2 distortion that still breaks the classifier.
- Domain-Agnostic Improvement in wannado.adv:
    Standard implementations hardcode ``atanh(2x - 1)`` assuming ``x in [0, 1]``,
    which produces NaNs on unbounded features and crashes on complex signals.
    Here, ``clip_bounds=(low, high)`` uses generalized ``tanh`` box mapping,
    while ``clip_bounds=None`` optimizes ``delta`` directly in real or complex64
    space—working seamlessly for wireless IQ signals and regression tensors.
- Extension Directions (for further research):
    1. Add domain penalties inside ``f_CW`` (e.g. spectral mask / PAPR penalty).
    2. Add L1 sparsity penalty -> EAD (Chen et al., AAAI 2018).
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
    coerce_clip_bounds,
    compute_perturbation_norm,
    eval_mode,
)
from wannado.adv.core.losses import _margin_loss
from wannado.adv.core.projections import within_eps_budget
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="cw_l2",
    name="Carlini & Wagner L2 Attack (C&W L2)",
    family="minimum_norm",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="high",
    supported_norms=("l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    parent_id=None,
    diff_from_parent="Lagrangian L2 + margin-hinge optimization with outer binary search over penalty constant c.",
    summary_plain="Finds near-minimal L2/energy perturbations via Adam optimization over a smooth logit-margin objective and binary search.",
    when_to_use="When measuring minimum jamming energy/distortion needed to break a classifier, or auditing defenses against gradient-based optimization.",
    when_not_to_use="When you need sub-second evaluation on huge datasets (use PGD/APGD/DeepFool first) or Linf-only threat models (use APGD/FAB).",
    extension_hooks=(
        "Add wireless spectral leakage or PAPR penalty terms directly to the objective",
        "Add L1 regularization for sparse minimum perturbations -> EAD (Chen et al., AAAI 2018)",
    ),
    module_path="wannado.adv.attacks.minimum_norm.cw_l2_carlini2017",
    citation=AttackCitation(
        key="carlini2017towards",
        title="Towards Evaluating the Robustness of Neural Networks",
        authors="Nicholas Carlini and David Wagner",
        venue="2017 IEEE Symposium on Security and Privacy (SP)",
        year=2017,
        url="https://arxiv.org/abs/1608.04644",
        bibtex=(
            "@inproceedings{carlini2017towards,\n"
            "  title={Towards Evaluating the Robustness of Neural Networks},\n"
            "  author={Carlini, Nicholas and Wagner, David},\n"
            "  booktitle={2017 IEEE Symposium on Security and Privacy (SP)},\n"
            "  pages={39--57},\n"
            "  year={2017},\n"
            "  organization={IEEE},\n"
            "  doi={10.1109/SP.2017.49},\n"
            "  url={https://arxiv.org/abs/1608.04644},\n"
            "}"
        ),
        ieee='N. Carlini and D. Wagner, "Towards evaluating the robustness of neural networks," in Proc. IEEE Symp. Secur. Privacy (SP), 2017, pp. 39-57.',
        apa="Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. 2017 IEEE Symposium on Security and Privacy (SP), 39-57.",
    ),
)


def _cw_margin_objective(
    logits: torch.Tensor,
    y: torch.Tensor,
    confidence: float,
    targeted: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return C&W hinge loss f(x') >= 0 and boolean success indicator."""
    margin = _margin_loss(logits, y)  # max_{j != y} z_j - z_y
    if targeted:
        # Want z_t >= max_{j != t} z_j + confidence -> penalize margin + confidence > 0.
        raw = margin + confidence
        succeeded = margin <= -confidence
    else:
        # Want max_{j != y} z_j >= z_y + confidence -> penalize -margin + confidence > 0.
        raw = -margin + confidence
        succeeded = margin >= confidence
    return torch.clamp(raw, min=0.0), succeeded


def _l2_sq_per_sample(delta: torch.Tensor) -> torch.Tensor:
    flat = delta.reshape(delta.shape[0], -1)
    if flat.is_complex():
        return (flat.real.pow(2) + flat.imag.pow(2)).sum(dim=1)
    return flat.pow(2).sum(dim=1)


def cw_l2(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float | None = None,
    norm: str = "l2",
    confidence: float = 0.0,
    lr: float = 0.01,
    steps: int = 200,
    binary_search_steps: int = 5,
    initial_const: float = 1e-2,
    abort_early: bool = True,
    targeted: bool = False,
    clip_bounds: tuple[Any, Any] | None = None,
) -> AttackResult:
    """Run Carlini & Wagner L2 attack (bounded tanh space or unbounded real/complex space).

    Args:
        model_or_fn: Differentiable ``nn.Module`` or callable mapping ``x -> (B, C)`` logits.
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Ground-truth labels (or target class when ``targeted=True``).
        eps: Optional maximum budget cutoff. When set (e.g. ``l2``, ``relative_l2``, or
            ``jsr_db``), any found perturbation exceeding ``eps_budget`` reverts to clean ``x``.
        norm: Distortion norm used if ``eps`` is provided (``"l2"``, ``"relative_l2"``, ``"jsr_db"``).
        confidence: Margin confidence ``kappa >= 0`` required for success.
        lr: Adam learning rate for inner optimization.
        steps: Maximum inner optimization iterations per binary-search step.
        binary_search_steps: Outer binary search rounds for balancing constant ``c``.
        initial_const: Initial weighting constant ``c`` on classification margin loss.
        abort_early: Stop inner Adam loop early if objective stops improving.
        targeted: If ``True``, steers prediction toward ``y`` with margin ``confidence``.
        clip_bounds: Optional ``(min, max)`` box bounds for real-valued inputs (uses tanh reparameterization).

    Returns:
        ``AttackResult`` containing minimum-L2 adversarial examples found across search steps.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if binary_search_steps < 1:
        raise ValueError(f"binary_search_steps must be >= 1, got {binary_search_steps}.")
    if clip_bounds is not None and x.is_complex():
        raise ValueError("clip_bounds is only supported for real-valued tensors.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    batch_size = x_clean.shape[0]
    real_dtype = x_clean.real.dtype if x_clean.is_complex() else x_clean.dtype
    view_shape = (batch_size,) + (1,) * (x_clean.ndim - 1)

    if clip_bounds is not None:
        low, high = coerce_clip_bounds(clip_bounds, x_clean)
        if bool((low >= high).any().item()):
            raise ValueError(f"clip_bounds requires low < high, got {clip_bounds}.")
        half_span = 0.5 * (high - low)
        mid = 0.5 * (high + low)
        # Map x in [low, high] -> (-1, 1) -> unconstrained tanh variable w.
        scaled = torch.clamp((x_clean - mid) / half_span, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        w_init = torch.atanh(scaled)

    lower_bound = torch.zeros(batch_size, device=x_clean.device, dtype=real_dtype)
    upper_bound = torch.full((batch_size,), 1e10, device=x_clean.device, dtype=real_dtype)
    const = torch.full((batch_size,), float(initial_const), device=x_clean.device, dtype=real_dtype)

    o_best_l2 = torch.full((batch_size,), float("inf"), device=x_clean.device, dtype=real_dtype)
    o_best_adv = x_clean.clone()
    total_steps = 0

    with eval_mode(model_or_fn):
        for outer_step in range(binary_search_steps):
            if clip_bounds is not None:
                opt_var = w_init.clone().detach().requires_grad_(True)
            else:
                opt_var = torch.zeros_like(x_clean, requires_grad=True)

            optimizer = torch.optim.Adam([opt_var], lr=lr)
            best_l2 = torch.full((batch_size,), float("inf"), device=x_clean.device, dtype=real_dtype)
            prev_loss = float("inf")

            for step_idx in range(steps):
                total_steps += 1
                optimizer.zero_grad()

                if clip_bounds is not None:
                    x_adv_cur = mid + half_span * torch.tanh(opt_var)
                else:
                    x_adv_cur = x_clean + opt_var

                logits = model_or_fn(x_adv_cur)
                l2_sq = _l2_sq_per_sample(x_adv_cur - x_clean)
                f_loss, succeeded = _cw_margin_objective(
                    logits, y, confidence=confidence, targeted=targeted
                )
                loss_per_sample = l2_sq + const * f_loss
                total_loss = loss_per_sample.sum()

                (grad_opt,) = torch.autograd.grad(total_loss, opt_var, only_inputs=True)
                opt_var.grad = grad_opt.detach()
                optimizer.step()

                with torch.no_grad():
                    l2_det = l2_sq.detach()
                    improved_inner = succeeded & (l2_det < best_l2)
                    best_l2 = torch.where(improved_inner, l2_det, best_l2)

                    improved_outer = succeeded & (l2_det < o_best_l2)
                    o_best_l2 = torch.where(improved_outer, l2_det, o_best_l2)
                    o_best_adv = torch.where(improved_outer.view(view_shape), x_adv_cur.detach(), o_best_adv)

                if abort_early and step_idx % max(steps // 10, 1) == 0 and step_idx > 0:
                    cur_val = float(total_loss.item())
                    if cur_val > prev_loss * 0.9999:
                        break
                    prev_loss = cur_val

            # Outer binary search update for per-sample penalty constant c.
            if outer_step < binary_search_steps - 1:
                found = best_l2 < float("inf")
                upper_bound = torch.where(found, torch.minimum(upper_bound, const), upper_bound)
                lower_bound = torch.where(~found, torch.maximum(lower_bound, const), lower_bound)
                has_finite_upper = upper_bound < 1e9
                const_mid = 0.5 * (lower_bound + upper_bound)
                const = torch.where(
                    found & has_finite_upper,
                    const_mid,
                    torch.where(~found & has_finite_upper, const_mid, torch.where(~found, const * 10.0, const)),
                )

    delta = o_best_adv - x_clean
    succeeded_mask = o_best_l2 < float("inf")
    if eps is not None:
        within_1d = within_eps_budget(delta, x_clean, eps=eps, norm=norm)
        within_budget = within_1d.view(view_shape)
        o_best_adv = torch.where(within_budget, o_best_adv, x_clean)
        delta = o_best_adv - x_clean
        succeeded_mask = succeeded_mask & within_1d

    dist_l2 = compute_perturbation_norm(delta, norm=norm, x_ref=x_clean)
    success_rate = float(succeeded_mask.float().mean().item())
    return AttackResult(
        x_adv=o_best_adv,
        perturbation=delta,
        eps=float(eps) if eps is not None else float(dist_l2.median().item()),
        norm=norm,
        steps_run=total_steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "success_rate": success_rate,
            "mean_distortion": float(dist_l2.mean().item()),
            "median_distortion": float(dist_l2.median().item()),
            "confidence": confidence,
            "binary_search_steps": binary_search_steps,
        },
    )
