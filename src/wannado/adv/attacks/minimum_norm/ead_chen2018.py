"""EAD: Elastic-Net Attacks to Deep Neural Networks (Chen et al., AAAI 2018).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Canonical L1 + L2 Elastic-Net Optimization Attack)
- Reference     : Chen, P.-Y., Sharma, Y., Zhang, H., Yi, J., & Hsieh, C.-J.
                  (2018). "EAD: Elastic-Net Attacks to Deep Neural Networks via
                  Adversarial Examples." AAAI 2018 (pp. 10-17).
- URL           : https://arxiv.org/abs/1709.04114
- Plain Intuition:
    C&W optimizes L2 energy smoothly, which tends to spread tiny changes over
    all coordinates. EAD adds an L1 sparsity penalty ``beta * ||delta||_1`` on
    top of C&W's L2 + hinge objective and solves it via FISTA (Fast Iterative
    Shrinkage-Thresholding Algorithm): each step takes a gradient step on the
    smooth L2 + margin terms and applies exact soft-thresholding shrinkage toward
    zero on the perturbation, producing sparse yet transferable L1 perturbations.
- Diff vs. C&W L2 (Carlini & Wagner, 2017):
    1. Solves ``beta * ||delta||_1 + ||delta||_2^2 + c * f_CW(x + delta)`` via FISTA.
    2. Supports ``decision_rule='EN'`` (Elastic-Net balance) and ``'L1'`` (pure L1 minimum).
- Why Chosen as a Baseline:
    Unifies L1 and L2 adversarial optimization (C&W L2 is literally the ``beta=0``
    special case of EAD) and produces highly transferable sparse perturbations.
- Extension Directions / Prompt Hooks:
    1. Tune ``beta`` from 0 (dense L2) to larger values (ultra-sparse L1).
    2. Use ``norm='jsr_db'`` with ``beta`` sparse penalty for multi-tone RF jamming.
===============================================================================
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import torch

from wannado.adv.attacks.minimum_norm.cw_l2_carlini2017 import _cw_margin_objective, _l2_sq_per_sample
from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    clamp_to_bounds,
    compute_perturbation_norm,
    eval_mode,
)
from wannado.adv.core.projections import within_eps_budget
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="ead",
    name="Elastic-Net Attack (EAD)",
    family="minimum_norm",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="high",
    supported_norms=("l1", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    parent_id="cw_l2",
    diff_from_parent="Generalizes C&W L2 by adding an L1 regularization term beta * ||delta||_1 solved via FISTA soft-thresholding shrinkage.",
    summary_plain="Optimizes Elastic-Net (L1 + L2) + C&W margin via FISTA shrinkage to craft sparse, transferable minimum-distortion attacks.",
    when_to_use="When measuring minimum L1 or Elastic-Net distortion margins, or generating sparse perturbations that generalize well.",
    when_not_to_use="When fast fixed-budget evaluation suffices (use apgd_l1 or apgd_ce) or on non-differentiable models.",
    extension_hooks=(
        "Swap `decision_rule` between `'EN'` (Elastic-Net) and `'L1'` (strictest L1 minimum)",
        "Add group-lasso structured shrinkage for patch or frequency-band sparsity",
    ),
    module_path="wannado.adv.attacks.minimum_norm.ead_chen2018",
    citation=AttackCitation(
        key="chen2018ead",
        title="EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples",
        authors="Pin-Yu Chen, Yash Sharma, Huan Zhang, Jinfeng Yi, and Cho-Jui Hsieh",
        venue="32nd AAAI Conference on Artificial Intelligence (AAAI)",
        year=2018,
        url="https://arxiv.org/abs/1709.04114",
        bibtex=(
            "@inproceedings{chen2018ead,\n"
            "  title={EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples},\n"
            "  author={Chen, Pin-Yu and Sharma, Yash and Zhang, Huan and Yi, Jinfeng and Hsieh, Cho-Jui},\n"
            "  booktitle={Proceedings of the 32nd AAAI Conference on Artificial Intelligence (AAAI)},\n"
            "  pages={10--17},\n"
            "  year={2018},\n"
            "  url={https://arxiv.org/abs/1709.04114},\n"
            "}"
        ),
        ieee='P.-Y. Chen, Y. Sharma, H. Zhang, J. Yi, and C.-J. Hsieh, "EAD: Elastic-net attacks to deep neural networks via adversarial examples," in Proc. AAAI Conf. Artif. Intell., 2018, pp. 10-17.',
        apa="Chen, P.-Y., Sharma, Y., Zhang, H., Yi, J., & Hsieh, C.-J. (2018). EAD: Elastic-net attacks to deep neural networks via adversarial examples. Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 10-17.",
    ),
)


def _soft_threshold_box(
    z: torch.Tensor,
    x_clean: torch.Tensor,
    thresh: torch.Tensor,
    clip_bounds: tuple[Any, Any] | None,
) -> torch.Tensor:
    """Proximal operator for beta * ||delta||_1 + indicator_{box}(x + delta), real & complex safe."""
    delta = z - x_clean
    if delta.is_complex():
        mag = delta.abs()
        shrunk = torch.sgn(delta) * (mag - thresh).clamp(min=0.0)
        return x_clean + shrunk

    shrunk = torch.sign(delta) * (delta.abs() - thresh).clamp(min=0.0)
    x_out = x_clean + shrunk
    return clamp_to_bounds(x_out, clip_bounds)


def ead(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float | None = None,
    norm: str = "l1",
    beta: float = 1e-2,
    decision_rule: str = "EN",
    confidence: float = 0.0,
    lr: float = 0.02,
    steps: int = 120,
    binary_search_steps: int = 5,
    initial_const: float = 1e-2,
    abort_early: bool = True,
    targeted: bool = False,
    clip_bounds: tuple[Any, Any] | None = None,
) -> AttackResult:
    """Run EAD Elastic-Net (L1 + L2) attack via FISTA optimization."""
    if decision_rule not in ("EN", "L1"):
        raise ValueError(f"decision_rule must be 'EN' or 'L1', got '{decision_rule}'.")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if binary_search_steps < 1:
        raise ValueError(f"binary_search_steps must be >= 1, got {binary_search_steps}.")
    if beta < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta}.")
    if clip_bounds is not None and x.is_complex():
        raise ValueError("clip_bounds is only supported for real-valued tensors.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    batch_size = x_clean.shape[0]
    real_dtype = x_clean.real.dtype if x_clean.is_complex() else x_clean.dtype
    view_shape = (batch_size,) + (1,) * (x_clean.ndim - 1)

    lower_bound = torch.zeros(batch_size, device=x_clean.device, dtype=real_dtype)
    upper_bound = torch.full((batch_size,), 1e10, device=x_clean.device, dtype=real_dtype)
    const = torch.full((batch_size,), float(initial_const), device=x_clean.device, dtype=real_dtype)

    o_best_score = torch.full((batch_size,), float("inf"), device=x_clean.device, dtype=real_dtype)
    o_best_adv = x_clean.clone()
    total_steps = 0

    with eval_mode(model_or_fn):
        for outer_step in range(binary_search_steps):
            x_k = x_clean.clone()
            y_k = x_clean.clone()
            t_k = 1.0

            best_score = torch.full((batch_size,), float("inf"), device=x_clean.device, dtype=real_dtype)
            prev_loss = float("inf")

            for step_idx in range(steps):
                total_steps += 1
                # Polynomial step-size decay (Chen et al., AAAI 2018, Eq. 5)
                alpha_k = lr * math.sqrt(max(1.0 - float(step_idx) / float(steps + 1), 1e-4))

                # Evaluate gradient of smooth term g(y_k) = ||y_k - x||_2^2 + c * f_CW(y_k)
                y_in = y_k.detach().requires_grad_(True)
                logits = model_or_fn(y_in)
                delta_y = y_in - x_clean
                l2_sq = _l2_sq_per_sample(delta_y)
                f_loss, _ = _cw_margin_objective(
                    logits, y, confidence=confidence, targeted=targeted
                )
                smooth_obj = (l2_sq + const * f_loss).sum()
                (grad_y,) = torch.autograd.grad(smooth_obj, y_in, only_inputs=True)

                with torch.no_grad():
                    # [AI-Prompt Hook]: FISTA proximal gradient step + exact L1 soft thresholding.
                    thresh = torch.full(view_shape, beta * alpha_k, device=x_clean.device, dtype=real_dtype)
                    x_next = _soft_threshold_box(
                        y_k - alpha_k * grad_y.detach(), x_clean, thresh, clip_bounds
                    )
                    t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k))
                    mom_coeff = (t_k - 1.0) / t_next
                    y_k = clamp_to_bounds(x_next + mom_coeff * (x_next - x_k), clip_bounds)
                    x_k = x_next
                    t_k = t_next

                    logits_x = model_or_fn(x_k)
                    delta_x = x_k - x_clean
                    l1_val = delta_x.abs().reshape(batch_size, -1).sum(dim=1).to(real_dtype)
                    l2_val = _l2_sq_per_sample(delta_x)
                    f_x, succeeded = _cw_margin_objective(
                        logits_x, y, confidence=confidence, targeted=targeted
                    )
                    en_score = l1_val if decision_rule == "L1" else (beta * l1_val + l2_val)
                    total_en_loss = float((beta * l1_val + l2_val + const * f_x).sum().item())

                    improved_inner = succeeded & (en_score < best_score)
                    best_score = torch.where(improved_inner, en_score, best_score)

                    improved_outer = succeeded & (en_score < o_best_score)
                    o_best_score = torch.where(improved_outer, en_score, o_best_score)
                    o_best_adv = torch.where(improved_outer.view(view_shape), x_k, o_best_adv)

                if abort_early and step_idx % max(steps // 10, 1) == 0 and step_idx > 0:
                    if total_en_loss > prev_loss * 0.9999:
                        break
                    prev_loss = total_en_loss

            if outer_step < binary_search_steps - 1:
                found = best_score < float("inf")
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
    if eps is not None:
        within_budget = within_eps_budget(delta, x_clean, eps=eps, norm=norm).view(view_shape)
        o_best_adv = torch.where(within_budget, o_best_adv, x_clean)
        delta = o_best_adv - x_clean

    dist = compute_perturbation_norm(delta, norm=norm, x_ref=x_clean)
    l0_dist = compute_perturbation_norm(delta, norm="l0")
    success_rate = float((o_best_score < float("inf")).float().mean().item())
    return AttackResult(
        x_adv=o_best_adv,
        perturbation=delta,
        eps=float(eps) if eps is not None else float(dist.median().item()),
        norm=norm,
        steps_run=total_steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "success_rate": success_rate,
            "decision_rule": decision_rule,
            "beta": beta,
            "mean_distortion": float(dist.mean().item()),
            "median_distortion": float(dist.median().item()),
            "mean_l0": float(l0_dist.mean().item()),
        },
    )
