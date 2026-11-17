"""VMI-FGSM: Variance-Tuned Momentum Iterative FGSM (Wang & He, CVPR 2021).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Modern Variance-Reduced Transferability SOTA Baseline)
- Reference     : Wang, X., & He, K. (2021). "Enhancing the Transferability of
                  Adversarial Attacks Through Variance Tuning." CVPR 2021 (pp. 1924-1933).
- URL           : https://arxiv.org/abs/2103.15571
- Plain Intuition:
    Standard MI-FGSM computes gradients only at the exact point ``x_t``, which
    can point toward sharp surrogate-specific quirks. VMI-FGSM probes ``N`` random
    neighbors inside a local radius ``beta * eps`` around ``x_t`` to estimate
    how much the gradient varies in the neighborhood (``v_{t+1} = g_nbr_mean - g_t``),
    and adds that variance-correction vector to the next step's gradient. This
    smooths out surrogate noise and significantly stabilizes cross-model transfer.
- Diff vs. MI-FGSM (Dong et al., 2018):
    Maintains a variance-tuning correction vector ``v_t`` computed over ``N``
    neighborhood probes around each iterate ``x_t`` inside ``beta * eps_budget``.
- Why Chosen as a Baseline:
    Purely optimization-level, domain-agnostic upgrade over MI-FGSM that works
    without spatial assumptions on vision, tabular, DRL, and complex wireless signals.
- Extension Directions / Prompt Hooks:
    1. Combine look-ahead Nesterov step with variance tuning -> VNI-FGSM.
    2. Use anisotropic covariance / channel-fading neighborhood draws for RF IQ signals.
===============================================================================
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    compute_attack_objective,
    eval_mode,
    make_generator,
    project_perturbation,
    random_init_perturbation,
    resolve_eps_budget,
    resolve_loss_fn,
    safe_l2_norm,
    step_direction,
)
from wannado.adv.core.losses import LossFn
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="vmi_fgsm",
    name="Variance-Tuned MI-FGSM (VMI-FGSM)",
    family="transferability",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="medium",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="mi_fgsm",
    diff_from_parent="Adds local neighborhood gradient variance tuning v_t across N random probes around each iterate.",
    summary_plain="Samples N local neighbors per step to cancel surrogate gradient variance and boost black-box transferability.",
    when_to_use="Strong domain-agnostic transfer attack when spatial transforms (DI/TI) do not apply or when attacking defended surrogates.",
    when_not_to_use="Under ultra-tight runtime budgets where evaluating num_variance_samples extra gradients per step is too slow (use ni_fgsm).",
    extension_hooks=(
        "Add Nesterov look-ahead before g_t evaluation -> VNI-FGSM",
        "Sample neighborhood perturbations along Rayleigh/phase manifold for complex wireless signals",
    ),
    module_path="wannado.adv.attacks.transferability.vmifgsm_wang2021",
    citation=AttackCitation(
        key="wang2021enhancing",
        title="Enhancing the Transferability of Adversarial Attacks Through Variance Tuning",
        authors="Xiaosen Wang and Kun He",
        venue="IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
        year=2021,
        url="https://arxiv.org/abs/2103.15571",
        bibtex=(
            "@inproceedings{wang2021enhancing,\n"
            "  title={Enhancing the Transferability of Adversarial Attacks Through Variance Tuning},\n"
            "  author={Wang, Xiaosen and He, Kun},\n"
            "  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},\n"
            "  pages={1924--1933},\n"
            "  year={2021},\n"
            "  url={https://arxiv.org/abs/2103.15571},\n"
            "}"
        ),
        ieee='X. Wang and K. He, "Enhancing the transferability of adversarial attacks through variance tuning," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 1924-1933.',
        apa="Wang, X., & He, K. (2021). Enhancing the transferability of adversarial attacks through variance tuning. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 1924-1933.",
    ),
)


def _sample_neighbor_offset(
    x_ref: torch.Tensor,
    radius: torch.Tensor,
    norm: str,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample a random local offset inside U(-radius, +radius) or radius-scaled sphere."""
    real_dtype = x_ref.real.dtype if x_ref.is_complex() else x_ref.dtype
    if norm == "linf" and not x_ref.is_complex():
        return (
            torch.rand(x_ref.shape, device=x_ref.device, dtype=real_dtype, generator=generator) * 2.0 - 1.0
        ) * radius

    if x_ref.is_complex():
        re = torch.rand(x_ref.shape, device=x_ref.device, dtype=real_dtype, generator=generator) * 2.0 - 1.0
        im = torch.rand(x_ref.shape, device=x_ref.device, dtype=real_dtype, generator=generator) * 2.0 - 1.0
        raw = torch.complex(re, im).to(x_ref.dtype)
    else:
        raw = torch.rand(x_ref.shape, device=x_ref.device, dtype=real_dtype, generator=generator) * 2.0 - 1.0

    # For L2 / JSR_dB constraints, match neighborhood scale per feature so radius stays proportional.
    if norm in ("l2", "relative_l2", "jsr_db"):
        b = x_ref.shape[0]
        raw_norm = safe_l2_norm(raw.reshape(b, -1), dim=1).view(radius.shape)
        unit = raw / raw_norm.clamp(min=1e-12)
        scale = torch.rand(radius.shape, device=x_ref.device, dtype=real_dtype, generator=generator).pow(0.5)
        return unit * (radius * scale)

    # Complex linf: scale magnitude within radius disk
    mag = raw.abs().clamp(min=1e-12)
    scale = torch.rand(x_ref.shape, device=x_ref.device, dtype=real_dtype, generator=generator)
    return (raw / mag) * (radius * scale)


def vmi_fgsm(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    step_size: float | None = None,
    steps: int = 10,
    decay: float = 1.0,
    beta: float = 1.5,
    num_variance_samples: int = 8,
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    random_start: bool = False,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run Variance-Tuned Momentum Iterative FGSM (VMI-FGSM)."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if decay < 0.0:
        raise ValueError(f"decay must be >= 0, got {decay}.")
    if beta < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta}.")
    if num_variance_samples < 1:
        raise ValueError(f"num_variance_samples must be >= 1, got {num_variance_samples}.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    loss_fn = resolve_loss_fn(loss, targeted=targeted)
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
            extra={
                "decay": decay,
                "beta": beta,
                "num_variance_samples": num_variance_samples,
            },
        )

    alpha = (
        eps_budget / float(steps)
        if step_size is None
        else torch.full_like(eps_budget, float(step_size))
    )
    generator = make_generator(seed, x_clean.device)

    if random_start:
        delta = random_init_perturbation(
            x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds, generator=generator
        )
    else:
        delta = torch.zeros_like(x_clean)

    momentum = torch.zeros_like(x_clean)
    variance = torch.zeros_like(x_clean)
    nbr_radius = beta * eps_budget
    view_shape = (x_clean.shape[0],) + (1,) * (x_clean.ndim - 1)

    with eval_mode(model_or_fn):
        for step_idx in range(steps):
            x_in = (x_clean + delta).detach().requires_grad_(True)
            objective = compute_attack_objective(
                model_or_fn, x_in, y, loss_fn, targeted=targeted
            )
            (grad,) = torch.autograd.grad(objective.sum(), x_in, only_inputs=True)
            grad = grad.detach()

            # Combine current gradient with variance correction v_t (Wang & He, 2021, Eq. 5).
            new_grad = grad + variance
            grad_l1 = new_grad.abs().reshape(x_clean.shape[0], -1).sum(dim=1).view(view_shape)
            momentum = decay * momentum + new_grad / grad_l1.clamp(min=1e-12)

            # Update variance estimate for next step (skip redundant sampling on the final step).
            if step_idx < steps - 1:
                global_grad = torch.zeros_like(grad)
                for _ in range(num_variance_samples):
                    offset = _sample_neighbor_offset(x_clean, nbr_radius, norm=norm, generator=generator)
                    x_nbr = (x_clean + delta + offset).detach()
                    if clip_bounds is not None:
                        x_nbr = x_nbr.clamp(min=clip_bounds[0], max=clip_bounds[1])
                    x_nbr = x_nbr.requires_grad_(True)
                    obj_n = compute_attack_objective(
                        model_or_fn, x_nbr, y, loss_fn, targeted=targeted
                    )
                    (g_n,) = torch.autograd.grad(obj_n.sum(), x_nbr, only_inputs=True)
                    global_grad = global_grad + g_n.detach()
                # [AI-Prompt Hook]: Wang & He (2021) Eq. (4) gradient variance v_{t+1} = mean(g_nbr) - g_t.
                variance = (global_grad / float(num_variance_samples)) - grad

            delta = delta + alpha * step_direction(momentum, norm=norm)
            delta = project_perturbation(
                delta, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds
            )

    return AttackResult(
        x_adv=x_clean + delta,
        perturbation=delta,
        eps=eps,
        norm=norm,
        steps_run=steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "decay": decay,
            "beta": beta,
            "num_variance_samples": num_variance_samples,
        },
    )
