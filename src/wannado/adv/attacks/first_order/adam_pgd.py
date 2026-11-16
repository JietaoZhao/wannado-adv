"""Adam-PGD: Projected Gradient Ascent with Adam Adaptive Moments.

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : A (Continuous-Signal / L2 Optimization Benchmark)
- Reference     : Combines Madry et al. (ICLR 2018) PGD projection framework
                  with Kingma & Ba (ICLR 2015) Adam adaptive moment estimation.
- URLs          : https://openreview.net/forum?id=rJzIBfZAb (PGD)
                  https://arxiv.org/abs/1412.6980 (Adam)
- Plain Intuition:
    Standard PGD uses ``sign(grad)`` or fixed-step L2 updates, which can
    oscillate wildly across directions with uneven curvature (common in raw I/Q
    wireless waveforms, regression targets, and continuous DRL states). Adam-PGD
    acts like an automatic transmission: it smooths gradients using a moving
    average ``m`` and scales down noisy, high-variance coordinates using ``sqrt(v)``.
- Diff vs. PGD (2018):
    Replaces raw ``step_direction(grad)`` with bias-corrected Adam moment ratios
    ``m_hat / (sqrt(v_hat) + eps)`` before projecting back onto the budget ball.
- Why Chosen as a Baseline:
    Frequently outperforms vanilla sign/L2-PGD on smooth continuous-valued tasks
    (complex baseband communication signals, channel estimators, and L2 budgets).
- Extension Directions (for further research):
    1. Pair with EoT expectations for stochastic wireless fading channels.
    2. Add cosine learning-rate annealing for tighter late-stage convergence.
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
    adam_step_direction,
    compute_attack_objective,
    eval_mode,
    make_generator,
    project_perturbation,
    random_init_perturbation,
    resolve_eps_budget,
    resolve_loss_fn,
    update_best_by_objective,
)
from wannado.adv.core.losses import LossFn
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="adam_pgd",
    name="Adam Projected Gradient Descent (Adam-PGD)",
    family="first_order",
    tier="A",
    threat_model="white_box",
    requires_training=False,
    compute_cost="low",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="pgd",
    diff_from_parent="Replaces raw sign/unit-L2 PGD updates with bias-corrected Adam first and second moment adaptive steps.",
    summary_plain="Replaces fixed PGD steps with Adam's adaptive learning-rate transmission—especially effective on continuous signals and L2 power budgets.",
    when_to_use="Continuous signals (wireless IQ, audio, sensor waveforms), regression losses, or L2/JSR-dB constraints where vanilla PGD oscillates.",
    when_not_to_use="When comparing strictly against standard Madry PGD or Croce & Hein APGD in CV classification benchmarks.",
    extension_hooks=(
        "Add cosine lr decay across steps",
        "Combine with EoT random channel sampling for wireless fading robustness",
    ),
    module_path="wannado.adv.attacks.first_order.adam_pgd",
    citation=AttackCitation(
        key="kingma2015adam",
        title="Adam: A Method for Stochastic Optimization (paired with Madry et al. PGD, ICLR 2018)",
        authors="Diederik P. Kingma and Jimmy Ba",
        venue="3rd International Conference on Learning Representations (ICLR)",
        year=2015,
        url="https://arxiv.org/abs/1412.6980",
        bibtex=(
            "@inproceedings{kingma2015adam,\n"
            "  author={Kingma, Diederik P. and Ba, Jimmy},\n"
            "  title={Adam: A Method for Stochastic Optimization},\n"
            "  booktitle={International Conference on Learning Representations (ICLR)},\n"
            "  year={2015},\n"
            "  url={https://arxiv.org/abs/1412.6980},\n"
            "}"
        ),
        ieee='D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. Int. Conf. Learn. Represent. (ICLR), 2015 (combined with Madry et al., ICLR 2018 PGD projection).',
        apa="Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. International Conference on Learning Representations (ICLR).",
    ),
)


def adam_pgd(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    lr: float | None = None,
    steps: int = 20,
    betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    norm: str = "l2",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    random_start: bool = True,
    restarts: int = 1,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run projected gradient ascent using Adam adaptive first and second moments.

    Args:
        model_or_fn: Differentiable ``nn.Module`` or callable mapping ``x -> outputs``.
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Ground-truth labels/targets (or target class when ``targeted=True``).
        eps: Perturbation budget (or JSR in dB when ``norm="jsr_db"``).
        lr: Step scale per iteration. If ``None``, auto-scales to ``min(1.0, 2.5 / steps) * eps_budget``.
        steps: Number of Adam optimization steps per restart.
        betas: Exponential decay rates ``(beta1, beta2)`` for first and second moments.
        adam_eps: Numerical stability constant in Adam denominator.
        norm: Constraint norm: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        loss: Built-in loss name (``"ce"``, ``"margin"``, ``"dlr"``, ``"mse"``) or callable.
        targeted: If ``True``, minimizes loss toward ``y``.
        random_start: Whether to initialize uniformly inside the constraint ball.
        restarts: Number of independent restarts; keeps the highest-objective sample per item.
        clip_bounds: Optional valid range ``(min, max)`` for real-valued inputs.
        seed: Optional random seed for reproducible restarts.

    Returns:
        ``AttackResult`` containing worst-case ``x_adv``, ``perturbation``, and run metadata.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if restarts < 1:
        raise ValueError(f"restarts must be >= 1, got {restarts}.")

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
            extra={"restarts": restarts, "betas": betas},
        )

    alpha = (
        eps_budget * min(1.0, 2.5 / steps)
        if lr is None
        else torch.full_like(eps_budget, float(lr))
    )
    beta1, beta2 = betas
    generator = make_generator(seed, x_clean.device)

    best_delta = torch.zeros_like(x_clean)
    best_obj = torch.full(
        (x_clean.shape[0],), float("-inf"), device=x_clean.device, dtype=eps_budget.dtype
    )

    with eval_mode(model_or_fn):
        for _ in range(restarts):
            if random_start:
                delta = random_init_perturbation(
                    x_clean,
                    eps_budget,
                    norm=norm,
                    clip_bounds=clip_bounds,
                    generator=generator,
                )
            else:
                delta = torch.zeros_like(x_clean)

            m = torch.zeros_like(x_clean)
            v = torch.zeros(x_clean.shape, device=x_clean.device, dtype=eps_budget.dtype)

            for step_idx in range(1, steps + 1):
                x_in = (x_clean + delta).detach().requires_grad_(True)
                objective = compute_attack_objective(
                    model_or_fn, x_in, y, loss_fn, targeted=targeted
                )
                (grad,) = torch.autograd.grad(objective.sum(), x_in, only_inputs=True)
                grad = grad.detach()

                # Compute real squared gradient magnitude so Adam works seamlessly on complex64 too.
                grad_sq = grad.real.pow(2) + grad.imag.pow(2) if grad.is_complex() else grad.pow(2)
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * grad_sq

                m_hat = m / (1.0 - math.pow(beta1, step_idx))
                v_hat = v / (1.0 - math.pow(beta2, step_idx))
                adam_update = m_hat / (v_hat.sqrt() + adam_eps)

                delta = delta + alpha * adam_step_direction(adam_update, norm=norm)
                delta = project_perturbation(
                    delta, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds
                )

            with torch.no_grad():
                final_obj = compute_attack_objective(
                    model_or_fn, x_clean + delta, y, loss_fn, targeted=targeted
                )
                best_delta, best_obj = update_best_by_objective(best_delta, best_obj, delta, final_obj)

    return AttackResult(
        x_adv=x_clean + best_delta,
        perturbation=best_delta,
        eps=eps,
        norm=norm,
        steps_run=steps * restarts,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={"restarts": restarts, "steps_per_restart": steps, "betas": betas},
    )
