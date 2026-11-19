"""PGD: Projected Gradient Descent Attack (Madry et al., ICLR 2018).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Gold-Standard First-Order White-Box Benchmark)
- Reference     : Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018).
                  "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR 2018.
- URL           : https://openreview.net/forum?id=rJzIBfZAb
- Plain Intuition:
    Instead of jumping once like FGSM, PGD first drops randomly inside the
    allowed perturbation ball (``random_start=True``) to escape flat spots near
    ``x``, then walks iteratively in small gradient steps and projects back onto
    the constraint ball after every step.
- Diff vs. FGSM (2015) & BIM (2017):
    1. Adds random uniform initialization inside the norm ball before iterating.
    2. Iterates for ``steps`` small updates with per-step projection.
    3. Supports multiple ``restarts`` and keeps the worst-case perturbation per sample.
- Why Chosen as a Baseline:
    Universally treated as the primary first-order white-box benchmark in almost
    all adversarial robustness papers (CV, wireless communications, DRL, etc.).
- Extension Directions (for further research):
    1. Add momentum to ``grad`` updates -> MI-FGSM (Dong et al., CVPR 2018).
    2. Replace ``step_direction`` with Adam optimizer -> Adam-PGD (strong for L2/IQ signals).
    3. Add automatic step-size halving & DLR loss -> APGD (Croce & Hein, ICML 2020).
    4. Average gradients across stochastic channel realizations -> EoT-PGD (Athalye et al., ICML 2018).
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
    project_perturbation,
    random_init_perturbation,
    resolve_eps_budget,
    resolve_loss_fn,
    step_direction,
)
from wannado.adv.core.losses import LossFn
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="pgd",
    name="Projected Gradient Descent (PGD)",
    family="first_order",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="low",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="fgsm",
    diff_from_parent="Replaces FGSM's single step with random initialization inside the ball, iterative small steps, and per-step projection (plus optional restarts).",
    summary_plain="Randomly starts inside the allowed budget and iteratively climbs the loss surface while projecting back inside constraints.",
    when_to_use="Primary white-box evaluation baseline for almost any differentiable model.",
    when_not_to_use="When the model contains non-differentiable steps (use SPSA/Square/HopSkipJump) or suffers from Softmax saturation (switch loss='margin' or use APGD-DLR).",
    extension_hooks=(
        "Accumulate gradient momentum -> MI-FGSM / NI-FGSM",
        "Use Adam moment estimates instead of sign/L2 step -> Adam-PGD",
        "Use adaptive step-size halving -> APGD-CE / APGD-DLR",
        "Monte-Carlo sample random channel/environment transforms per step -> EoT-PGD",
    ),
    module_path="wannado.adv.attacks.first_order.pgd_madry2018",
    citation=AttackCitation(
        key="madry2018towards",
        title="Towards Deep Learning Models Resistant to Adversarial Attacks",
        authors="Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu",
        venue="6th International Conference on Learning Representations (ICLR)",
        year=2018,
        url="https://openreview.net/forum?id=rJzIBfZAb",
        bibtex=(
            "@inproceedings{madry2018towards,\n"
            "  title={Towards Deep Learning Models Resistant to Adversarial Attacks},\n"
            "  author={Aleksander Madry and Aleksandar Makelov and Ludwig Schmidt and Dimitris Tsipras and Adrian Vladu},\n"
            "  booktitle={International Conference on Learning Representations (ICLR)},\n"
            "  year={2018},\n"
            "  url={https://openreview.net/forum?id=rJzIBfZAb},\n"
            "}"
        ),
        ieee='A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, "Towards deep learning models resistant to adversarial attacks," in Proc. Int. Conf. Learn. Represent. (ICLR), 2018.',
        apa="Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. International Conference on Learning Representations (ICLR).",
    ),
)


def pgd(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    step_size: float | None = None,
    steps: int = 20,
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    random_start: bool = True,
    restarts: int = 1,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run multi-step PGD with optional random start and multi-restart selection.

    Args:
        model_or_fn: Differentiable ``nn.Module`` or callable mapping ``x -> outputs``.
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Ground-truth labels/targets (or target class when ``targeted=True``).
        eps: Perturbation budget (or JSR in dB when ``norm="jsr_db"``).
        step_size: Per-iteration step length. If ``None``, auto-scales to
            ``min(1.0, 2.5 / steps) * eps_budget`` so it adapts cleanly across norms and JSR dB.
        steps: Number of projected gradient steps per restart.
        norm: Constraint norm: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        loss: Built-in loss name (``"ce"``, ``"margin"``, ``"mse"``) or custom callable.
        targeted: If ``True``, minimizes loss toward ``y`` instead of maximizing error.
        random_start: Whether to initialize uniformly inside the constraint ball.
        restarts: Number of independent restarts; keeps the highest-objective sample per item.
        clip_bounds: Optional valid value range ``(min, max)`` for real-valued inputs.
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
            extra={"restarts": restarts},
        )

    alpha = (
        eps_budget * min(1.0, 2.5 / steps)
        if step_size is None
        else torch.full_like(eps_budget, float(step_size))
    )

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=x_clean.device)
        generator.manual_seed(seed)

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

            for _step in range(steps):
                x_in = (x_clean + delta).detach().requires_grad_(True)
                objective = compute_attack_objective(
                    model_or_fn, x_in, y, loss_fn, targeted=targeted
                )
                (grad,) = torch.autograd.grad(objective.sum(), x_in, only_inputs=True)

                # Gradient ascent step followed by norm-ball + optional box projection.
                delta = delta + alpha * step_direction(grad.detach(), norm=norm)
                delta = project_perturbation(
                    delta, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds
                )

            with torch.no_grad():
                final_obj = compute_attack_objective(
                    model_or_fn, x_clean + delta, y, loss_fn, targeted=targeted
                )
                improve_mask = final_obj > best_obj
                best_obj = torch.where(improve_mask, final_obj, best_obj)
                mask_view = improve_mask.view((x_clean.shape[0],) + (1,) * (x_clean.ndim - 1))
                best_delta = torch.where(mask_view, delta, best_delta)

    return AttackResult(
        x_adv=x_clean + best_delta,
        perturbation=best_delta,
        eps=eps,
        norm=norm,
        steps_run=steps * restarts,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={"restarts": restarts, "steps_per_restart": steps},
    )
