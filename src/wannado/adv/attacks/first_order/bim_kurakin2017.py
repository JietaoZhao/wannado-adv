"""BIM / I-FGSM: Basic Iterative Method (Kurakin et al., ICLR 2017 Workshop).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : A (Iterative Bridge Between FGSM and PGD)
- Reference     : Kurakin, A., Goodfellow, I. J., & Bengio, S. (2017).
                  "Adversarial examples in the physical world." ICLR 2017 Workshop.
- URL           : https://arxiv.org/abs/1607.02533
- Plain Intuition:
    Instead of taking one giant step like FGSM, BIM starts right at ``x``
    (no random start) and takes ``steps`` smaller gradient steps, projecting
    back inside the ``eps`` constraint ball after each step.
- Diff vs. FGSM (2015) & PGD (2018):
    1. Unlike FGSM: splits a single jump into multiple small projected steps,
       allowing it to curve around non-linear decision boundaries.
    2. Unlike PGD: starts deterministically at ``x`` (``random_start=False``)
       without multi-restart exploration.
- Why Chosen as a Baseline:
    Isolates the benefit of iterative optimization over single-step FGSM
    without stochastic variance from random initialization.
- Extension Directions (for further research):
    1. Add uniform random initialization -> PGD (Madry et al., ICLR 2018).
    2. Accumulate gradient momentum -> MI-FGSM (Dong et al., CVPR 2018).
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
    resolve_eps_budget,
    resolve_loss_fn,
    step_direction,
)
from wannado.adv.core.losses import LossFn
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="bim",
    name="Basic Iterative Method (BIM / I-FGSM)",
    family="first_order",
    tier="A",
    threat_model="white_box",
    requires_training=False,
    compute_cost="low",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="fgsm",
    diff_from_parent="Splits FGSM's single full step into multiple smaller iterative ascent steps with per-step projection (deterministic start at x).",
    summary_plain="Deterministic multi-step iterative FGSM without random initialization.",
    when_to_use="When you want a reproducible deterministic iterative baseline or to compare directly against FGSM without random-seed variance.",
    when_not_to_use="When testing models with sharp local gradient masking near x (prefer PGD with random_start=True or APGD).",
    extension_hooks=(
        "Add random start inside the ball -> PGD (Madry 2018)",
        "Add gradient momentum for black-box transferability -> MI-FGSM (Dong 2018)",
    ),
    module_path="wannado.adv.attacks.first_order.bim_kurakin2017",
    citation=AttackCitation(
        key="kurakin2017adversarial",
        title="Adversarial examples in the physical world",
        authors="Alexey Kurakin, Ian J. Goodfellow, and Samy Bengio",
        venue="International Conference on Learning Representations (ICLR), Workshop Track",
        year=2017,
        url="https://arxiv.org/abs/1607.02533",
        bibtex=(
            "@inproceedings{kurakin2017adversarial,\n"
            "  title={Adversarial examples in the physical world},\n"
            "  author={Alexey Kurakin and Ian J. Goodfellow and Samy Bengio},\n"
            "  booktitle={International Conference on Learning Representations (ICLR), Workshop Track},\n"
            "  year={2017},\n"
            "  url={https://arxiv.org/abs/1607.02533},\n"
            "}"
        ),
        ieee='A. Kurakin, I. J. Goodfellow, and S. Bengio, "Adversarial examples in the physical world," in Proc. Int. Conf. Learn. Represent. (ICLR) Workshop, 2017.',
        apa="Kurakin, A., Goodfellow, I. J., & Bengio, S. (2017). Adversarial examples in the physical world. International Conference on Learning Representations (ICLR) Workshop.",
    ),
)


def bim(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    step_size: float | None = None,
    steps: int = 10,
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    clip_bounds: tuple[float, float] | None = None,
) -> AttackResult:
    """Run deterministic iterative FGSM (BIM) starting cleanly from ``x``.

    Args:
        model_or_fn: Differentiable ``nn.Module`` or callable mapping ``x -> outputs``.
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Ground-truth labels/targets (or target class when ``targeted=True``).
        eps: Perturbation budget (or JSR in dB when ``norm="jsr_db"``).
        step_size: Per-iteration step size. If ``None``, uses ``min(1.0, 1.25 / steps) * eps_budget``.
        steps: Number of projected gradient steps.
        norm: Constraint norm: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        loss: Built-in loss name (``"ce"``, ``"margin"``, ``"dlr"``, ``"mse"``) or callable.
        targeted: If ``True``, minimizes loss toward ``y``.
        clip_bounds: Optional valid range ``(min, max)`` for real-valued inputs.

    Returns:
        ``AttackResult`` containing ``x_adv``, ``perturbation``, and run metadata.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")

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
        )

    alpha = (
        eps_budget * min(1.0, 1.25 / steps)
        if step_size is None
        else torch.full_like(eps_budget, float(step_size))
    )
    delta = torch.zeros_like(x_clean)

    with eval_mode(model_or_fn):
        for _ in range(steps):
            x_in = (x_clean + delta).detach().requires_grad_(True)
            objective = compute_attack_objective(
                model_or_fn, x_in, y, loss_fn, targeted=targeted
            )
            (grad,) = torch.autograd.grad(objective.sum(), x_in, only_inputs=True)

            delta = delta + alpha * step_direction(grad.detach(), norm=norm)
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
    )
