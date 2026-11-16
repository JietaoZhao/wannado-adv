"""FGSM: Fast Gradient Sign Method (Goodfellow et al., ICLR 2015).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Consensus Baseline — fastest linear vulnerability check)
- Reference     : Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015).
                  "Explaining and Harnessing Adversarial Examples." ICLR 2015.
- URL           : https://arxiv.org/abs/1412.6572
- Plain Intuition:
    Linearizes the model loss around input ``x`` and takes ONE full step of
    size ``eps`` along the steepest gradient direction. Think of it as testing
    whether a single straight-line push breaks your model.
- Why Chosen as a Baseline:
    Every adversarial robustness evaluation starts here (1 forward + 1 backward
    pass). If FGSM already collapses performance, the model lacks basic local
    robustness; if FGSM fails while black-box Square Attack succeeds, your model
    likely exhibits gradient masking / obfuscated gradients.
- Extension Directions (for further research):
    1. Add uniform random start before the gradient step -> R+FGSM (Tramèr 2018).
    2. Split into N projected steps -> BIM (Kurakin 2017) / PGD (Madry 2018).
    3. Average gradients over stochastic wireless channels -> EoT-PGD (Athalye 2018).
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
    id="fgsm",
    name="Fast Gradient Sign Method (FGSM)",
    family="first_order",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="very_low",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id=None,
    diff_from_parent="Origin of fast first-order gradient attacks: single-step linear ascent.",
    summary_plain="Takes one full-budget step along the input gradient direction to rapidly probe linear vulnerability.",
    when_to_use="Always run first as a 1-second sanity baseline and speed benchmark.",
    when_not_to_use="Never claim a model is robust solely because it resists FGSM; curved or masked loss surfaces easily fool single-step attacks.",
    extension_hooks=(
        "Add random initialization -> R+FGSM / Fast-FGSM",
        "Iterate with small steps and projection -> BIM / PGD",
        "Average gradient across random channel draws -> EoT-FGSM",
    ),
    module_path="wannado.adv.attacks.first_order.fgsm_goodfellow2015",
    citation=AttackCitation(
        key="goodfellow2015explaining",
        title="Explaining and Harnessing Adversarial Examples",
        authors="Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy",
        venue="3rd International Conference on Learning Representations (ICLR)",
        year=2015,
        url="https://arxiv.org/abs/1412.6572",
        bibtex=(
            "@inproceedings{goodfellow2015explaining,\n"
            "  title={Explaining and Harnessing Adversarial Examples},\n"
            "  author={Ian J. Goodfellow and Jonathon Shlens and Christian Szegedy},\n"
            "  booktitle={3rd International Conference on Learning Representations, {ICLR} 2015, "
            "San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings},\n"
            "  year={2015},\n"
            "  url={https://arxiv.org/abs/1412.6572},\n"
            "}"
        ),
        ieee='I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and harnessing adversarial examples," in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.',
        apa="Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. International Conference on Learning Representations (ICLR).",
    ),
)


def fgsm(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    clip_bounds: tuple[float, float] | None = None,
) -> AttackResult:
    """Run single-step FGSM (and its L2/JSR normalized gradient counterpart).

    Args:
        model_or_fn: Differentiable ``nn.Module`` or callable mapping ``x -> outputs``.
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Ground-truth labels/targets (or target class when ``targeted=True``).
        eps: Perturbation budget (or JSR in dB when ``norm="jsr_db"``).
        norm: Constraint norm: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        loss: Built-in loss name (``"ce"``, ``"margin"``, ``"mse"``) or custom callable.
        targeted: If ``True``, minimizes loss toward ``y`` instead of maximizing error.
        clip_bounds: Optional valid value range ``(min, max)`` for real-valued inputs.

    Returns:
        ``AttackResult`` containing ``x_adv``, ``perturbation``, and run metadata.
    """
    t0 = time.perf_counter()
    loss_fn = resolve_loss_fn(loss)
    eps_budget = resolve_eps_budget(x, eps=eps, norm=norm)

    if (eps_budget == 0).all():
        x_clean = x.detach().clone()
        return AttackResult(
            x_adv=x_clean,
            perturbation=torch.zeros_like(x_clean),
            eps=eps,
            norm=norm,
            steps_run=0,
            attack_id=META.id,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    with eval_mode(model_or_fn):
        x_in = x.detach().clone().requires_grad_(True)
        objective = compute_attack_objective(model_or_fn, x_in, y, loss_fn, targeted=targeted)
        (grad,) = torch.autograd.grad(objective.sum(), x_in, only_inputs=True)

    # Single full-budget step along the norm-matched steepest ascent direction.
    delta = eps_budget * step_direction(grad.detach(), norm=norm)
    delta = project_perturbation(delta, x.detach(), eps_budget, norm=norm, clip_bounds=clip_bounds)
    x_adv = x.detach() + delta

    return AttackResult(
        x_adv=x_adv,
        perturbation=delta,
        eps=eps,
        norm=norm,
        steps_run=1,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )
