"""R+FGSM: Random-Start Single-Step FGSM (Tramèr et al., ICLR 2018).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : A (Fast Gradient-Masking Escape Baseline)
- Reference     : Tramèr, F., Kurakin, A., Papernot, N., Goodfellow, I., Boneh, D.,
                  & McDaniel, P. (2018). "Ensemble Adversarial Training: Attacks
                  and Defenses." ICLR 2018.
- URL           : https://arxiv.org/abs/1705.07204
- Plain Intuition:
    Adversarially trained models often flatten or distort gradients right next
    to the clean point ``x``. R+FGSM first jumps a small random distance
    ``alpha`` away from ``x`` to escape that flat zone, then takes a single
    gradient step of remaining budget ``eps - alpha``.
- Diff vs. FGSM (2015):
    Adds a controlled random step ``alpha < eps`` before evaluating the gradient.
- Why Chosen as a Baseline:
    Reveals whether a model defended against vanilla FGSM merely suffers from
    local gradient masking near ``x`` while still failing at almost single-step cost.
- Extension Directions (for further research):
    1. Increase random step size & multi-step iteration -> PGD (Madry et al., 2018).
    2. Repeat random start with larger step size -> Fast-FGSM (Wong et al., 2020).
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
    id="rfgsm",
    name="Random-Start FGSM (R+FGSM)",
    family="first_order",
    tier="A",
    threat_model="white_box",
    requires_training=False,
    compute_cost="very_low",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="fgsm",
    diff_from_parent="Prepends a random step of size alpha (default 0.25 * eps) before computing the single-step FGSM gradient update of size (eps - alpha).",
    summary_plain="Takes a small random hop to escape flat gradients near x, then executes one fast gradient step.",
    when_to_use="Fast check against models showing suspiciously high FGSM robustness (e.g. adversarially trained or locally smoothed models).",
    when_not_to_use="Not a substitute for full multi-step PGD or APGD when certifying strong white-box robustness.",
    extension_hooks=(
        "Continue iterating from the perturbed point -> PGD (Madry 2018)",
        "Use multiple random draws and pick worst-case -> Multi-restart R+FGSM",
    ),
    module_path="wannado.adv.attacks.first_order.rfgsm_tramer2018",
    citation=AttackCitation(
        key="tramer2018ensemble",
        title="Ensemble Adversarial Training: Attacks and Defenses",
        authors="Florian Tram{\\`e}r, Alexey Kurakin, Nicolas Papernot, Ian Goodfellow, Dan Boneh, and Patrick McDaniel",
        venue="6th International Conference on Learning Representations (ICLR)",
        year=2018,
        url="https://arxiv.org/abs/1705.07204",
        bibtex=(
            "@inproceedings{tramer2018ensemble,\n"
            "  title={Ensemble Adversarial Training: Attacks and Defenses},\n"
            "  author={Florian Tram{\\`e}r and Alexey Kurakin and Nicolas Papernot and "
            "Ian Goodfellow and Dan Boneh and Patrick McDaniel},\n"
            "  booktitle={International Conference on Learning Representations (ICLR)},\n"
            "  year={2018},\n"
            "  url={https://arxiv.org/abs/1705.07204},\n"
            "}"
        ),
        ieee='F. Tramèr, A. Kurakin, N. Papernot, I. Goodfellow, D. Boneh, and P. McDaniel, "Ensemble adversarial training: Attacks and defenses," in Proc. Int. Conf. Learn. Represent. (ICLR), 2018.',
        apa="Tramèr, F., Kurakin, A., Papernot, N., Goodfellow, I., Boneh, D., & McDaniel, P. (2018). Ensemble adversarial training: Attacks and defenses. International Conference on Learning Representations (ICLR).",
    ),
)


def rfgsm(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    alpha_ratio: float = 0.25,
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run R+FGSM: random step of ``alpha_ratio * eps`` followed by ``(1 - alpha_ratio) * eps`` FGSM.

    Args:
        model_or_fn: Differentiable ``nn.Module`` or callable mapping ``x -> outputs``.
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Ground-truth labels/targets (or target class when ``targeted=True``).
        eps: Total perturbation budget (or JSR in dB when ``norm="jsr_db"``).
        alpha_ratio: Fraction of ``eps_budget`` spent on the initial random step (``0 < alpha_ratio < 1``).
        norm: Constraint norm: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        loss: Built-in loss name (``"ce"``, ``"margin"``, ``"dlr"``, ``"mse"``) or callable.
        targeted: If ``True``, minimizes loss toward ``y``.
        clip_bounds: Optional valid range ``(min, max)`` for real-valued inputs.
        seed: Optional random seed for reproducible initial step.

    Returns:
        ``AttackResult`` containing ``x_adv``, ``perturbation``, and run metadata.
    """
    if not (0.0 <= alpha_ratio <= 1.0):
        raise ValueError(f"alpha_ratio must lie in [0, 1], got {alpha_ratio}.")

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

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=x_clean.device)
        generator.manual_seed(seed)

    real_dtype = eps_budget.dtype
    if x_clean.is_complex():
        re = torch.randn(x_clean.shape, device=x_clean.device, dtype=real_dtype, generator=generator)
        im = torch.randn(x_clean.shape, device=x_clean.device, dtype=real_dtype, generator=generator)
        noise = torch.complex(re, im).to(x_clean.dtype)
    else:
        noise = torch.randn(x_clean.shape, device=x_clean.device, dtype=real_dtype, generator=generator).to(
            x_clean.dtype
        )

    # Step 1: Random step of size alpha = alpha_ratio * eps_budget to escape local gradient masking.
    alpha_budget = eps_budget * alpha_ratio
    delta_rand = alpha_budget * step_direction(noise, norm=norm)
    if norm == "linf" and x_clean.is_complex():
        # Random phase on complex Linf step
        phase = (
            torch.rand(x_clean.shape, device=x_clean.device, dtype=real_dtype, generator=generator)
            * (2.0 * math.pi)
        )
        delta_rand = torch.polar(torch.ones_like(phase), phase).to(x_clean.dtype) * alpha_budget
    delta_rand = project_perturbation(
        delta_rand, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds
    )

    with eval_mode(model_or_fn):
        x_mid = (x_clean + delta_rand).detach().requires_grad_(True)
        objective = compute_attack_objective(model_or_fn, x_mid, y, loss_fn, targeted=targeted)
        (grad,) = torch.autograd.grad(objective.sum(), x_mid, only_inputs=True)

    # Step 2: Single gradient step using the remaining budget (1 - alpha_ratio) * eps_budget.
    step_budget = eps_budget * (1.0 - alpha_ratio)
    delta = delta_rand + step_budget * step_direction(grad.detach(), norm=norm)
    delta = project_perturbation(delta, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds)

    return AttackResult(
        x_adv=x_clean + delta,
        perturbation=delta,
        eps=eps,
        norm=norm,
        steps_run=1,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={"alpha_ratio": alpha_ratio},
    )
