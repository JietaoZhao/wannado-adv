"""NI-FGSM: Nesterov Iterative Fast Gradient Sign Method (Lin et al., ICLR 2020).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Canonical Momentum-Nesterov Transfer Attack)
- Reference     : Lin, J., Song, C., He, K., Wang, L., & Hopcroft, J. E. (2020).
                  "Nesterov Accelerated Gradient and Scale Invariance for
                  Adversarial Attacks." ICLR 2020.
- URL           : https://arxiv.org/abs/1908.06281
- Plain Intuition:
    MI-FGSM checks the gradient at the current point and then moves along its
    accumulated velocity. NI-FGSM takes a "look-ahead" half-step along the current
    momentum trajectory *first* (`x_nes = x_t + step_size * decay * momentum`),
    evaluates the gradient at that future point, and then applies the update.
    Just like Nesterov accelerated gradient in optimization, looking ahead
    detects upcoming curvature earlier and yields smoother, more transferable
    adversarial perturbations across unseen black-box architectures.
- Diff vs. MI-FGSM (Dong et al., 2018):
    1. Evaluates the gradient at the Nesterov look-ahead point ``x_nes`` instead
       of the current iterate ``x_t``.
    2. Reuses L1-normalized momentum accumulation and step direction afterwards.
- Why Chosen as a Baseline:
    Direct drop-in upgrade over MI-FGSM in virtually every black-box transfer
    benchmark.
- Extension Directions / Prompt Hooks:
    1. Average look-ahead gradients over dyadic input scales -> SI-NI-FGSM.
    2. Inject local neighborhood variance correction -> VNI-FGSM.
    3. Swap ``step_direction`` or add multi-surrogate ensemble logits at look-ahead.
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
    step_direction,
)
from wannado.adv.core.losses import LossFn
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="ni_fgsm",
    name="Nesterov Iterative FGSM (NI-FGSM)",
    family="transferability",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="low",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="mi_fgsm",
    diff_from_parent="Applies Nesterov look-ahead (x + alpha * decay * momentum) before evaluating the L1-normalized gradient.",
    summary_plain="Upgrades MI-FGSM with Nesterov look-ahead momentum so updates anticipate curvature and transfer better across unknown models.",
    when_to_use="Strong default baseline for surrogate-to-victim transfer attacks across classification, regression, or wireless IQ models.",
    when_not_to_use="Against randomized or non-differentiable defenses without EoT/BPDA, or for minimum-norm boundary distance measurement.",
    extension_hooks=(
        "Average look-ahead gradients over scaled input copies -> SI-NI-FGSM",
        "Add input diversity or translation-invariant kernel -> DI-NI-FGSM / TI-NI-FGSM",
        "Substitute ensemble surrogate forward pass inside look-ahead step",
    ),
    module_path="wannado.adv.attacks.transferability.nifgsm_lin2020",
    citation=AttackCitation(
        key="lin2020nesterov",
        title="Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks",
        authors="Jiadong Lin, Chuanbiao Song, Kun He, Liwei Wang, and John E. Hopcroft",
        venue="8th International Conference on Learning Representations (ICLR)",
        year=2020,
        url="https://arxiv.org/abs/1908.06281",
        bibtex=(
            "@inproceedings{lin2020nesterov,\n"
            "  title={Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks},\n"
            "  author={Lin, Jiadong and Song, Chuanbiao and He, Kun and Wang, Liwei and Hopcroft, John E.},\n"
            "  booktitle={International Conference on Learning Representations (ICLR)},\n"
            "  year={2020},\n"
            "  url={https://arxiv.org/abs/1908.06281},\n"
            "}"
        ),
        ieee='J. Lin, C. Song, K. He, L. Wang, and J. E. Hopcroft, "Nesterov accelerated gradient and scale invariance for adversarial attacks," in Proc. Int. Conf. Learn. Represent. (ICLR), 2020.',
        apa="Lin, J., Song, C., He, K., Wang, L., & Hopcroft, J. E. (2020). Nesterov accelerated gradient and scale invariance for adversarial attacks. International Conference on Learning Representations (ICLR).",
    ),
)


def ni_fgsm(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    step_size: float | None = None,
    steps: int = 10,
    decay: float = 1.0,
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    random_start: bool = False,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run Nesterov Iterative FGSM (NI-FGSM) for transferable perturbations."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if decay < 0.0:
        raise ValueError(f"decay must be >= 0, got {decay}.")

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
            extra={"decay": decay},
        )

    alpha = (
        eps_budget / float(steps)
        if step_size is None
        else torch.full_like(eps_budget, float(step_size))
    )

    if random_start:
        generator = make_generator(seed, x_clean.device)
        delta = random_init_perturbation(
            x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds, generator=generator
        )
    else:
        delta = torch.zeros_like(x_clean)

    momentum = torch.zeros_like(x_clean)
    view_shape = (x_clean.shape[0],) + (1,) * (x_clean.ndim - 1)

    with eval_mode(model_or_fn):
        for _ in range(steps):
            # [AI-Prompt Hook]: Look ahead along existing momentum before evaluating gradient (Lin et al., 2020, Eq. 5).
            # Extending to VNI-FGSM only requires adding a variance-tuning vector v_t to `grad` below.
            lookahead = delta + alpha * decay * momentum
            x_nes = (x_clean + lookahead).detach().requires_grad_(True)
            objective = compute_attack_objective(
                model_or_fn, x_nes, y, loss_fn, targeted=targeted
            )
            (grad,) = torch.autograd.grad(objective.sum(), x_nes, only_inputs=True)
            grad = grad.detach()

            grad_l1 = grad.abs().reshape(x_clean.shape[0], -1).sum(dim=1).view(view_shape)
            momentum = decay * momentum + grad / grad_l1.clamp(min=1e-12)

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
        extra={"decay": decay},
    )
