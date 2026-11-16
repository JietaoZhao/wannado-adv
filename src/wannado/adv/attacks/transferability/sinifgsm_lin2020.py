"""SI-NI-FGSM: Scale-Invariant Nesterov Iterative FGSM (Lin et al., ICLR 2020).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Transfer Benchmark Combining Nesterov + Scale Invariance)
- Reference     : Lin, J., Song, C., He, K., Wang, L., & Hopcroft, J. E. (2020).
                  "Nesterov Accelerated Gradient and Scale Invariance for
                  Adversarial Attacks." ICLR 2020.
- URL           : https://arxiv.org/abs/1908.06281
- Plain Intuition:
    Neural networks often maintain consistent predictions under uniform amplitude
    scaling ``x / 2^i``. Evaluating the gradient not only at ``x_nes``, but averaging
    gradients across ``m`` dyadically scaled copies ``x_nes / 2^0, x_nes / 2^1, ...``
    regularizes the perturbation so it does not overfit to a single activation
    regime of the surrogate model. For complex IQ wireless signals, linear amplitude
    scaling keeps constellation phase intact while probing varying SNR gains.
- Diff vs. NI-FGSM (Lin et al., 2020):
    Replaces single-scale gradient at ``x_nes`` with the average gradient over
    ``m = scale_copies`` dyadic amplitude scales ``x_nes / (2 ** i)``.
- Why Chosen as a Baseline:
    Canonical companion to NI-FGSM and a foundation for Admix / multi-spectrum
    transferability research.
- Extension Directions / Prompt Hooks:
    1. Replace dyadic power-of-2 scales with domain-specific transforms (e.g.
       brightness/gain curves, complex carrier phase offsets, or Admix blending).
    2. Combine with DI-FGSM + TI-FGSM -> SI-NI-TI-DIM ensemble benchmark.
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
    id="sini_fgsm",
    name="Scale-Invariant Nesterov Iterative FGSM (SI-NI-FGSM)",
    family="transferability",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="medium",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="ni_fgsm",
    diff_from_parent="Averages Nesterov look-ahead gradients across m dyadic input amplitude scales x_nes / 2^i (Lin et al., ICLR 2020).",
    summary_plain="Combines Nesterov look-ahead momentum with multi-scale gradient averaging (x / 2^i) to prevent surrogate overfitting.",
    when_to_use="When transferring attacks across architectures or gain/amplitude regimes (images or linear wireless receivers).",
    when_not_to_use="Against scale-sensitive preprocessing where amplitude halving collapses clean accuracy (set scale_copies smaller or use ni_fgsm/vmi_fgsm).",
    extension_hooks=(
        "Swap dyadic scaling `x / 2^i` with Admix image mixing or wireless SNR gain sweeps",
        "Compose with DI spatial diversity or TI gradient smoothing kernel",
    ),
    module_path="wannado.adv.attacks.transferability.sinifgsm_lin2020",
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


def sini_fgsm(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    step_size: float | None = None,
    steps: int = 10,
    decay: float = 1.0,
    scale_copies: int = 5,
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    random_start: bool = False,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run Scale-Invariant Nesterov Iterative FGSM (SI-NI-FGSM)."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if decay < 0.0:
        raise ValueError(f"decay must be >= 0, got {decay}.")
    if scale_copies < 1:
        raise ValueError(f"scale_copies must be >= 1, got {scale_copies}.")

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
            extra={"decay": decay, "scale_copies": scale_copies},
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
            lookahead = delta + alpha * decay * momentum
            x_nes = (x_clean + lookahead).detach().requires_grad_(True)
            grad_sum = torch.zeros_like(x_clean)

            # [AI-Prompt Hook]: Scale-Invariant Method (Lin et al., 2020, Eq. 6-7).
            # To upgrade toward Admix, blend x_nes with random other-class samples before scaling.
            for scale_idx in range(scale_copies):
                x_scaled = x_nes / float(2**scale_idx)
                obj_s = compute_attack_objective(
                    model_or_fn, x_scaled, y, loss_fn, targeted=targeted
                )
                (grad_s,) = torch.autograd.grad(obj_s.sum(), x_nes, only_inputs=True)
                grad_sum = grad_sum + grad_s.detach()

            grad_avg = grad_sum / float(scale_copies)
            grad_l1 = grad_avg.abs().reshape(x_clean.shape[0], -1).sum(dim=1).view(view_shape)
            momentum = decay * momentum + grad_avg / grad_l1.clamp(min=1e-12)

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
        extra={"decay": decay, "scale_copies": scale_copies},
    )
