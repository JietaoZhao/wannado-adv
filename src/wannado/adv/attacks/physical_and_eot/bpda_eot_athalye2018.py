"""BPDA+EoT: Backward Pass Differentiable Approximation + EoT (Athalye et al., ICML 2018).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Canonical Adaptive Attack Against Obfuscated Gradients)
- Reference     : Athalye, A., Carlini, N., & Wagner, D. (2018). "Obfuscated
                  Gradients Give a False Sense of Security: Circumventing
                  Defenses to Adversarial Examples." ICML 2018 (Best Paper Award,
                  PMLR 80:274-283).
- URL           : https://arxiv.org/abs/1802.00420
- Plain Intuition:
    Carlini, Athalye & Wagner proved that 7 out of 9 ICLR 2018 defenses did NOT
    make models robust—they merely shattered gradients via non-differentiable
    preprocessing (quantization, bit-depth reduction, JPEG, denoising) or hid
    them behind randomness. BPDA+EoT breaks BOTH simultaneously:
      - **Forward pass**: runs the exact non-differentiable ``defense_fn(x)``.
      - **Backward pass**: substitutes a smooth ``surrogate_fn`` (default: the
        identity Straight-Through Estimator, g(x) ~= x) for VJP backpropagation.
      - **EoT**: Monte Carlo averages gradients over ``eot_samples`` random
        draws if the defense or model is also stochastic.
- Diff vs. EoT-PGD (Athalye et al., ICML 2018):
    Wraps any non-differentiable ``defense_fn`` in a custom ``torch.autograd.Function``
    so gradients bypass shattered/quantized/no-grad defense layers via straight-through
    or custom surrogate backpropagation, combined with EoT averaging.
- Why Chosen as a Baseline:
    Mandatory adaptive white-box baseline whenever a model includes quantization,
    JPEG/compression, randomized smoothing, or RF ADC bit-truncation defenses.
- Extension Directions / Prompt Hooks:
    1. Supply an explicit smooth ``surrogate_fn`` (e.g., soft-sign or soft-tanh)
       when identity straight-through is imprecise.
    2. Pair directly with ``audit_robustness()`` after catching obfuscated grads.
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
    update_best_by_objective,
)
from wannado.adv.core.losses import LossFn
from wannado.adv.core.types import AttackCitation

DefenseFn = Callable[[torch.Tensor], torch.Tensor]

META = AttackMeta(
    id="bpda_eot",
    name="BPDA + EoT PGD",
    family="physical_and_eot",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="medium",
    supported_norms=("linf", "l2", "l1", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="eot_pgd",
    diff_from_parent="Wraps non-differentiable preprocessing defenses in a custom autograd Function (straight-through or smooth surrogate backward pass) combined with EoT sampling.",
    summary_plain="Circumvents shattered, quantized, or non-differentiable preprocessing defenses via Backward Pass Differentiable Approximation + EoT.",
    when_to_use="When a differentiable base classifier sits behind a non-differentiable or stochastic preprocessor (quantizer, bit-depth reduction, ADC truncation, JPEG).",
    when_not_to_use="When the entire victim model weights and architecture are completely hidden black-box (use square_attack or nes).",
    extension_hooks=(
        "Supply explicit differentiable `surrogate_fn` instead of straight-through identity",
        "Simulate RF ADC quantization + Rayleigh fading inside `defense_fn`",
    ),
    module_path="wannado.adv.attacks.physical_and_eot.bpda_eot_athalye2018",
    citation=AttackCitation(
        key="athalye2018obfuscated",
        title="Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples",
        authors="Anish Athalye, Nicholas Carlini, and David Wagner",
        venue="35th International Conference on Machine Learning (ICML)",
        year=2018,
        url="https://arxiv.org/abs/1802.00420",
        bibtex=(
            "@inproceedings{athalye2018obfuscated,\n"
            "  title={Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples},\n"
            "  author={Athalye, Anish and Carlini, Nicholas and Wagner, David},\n"
            "  booktitle={Proceedings of the 35th International Conference on Machine Learning},\n"
            "  pages={274--283},\n"
            "  year={2018},\n"
            "  volume={80},\n"
            "  series={Proceedings of Machine Learning Research},\n"
            "  publisher={PMLR},\n"
            "  url={https://arxiv.org/abs/1802.00420},\n"
            "}"
        ),
        ieee='A. Athalye, N. Carlini, and D. Wagner, "Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples," in Proc. 35th Int. Conf. Mach. Learn. (ICML), PMLR 80, 2018, pp. 274-283.',
        apa="Athalye, A., Carlini, N., & Wagner, D. (2018). Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80, 274-283.",
    ),
)


class _BPDAStraightThrough(torch.autograd.Function):
    """Evaluate exact non-differentiable ``defense_fn`` in forward, identity Jacobian in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, defense_fn: DefenseFn) -> torch.Tensor:  # type: ignore[override]
        with torch.no_grad():
            return defense_fn(x).detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return grad_output, None


def _bpda_forward(
    x: torch.Tensor,
    defense_fn: DefenseFn | None,
    surrogate_fn: DefenseFn | None,
) -> torch.Tensor:
    """Route forward through ``defense_fn`` while backpropagating via ``surrogate_fn`` (or identity)."""
    if defense_fn is None:
        return x
    if surrogate_fn is None:
        return _BPDAStraightThrough.apply(x, defense_fn)
    # Surrogate trick: y = surrogate(x) + stop_grad(defense(x) - surrogate(x)) matches exact
    # forward `defense(x)` and exact backward Jacobian of `surrogate(x)`!
    sur_out = surrogate_fn(x)
    with torch.no_grad():
        exact_out = defense_fn(x)
    return sur_out + (exact_out - sur_out.detach())


def bpda_eot(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    step_size: float | None = None,
    steps: int = 25,
    eot_samples: int = 4,
    defense_fn: DefenseFn | None = None,
    surrogate_fn: DefenseFn | None = None,
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    random_start: bool = True,
    restarts: int = 1,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run BPDA + EoT PGD to break non-differentiable/shattered + stochastic defenses."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if eot_samples < 1:
        raise ValueError(f"eot_samples must be >= 1, got {eot_samples}.")
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
            extra={"eot_samples": eot_samples, "restarts": restarts},
        )

    alpha = (
        eps_budget * min(1.0, 2.5 / float(steps))
        if step_size is None
        else torch.full_like(eps_budget, float(step_size))
    )
    generator = make_generator(seed, x_clean.device)

    best_delta = torch.zeros_like(x_clean)
    best_obj = torch.full(
        (x_clean.shape[0],), float("-inf"), device=x_clean.device, dtype=eps_budget.dtype
    )

    with eval_mode(model_or_fn), eval_mode(defense_fn) if defense_fn is not None else eval_mode(lambda z: z):
        for _ in range(restarts):
            if random_start:
                delta = random_init_perturbation(
                    x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds, generator=generator
                )
            else:
                delta = torch.zeros_like(x_clean)

            for _ in range(steps):
                grad_acc = torch.zeros_like(x_clean)
                x_in = (x_clean + delta).detach().requires_grad_(True)

                # [AI-Prompt Hook]: Combines BPDA surrogate/straight-through backward pass with
                # K Monte Carlo EoT draws against non-differentiable + randomized defenses.
                for _ in range(eot_samples):
                    x_def = _bpda_forward(x_in, defense_fn, surrogate_fn)
                    obj_k = compute_attack_objective(
                        model_or_fn, x_def, y, loss_fn, targeted=targeted
                    )
                    (grad_k,) = torch.autograd.grad(obj_k.sum(), x_in, only_inputs=True)
                    grad_acc = grad_acc + grad_k.detach() / float(eot_samples)

                delta = delta + alpha * step_direction(grad_acc, norm=norm)
                delta = project_perturbation(
                    delta, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds
                )

            with torch.no_grad():
                expected_obj = torch.zeros(
                    x_clean.shape[0], device=x_clean.device, dtype=eps_budget.dtype
                )
                x_eval_in = x_clean + delta
                for _ in range(eot_samples):
                    x_def_eval = defense_fn(x_eval_in) if defense_fn is not None else x_eval_in
                    expected_obj = expected_obj + compute_attack_objective(
                        model_or_fn, x_def_eval, y, loss_fn, targeted=targeted
                    ) / float(eot_samples)
                best_delta, best_obj = update_best_by_objective(best_delta, best_obj, delta, expected_obj)

    return AttackResult(
        x_adv=x_clean + best_delta,
        perturbation=best_delta,
        eps=eps,
        norm=norm,
        steps_run=steps * restarts,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "eot_samples": eot_samples,
            "restarts": restarts,
            "used_custom_surrogate": surrogate_fn is not None,
            "expected_objective_mean": float(best_obj.mean().item()),
        },
    )
