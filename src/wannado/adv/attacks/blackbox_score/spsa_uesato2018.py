"""SPSA: Simultaneous Perturbation Stochastic Approximation Attack (Uesato et al., ICML 2018).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : A (Classic Gradient-Free Finite-Difference Score Baseline)
- Reference     : Uesato, J., O'Donoghue, B., van den Oord, A., & Kohli, P. (2018).
                  "Adversarial Risk and the Dangers of Evaluating Against Weak
                  Attacks." ICML 2018, PMLR 80:5025-5034.
- URL           : https://proceedings.mlr.press/v80/uesato18a.html
- Plain Intuition:
    When a model blocks gradients (discontinuous layers, black-box API, or
    non-differentiable simulator), SPSA still estimates the gradient direction
    by evaluating pairs of random +/- Rademacher probes around the current
    input and driving an Adam optimizer on those stochastic finite differences.
- Diff vs. White-Box Adam-PGD:
    1. Requires zero backward passes or autograd graph access (score queries only).
    2. Estimates gradients via Rademacher two-sided finite differences.
    3. Uses margin loss by default to avoid Softmax finite-difference saturation.
- Why Chosen as a Baseline:
    The canonical gradient-free sanity check for detecting gradient obfuscation
    and evaluating black-box score-access threat models.
- Extension Directions (for further research):
    1. Replace finite-difference Adam with query-efficient random search -> Square Attack.
    2. Replace logit queries with hard-label boundary queries -> HopSkipJump.
    3. Use structured/antithetic or variance-reduced basis probes (NES / Bandits).
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
    clamp_to_bounds,
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
    id="spsa",
    name="SPSA Black-Box Attack",
    family="blackbox_score",
    tier="A",
    threat_model="black_box_score",
    requires_training=False,
    compute_cost="high",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="pgd",
    diff_from_parent="Replaces exact autograd backprop with Rademacher simultaneous perturbation finite differences fed into projected Adam ascent.",
    summary_plain="Gradient-free score-based attack that estimates ascent directions via random +/- probe pairs and updates with Adam.",
    when_to_use="When models contain non-differentiable blocks (quantization, custom simulators) or to expose gradient masking when scores/logits are observable.",
    when_not_to_use="When only hard decisions are returned (use HopSkipJump), when strict query budgets are tight (use Square Attack), or when full gradients exist (use APGD).",
    extension_hooks=(
        "Swap Rademacher probes for spatial/burst random search -> Square Attack (ECCV 2020)",
        "Switch from score queries to label-only queries -> HopSkipJump (IEEE S&P 2020)",
    ),
    module_path="wannado.adv.attacks.blackbox_score.spsa_uesato2018",
    citation=AttackCitation(
        key="uesato2018adversarial",
        title="Adversarial Risk and the Dangers of Evaluating Against Weak Attacks",
        authors="Jonathan Uesato, Brendan O'Donoghue, Aaron van den Oord, and Pushmeet Kohli",
        venue="35th International Conference on Machine Learning (ICML)",
        year=2018,
        url="https://proceedings.mlr.press/v80/uesato18a.html",
        bibtex=(
            "@inproceedings{uesato2018adversarial,\n"
            "  title={Adversarial Risk and the Dangers of Evaluating Against Weak Attacks},\n"
            "  author={Uesato, Jonathan and O'Donoghue, Brendan and van den Oord, A{\\\"a}ron and Kohli, Pushmeet},\n"
            "  booktitle={Proceedings of the 35th International Conference on Machine Learning},\n"
            "  pages={5025--5034},\n"
            "  year={2018},\n"
            "  volume={80},\n"
            "  series={Proceedings of Machine Learning Research},\n"
            "  publisher={PMLR},\n"
            "  url={https://proceedings.mlr.press/v80/uesato18a.html},\n"
            "}"
        ),
        ieee='J. Uesato, B. O\'Donoghue, A. van den Oord, and P. Kohli, "Adversarial risk and the dangers of evaluating against weak attacks," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 80, 2018, pp. 5025-5034.',
        apa="Uesato, J., O'Donoghue, B., van den Oord, A., & Kohli, P. (2018). Adversarial risk and the dangers of evaluating against weak attacks. Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80, 5025-5034.",
    ),
)


def _sample_rademacher(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    is_complex: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample symmetric Rademacher probes (independent real/imag {-1, +1} for complex tensors)."""
    real_dtype = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
    if is_complex:
        re = (
            torch.randint(0, 2, shape, device=device, generator=generator, dtype=torch.int64).to(real_dtype)
            * 2.0
            - 1.0
        )
        im = (
            torch.randint(0, 2, shape, device=device, generator=generator, dtype=torch.int64).to(real_dtype)
            * 2.0
            - 1.0
        )
        return torch.complex(re, im).to(dtype)
    return (
        torch.randint(0, 2, shape, device=device, generator=generator, dtype=torch.int64).to(dtype) * 2.0
        - 1.0
    )


def _estimate_spsa_gradient(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_adv: torch.Tensor,
    y: torch.Tensor,
    loss_fn: LossFn,
    targeted: bool,
    spsa_delta: float,
    nb_sample: int,
    max_batch_size: int,
    clip_bounds: tuple[float, float] | None,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, int]:
    """Estimate ascent gradient at ``x_adv`` via chunked two-sided SPSA probes."""
    batch_size = x_adv.shape[0]
    probe_shape = (nb_sample, batch_size, *x_adv.shape[1:])
    probes = _sample_rademacher(
        probe_shape,
        dtype=x_adv.dtype,
        device=x_adv.device,
        is_complex=x_adv.is_complex(),
        generator=generator,
    )

    x_plus = x_adv.unsqueeze(0) + spsa_delta * probes
    x_minus = x_adv.unsqueeze(0) - spsa_delta * probes
    if clip_bounds is not None:
        x_plus = clamp_to_bounds(x_plus, clip_bounds)
        x_minus = clamp_to_bounds(x_minus, clip_bounds)

    flat_plus = x_plus.reshape(nb_sample * batch_size, *x_adv.shape[1:])
    flat_minus = x_minus.reshape(nb_sample * batch_size, *x_adv.shape[1:])
    y_rep = y.repeat(nb_sample)

    obj_plus_chunks: list[torch.Tensor] = []
    obj_minus_chunks: list[torch.Tensor] = []
    total_items = nb_sample * batch_size
    chunk_size = max(batch_size, max_batch_size)
    queries = 0

    for start in range(0, total_items, chunk_size):
        end = min(start + chunk_size, total_items)
        y_slice = y_rep[start:end]
        obj_plus_chunks.append(
            compute_attack_objective(model_or_fn, flat_plus[start:end], y_slice, loss_fn, targeted=targeted)
        )
        obj_minus_chunks.append(
            compute_attack_objective(model_or_fn, flat_minus[start:end], y_slice, loss_fn, targeted=targeted)
        )
        queries += 2 * (end - start)

    obj_plus = torch.cat(obj_plus_chunks, dim=0).view(nb_sample, batch_size)
    obj_minus = torch.cat(obj_minus_chunks, dim=0).view(nb_sample, batch_size)
    diff = (obj_plus - obj_minus) / (2.0 * spsa_delta)
    diff_view = diff.view((nb_sample, batch_size) + (1,) * (x_adv.ndim - 1))

    # For both real and Wirtinger complex derivatives, E[diff * v] equals nabla L(x).
    grad_est = (diff_view * probes).mean(dim=0)
    return grad_est, queries


def spsa(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.05,
    lr: float | None = None,
    steps: int = 30,
    spsa_delta: float = 0.01,
    nb_sample: int = 64,
    max_batch_size: int = 256,
    betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    norm: str = "linf",
    loss: str | LossFn | None = "margin",
    targeted: bool = False,
    random_start: bool = True,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run SPSA black-box finite-difference attack with projected Adam updates.

    Args:
        model_or_fn: Callable or ``nn.Module`` returning scores/logits (no gradient needed).
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Ground-truth labels/targets (or target class when ``targeted=True``).
        eps: Perturbation budget (or JSR in dB when ``norm="jsr_db"``).
        lr: Per-step update scale. If ``None``, auto-scales to ``min(1.0, 2.5 / steps) * eps_budget``.
        steps: Number of SPSA optimization steps.
        spsa_delta: Finite-difference probe magnitude around the current iterate.
        nb_sample: Number of two-sided Rademacher probe pairs per step (``2 * nb_sample`` queries/step).
        max_batch_size: Maximum batch items evaluated per forward chunk during SPSA probing.
        betas: Adam momentum decay rates ``(beta1, beta2)``.
        adam_eps: Numerical stability constant in Adam denominator.
        norm: Constraint norm: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        loss: Built-in loss name (defaults to ``"margin"`` per Uesato et al. 2018) or callable.
        targeted: If ``True``, minimizes loss toward ``y``.
        random_start: Whether to initialize randomly inside the norm ball.
        clip_bounds: Optional valid value range ``(min, max)`` for real-valued inputs.
        seed: Optional random seed for reproducible probes and initialization.

    Returns:
        ``AttackResult`` containing best perturbation and black-box query count.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if nb_sample < 1:
        raise ValueError(f"nb_sample must be >= 1, got {nb_sample}.")
    if spsa_delta <= 0.0:
        raise ValueError(f"spsa_delta must be > 0, got {spsa_delta}.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    batch_size = x_clean.shape[0]
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
            extra={"queries": 0, "nb_sample": nb_sample},
        )

    alpha = (
        eps_budget * min(1.0, 2.5 / steps)
        if lr is None
        else torch.full_like(eps_budget, float(lr))
    )
    beta1, beta2 = betas
    generator = make_generator(seed, x_clean.device)

    with eval_mode(model_or_fn), torch.no_grad():
        if random_start:
            delta = random_init_perturbation(
                x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds, generator=generator
            )
        else:
            delta = torch.zeros_like(x_clean)

        best_delta = delta.clone()
        best_obj = compute_attack_objective(
            model_or_fn, x_clean + delta, y, loss_fn, targeted=targeted
        )
        total_queries = batch_size

        m = torch.zeros_like(x_clean)
        v = torch.zeros(x_clean.shape, device=x_clean.device, dtype=eps_budget.dtype)

        for step_idx in range(1, steps + 1):
            grad_est, step_queries = _estimate_spsa_gradient(
                model_or_fn=model_or_fn,
                x_adv=x_clean + delta,
                y=y,
                loss_fn=loss_fn,
                targeted=targeted,
                spsa_delta=spsa_delta,
                nb_sample=nb_sample,
                max_batch_size=max_batch_size,
                clip_bounds=clip_bounds,
                generator=generator,
            )
            total_queries += step_queries

            grad_sq = grad_est.real.pow(2) + grad_est.imag.pow(2) if grad_est.is_complex() else grad_est.pow(2)
            m = beta1 * m + (1.0 - beta1) * grad_est
            v = beta2 * v + (1.0 - beta2) * grad_sq

            m_hat = m / (1.0 - math.pow(beta1, step_idx))
            v_hat = v / (1.0 - math.pow(beta2, step_idx))
            adam_update = m_hat / (v_hat.sqrt() + adam_eps)

            delta = delta + alpha * adam_step_direction(adam_update, norm=norm)
            delta = project_perturbation(
                delta, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds
            )

            cand_obj = compute_attack_objective(
                model_or_fn, x_clean + delta, y, loss_fn, targeted=targeted
            )
            total_queries += batch_size
            best_delta, best_obj = update_best_by_objective(best_delta, best_obj, delta, cand_obj)

    return AttackResult(
        x_adv=x_clean + best_delta,
        perturbation=best_delta,
        eps=eps,
        norm=norm,
        steps_run=steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={"queries": total_queries, "nb_sample": nb_sample, "spsa_delta": spsa_delta},
    )
