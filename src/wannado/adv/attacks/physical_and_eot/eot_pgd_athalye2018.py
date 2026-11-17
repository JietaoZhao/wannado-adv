"""EoT-PGD: Expectation Over Transformation PGD (Athalye et al., ICML 2018).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Gold-Standard Stochastic Defense & Physical/Channel Robustness Attack)
- Reference     : Athalye, A., Engstrom, L., Ilyas, A., & Kwok, K. (2018).
                  "Synthesizing Robust Adversarial Examples." ICML 2018, PMLR 80:284-293.
- URL           : https://arxiv.org/abs/1707.07397
- Plain Intuition:
    Ordinary PGD optimizes perturbations for ONE exact frozen tensor `x + delta`.
    In the real world—camera rotation/lighting, randomized smoothing defenses, or
    wireless fading channels `h * (x + delta) + n`—random nuisance transforms
    break brittle single-point attacks. EoT-PGD takes `K` Monte Carlo transformation
    samples at every gradient step and climbs the EXPECTED objective gradient so
    adversarial perturbations remain potent across the whole transform distribution.
- Diff vs. Standard PGD (2018):
    1. Wraps inputs through stochastic `transform(x_adv)` inside autograd.
    2. Averages `K = eot_samples` independent Monte Carlo gradients at every step.
    3. Evaluates candidate perturbations using expected stochastic objective.
- Why Chosen as a Baseline:
    The mandatory evaluation baseline for randomized defenses (BPDA+EoT, randomized
    smoothing) as well as physical CV and stochastic wireless channel jamming.
- Extension Directions (for further research):
    1. Replace fixed step size with APGD-EoT step-size halving.
    2. Pair with Straight-Through / BPDA surrogate layers for quantized preprocessing.
    3. Supply domain transforms: camera warps (CV) or Rayleigh/CFO wireless channels.
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

TransformFn = Callable[[torch.Tensor], torch.Tensor]

META = AttackMeta(
    id="eot_pgd",
    name="Expectation Over Transformation PGD (EoT-PGD)",
    family="physical_and_eot",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="medium",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="pgd",
    diff_from_parent="Monte-Carlo averages autograd gradients and objective checks across K independent draws of a stochastic transform or stochastic victim model.",
    summary_plain="PGD averaged across random physical, preprocessing, or wireless-channel transforms so perturbations survive stochastic noise.",
    when_to_use="When attacking randomized defenses, noisy physical sensors, or stochastic wireless fading/AWGN channels.",
    when_not_to_use="On purely deterministic static models where K=1 standard PGD or APGD is K times faster.",
    extension_hooks=(
        "Plug in Rayleigh fading + AWGN + phase offset wireless channel as transform",
        "Combine EoT gradient averaging with APGD adaptive step-size halving",
    ),
    module_path="wannado.adv.attacks.physical_and_eot.eot_pgd_athalye2018",
    citation=AttackCitation(
        key="athalye2018synthesizing",
        title="Synthesizing Robust Adversarial Examples",
        authors="Anish Athalye, Logan Engstrom, Andrew Ilyas, and Kevin Kwok",
        venue="35th International Conference on Machine Learning (ICML)",
        year=2018,
        url="https://arxiv.org/abs/1707.07397",
        bibtex=(
            "@inproceedings{athalye2018synthesizing,\n"
            "  title={Synthesizing Robust Adversarial Examples},\n"
            "  author={Athalye, Anish and Engstrom, Logan and Ilyas, Andrew and Kwok, Kevin},\n"
            "  booktitle={Proceedings of the 35th International Conference on Machine Learning},\n"
            "  pages={284--293},\n"
            "  year={2018},\n"
            "  volume={80},\n"
            "  series={Proceedings of Machine Learning Research},\n"
            "  publisher={PMLR},\n"
            "  url={https://arxiv.org/abs/1707.07397},\n"
            "}"
        ),
        ieee='A. Athalye, L. Engstrom, A. Ilyas, and K. Kwok, "Synthesizing robust adversarial examples," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 80, 2018, pp. 284-293.',
        apa="Athalye, A., Engstrom, L., Ilyas, A., & Kwok, K. (2018). Synthesizing robust adversarial examples. Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80, 284-293.",
    ),
)


def _apply_transform(
    x: torch.Tensor,
    transform: TransformFn | None,
    noise_std: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Apply user differentiable transform and/or optional isotropic Gaussian jitter."""
    out = transform(x) if transform is not None else x
    if noise_std > 0.0:
        if out.is_complex():
            real_dtype = torch.float32 if out.dtype == torch.complex64 else torch.float64
            sigma = noise_std / math.sqrt(2.0)
            re = torch.randn(out.shape, device=out.device, dtype=real_dtype, generator=generator) * sigma
            im = torch.randn(out.shape, device=out.device, dtype=real_dtype, generator=generator) * sigma
            out = out + torch.complex(re, im).to(out.dtype)
        else:
            noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator) * noise_std
            out = out + noise
    return out


def eot_pgd(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    step_size: float | None = None,
    steps: int = 20,
    eot_samples: int = 8,
    transform: TransformFn | None = None,
    noise_std: float = 0.0,
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    random_start: bool = True,
    restarts: int = 1,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run Expectation Over Transformation PGD over stochastic transforms or models.

    Args:
        model_or_fn: Differentiable (potentially stochastic) ``nn.Module`` or callable.
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Ground-truth labels/targets (or target class when ``targeted=True``).
        eps: Perturbation budget (or JSR in dB when ``norm="jsr_db"``).
        step_size: Per-step magnitude. If ``None``, auto-scales to ``min(1.0, 2.5 / steps) * eps_budget``.
        steps: Number of projected gradient steps per restart.
        eot_samples: Number of Monte Carlo transformation/forward draws averaged per step.
        transform: Optional differentiable stochastic callable mapping ``x_adv -> x_transformed``.
            If ``None``, evaluates ``model_or_fn`` directly (useful when randomness is inside
            ``model_or_fn`` or paired with ``noise_std > 0``).
        noise_std: Optional built-in isotropic real/complex Gaussian noise std added per MC draw.
        norm: Constraint norm: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        loss: Built-in loss name (``"ce"``, ``"margin"``, ``"dlr"``, ``"mse"``) or callable.
        targeted: If ``True``, minimizes expected loss toward ``y``.
        random_start: Whether to initialize randomly inside the norm ball.
        restarts: Number of independent restarts; keeps highest expected-objective perturbation.
        clip_bounds: Optional valid value range ``(min, max)`` for real-valued inputs.
        seed: Optional random seed for reproducible restarts and built-in noise draws.

    Returns:
        ``AttackResult`` containing robust perturbation and EoT run metadata.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if eot_samples < 1:
        raise ValueError(f"eot_samples must be >= 1, got {eot_samples}.")
    if restarts < 1:
        raise ValueError(f"restarts must be >= 1, got {restarts}.")
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}.")

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
        eps_budget * min(1.0, 2.5 / steps)
        if step_size is None
        else torch.full_like(eps_budget, float(step_size))
    )
    generator = make_generator(seed, x_clean.device)

    best_delta = torch.zeros_like(x_clean)
    best_obj = torch.full(
        (x_clean.shape[0],), float("-inf"), device=x_clean.device, dtype=eps_budget.dtype
    )

    with eval_mode(model_or_fn):
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

                for _ in range(eot_samples):
                    x_t = _apply_transform(x_in, transform, noise_std, generator)
                    obj_k = compute_attack_objective(
                        model_or_fn, x_t, y, loss_fn, targeted=targeted
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
                for _ in range(eot_samples):
                    x_t = _apply_transform(x_clean + delta, transform, noise_std, generator)
                    expected_obj = expected_obj + compute_attack_objective(
                        model_or_fn, x_t, y, loss_fn, targeted=targeted
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
            "expected_objective_mean": float(best_obj.mean().item()),
        },
    )
