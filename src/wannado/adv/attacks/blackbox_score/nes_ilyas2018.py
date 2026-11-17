"""NES: Natural Evolution Strategies Query-Limited Black-Box Attack (Ilyas et al., ICML 2018).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Classic Query-Limited Black-Box Gradient Estimator)
- Reference     : Ilyas, A., Engstrom, L., Athalye, A., & Lin, J. (2018).
                  "Black-box Adversarial Attacks with Limited Queries and
                  Information." ICML 2018, PMLR 80:2137-2146.
- URL           : https://arxiv.org/abs/1804.08598
- Plain Intuition:
    When only confidence scores (logits/probabilities) are visible, NES turns
    derivative-free Gaussian smoothing into an unbiased search-gradient oracle.
    Around the current candidate ``x_adv``, it evaluates ``N`` antithetic
    Gaussian pairs ``x_adv + sigma * u_i`` and ``x_adv - sigma * u_i`` and
    weights each perturbation ``u_i`` by the objective difference ``J(+) - J(-)``.
    That estimated gradient plugs straight into projected gradient ascent (PGD).
- Diff vs. SPSA (Uesato et al., 2018):
    1. Samples isotropic Gaussian probes u ~ N(0, I) instead of Rademacher {+-1}
       and divides by search std ``sigma`` weighted directly as NES score-function
       estimator.
    2. Widely used baseline in query-limited API & commercial vision attacks.
- Why Chosen as a Baseline:
    The seminal query-limited score-based black-box gradient estimator.
- Extension Directions / Prompt Hooks:
    1. Replace isotropic Gaussian covariance with data-dependent / band-limited
       priors (Bandits-TD, Ilyas et al. ICLR 2019).
    2. Use ``norm='jsr_db'`` to probe non-differentiable wireless RF demodulators.
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
    clamp_to_bounds,
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

META = AttackMeta(
    id="nes",
    name="Natural Evolution Strategies Attack (NES)",
    family="blackbox_score",
    tier="S",
    threat_model="black_box_score",
    requires_training=False,
    compute_cost="high",
    supported_norms=("linf", "l2", "l1", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="pgd",
    diff_from_parent="Replaces exact autograd backprop with antithetic Gaussian NES score-function gradient estimates using only forward queries.",
    summary_plain="Estimates gradients from forward-only confidence queries via antithetic Gaussian search directions and runs PGD.",
    when_to_use="Against non-differentiable or gradient-obfuscated models that still return continuous scores or logits.",
    when_not_to_use="When hard decisions only are exposed (use boundary or hopskipjump) or when white-box autograd works.",
    extension_hooks=(
        "Add time/data priors to Gaussian covariance -> Bandits-TD",
        "Adjust search std `sigma` and `nb_sample` for noisy/stochastic RF channels",
    ),
    module_path="wannado.adv.attacks.blackbox_score.nes_ilyas2018",
    citation=AttackCitation(
        key="ilyas2018black",
        title="Black-box Adversarial Attacks with Limited Queries and Information",
        authors="Andrew Ilyas, Logan Engstrom, Anish Athalye, and Jessy Lin",
        venue="35th International Conference on Machine Learning (ICML)",
        year=2018,
        url="https://arxiv.org/abs/1804.08598",
        bibtex=(
            "@inproceedings{ilyas2018black,\n"
            "  title={Black-box Adversarial Attacks with Limited Queries and Information},\n"
            "  author={Ilyas, Andrew and Engstrom, Logan and Athalye, Anish and Lin, Jessy},\n"
            "  booktitle={Proceedings of the 35th International Conference on Machine Learning},\n"
            "  pages={2137--2146},\n"
            "  year={2018},\n"
            "  volume={80},\n"
            "  series={Proceedings of Machine Learning Research},\n"
            "  publisher={PMLR},\n"
            "  url={https://arxiv.org/abs/1804.08598},\n"
            "}"
        ),
        ieee='A. Ilyas, L. Engstrom, A. Athalye, and J. Lin, "Black-box adversarial attacks with limited queries and information," in Proc. 35th Int. Conf. Mach. Learn. (ICML), PMLR 80, 2018, pp. 2137-2146.',
        apa="Ilyas, A., Engstrom, L., Athalye, A., & Lin, J. (2018). Black-box adversarial attacks with limited queries and information. Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80, 2137-2146.",
    ),
)


def _sample_gaussian(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    is_complex: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample isotropic real or circularly symmetric complex Gaussian directions."""
    real_dtype = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
    if is_complex:
        scale = 1.0 / math.sqrt(2.0)
        re = torch.randn(shape, device=device, dtype=real_dtype, generator=generator) * scale
        im = torch.randn(shape, device=device, dtype=real_dtype, generator=generator) * scale
        return torch.complex(re, im).to(dtype)
    return torch.randn(shape, device=device, dtype=real_dtype, generator=generator).to(dtype)


def _estimate_nes_gradient(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_adv: torch.Tensor,
    y: torch.Tensor,
    loss_fn: LossFn,
    targeted: bool,
    sigma: float,
    nb_sample: int,
    max_batch_size: int,
    clip_bounds: tuple[float, float] | None,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, int]:
    """Estimate ascent gradient via antithetic Gaussian NES pairs."""
    batch_size = x_adv.shape[0]
    probe_shape = (nb_sample, batch_size, *x_adv.shape[1:])
    u = _sample_gaussian(
        probe_shape,
        dtype=x_adv.dtype,
        device=x_adv.device,
        is_complex=x_adv.is_complex(),
        generator=generator,
    )

    x_plus = x_adv.unsqueeze(0) + sigma * u
    x_minus = x_adv.unsqueeze(0) - sigma * u
    if clip_bounds is not None:
        x_plus = clamp_to_bounds(x_plus, clip_bounds)
        x_minus = clamp_to_bounds(x_minus, clip_bounds)

    all_probes = torch.cat(
        [x_plus.reshape(-1, *x_adv.shape[1:]), x_minus.reshape(-1, *x_adv.shape[1:])],
        dim=0,
    )
    y_rep = y.unsqueeze(0).expand(nb_sample, *y.shape).reshape(-1, *y.shape[1:])
    y_all = torch.cat([y_rep, y_rep], dim=0)

    obj_chunks: list[torch.Tensor] = []
    total_evals = int(all_probes.shape[0])
    for start in range(0, total_evals, max(1, max_batch_size)):
        end = min(start + max(1, max_batch_size), total_evals)
        obj_chunks.append(
            compute_attack_objective(
                model_or_fn, all_probes[start:end], y_all[start:end], loss_fn, targeted=targeted
            )
        )

    all_obj = torch.cat(obj_chunks, dim=0)
    half = nb_sample * batch_size
    obj_plus = all_obj[:half].view(nb_sample, batch_size)
    obj_minus = all_obj[half:].view(nb_sample, batch_size)

    diff = (obj_plus - obj_minus).view((nb_sample, batch_size) + (1,) * (x_adv.ndim - 1))
    # [AI-Prompt Hook]: Ilyas et al. (2018) Eq. (2) antithetic NES gradient estimator.
    # Replace scalar `diff` normalization with rank-transformation for extreme score invariance.
    grad_est = (diff * u).sum(dim=0) / float(2.0 * nb_sample * sigma)
    return grad_est, total_evals


def nes(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    step_size: float | None = None,
    steps: int = 30,
    sigma: float = 0.01,
    nb_sample: int = 40,
    momentum: float = 0.9,
    norm: str = "linf",
    loss: str | LossFn | None = "margin",
    targeted: bool = False,
    random_start: bool = True,
    max_batch_size: int = 512,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run NES score-based black-box attack (Ilyas et al., ICML 2018)."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")
    if nb_sample < 1:
        raise ValueError(f"nb_sample must be >= 1, got {nb_sample}.")
    if momentum < 0.0:
        raise ValueError(f"momentum must be >= 0, got {momentum}.")

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
            extra={"queries": 0, "nb_sample": nb_sample, "sigma": sigma},
        )

    alpha = (
        eps_budget * min(1.0, 2.5 / float(steps))
        if step_size is None
        else torch.full_like(eps_budget, float(step_size))
    )
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
        queries = int(x_clean.shape[0])
        vel = torch.zeros_like(x_clean)

        for _ in range(steps):
            grad_est, step_queries = _estimate_nes_gradient(
                model_or_fn=model_or_fn,
                x_adv=x_clean + delta,
                y=y,
                loss_fn=loss_fn,
                targeted=targeted,
                sigma=sigma,
                nb_sample=nb_sample,
                max_batch_size=max_batch_size,
                clip_bounds=clip_bounds,
                generator=generator,
            )
            queries += step_queries

            vel = momentum * vel + grad_est
            delta = delta + alpha * step_direction(vel, norm=norm)
            delta = project_perturbation(
                delta, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds
            )
            cand_obj = compute_attack_objective(
                model_or_fn, x_clean + delta, y, loss_fn, targeted=targeted
            )
            queries += int(x_clean.shape[0])
            best_delta, best_obj = update_best_by_objective(best_delta, best_obj, delta, cand_obj)

    return AttackResult(
        x_adv=x_clean + best_delta,
        perturbation=best_delta,
        eps=eps,
        norm=norm,
        steps_run=steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "queries": queries,
            "nb_sample": nb_sample,
            "sigma": sigma,
            "momentum": momentum,
        },
    )
