"""HopSkipJumpAttack: Query-Efficient Decision-Based Attack (Chen et al., IEEE S&P 2020).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Gold-Standard Label-Only / Decision-Based Black-Box Benchmark)
- Reference     : Chen, J., Jordan, M. I., & Wainwright, M. J. (2020).
                  "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack."
                  2020 IEEE Symposium on Security and Privacy (SP) (pp. 1277-1294).
- URL           : https://arxiv.org/abs/1904.02144
- Plain Intuition:
    Hard-label models return ONLY the predicted class index—no gradients, no
    probabilities, no logits. HopSkipJump starts from an adversarial point in the
    target region, binary-searches ("jumps") onto the decision boundary, estimates
    the boundary normal using Monte Carlo random +/- probes ("hops"), takes a
    geometric step along that boundary direction ("skips"), and jumps back to the
    boundary—steadily shrinking the perturbation norm without needing logits.
- Diff vs. Square Attack & SPSA:
    1. Needs ONLY hard class predictions (`argmax` / label oracle), not scores.
    2. Optimizes minimum boundary distortion via binary search + boundary gradient.
    3. Immune to score/logit masking (because it never reads logits at all).
- Why Chosen as a Baseline:
    Standard decision-based black-box evaluation baseline across vision, tabular,
    and wireless RF classification when APIs reveal only final decisions.
- Extension Directions (for further research):
    1. Use subspace/low-frequency noise sampling (QEBA / SurFree) for high-res images.
    2. Use multi-sample geometric progression schedules for custom non-Euclidean metrics.
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
    compute_perturbation_norm,
    eval_mode,
    make_generator,
    resolve_eps_budget,
    safe_l2_norm,
)
from wannado.adv.core.projections import within_eps_budget
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="hopskipjump",
    name="HopSkipJumpAttack (HSJA)",
    family="blackbox_decision",
    tier="S",
    threat_model="black_box_decision",
    requires_training=False,
    compute_cost="high",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    parent_id="boundary",
    diff_from_parent="Replaces random walk along the boundary with query-efficient Monte Carlo boundary-normal gradient estimation, geometric step-size search, and binary boundary projection.",
    summary_plain="Label-only black-box attack that minimizes perturbation distance using binary search and Monte Carlo boundary gradient estimation.",
    when_to_use="When you only observe final hard decisions (class labels) with zero access to gradients, probabilities, or logits.",
    when_not_to_use="When gradients or logit scores are available (use APGD/C&W/FAB or Square Attack for far lower query cost) or for continuous regression without decision boundaries.",
    extension_hooks=(
        "Swap isotropic Monte Carlo probes for frequency-subspace probes (QEBA)",
        "Provide domain-specific x_init seeds for harder targeted attacks",
    ),
    module_path="wannado.adv.attacks.blackbox_decision.hopskipjump_chen2020",
    citation=AttackCitation(
        key="chen2020hopskipjumpattack",
        title="HopSkipJumpAttack: A Query-Efficient Decision-Based Attack",
        authors="Jianbo Chen, Michael I. Jordan, and Martin J. Wainwright",
        venue="2020 IEEE Symposium on Security and Privacy (SP)",
        year=2020,
        url="https://arxiv.org/abs/1904.02144",
        bibtex=(
            "@inproceedings{chen2020hopskipjumpattack,\n"
            "  title={HopSkipJumpAttack: A Query-Efficient Decision-Based Attack},\n"
            "  author={Chen, Jianbo and Jordan, Michael I. and Wainwright, Martin J.},\n"
            "  booktitle={2020 IEEE Symposium on Security and Privacy (SP)},\n"
            "  pages={1277--1294},\n"
            "  year={2020},\n"
            "  organization={IEEE},\n"
            "  doi={10.1109/SP40000.2020.00045},\n"
            "  url={https://arxiv.org/abs/1904.02144},\n"
            "}"
        ),
        ieee='J. Chen, M. I. Jordan, and M. J. Wainwright, "HopSkipJumpAttack: A query-efficient decision-based attack," in Proc. IEEE Symp. Secur. Privacy (SP), 2020, pp. 1277-1294, doi: 10.1109/SP40000.2020.00045.',
        apa="Chen, J., Jordan, M. I., & Wainwright, M. J. (2020). HopSkipJumpAttack: A query-efficient decision-based attack. 2020 IEEE Symposium on Security and Privacy (SP), 1277-1294.",
    ),
)


def _query_decisions(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Query ONLY hard predicted class IDs from either logits or label-only oracle."""
    out = model_or_fn(z)
    if out.ndim >= 2:
        preds = out.argmax(dim=-1).reshape(-1).to(torch.long)
    elif out.ndim == 1:
        preds = out.to(torch.long)
    else:
        preds = out.reshape(1).to(torch.long)
    return preds, z.shape[0]


def _is_adversarial(preds: torch.Tensor, ref_label: int, targeted: bool) -> torch.Tensor:
    return preds == ref_label if targeted else preds != ref_label


def _clip(z: torch.Tensor, clip_bounds: tuple[float, float] | None) -> torch.Tensor:
    if clip_bounds is None:
        return z
    return torch.clamp(z, min=clip_bounds[0], max=clip_bounds[1])


def _norm_distance(x_ref: torch.Tensor, x_cand: torch.Tensor, norm: str) -> float:
    diff = (x_cand - x_ref).reshape(-1)
    if norm == "linf":
        return float(diff.abs().max().item())
    return float(safe_l2_norm(diff, dim=None).item())


def _project_interpolation(
    x_ref: torch.Tensor,
    x_adv: torch.Tensor,
    alpha: float,
    norm: str,
) -> torch.Tensor:
    """Interpolate toward ``x_ref`` in L2 (scalar fraction) or Linf (elementwise clamp)."""
    if norm == "linf":
        delta = x_adv - x_ref
        if delta.is_complex():
            mag = delta.abs()
            scaled = delta * (alpha / mag.clamp(min=1e-12))
            return x_ref + torch.where(mag > alpha, scaled, delta)
        return x_ref + torch.clamp(delta, min=-alpha, max=alpha)
    return (1.0 - alpha) * x_ref + alpha * x_adv


def _binary_search_boundary(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_ref: torch.Tensor,
    x_adv: torch.Tensor,
    ref_label: int,
    targeted: bool,
    norm: str,
    theta: float,
) -> tuple[torch.Tensor, int]:
    """Project an adversarial sample strictly onto the decision boundary via bisection."""
    queries = 0
    if norm == "linf":
        low = 0.0
        high = _norm_distance(x_ref, x_adv, norm="linf")
        if high <= 1e-12:
            return x_adv, 0
        tol = theta * max(high, 1.0)
        best = x_adv
        while (high - low) > tol:
            mid = 0.5 * (low + high)
            cand = _project_interpolation(x_ref, x_adv, mid, norm="linf")
            preds, q = _query_decisions(model_or_fn, cand)
            queries += q
            if bool(_is_adversarial(preds, ref_label, targeted)[0].item()):
                high = mid
                best = cand
            else:
                low = mid
        return best, queries

    low = 0.0
    high = 1.0
    best = x_adv
    while (high - low) > theta:
        mid = 0.5 * (low + high)
        cand = _project_interpolation(x_ref, x_adv, mid, norm=norm)
        preds, q = _query_decisions(model_or_fn, cand)
        queries += q
        if bool(_is_adversarial(preds, ref_label, targeted)[0].item()):
            high = mid
            best = cand
        else:
            low = mid
    return best, queries


def _sample_gaussian(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    is_complex: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    real_dtype = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
    if is_complex:
        re = torch.randn(shape, device=device, dtype=real_dtype, generator=generator)
        im = torch.randn(shape, device=device, dtype=real_dtype, generator=generator)
        return torch.complex(re, im).to(dtype)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def _initialize_adversarial(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_single: torch.Tensor,
    ref_label: int,
    targeted: bool,
    x_init_pool: torch.Tensor | None,
    clip_bounds: tuple[float, float] | None,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor | None, int]:
    """Find an initial adversarial point on the wrong/target side using label queries only."""
    queries = 0
    if x_init_pool is not None:
        preds, q = _query_decisions(model_or_fn, _clip(x_init_pool, clip_bounds))
        queries += q
        adv_mask = _is_adversarial(preds, ref_label, targeted)
        if adv_mask.any():
            idx = int(torch.nonzero(adv_mask, as_tuple=False)[0, 0].item())
            return _clip(x_init_pool[idx : idx + 1], clip_bounds), queries

    batch_try = 32
    real_dtype = torch.float32 if x_single.dtype in (torch.float32, torch.complex64) else torch.float64
    if clip_bounds is not None:
        low, high = clip_bounds
        for _ in range(10):
            cands = (
                torch.rand((batch_try, *x_single.shape[1:]), device=x_single.device, dtype=real_dtype, generator=generator)
                * (high - low)
                + low
            ).to(x_single.dtype)
            preds, q = _query_decisions(model_or_fn, cands)
            queries += q
            adv_mask = _is_adversarial(preds, ref_label, targeted)
            if adv_mask.any():
                idx = int(torch.nonzero(adv_mask, as_tuple=False)[0, 0].item())
                return cands[idx : idx + 1], queries

    rms = float(
        safe_l2_norm(x_single.reshape(-1), dim=None).item() / math.sqrt(max(x_single.numel(), 1))
    )
    base_scale = max(rms, 1.0)
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        noise = _sample_gaussian(
            (batch_try, *x_single.shape[1:]),
            dtype=x_single.dtype,
            device=x_single.device,
            is_complex=x_single.is_complex(),
            generator=generator,
        )
        cands = _clip(x_single + (mult * base_scale) * noise, clip_bounds)
        preds, q = _query_decisions(model_or_fn, cands)
        queries += q
        adv_mask = _is_adversarial(preds, ref_label, targeted)
        if adv_mask.any():
            idx = int(torch.nonzero(adv_mask, as_tuple=False)[0, 0].item())
            return cands[idx : idx + 1], queries

    return None, queries


def _approximate_gradient(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_boundary: torch.Tensor,
    ref_label: int,
    targeted: bool,
    num_evals: int,
    delta_probe: float,
    clip_bounds: tuple[float, float] | None,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, int]:
    """Monte Carlo boundary normal estimation using binary indicator centering (Chen et al. Eq. 13)."""
    noise = _sample_gaussian(
        (num_evals, *x_boundary.shape[1:]),
        dtype=x_boundary.dtype,
        device=x_boundary.device,
        is_complex=x_boundary.is_complex(),
        generator=generator,
    )
    norms = safe_l2_norm(noise.reshape(num_evals, -1), dim=1).clamp(min=1e-12)
    norms_view = norms.to(torch.float32 if x_boundary.dtype in (torch.float32, torch.complex64) else torch.float64)
    u = noise / norms_view.view((num_evals,) + (1,) * (x_boundary.ndim - 1))

    perturbed = _clip(x_boundary + delta_probe * u, clip_bounds)
    if clip_bounds is not None:
        # Match actual clipped direction when probes hit boundary walls.
        u_eff = (perturbed - x_boundary) / max(delta_probe, 1e-12)
    else:
        u_eff = u

    preds, queries = _query_decisions(model_or_fn, perturbed)
    adv_mask = _is_adversarial(preds, ref_label, targeted)
    fval = torch.where(adv_mask, 1.0, -1.0).to(norms_view.dtype)
    fval_mean = float(fval.mean().item())

    if fval_mean == 1.0:
        weights = fval
    elif fval_mean == -1.0:
        weights = fval
    else:
        weights = fval - fval_mean

    weights_view = weights.view((num_evals,) + (1,) * (x_boundary.ndim - 1))
    grad_est = (weights_view * u_eff).mean(dim=0, keepdim=True)
    g_norm = float(safe_l2_norm(grad_est.reshape(-1), dim=None).item())
    if g_norm <= 1e-12:
        return u[:1], queries
    return grad_est / g_norm, queries


def _hopskipjump_single_sample(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_single: torch.Tensor,
    ref_label: int,
    targeted: bool,
    norm: str,
    steps: int,
    initial_num_evals: int,
    max_num_evals: int,
    stepsize_search: str,
    gamma: float,
    x_init_pool: torch.Tensor | None,
    clip_bounds: tuple[float, float] | None,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, int, bool]:
    """Run HopSkipJump on one sample ``(1, ...)`` returning ``(x_adv, queries, success)``."""
    queries = 0
    x_adv, init_q = _initialize_adversarial(
        model_or_fn=model_or_fn,
        x_single=x_single,
        ref_label=ref_label,
        targeted=targeted,
        x_init_pool=x_init_pool,
        clip_bounds=clip_bounds,
        generator=generator,
    )
    queries += init_q
    if x_adv is None:
        return x_single.clone(), queries, False

    d_real = x_single[0].numel() * (2 if x_single.is_complex() else 1)
    if norm == "linf":
        theta = gamma / (max(d_real, 1) ** 2)
    else:
        theta = gamma / (max(d_real, 1) ** 1.5)
    theta = min(max(theta, 1e-5), 1e-3)

    x_adv, bs_q = _binary_search_boundary(
        model_or_fn, x_single, x_adv, ref_label, targeted, norm, theta
    )
    queries += bs_q
    dist = _norm_distance(x_single, x_adv, norm=norm)

    for step_idx in range(1, steps + 1):
        delta_probe = math.sqrt(max(d_real, 1)) * theta * max(dist, 1e-6)
        num_evals = min(max_num_evals, int(round(initial_num_evals * math.sqrt(step_idx))))
        num_evals = max(num_evals, 1)

        grad_est, grad_q = _approximate_gradient(
            model_or_fn=model_or_fn,
            x_boundary=x_adv,
            ref_label=ref_label,
            targeted=targeted,
            num_evals=num_evals,
            delta_probe=delta_probe,
            clip_bounds=clip_bounds,
            generator=generator,
        )
        queries += grad_q

        if norm == "linf":
            update = torch.sgn(grad_est) if grad_est.is_complex() else torch.sign(grad_est)
        else:
            update = grad_est

        epsilon = max(dist, 1e-6) / math.sqrt(step_idx)
        cand_accepted = None

        if stepsize_search == "geometric_progression":
            for _ in range(20):
                cand = _clip(x_adv + epsilon * update, clip_bounds)
                preds, q = _query_decisions(model_or_fn, cand)
                queries += q
                if bool(_is_adversarial(preds, ref_label, targeted)[0].item()):
                    cand_accepted = cand
                    break
                epsilon *= 0.5
        elif stepsize_search == "grid_search":
            grid = torch.logspace(-4, 0, steps=16, device=x_single.device) * max(dist, 1e-6)
            view_shape = (16,) + (1,) * (x_single.ndim - 1)
            cands = _clip(x_adv + grid.view(view_shape).to(update.real.dtype if update.is_complex() else update.dtype) * update, clip_bounds)
            preds, q = _query_decisions(model_or_fn, cands)
            queries += q
            adv_mask = _is_adversarial(preds, ref_label, targeted)
            if adv_mask.any():
                dists = [
                    _norm_distance(x_single, cands[k : k + 1], norm=norm) if bool(adv_mask[k]) else float("inf")
                    for k in range(16)
                ]
                best_k = int(min(range(16), key=lambda k: dists[k]))
                cand_accepted = cands[best_k : best_k + 1]
        else:
            raise ValueError(
                f"Unknown stepsize_search '{stepsize_search}'. Expected 'geometric_progression' or 'grid_search'."
            )

        if cand_accepted is not None:
            x_next, bs_q2 = _binary_search_boundary(
                model_or_fn, x_single, cand_accepted, ref_label, targeted, norm, theta
            )
            queries += bs_q2
            new_dist = _norm_distance(x_single, x_next, norm=norm)
            if new_dist <= dist + 1e-8:
                x_adv = x_next
                dist = new_dist

    return x_adv, queries, True


def hopskipjump(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    eps: float | None = None,
    steps: int = 20,
    initial_num_evals: int = 50,
    max_num_evals: int = 1000,
    stepsize_search: str = "geometric_progression",
    gamma: float = 1.0,
    norm: str = "l2",
    targeted: bool = False,
    x_init: torch.Tensor | None = None,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run HopSkipJump decision-based black-box attack using hard labels only.

    Args:
        model_or_fn: Callable or ``nn.Module`` returning either logits ``(B, C)`` or hard
            1D integer labels ``(B,)``. Only ``argmax`` decisions are ever read.
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Clean ground-truth labels (optional if untargeted), or target labels if ``targeted=True``.
        eps: Optional perturbation budget cutoff (reverts samples exceeding ``eps`` to clean ``x``).
        steps: Number of boundary refinement iterations per sample.
        initial_num_evals: Initial Monte Carlo boundary probes ``B_0`` (grows as ``B_0 * sqrt(t)``).
        max_num_evals: Upper cap on Monte Carlo probes per iteration.
        stepsize_search: ``"geometric_progression"`` (default) or ``"grid_search"``.
        gamma: Binary-search threshold factor controlling boundary precision ``theta``.
        norm: Distance metric: ``"l2"``, ``"linf"``, ``"relative_l2"``, or ``"jsr_db"``.
        targeted: If ``True``, requires explicit target labels ``y`` and steers decisions to ``y``.
        x_init: Optional pool of candidate initial adversarial samples (especially helpful for targeted).
        clip_bounds: Optional valid value range ``(min, max)`` for real-valued inputs.
        seed: Optional random seed for reproducible boundary probes and initialization.

    Returns:
        ``AttackResult`` containing boundary-minimized perturbations and query counts.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if clip_bounds is not None and x.is_complex():
        raise ValueError("clip_bounds is only supported for real-valued tensors.")
    if targeted and y is None:
        raise ValueError("y target labels must be provided when targeted=True.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    batch_size = x_clean.shape[0]
    if eps is not None:
        eps_budget = resolve_eps_budget(x_clean, eps=eps, norm=norm).reshape(batch_size)
        if (eps_budget == 0).all():
            return AttackResult(
                x_adv=x_clean.clone(),
                perturbation=torch.zeros_like(x_clean),
                eps=float(eps),
                norm=norm,
                steps_run=0,
                attack_id=META.id,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                extra={"queries": 0, "mean_distortion": 0.0, "success_rate": 0.0},
            )
    else:
        eps_budget = None

    generator = make_generator(seed, x_clean.device)
    x_adv = x_clean.clone()
    total_queries = 0
    success_count = 0

    with eval_mode(model_or_fn), torch.no_grad():
        clean_preds, q0 = _query_decisions(model_or_fn, x_clean)
        total_queries += q0
        ref_labels = y if y is not None else clean_preds

        for i in range(batch_size):
            ref_i = int(ref_labels[i].item())
            clean_pred_i = int(clean_preds[i].item())
            already_adv = (clean_pred_i == ref_i) if targeted else (y is not None and clean_pred_i != ref_i)
            if already_adv:
                success_count += 1
                continue

            pool_i = x_init[i : i + 1] if (x_init is not None and x_init.shape[0] == batch_size) else x_init
            adv_i, q_i, ok_i = _hopskipjump_single_sample(
                model_or_fn=model_or_fn,
                x_single=x_clean[i : i + 1],
                ref_label=ref_i,
                targeted=targeted,
                norm=norm,
                steps=steps,
                initial_num_evals=initial_num_evals,
                max_num_evals=max_num_evals,
                stepsize_search=stepsize_search,
                gamma=gamma,
                x_init_pool=pool_i,
                clip_bounds=clip_bounds,
                generator=generator,
            )
            total_queries += q_i
            if ok_i:
                x_adv[i] = adv_i[0]
                success_count += 1

    delta = x_adv - x_clean
    if eps is not None:
        within_budget = within_eps_budget(delta, x_clean, eps=eps, norm=norm).view(
            (batch_size,) + (1,) * (x_clean.ndim - 1)
        )
        x_adv = torch.where(within_budget, x_adv, x_clean)
        delta = x_adv - x_clean

    dist_norm = compute_perturbation_norm(delta, norm=norm, x_ref=x_clean)
    return AttackResult(
        x_adv=x_adv,
        perturbation=delta,
        eps=float(eps) if eps is not None else float(dist_norm.median().item()),
        norm=norm,
        steps_run=steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "queries": total_queries,
            "mean_queries": float(total_queries / max(batch_size, 1)),
            "mean_distortion": float(dist_norm.mean().item()),
            "median_distortion": float(dist_norm.median().item()),
            "success_rate": float(success_count / max(batch_size, 1)),
        },
    )
