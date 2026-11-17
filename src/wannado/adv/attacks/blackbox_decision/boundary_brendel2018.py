"""Boundary Attack: Original Hard-Label Decision-Based Attack (Brendel et al., ICLR 2018).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Seminal Decision-Based Hard-Label Attack Origin)
- Reference     : Brendel, W., Rauber, J., & Bethge, M. (2018).
                  "Decision-Based Adversarial Attacks: Reliable Attacks Against
                  Black-Box Machine Learning Models." ICLR 2018.
- URL           : https://arxiv.org/abs/1712.04248
- Plain Intuition:
    How do you attack a model that ONLY returns a hard class label ("cat")
    with NO probabilities and NO gradients? Start from a point that is ALREADY
    misclassified (random noise or another sample), shrink along the line toward
    the clean input via binary search until you sit right on the decision
    boundary, and then do an alternating random walk:
      1. Take a small orthogonal step along the sphere around ``x_clean``.
      2. Take a small inward step toward ``x_clean``.
    Keep moves that remain adversarial, shrinking step sizes adaptively whenever
    too many proposals fall back on the clean side.
- Diff vs. HopSkipJumpAttack (Chen et al., 2020):
    Boundary Attack uses purely random orthogonal sphere steps (zero boundary
    gradient estimation), serving as the simpler ancestor to HSJA.
- Why Chosen as a Baseline:
    Invented the entire decision-based threat model; foundational baseline when
    evaluating label-only black-box APIs.
- Extension Directions / Prompt Hooks:
    1. Replace isotropic orthogonal probes with low-frequency / Perlin noise
       proposals to cut query counts on images.
    2. Initialize from clean-class boundary blends (e.g. modulation constellation
       neighbors in wireless RF).
===============================================================================
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    clamp_to_bounds,
    coerce_clip_bounds,
    compute_perturbation_norm,
    eval_mode,
    make_generator,
    predict_labels,
    safe_l2_norm,
)
from wannado.adv.core.projections import within_eps_budget
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="boundary",
    name="Boundary Attack",
    family="blackbox_decision",
    tier="S",
    threat_model="black_box_decision",
    requires_training=False,
    compute_cost="high",
    supported_norms=("l2", "linf", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    parent_id=None,
    diff_from_parent="Origin of decision-based black-box attacks: walks randomly along the hard-label decision boundary while stepping toward the clean sample.",
    summary_plain="Original label-only black-box attack that walks along the decision boundary from an adversarial start toward clean x.",
    when_to_use="When benchmarking label-only models against the classic non-gradient random-walk baseline alongside HopSkipJump.",
    when_not_to_use="When gradients/scores are exposed or lowest possible query count is critical (use hopskipjump or square_attack).",
    extension_hooks=(
        "Swap Gaussian orthogonal proposals for DCT/Perlin priors for faster spatial convergence",
        "Supply warm-start `x_init` from surrogate transfer or targeted class prototypes",
    ),
    module_path="wannado.adv.attacks.blackbox_decision.boundary_brendel2018",
    citation=AttackCitation(
        key="brendel2018decision",
        title="Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models",
        authors="Wieland Brendel, Jonas Rauber, and Matthias Bethge",
        venue="6th International Conference on Learning Representations (ICLR)",
        year=2018,
        url="https://arxiv.org/abs/1712.04248",
        bibtex=(
            "@inproceedings{brendel2018decision,\n"
            "  title={Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models},\n"
            "  author={Brendel, Wieland and Rauber, Jonas and Bethge, Matthias},\n"
            "  booktitle={International Conference on Learning Representations (ICLR)},\n"
            "  year={2018},\n"
            "  url={https://arxiv.org/abs/1712.04248},\n"
            "}"
        ),
        ieee='W. Brendel, J. Rauber, and M. Bethge, "Decision-based adversarial attacks: Reliable attacks against black-box machine learning models," in Proc. Int. Conf. Learn. Represent. (ICLR), 2018.',
        apa="Brendel, W., Rauber, J., & Bethge, M. (2018). Decision-based adversarial attacks: Reliable attacks against black-box machine learning models. International Conference on Learning Representations (ICLR).",
    ),
)


def _hard_decisions(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Query ONLY predicted class indices regardless of whether oracle returns logits or class IDs."""
    return predict_labels(model_or_fn(z)), int(z.shape[0])


def _is_adv(preds: torch.Tensor, ref_label: torch.Tensor, targeted: bool) -> torch.Tensor:
    return (preds == ref_label) if targeted else (preds != ref_label)


def _sample_like(
    template: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample standard normal real or complex tensor matching ``template`` shape and dtype."""
    if template.is_complex():
        real_dtype = torch.float32 if template.dtype == torch.complex64 else torch.float64
        re = torch.randn(template.shape, device=template.device, dtype=real_dtype, generator=generator)
        im = torch.randn(template.shape, device=template.device, dtype=real_dtype, generator=generator)
        return torch.complex(re, im).to(template.dtype)
    return torch.randn(template.shape, device=template.device, dtype=template.dtype, generator=generator)


def boundary(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    eps: float | None = None,
    norm: str = "l2",
    steps: int = 80,
    spherical_step: float = 0.05,
    source_step: float = 0.05,
    step_adaptation: float = 1.5,
    targeted: bool = False,
    clip_bounds: tuple[Any, Any] | None = None,
    x_init: torch.Tensor | None = None,
    max_queries: int | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run Boundary Attack (Brendel et al., ICLR 2018) on a hard-label oracle."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if max_queries is not None and max_queries < 1:
        raise ValueError(f"max_queries must be >= 1, got {max_queries}.")
    if clip_bounds is not None and x.is_complex():
        raise ValueError("clip_bounds is only supported for real-valued tensors.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    batch_size = x_clean.shape[0]
    generator = make_generator(seed, x_clean.device)
    queries = 0

    with eval_mode(model_or_fn), torch.no_grad():
        clean_preds, q = _hard_decisions(model_or_fn, x_clean)
        queries += q
        ref_y = y.to(torch.long) if y is not None else clean_preds

        x_adv = x_clean.clone()

        for idx in range(batch_size):
            if max_queries is not None and queries >= max_queries:
                break
            x_i = x_clean[idx : idx + 1]
            label_i = ref_y[idx : idx + 1]
            if _is_adv(clean_preds[idx : idx + 1], label_i, targeted).item():
                continue

            # Step 1: Locate an initial adversarial starting point.
            adv_cur: torch.Tensor | None = None
            if x_init is not None:
                cand = clamp_to_bounds(x_init[idx : idx + 1].detach(), clip_bounds)
                p, q = _hard_decisions(model_or_fn, cand)
                queries += q
                if _is_adv(p, label_i, targeted).item():
                    adv_cur = cand

            if adv_cur is None and batch_size > 1:
                # Search other batch samples first (reliable for both real & complex/targeted).
                if targeted:
                    matches = (clean_preds == label_i[0]).nonzero(as_tuple=False).squeeze(1)
                else:
                    matches = (clean_preds != label_i[0]).nonzero(as_tuple=False).squeeze(1)
                if matches.numel() > 0:
                    rand_pos = int(
                        torch.randint(
                            0,
                            matches.numel(),
                            (),
                            device=x_clean.device,
                            generator=generator,
                        ).item()
                    )
                    pick = int(matches[rand_pos].item())
                    adv_cur = x_clean[pick : pick + 1].clone()

            if adv_cur is None:
                # Random perturbation search over increasing scales.
                base_scale = float(max(x_i.abs().mean().item(), 1.0))
                lo_hi = coerce_clip_bounds(clip_bounds, x_i) if clip_bounds is not None else None
                for mult in (0.5, 1.0, 2.0, 5.0, 10.0):
                    for _ in range(15):
                        if max_queries is not None and queries >= max_queries:
                            break
                        if lo_hi is not None:
                            lo_c, hi_c = lo_hi
                            cand = (
                                torch.rand(
                                    x_i.shape,
                                    device=x_i.device,
                                    dtype=x_i.dtype,
                                    generator=generator,
                                )
                                * (hi_c - lo_c)
                                + lo_c
                            )
                        else:
                            cand = x_i + _sample_like(x_i, generator) * (base_scale * mult)
                        p, q = _hard_decisions(model_or_fn, cand)
                        queries += q
                        if _is_adv(p, label_i, targeted).item():
                            adv_cur = cand
                            break
                    if adv_cur is not None or (max_queries is not None and queries >= max_queries):
                        break

            if adv_cur is None:
                continue

            # Binary search along segment [x_i, adv_cur] to land tightly on the boundary.
            lo_b, hi_b = 0.0, 1.0
            for _ in range(10):
                if max_queries is not None and queries >= max_queries:
                    break
                mid_b = 0.5 * (lo_b + hi_b)
                mid_x = (1.0 - mid_b) * x_i + mid_b * adv_cur
                p, q = _hard_decisions(model_or_fn, mid_x)
                queries += q
                if _is_adv(p, label_i, targeted).item():
                    hi_b = mid_b
                    adv_cur = mid_x
                else:
                    lo_b = mid_b

            # Step 2: Brendel et al. (2018) alternating orthogonal + inward walk on boundary.
            sph_step = spherical_step
            src_step = source_step
            orth_successes = 0
            orth_trials = 0

            for step_idx in range(steps):
                if max_queries is not None and queries >= max_queries:
                    break
                unpert = adv_cur - x_i
                dist_cur = float(safe_l2_norm(unpert.reshape(-1), dim=None).item())
                if dist_cur <= 1e-8:
                    break

                # Sample random noise orthogonal to `(adv_cur - x_i)` around x_i sphere.
                noise = _sample_like(x_i, generator)
                # Real inner product <noise, unpert> works cleanly in real & complex (Re(n * conj(u))).
                if x_i.is_complex():
                    dot = (noise.real * unpert.real + noise.imag * unpert.imag).sum()
                else:
                    dot = (noise * unpert).sum()
                noise_orth = noise - (dot / (dist_cur * dist_cur + 1e-12)) * unpert
                orth_norm = float(safe_l2_norm(noise_orth.reshape(-1), dim=None).item())
                if orth_norm <= 1e-12:
                    continue
                noise_orth = noise_orth / orth_norm

                # Move along sphere of radius dist_cur, then re-normalize distance to dist_cur.
                # [AI-Prompt Hook]: Replace Gaussian `noise_orth` with Perlin/DCT basis for faster CV walk.
                cand_sph = clamp_to_bounds(adv_cur + (sph_step * dist_cur) * noise_orth, clip_bounds)
                cand_delta = cand_sph - x_i
                cand_delta = cand_delta * (
                    dist_cur / max(float(safe_l2_norm(cand_delta.reshape(-1), dim=None).item()), 1e-12)
                )
                cand_sph = clamp_to_bounds(x_i + cand_delta, clip_bounds)

                p_sph, q = _hard_decisions(model_or_fn, cand_sph)
                queries += q
                orth_trials += 1
                if _is_adv(p_sph, label_i, targeted).item():
                    orth_successes += 1
                    # Try inward step toward clean x_i along the accepted sphere candidate.
                    cand_in = clamp_to_bounds((1.0 - src_step) * cand_sph + src_step * x_i, clip_bounds)
                    p_in, q = _hard_decisions(model_or_fn, cand_in)
                    queries += q
                    if _is_adv(p_in, label_i, targeted).item():
                        adv_cur = cand_in
                        src_step = min(src_step * step_adaptation, 0.5)
                    else:
                        adv_cur = cand_sph
                        src_step = max(src_step / step_adaptation, 1e-4)

                # Adapt spherical step every 10 orthogonal proposals to keep acceptance ~30-70%.
                if (step_idx + 1) % 10 == 0 and orth_trials > 0:
                    acc_rate = orth_successes / float(orth_trials)
                    if acc_rate < 0.25:
                        sph_step = max(sph_step / step_adaptation, 1e-4)
                    elif acc_rate > 0.70:
                        sph_step = min(sph_step * step_adaptation, 0.5)
                    orth_successes = 0
                    orth_trials = 0

            x_adv[idx] = adv_cur[0]

    delta = x_adv - x_clean
    if eps is not None:
        within = within_eps_budget(delta, x_clean, eps=eps, norm=norm).view(
            (batch_size,) + (1,) * (x_clean.ndim - 1)
        )
        x_adv = torch.where(within, x_adv, x_clean)
        delta = x_adv - x_clean

    dist = compute_perturbation_norm(delta, norm=norm, x_ref=x_clean)
    return AttackResult(
        x_adv=x_adv,
        perturbation=delta,
        eps=float(eps) if eps is not None else float(dist.median().item()),
        norm=norm,
        steps_run=steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "queries": queries,
            "mean_distortion": float(dist.mean().item()),
            "median_distortion": float(dist.median().item()),
        },
    )
