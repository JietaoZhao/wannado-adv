"""Square Attack: Query-Efficient Black-Box Random Search (Andriushchenko et al., ECCV 2020).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Gold-Standard Score-Based Black-Box Benchmark; AutoAttack Member)
- Reference     : Andriushchenko, M., Croce, F., Flammarion, N., & Hein, M. (2020).
                  "Square Attack: a query-efficient black-box adversarial attack
                  via random search." ECCV 2020 (pp. 484-501).
- URL           : https://arxiv.org/abs/1912.00049
- Plain Intuition:
    Finite-difference estimators (like SPSA) burn dozens of queries per step just
    to estimate gradients, and fail if gradients are flat or noisy. Square Attack
    skips gradients completely: it starts on the constraint boundary (stripes for
    images, dense vectors for 1D signals) and runs greedy hill-climbing using
    localized, shrinking square/burst patches—accepting any patch that improves
    the attack margin and keeping the perturbation tightly on the feasible budget.
- Diff vs. SPSA (2018):
    1. Pure randomized hill-climbing (no gradient estimation whatsoever).
    2. Immune to non-differentiable preprocessing and loss rescaling obfuscation.
    3. Dramatically fewer queries needed via coarse-to-fine patch size halving.
- Why Chosen as a Baseline:
    The score-based black-box pillar of AutoAttack (Croce & Hein, ICML 2020),
    used everywhere to test defenses without assuming gradient access.
- Extension Directions (for further research):
    1. Run APGD-CE + APGD-DLR + FAB + Square Attack in sequence -> AutoAttack.
    2. Replace logit queries with hard-label queries -> HopSkipJump.
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
    eval_mode,
    make_generator,
    outputs_to_objective,
    project_perturbation,
    random_init_perturbation,
    resolve_eps_budget,
    resolve_loss_fn,
    safe_l2_norm,
)
from wannado.adv.core.losses import LossFn
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="square_attack",
    name="Square Attack",
    family="blackbox_score",
    tier="S",
    threat_model="black_box_score",
    requires_training=False,
    compute_cost="medium",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id=None,
    diff_from_parent="Origin of query-efficient boundary random search: samples shrinking localized square patches (4D) or contiguous windows (1D/ND) without gradient estimation.",
    summary_plain="Gradient-free score-based black-box attack that greedily proposes shrinking localized patches/bursts along the constraint boundary.",
    when_to_use="Strongest general-purpose score-based black-box attack and essential sanity check against white-box gradient masking.",
    when_not_to_use="When only top-1 hard labels are observable without confidence/logit scores (use HopSkipJump).",
    extension_hooks=(
        "Chain with APGD-CE, APGD-DLR, and FAB -> AutoAttack",
        "Customize patch geometry for frequency-domain or spectrogram bursts",
    ),
    module_path="wannado.adv.attacks.blackbox_score.square_andriushchenko2020",
    citation=AttackCitation(
        key="andriushchenko2020square",
        title="Square Attack: a query-efficient black-box adversarial attack via random search",
        authors="Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, and Matthias Hein",
        venue="European Conference on Computer Vision (ECCV)",
        year=2020,
        url="https://arxiv.org/abs/1912.00049",
        bibtex=(
            "@inproceedings{andriushchenko2020square,\n"
            "  title={Square attack: a query-efficient black-box adversarial attack via random search},\n"
            "  author={Andriushchenko, Maksym and Croce, Francesco and Flammarion, Nicolas and Hein, Matthias},\n"
            "  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Proceedings, Part XXIV},\n"
            "  pages={484--501},\n"
            "  year={2020},\n"
            "  publisher={Springer},\n"
            "  url={https://arxiv.org/abs/1912.00049},\n"
            "}"
        ),
        ieee='M. Andriushchenko, F. Croce, N. Flammarion, and M. Hein, "Square attack: A query-efficient black-box adversarial attack via random search," in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020, pp. 484-501.',
        apa="Andriushchenko, M., Croce, F., Flammarion, N., & Hein, M. (2020). Square attack: A query-efficient black-box adversarial attack via random search. European Conference on Computer Vision (ECCV), 484-501.",
    ),
)


def _p_selection(p_init: float, it: int, n_iters: int) -> float:
    """Piecewise-constant shrinking schedule from Andriushchenko et al. (ECCV 2020)."""
    if n_iters < 2000:
        # Preserve coarse-to-fine exploration under tight query budgets.
        frac_thresholds = (0.04, 0.10, 0.18, 0.28, 0.40, 0.55, 0.70, 0.85, 0.93)
        frac = it / max(n_iters, 1)
        p = p_init
        for thresh in frac_thresholds:
            if frac <= thresh:
                return p
            p *= 0.5
        return p

    it_scaled = int((it / max(n_iters, 1)) * 10000)
    thresholds = (10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000)
    p = p_init
    for thresh in thresholds:
        if it_scaled <= thresh:
            return p
        p *= 0.5
    return p


def _sample_unit_linf(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    is_complex: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample extreme Linf values: {-1, +1} for real or unit-circle phases for complex."""
    real_dtype = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
    if is_complex:
        phase = torch.rand(shape, device=device, dtype=real_dtype, generator=generator) * (2.0 * math.pi)
        return torch.polar(torch.ones_like(phase), phase).to(dtype)
    signs = torch.randint(0, 2, shape, device=device, generator=generator, dtype=torch.int64).to(dtype)
    return signs * 2.0 - 1.0


def _sample_l2_noise(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    is_complex: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample real or complex Gaussian tensor."""
    real_dtype = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
    if is_complex:
        re = torch.randn(shape, device=device, dtype=real_dtype, generator=generator)
        im = torch.randn(shape, device=device, dtype=real_dtype, generator=generator)
        return torch.complex(re, im).to(dtype)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def _pseudo_gaussian_window_2d(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Symmetric 2D tent/radial profile used inside Square Attack L2 square updates."""
    coords = torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    profile = 1.0 - torch.maximum(grid_y.abs(), grid_x.abs()) * 0.75
    norm = safe_l2_norm(profile.reshape(-1), dim=None).clamp(min=1e-12)
    return profile / norm


def _init_square_perturbation(
    x_clean: torch.Tensor,
    eps_budget: torch.Tensor,
    norm: str,
    clip_bounds: tuple[float, float] | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Initialize perturbation on/near the boundary (vertical stripes for 4D images)."""
    b = x_clean.shape[0]
    if x_clean.ndim == 4:
        _, c, _, w = x_clean.shape
        if norm == "linf":
            stripes = _sample_unit_linf(
                (b, c, 1, w), x_clean.dtype, x_clean.device, x_clean.is_complex(), generator
            )
            delta = stripes.expand_as(x_clean) * eps_budget
        else:
            stripes = _sample_l2_noise(
                (b, c, 1, w), x_clean.dtype, x_clean.device, x_clean.is_complex(), generator
            ).expand_as(x_clean)
            s_norm = safe_l2_norm(stripes.reshape(b, -1), dim=1)
            s_norm = s_norm.to(eps_budget.dtype).view_as(eps_budget).clamp(min=1e-12)
            delta = stripes * (eps_budget / s_norm)
        return project_perturbation(delta, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds)

    if norm == "linf":
        delta = (
            _sample_unit_linf(
                x_clean.shape, x_clean.dtype, x_clean.device, x_clean.is_complex(), generator
            )
            * eps_budget
        )
        return project_perturbation(delta, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds)

    return random_init_perturbation(
        x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds, generator=generator
    )


def _propose_candidate(
    delta: torch.Tensor,
    x_clean: torch.Tensor,
    eps_budget: torch.Tensor,
    p: float,
    norm: str,
    clip_bounds: tuple[float, float] | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Propose localized square (4D) or contiguous burst-window (ND) candidate updates."""
    b = delta.shape[0]
    cand = delta.clone()

    if delta.ndim == 4:
        _, c, h, w = delta.shape
        s = max(1, min(h, w, int(round(math.sqrt(p * h * w)))))
        if norm == "linf":
            # Fully vectorized mask update avoids per-sample .item() GPU/MPS barriers.
            h0 = torch.randint(0, h - s + 1, (b, 1, 1, 1), device=delta.device, generator=generator)
            w0 = torch.randint(0, w - s + 1, (b, 1, 1, 1), device=delta.device, generator=generator)
            rows = torch.arange(h, device=delta.device).view(1, 1, h, 1)
            cols = torch.arange(w, device=delta.device).view(1, 1, 1, w)
            mask = (rows >= h0) & (rows < h0 + s) & (cols >= w0) & (cols < w0 + s)
            patches = _sample_unit_linf(
                (b, c, 1, 1), delta.dtype, delta.device, delta.is_complex(), generator
            ) * eps_budget.reshape(b, 1, 1, 1)
            same_in_square = (~mask) | ((cand.real - patches.real).abs() <= 1e-6)
            all_same = same_in_square.reshape(b, -1).all(dim=1).view(b, 1, 1, 1)
            patches = torch.where(all_same, -patches, patches)
            cand = torch.where(mask, patches.expand_as(cand), cand)
            return project_perturbation(cand, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds)

        real_dtype = eps_budget.dtype
        window_2d = _pseudo_gaussian_window_2d(s, delta.device, real_dtype).unsqueeze(0).to(delta.dtype)
        h0s = torch.randint(0, h - s + 1, (b,), device=delta.device, generator=generator).tolist()
        w0s = torch.randint(0, w - s + 1, (b,), device=delta.device, generator=generator).tolist()
        do_erase = s <= min(h, w) // 2
        h2s = torch.randint(0, h - s + 1, (b,), device=delta.device, generator=generator).tolist() if do_erase else []
        w2s = torch.randint(0, w - s + 1, (b,), device=delta.device, generator=generator).tolist() if do_erase else []
        ch_weights = _sample_l2_noise(
            (b, c, 1, 1), delta.dtype, delta.device, delta.is_complex(), generator
        )
        eps_sq_all = eps_budget.reshape(b).pow(2)

        for i in range(b):
            h0_i, w0_i = h0s[i], w0s[i]
            if do_erase:
                cand[i, :, h2s[i] : h2s[i] + s, w2s[i] : w2s[i] + s] = 0
            cand[i, :, h0_i : h0_i + s, w0_i : w0_i + s] = 0
            outside_sq = float((cand[i].abs().pow(2)).sum().item())
            eps_i_sq = float(eps_sq_all[i].item())
            rem_sq = max(eps_i_sq - outside_sq, 0.0)
            if rem_sq <= 1e-12 * max(eps_i_sq, 1e-24):
                rem_sq = eps_i_sq * 0.25
                cand[i] = cand[i] * math.sqrt(0.75)
            bump = ch_weights[i] * window_2d
            bump_norm = float(safe_l2_norm(bump.reshape(-1), dim=None).item())
            cand[i, :, h0_i : h0_i + s, w0_i : w0_i + s] = bump * (math.sqrt(rem_sq) / max(bump_norm, 1e-12))

        return project_perturbation(cand, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds)

    # Arbitrary-rank tensor (1D/2D/3D: wireless IQ frames, tabular, time series, DRL states)
    cand_flat = cand.reshape(b, -1)
    dim = cand_flat.shape[1]
    s = max(1, min(dim, int(round(p * dim))))
    if norm == "linf":
        pos1 = torch.randint(0, dim - s + 1, (b, 1), device=delta.device, generator=generator)
        idx = torch.arange(dim, device=delta.device).view(1, dim)
        mask = (idx >= pos1) & (idx < pos1 + s)
        patches = _sample_unit_linf(
            (b, 1), delta.dtype, delta.device, delta.is_complex(), generator
        ) * eps_budget.reshape(b, 1)
        same_in_win = (~mask) | ((cand_flat.real - patches.real).abs() <= 1e-6)
        all_same = same_in_win.all(dim=1, keepdim=True)
        patches = torch.where(all_same, -patches, patches)
        cand_flat = torch.where(mask, patches.expand_as(cand_flat), cand_flat)
        cand = cand_flat.view_as(delta)
        return project_perturbation(cand, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds)

    pos1s = torch.randint(0, dim - s + 1, (b,), device=delta.device, generator=generator).tolist()
    do_erase = s <= dim // 2
    pos2s = torch.randint(0, dim - s + 1, (b,), device=delta.device, generator=generator).tolist() if do_erase else []
    patches_l2 = _sample_l2_noise(
        (b, s), delta.dtype, delta.device, delta.is_complex(), generator
    )
    eps_sq_all = eps_budget.reshape(b).pow(2)
    for i in range(b):
        pos1_i = pos1s[i]
        if do_erase:
            cand_flat[i, pos2s[i] : pos2s[i] + s] = 0
        cand_flat[i, pos1_i : pos1_i + s] = 0
        outside_sq = float((cand_flat[i].abs().pow(2)).sum().item())
        eps_i_sq = float(eps_sq_all[i].item())
        rem_sq = max(eps_i_sq - outside_sq, 0.0)
        if rem_sq <= 1e-12 * max(eps_i_sq, 1e-24):
            rem_sq = eps_i_sq * 0.25
            cand_flat[i] = cand_flat[i] * math.sqrt(0.75)
        patch = patches_l2[i]
        p_norm = float(safe_l2_norm(patch, dim=None).item())
        cand_flat[i, pos1_i : pos1_i + s] = patch * (math.sqrt(rem_sq) / max(p_norm, 1e-12))

    cand = cand_flat.view_as(delta)
    return project_perturbation(cand, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds)


def square_attack(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.05,
    max_queries: int = 500,
    p_init: float = 0.8,
    norm: str = "linf",
    loss: str | LossFn | None = "margin",
    targeted: bool = False,
    early_stop: bool = True,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run Square Attack score-based black-box random search.

    Args:
        model_or_fn: Callable or ``nn.Module`` returning scores/logits (no gradient needed).
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Ground-truth labels/targets (or target class when ``targeted=True``).
        eps: Perturbation budget (or JSR in dB when ``norm="jsr_db"``).
        max_queries: Maximum number of black-box forward evaluations per sample.
        p_init: Initial fraction of area/elements modified per square/window.
        norm: Constraint norm: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        loss: Built-in loss name (defaults to ``"margin"``) or callable.
        targeted: If ``True``, steers toward target class ``y``.
        early_stop: Stop querying individual classification samples once fooled.
        clip_bounds: Optional valid value range ``(min, max)`` for real-valued inputs.
        seed: Optional random seed for reproducible random search.

    Returns:
        ``AttackResult`` containing adversarial perturbations and query usage stats.
    """
    if max_queries < 1:
        raise ValueError(f"max_queries must be >= 1, got {max_queries}.")
    if not 0.0 < p_init <= 1.0:
        raise ValueError(f"p_init must be in (0, 1], got {p_init}.")

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
            extra={"queries": 0, "mean_queries": 0.0},
        )

    generator = make_generator(seed, x_clean.device)
    can_check_class_stop = early_stop and (loss in (None, "ce", "margin", "dlr"))
    queries_per_sample = torch.zeros(batch_size, device=x_clean.device, dtype=torch.long)
    iters_run = 1

    with eval_mode(model_or_fn), torch.no_grad():
        best_delta = _init_square_perturbation(
            x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds, generator=generator
        )
        outputs_init = model_or_fn(x_clean + best_delta)
        best_obj = outputs_to_objective(outputs_init, y, loss_fn, targeted=targeted)
        queries_per_sample += 1

        if can_check_class_stop and outputs_init.ndim == 2:
            preds_init = outputs_init.argmax(dim=1)
            active = preds_init != y if targeted else preds_init == y
        else:
            active = torch.ones(batch_size, device=x_clean.device, dtype=torch.bool)

        for it in range(1, max_queries):
            if not active.any():
                break
            iters_run = it + 1
            active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)

            p = _p_selection(p_init, it, max_queries)
            cand_delta = _propose_candidate(
                delta=best_delta[active_idx],
                x_clean=x_clean[active_idx],
                eps_budget=eps_budget[active_idx],
                p=p,
                norm=norm,
                clip_bounds=clip_bounds,
                generator=generator,
            )
            outputs_cand = model_or_fn(x_clean[active_idx] + cand_delta)
            cand_obj = outputs_to_objective(
                outputs_cand, y[active_idx], loss_fn, targeted=targeted
            )
            queries_per_sample[active_idx] += 1

            improved = cand_obj > best_obj[active_idx]
            if improved.any():
                imp_idx = active_idx[improved]
                best_obj[imp_idx] = cand_obj[improved]
                best_delta[imp_idx] = cand_delta[improved]

            if can_check_class_stop and outputs_cand.ndim == 2:
                preds_cand = outputs_cand.argmax(dim=1)
                solved = preds_cand == y[active_idx] if targeted else preds_cand != y[active_idx]
                if solved.any():
                    active[active_idx[solved]] = False

    return AttackResult(
        x_adv=x_clean + best_delta,
        perturbation=best_delta,
        eps=eps,
        norm=norm,
        steps_run=iters_run,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "queries": int(queries_per_sample.sum().item()),
            "mean_queries": float(queries_per_sample.float().mean().item()),
            "max_queries": max_queries,
        },
    )
