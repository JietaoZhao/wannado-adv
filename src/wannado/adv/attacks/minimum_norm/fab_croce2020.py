"""FAB: Fast Adaptive Boundary Attack (Croce & Hein, ICML 2020).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Current Minimum-Norm Geometric Boundary SOTA)
- Reference     : Croce, F., & Hein, M. (2020). "Minimally distorted Adversarial
                  Examples with a Fast Adaptive Boundary Attack." ICML 2020
                  (PMLR 119:2196-2205).
- URL           : https://proceedings.mlr.press/v119/croce20a.html
- Plain Intuition:
    DeepFool steps onto the closest decision boundary, which can drift away
    from ``x_clean`` along a curved surface. FAB fixes this by projecting BOTH
    the current point ``x_curr`` AND the original point ``x_clean`` onto the
    linearized boundary, taking a biased boundary-tracking step capped by
    ``alpha_max``, then executing a backward step ``beta = 0.9`` toward
    ``x_clean`` so the iterate tightly hugs the minimal-distortion boundary.
- Diff vs. DeepFool (2016):
    1. Biased step mixing projections from ``x_curr`` and ``x_clean`` (capped by ``alpha_max``).
    2. Backward contraction step ``beta = 0.9`` pulling iterates back toward ``x_clean``.
    3. Multi-restart exploration initialized near the best known distortion radius.
- Why Chosen as a Baseline:
    One of the 4 pillars of AutoAttack (ICML 2020); finds tighter minimal L2 and
    Linf adversarial examples than DeepFool and avoids gradient-scaling traps.
- Extension Directions (for further research):
    1. Use FAB-T (targeted FAB) across top candidate classes on many-class tasks.
    2. Pair with APGD-CE, APGD-DLR, and Square Attack -> full AutoAttack suite.
===============================================================================
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    compute_perturbation_norm,
    eval_mode,
    random_init_perturbation,
    safe_l2_norm,
    step_direction,
)
from wannado.adv.core.projections import within_eps_budget
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="fab",
    name="Fast Adaptive Boundary Attack (FAB)",
    family="minimum_norm",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="medium",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    parent_id="deepfool",
    diff_from_parent="Improves DeepFool with biased forward projections toward x_clean (alpha_max), a backward contraction step (beta=0.9), and scaled restarts.",
    summary_plain="Geometric boundary SOTA that tracks the decision boundary while contracting back toward x_clean for minimal L2/Linf distortion.",
    when_to_use="When measuring exact minimal Linf/L2 robustness margins or building an AutoAttack-grade evaluation suite.",
    when_not_to_use="For non-classification regression tasks without logits (use PGD/Adam-PGD with custom loss).",
    extension_hooks=(
        "Combine with APGD-CE, APGD-DLR, and Square Attack -> AutoAttack",
        "Run targeted FAB over top-K runner-up classes on many-class tasks",
    ),
    module_path="wannado.adv.attacks.minimum_norm.fab_croce2020",
    citation=AttackCitation(
        key="croce2020minimally",
        title="Minimally distorted Adversarial Examples with a Fast Adaptive Boundary Attack",
        authors="Francesco Croce and Matthias Hein",
        venue="37th International Conference on Machine Learning (ICML)",
        year=2020,
        url="https://proceedings.mlr.press/v119/croce20a.html",
        bibtex=(
            "@inproceedings{croce2020minimally,\n"
            "  title={Minimally distorted Adversarial Examples with a Fast Adaptive Boundary Attack},\n"
            "  author={Croce, Francesco and Hein, Matthias},\n"
            "  booktitle={Proceedings of the 37th International Conference on Machine Learning},\n"
            "  pages={2196--2205},\n"
            "  year={2020},\n"
            "  volume={119},\n"
            "  series={Proceedings of Machine Learning Research},\n"
            "  publisher={PMLR},\n"
            "  url={https://proceedings.mlr.press/v119/croce20a.html},\n"
            "}"
        ),
        ieee='F. Croce and M. Hein, "Minimally distorted adversarial examples with a fast adaptive boundary attack," in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR, 2020, pp. 2196-2205.',
        apa="Croce, F., & Hein, M. (2020). Minimally distorted adversarial examples with a fast adaptive boundary attack. Proceedings of the 37th International Conference on Machine Learning (ICML), 2196-2205.",
    ),
)


def _norm_p(x: torch.Tensor, norm: str) -> float:
    if norm == "linf":
        return float(x.abs().max().item())
    return float(safe_l2_norm(x.reshape(-1), dim=None).item())


def _project_hyperplane_box(
    x_ref: torch.Tensor,
    w: torch.Tensor,
    h_val: float,
    norm: str,
    clip_bounds: tuple[float, float] | None,
) -> torch.Tensor:
    """Compute minimal-norm displacement ``d`` from ``x_ref`` to ``h_val + <w, d> = 0``."""
    if clip_bounds is None:
        if norm == "linf":
            w_l1 = float(w.abs().sum().item())
            if w_l1 <= 1e-12:
                return torch.zeros_like(x_ref)
            sign_factor = -1.0 if h_val >= 0 else 1.0
            return (abs(h_val) / w_l1) * sign_factor * step_direction(w, norm="linf")
        w_l2_sq = float((w.abs().pow(2)).sum().item())
        if w_l2_sq <= 1e-12:
            return torch.zeros_like(x_ref)
        return (-h_val / w_l2_sq) * w

    # Real tensor with box constraints [low, high]: solve via exact 1D monotonic bisection.
    low, high = clip_bounds
    d_low = low - x_ref
    d_high = high - x_ref
    sign_h = -1.0 if h_val >= 0 else 1.0
    # Target linear functional value: <w, d> = -h_val => sign_h * <w, d> = |h_val|.
    target = abs(h_val)
    if target <= 1e-12:
        return torch.zeros_like(x_ref)

    if norm == "linf":
        direction = sign_h * torch.sign(w)
        r_hi = max(float(d_low.abs().max().item()), float(d_high.abs().max().item()), 1e-6)
        candidate_fn = lambda r: torch.clamp(r * direction, min=d_low, max=d_high)
    else:
        w_l2_sq = float(w.pow(2).sum().item())
        if w_l2_sq <= 1e-12:
            return torch.zeros_like(x_ref)
        direction = sign_h * w
        r_max_box = float(torch.maximum(d_low.abs(), d_high.abs()).max().item())
        w_min_nz = float(w[w.abs() > 1e-12].abs().min().item()) if (w.abs() > 1e-12).any() else 1.0
        r_hi = max(r_max_box / max(w_min_nz, 1e-6), target / w_l2_sq)
        candidate_fn = lambda r: torch.clamp(r * direction, min=d_low, max=d_high)

    max_achievable = float((sign_h * (w * candidate_fn(r_hi)).sum()).item())
    if max_achievable <= target:
        return candidate_fn(r_hi)

    r_lo = 0.0
    for _ in range(30):
        r_mid = 0.5 * (r_lo + r_hi)
        val = float((sign_h * (w * candidate_fn(r_mid)).sum()).item())
        if val < target:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return candidate_fn(r_hi)


def _fab_single_sample(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_single: torch.Tensor,
    y_orig: int,
    norm: str,
    steps: int,
    restarts: int,
    alpha_max: float,
    beta: float,
    eta: float,
    max_classes: int,
    targeted: bool,
    target_label: int | None,
    clip_bounds: tuple[float, float] | None,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, int]:
    """Run Algorithm 1 of Croce & Hein (ICML 2020) on a single sample ``(1, ...)``."""
    best_adv = x_single.clone()
    best_dist = float("inf")
    total_steps = 0

    with torch.no_grad():
        logits_clean = model_or_fn(x_single)
        num_classes = logits_clean.shape[1]
        sorted_classes = logits_clean.argsort(dim=1, descending=True)[0]
        if targeted:
            candidate_classes = [target_label]  # type: ignore[list-item]
        else:
            candidate_classes = [
                int(c.item()) for c in sorted_classes if int(c.item()) != y_orig
            ][: max(1, min(num_classes - 1, max_classes))]

    for r_idx in range(restarts):
        if r_idx == 0:
            x_curr = x_single.clone()
        else:
            # Restart inside a ball around x_clean scaled to 0.5 * best_dist (or small probe).
            probe_radius = 0.5 * best_dist if best_dist < float("inf") else 0.1
            eps_probe = torch.full(
                (1,) + (1,) * (x_single.ndim - 1),
                probe_radius,
                device=x_single.device,
                dtype=x_single.real.dtype if x_single.is_complex() else x_single.dtype,
            )
            proj_norm = "linf" if norm == "linf" else "l2"
            x_curr = x_single + random_init_perturbation(
                x_single, eps_probe, norm=proj_norm, clip_bounds=clip_bounds, generator=generator
            )

        for _ in range(steps):
            total_steps += 1
            x_in = x_curr.detach().requires_grad_(True)
            logits = model_or_fn(x_in)
            pred = int(logits.argmax(dim=1)[0].item())

            is_fooled = (pred == target_label) if targeted else (pred != y_orig)
            if is_fooled:
                cur_dist = _norm_p(x_curr - x_single, norm=norm)
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_adv = x_curr.detach().clone()

            # Compute closest linearized boundary hyperplane among candidate classes.
            if targeted:
                other_classes = [c for c in range(logits.shape[1]) if c != target_label]
                ref_cls = (
                    pred
                    if pred != target_label
                    else max(other_classes, key=lambda c: float(logits[0, c].item()))
                )
            else:
                ref_cls = y_orig

            (grad_orig,) = torch.autograd.grad(logits[0, ref_cls], x_in, retain_graph=True)
            grad_orig = grad_orig.detach()

            min_hp_dist = float("inf")
            best_w = None
            best_b = 0.0

            for idx, k in enumerate(candidate_classes):
                retain = idx < len(candidate_classes) - 1
                (grad_k,) = torch.autograd.grad(logits[0, k], x_in, retain_graph=retain)
                w_k = grad_k.detach() - grad_orig
                b_k = float((logits[0, k] - logits[0, ref_cls]).detach().item())

                if norm == "linf":
                    dual_norm = float(w_k.abs().sum().item())
                else:
                    dual_norm = float(safe_l2_norm(w_k.reshape(-1), dim=None).item())
                if dual_norm <= 1e-12:
                    continue

                hp_dist = abs(b_k) / dual_norm
                if hp_dist < min_hp_dist:
                    min_hp_dist = hp_dist
                    best_w = w_k
                    best_b = b_k

            if best_w is None:
                break

            # Project both x_curr and x_clean onto the linearized boundary hyperplane.
            d_curr = _project_hyperplane_box(x_curr, best_w, best_b, norm=norm, clip_bounds=clip_bounds)
            diff_clean = x_single - x_curr
            inner_clean = float(
                (best_w.conj() * diff_clean).real.sum().item()
                if best_w.is_complex()
                else (best_w * diff_clean).sum().item()
            )
            h_clean = best_b + inner_clean
            d_orig = _project_hyperplane_box(x_single, best_w, h_clean, norm=norm, clip_bounds=clip_bounds)

            n1 = _norm_p(d_curr, norm=norm)
            n2 = _norm_p(d_orig, norm=norm)
            alpha = min(n1 / (n1 + n2 + 1e-12), alpha_max)

            x_forward = (1.0 - alpha) * (x_curr + eta * d_curr) + alpha * (x_single + eta * d_orig)
            if is_fooled:
                # Backward contraction step only once across the boundary to tighten distortion.
                x_curr = (1.0 - beta) * x_single + beta * x_forward
            else:
                x_curr = x_forward
            if clip_bounds is not None:
                x_curr = torch.clamp(x_curr, min=clip_bounds[0], max=clip_bounds[1])

        with torch.no_grad():
            pred_end = int(model_or_fn(x_curr).argmax(dim=1)[0].item())
            is_fooled = (pred_end == target_label) if targeted else (pred_end != y_orig)
            if is_fooled:
                cur_dist = _norm_p(x_curr - x_single, norm=norm)
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_adv = x_curr.detach().clone()

    return best_adv, total_steps


def fab(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    eps: float | None = None,
    steps: int = 25,
    restarts: int = 1,
    alpha_max: float = 0.1,
    beta: float = 0.9,
    eta: float = 1.05,
    max_classes: int = 10,
    norm: str = "l2",
    targeted: bool = False,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run Fast Adaptive Boundary (FAB) attack for minimal L2 or Linf distortion.

    Args:
        model_or_fn: Differentiable ``nn.Module`` or callable mapping ``x -> (B, C)`` logits.
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Ground-truth labels (or target class when ``targeted=True``). Optional when untargeted.
        eps: Optional maximum budget cutoff. Perturbations exceeding ``eps_budget`` revert to ``x``.
        steps: FAB iterations per restart.
        restarts: Number of restarts (restart 0 starts at ``x``; subsequent restarts probe near best norm).
        alpha_max: Cap on biased projection mixing coefficient ``alpha`` (paper default ``0.1``).
        beta: Backward contraction factor toward ``x_clean`` (paper default ``0.9``).
        eta: Boundary overshoot factor (paper default ``1.05``).
        max_classes: Number of top alternative classes evaluated per step.
        norm: Distortion geometry: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        targeted: If ``True``, searches minimal perturbation reaching target label ``y``.
        clip_bounds: Optional valid range ``(min, max)`` for real-valued inputs.
        seed: Optional random seed for restarts ``> 1``.

    Returns:
        ``AttackResult`` containing minimal boundary adversarial examples and distortion stats.
    """
    if steps < 1 or restarts < 1:
        raise ValueError("steps and restarts must both be >= 1.")
    if clip_bounds is not None and x.is_complex():
        raise ValueError("clip_bounds is only supported for real-valued tensors.")
    if targeted and y is None:
        raise ValueError("y (target labels) must be provided when targeted=True.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    batch_size = x_clean.shape[0]
    x_adv = x_clean.clone()
    total_steps = 0

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=x_clean.device)
        generator.manual_seed(seed)

    with eval_mode(model_or_fn):
        with torch.no_grad():
            clean_preds = model_or_fn(x_clean).argmax(dim=1)

        for idx in range(batch_size):
            pred_i = int(clean_preds[idx].item())
            if targeted:
                target_i = int(y[idx].item())  # type: ignore[index]
                if pred_i == target_i:
                    continue
                y_orig_i = pred_i
            else:
                if y is not None and pred_i != int(y[idx].item()):
                    continue
                y_orig_i = int(y[idx].item()) if y is not None else pred_i
                target_i = None

            adv_i, s_i = _fab_single_sample(
                model_or_fn=model_or_fn,
                x_single=x_clean[idx : idx + 1],
                y_orig=y_orig_i,
                norm=norm,
                steps=steps,
                restarts=restarts,
                alpha_max=alpha_max,
                beta=beta,
                eta=eta,
                max_classes=max_classes,
                targeted=targeted,
                target_label=target_i,
                clip_bounds=clip_bounds,
                generator=generator,
            )
            x_adv[idx] = adv_i[0]
            total_steps += s_i

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
        steps_run=total_steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "mean_distortion": float(dist_norm.mean().item()),
            "median_distortion": float(dist_norm.median().item()),
            "alpha_max": alpha_max,
            "beta": beta,
            "eta": eta,
            "restarts": restarts,
        },
    )
