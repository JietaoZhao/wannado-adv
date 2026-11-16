"""L1-APGD: Mind-the-Box Sparse Adaptive PGD (Croce & Hein, ICML 2021).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (State-of-the-Art Sparse L1 White-Box Attack)
- Reference     : Croce, F., & Hein, M. (2021). "Mind the Box: l1-APGD for
                  Sparse Adversarial Attacks on Image Classifiers." ICML 2021,
                  PMLR 139:2201-2211.
- URL           : https://arxiv.org/abs/2103.01208
- Plain Intuition:
    Naive L1 PGD attacks suffer from two bugs: (1) they project onto the L1
    ball first and clip to the box [0, 1] afterward (losing distortion efficiency),
    and (2) exact steepest L1 descent updates only 1 coordinate per step, which
    gets stuck when features hit box bounds or vary smoothly. ``l1-APGD`` fixes
    both: it projects onto the **exact intersection** ``B_1(x, eps) cap box``
    and dynamically schedules a sparse top-k coordinate ascent mask that concentrates
    updates onto the active feasible coordinates with largest |grad|.
- Diff vs. APGD-CE (Croce & Hein, 2020):
    1. Exact Euclidean projection onto the intersection of the L1 ball and box bounds.
    2. Feasible-box gradient masking with scheduled sparsity ``k`` (from broad
       exploration down to 1-coordinate fine tuning) normalized to unit L1 step.
- Why Chosen as a Baseline:
    The canonical parameter-free benchmark for sparse L1 robustness evaluation.
- Extension Directions / Prompt Hooks:
    1. Swap Cross-Entropy for DLR loss (`loss="dlr"`) for Softmax-saturated models.
    2. Combine with sparse black-box probing for an L1-AutoAttack suite.
===============================================================================
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from wannado.adv.attacks.first_order.apgd_ce_croce2020 import compute_apgd_checkpoints
from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    compute_attack_objective,
    compute_perturbation_norm,
    eval_mode,
    make_generator,
    project_perturbation,
    random_init_perturbation,
    resolve_eps_budget,
    resolve_loss_fn,
)
from wannado.adv.core.losses import LossFn
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="apgd_l1",
    name="L1 Auto-PGD (Mind the Box)",
    family="first_order",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="medium",
    supported_norms=("l1",),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="apgd_ce",
    diff_from_parent="Adapts Auto-PGD to sparse L1 threat models using exact box-intersected L1 projection and sparsity-scheduled steepest coordinate updates.",
    summary_plain="Sparse L1-APGD: computes exact box+L1 projections and top-k steepest coordinate updates with adaptive step-size halving.",
    when_to_use="Whenever evaluating sparse fixed-budget L1 robustness (fewer pixels/bins perturbed strongly rather than dense small noise).",
    when_not_to_use="For dense L2/Linf perturbations or wireless JSR energy budgets (use apgd_ce, apgd_dlr, or sparsefool for unconstrained L0).",
    extension_hooks=(
        "Pass `loss='dlr'` against Softmax-saturated classifiers under L1 budgets",
        "Customize top-k sparsity annealing schedule inside `_l1_steepest_step`",
    ),
    module_path="wannado.adv.attacks.first_order.apgd_l1_croce2021",
    citation=AttackCitation(
        key="croce2021mind",
        title="Mind the Box: l1-APGD for Sparse Adversarial Attacks on Image Classifiers",
        authors="Francesco Croce and Matthias Hein",
        venue="38th International Conference on Machine Learning (ICML)",
        year=2021,
        url="https://arxiv.org/abs/2103.01208",
        bibtex=(
            "@inproceedings{croce2021mind,\n"
            "  title={Mind the Box: $l_1$-APGD for Sparse Adversarial Attacks on Image Classifiers},\n"
            "  author={Croce, Francesco and Hein, Matthias},\n"
            "  booktitle={Proceedings of the 38th International Conference on Machine Learning},\n"
            "  pages={2201--2211},\n"
            "  year={2021},\n"
            "  volume={139},\n"
            "  series={Proceedings of Machine Learning Research},\n"
            "  publisher={PMLR},\n"
            "  url={https://arxiv.org/abs/2103.01208},\n"
            "}"
        ),
        ieee='F. Croce and M. Hein, "Mind the box: l1-APGD for sparse adversarial attacks on image classifiers," in Proc. 38th Int. Conf. Mach. Learn. (ICML), PMLR 139, 2021, pp. 2201-2211.',
        apa="Croce, F., & Hein, M. (2021). Mind the box: l1-APGD for sparse adversarial attacks on image classifiers. Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139, 2201-2211.",
    ),
)


def _l1_steepest_step(
    grad: torch.Tensor,
    x_adv: torch.Tensor,
    top_k: int,
    clip_bounds: tuple[float, float] | None,
) -> torch.Tensor:
    """Compute sparsity-controlled unit-L1 steepest ascent direction inside box bounds."""
    b = grad.shape[0]
    flat_g = grad.reshape(b, -1)
    dim = flat_g.shape[1]

    if clip_bounds is not None and not grad.is_complex():
        low, high = clip_bounds
        flat_x = x_adv.reshape(b, -1)
        # Mask out coordinates already pinned against the box boundary in the gradient direction.
        at_low = (flat_x <= low + 1e-6) & (flat_g < 0)
        at_high = (flat_x >= high - 1e-6) & (flat_g > 0)
        feasible = ~(at_low | at_high)
        flat_g = torch.where(feasible, flat_g, torch.zeros_like(flat_g))

    k = min(max(1, top_k), dim)
    abs_g = flat_g.abs()
    _, idx = torch.topk(abs_g, k=k, dim=1)
    mask = torch.zeros_like(abs_g)
    mask.scatter_(1, idx, 1.0)

    masked_g = flat_g * (mask.to(flat_g.dtype) if flat_g.is_complex() else mask)
    l1_norm = masked_g.abs().sum(dim=1, keepdim=True)
    # Fall back to unconstrained unit-sgn direction if feasible box is locally flat.
    fallback = torch.sgn(flat_g) if flat_g.is_complex() else torch.sign(flat_g)
    fallback_norm = fallback.abs().sum(dim=1, keepdim=True).clamp(min=1e-12)
    dir_flat = torch.where(
        l1_norm > 1e-12,
        masked_g / l1_norm.clamp(min=1e-12),
        fallback / fallback_norm,
    )
    return dir_flat.reshape_as(grad)


def apgd_l1(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 3.0,
    steps: int = 50,
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    random_start: bool = True,
    restarts: int = 1,
    rho: float = 0.75,
    momentum: float = 0.75,
    sparsity_init_frac: float = 0.25,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
    norm: str = "l1",
) -> AttackResult:
    """Run Mind-the-Box L1 Auto-PGD (Croce & Hein, ICML 2021)."""
    if norm != "l1":
        raise ValueError(f"apgd_l1 strictly targets norm='l1', got '{norm}'.")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if restarts < 1:
        raise ValueError(f"restarts must be >= 1, got {restarts}.")
    if not 0.0 < sparsity_init_frac <= 1.0:
        raise ValueError(f"sparsity_init_frac must be in (0, 1], got {sparsity_init_frac}.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    loss_fn = resolve_loss_fn(loss, targeted=targeted)
    eps_budget = resolve_eps_budget(x_clean, eps=eps, norm="l1")

    if (eps_budget == 0).all():
        return AttackResult(
            x_adv=x_clean.clone(),
            perturbation=torch.zeros_like(x_clean),
            eps=eps,
            norm="l1",
            steps_run=0,
            attack_id=META.id,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            extra={"restarts": restarts, "l0_mean": 0.0},
        )

    checkpoints = compute_apgd_checkpoints(steps)
    checkpoint_set = set(checkpoints)
    batch_size = x_clean.shape[0]
    dim = x_clean[0].numel()
    view_shape = (batch_size,) + (1,) * (x_clean.ndim - 1)
    generator = make_generator(seed, x_clean.device)

    global_best_delta = torch.zeros_like(x_clean)
    global_best_obj = torch.full(
        (batch_size,), float("-inf"), device=x_clean.device, dtype=eps_budget.dtype
    )

    k_start = max(1, int(round(sparsity_init_frac * dim)))

    with eval_mode(model_or_fn):
        for _ in range(restarts):
            if random_start:
                delta = random_init_perturbation(
                    x_clean, eps_budget, norm="l1", clip_bounds=clip_bounds, generator=generator
                )
            else:
                delta = torch.zeros_like(x_clean)

            x_in = (x_clean + delta).detach().requires_grad_(True)
            obj_init = compute_attack_objective(
                model_or_fn, x_in, y, loss_fn, targeted=targeted
            )
            (grad,) = torch.autograd.grad(obj_init.sum(), x_in, only_inputs=True)
            obj_prev = obj_init.detach()
            grad = grad.detach()

            best_delta = delta.clone()
            best_obj = obj_prev.clone()
            best_grad = grad.clone()

            delta_old = delta.clone()
            eta = 1.0 * eps_budget.clone()
            prev_cp = 0
            best_obj_prev_cp = best_obj.clone()
            reduced_last_check = torch.zeros(batch_size, dtype=torch.bool, device=x_clean.device)
            improved_count = torch.zeros(batch_size, dtype=torch.long, device=x_clean.device)

            for k in range(steps):
                # Anneal active coordinate count from broad L1 exploration toward ultra-sparse updates.
                frac_left = 1.0 - float(k) / float(max(steps - 1, 1))
                top_k = max(1, int(round(1 + (k_start - 1) * (frac_left**2))))

                # [AI-Prompt Hook]: Exact L1-box steepest coordinate direction + exact box-L1 projection.
                steep = _l1_steepest_step(
                    grad, x_clean + delta, top_k=top_k, clip_bounds=clip_bounds
                )
                z_delta = project_perturbation(
                    delta + eta * steep, x_clean, eps_budget, norm="l1", clip_bounds=clip_bounds
                )
                a = 1.0 if k == 0 else momentum
                delta_next = project_perturbation(
                    delta + a * (z_delta - delta) + (1.0 - a) * (delta - delta_old),
                    x_clean,
                    eps_budget,
                    norm="l1",
                    clip_bounds=clip_bounds,
                )
                delta_old = delta
                delta = delta_next

                is_last = k == steps - 1
                if is_last:
                    with torch.no_grad():
                        obj_new = compute_attack_objective(
                            model_or_fn, x_clean + delta, y, loss_fn, targeted=targeted
                        )
                    grad_new = grad
                else:
                    x_in = (x_clean + delta).detach().requires_grad_(True)
                    obj_new = compute_attack_objective(
                        model_or_fn, x_in, y, loss_fn, targeted=targeted
                    )
                    (grad_new,) = torch.autograd.grad(obj_new.sum(), x_in, only_inputs=True)
                    obj_new = obj_new.detach()
                    grad_new = grad_new.detach()

                improved_count += (obj_new > obj_prev).long()
                obj_prev = obj_new

                improve_best = obj_new > best_obj
                best_obj = torch.where(improve_best, obj_new, best_obj)
                improve_view = improve_best.view(view_shape)
                best_delta = torch.where(improve_view, delta, best_delta)
                best_grad = torch.where(improve_view, grad_new, best_grad)

                step_idx = k + 1
                if step_idx in checkpoint_set and not is_last:
                    window_len = max(step_idx - prev_cp, 1)
                    cond_osc = improved_count.float() < (rho * window_len)
                    cond_stall = (~reduced_last_check) & (best_obj <= best_obj_prev_cp)
                    reduce_mask = cond_osc | cond_stall

                    if reduce_mask.any():
                        reduce_view = reduce_mask.view(view_shape)
                        eta = torch.where(reduce_view, eta * 0.5, eta)
                        delta = torch.where(reduce_view, best_delta, delta)
                        delta_old = torch.where(reduce_view, best_delta, delta_old)
                        grad_new = torch.where(reduce_view, best_grad, grad_new)
                        obj_prev = torch.where(reduce_mask, best_obj, obj_prev)

                    reduced_last_check = reduce_mask
                    prev_cp = step_idx
                    best_obj_prev_cp = best_obj.clone()
                    improved_count.zero_()

                grad = grad_new

            improve_global = best_obj > global_best_obj
            global_best_obj = torch.where(improve_global, best_obj, global_best_obj)
            global_best_delta = torch.where(
                improve_global.view(view_shape), best_delta, global_best_delta
            )

    l0_mean = float(compute_perturbation_norm(global_best_delta, "l0").mean().item())
    return AttackResult(
        x_adv=x_clean + global_best_delta,
        perturbation=global_best_delta,
        eps=eps,
        norm="l1",
        steps_run=steps * restarts,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "restarts": restarts,
            "checkpoints": checkpoints,
            "l0_mean": l0_mean,
        },
    )
