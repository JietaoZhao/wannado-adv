"""APGD-CE: Auto-PGD with Cross-Entropy Loss (Croce & Hein, ICML 2020).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Current White-Box Parameter-Free SOTA Benchmark)
- Reference     : Croce, F., & Hein, M. (2020). "Reliable evaluation of
                  adversarial robustness with an ensemble of diverse
                  parameter-free attacks." ICML 2020 (PMLR 119:2206-2216).
- URL           : https://proceedings.mlr.press/v119/croce20b.html
- Plain Intuition:
    Hand-tuned fixed step sizes often overshoot late in PGD or stall when
    gradients fluctuate. APGD removes step-size guesswork: it starts with an
    exploratory large step ``2 * eps``, mixes momentum ``0.75``, and monitors
    progress at scheduled checkpoints ``w_j``. Whenever fewer than ``rho=75%``
    of recent steps improved the objective—or overall progress stalled—it halves
    that sample's step size and teleports back to its best point found so far.
- Diff vs. PGD (2018):
    1. Momentum-blended projected updates (weight ``a = 0.75`` after step 0).
    2. Per-sample adaptive step-size halving at checkpoints ``w_j`` with rollback
       to ``best_delta`` (zero manual step-size tuning needed).
- Why Chosen as a Baseline:
    Part of AutoAttack (ICML 2020), the universal gold standard for reliable
    white-box robustness evaluation without false security from bad step sizes.
- Extension Directions (for further research):
    1. Swap Cross-Entropy for Difference of Logits Ratio -> APGD-DLR.
    2. Pair with FAB and black-box Square Attack -> full AutoAttack suite.
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
    project_perturbation,
    random_init_perturbation,
    resolve_eps_budget,
    resolve_loss_fn,
    step_direction,
)
from wannado.adv.core.losses import LossFn
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="apgd_ce",
    name="Auto-PGD Cross-Entropy (APGD-CE)",
    family="first_order",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="medium",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="pgd",
    diff_from_parent="Adds parameter-free adaptive step-size halving at checkpoints w_j, rollback to best iterate, and momentum blending (alpha=0.75).",
    summary_plain="Parameter-free strongest white-box PGD: automatically halves step sizes and rolls back to the best perturbation when progress stalls.",
    when_to_use="Whenever you want maximum white-box attack strength without manually tuning step_size.",
    when_not_to_use="If your model rescales logits so Softmax saturates to 0/1 (pair with or switch to apgd_dlr).",
    extension_hooks=(
        "Switch loss='dlr' to defeat Softmax gradient saturation -> APGD-DLR",
        "Add stochastic environment expectations inside objective -> EoT-APGD",
    ),
    module_path="wannado.adv.attacks.first_order.apgd_ce_croce2020",
    citation=AttackCitation(
        key="croce2020reliable",
        title="Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks",
        authors="Francesco Croce and Matthias Hein",
        venue="37th International Conference on Machine Learning (ICML)",
        year=2020,
        url="https://proceedings.mlr.press/v119/croce20b.html",
        bibtex=(
            "@inproceedings{croce2020reliable,\n"
            "  title={Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks},\n"
            "  author={Croce, Francesco and Hein, Matthias},\n"
            "  booktitle={Proceedings of the 37th International Conference on Machine Learning},\n"
            "  pages={2206--2216},\n"
            "  year={2020},\n"
            "  volume={119},\n"
            "  series={Proceedings of Machine Learning Research},\n"
            "  publisher={PMLR},\n"
            "  url={https://proceedings.mlr.press/v119/croce20b.html},\n"
            "}"
        ),
        ieee='F. Croce and M. Hein, "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks," in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR, 2020, pp. 2206-2216.',
        apa="Croce, F., & Hein, M. (2020). Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. Proceedings of the 37th International Conference on Machine Learning (ICML), 2206-2216.",
    ),
)


def compute_apgd_checkpoints(steps: int) -> list[int]:
    """Compute Croce & Hein (2020) step-size check-points w_j in 1..steps."""
    if steps < 2:
        return []
    p_prev, p_curr = 0.0, 0.22
    checkpoints: list[int] = []
    while p_curr < 1.0:
        cp = int(p_curr * steps)
        if 0 < cp < steps and (not checkpoints or cp > checkpoints[-1]):
            checkpoints.append(cp)
        p_next = p_curr + max(p_curr - p_prev - 0.03, 0.06)
        p_prev, p_curr = p_curr, p_next
    if not checkpoints or checkpoints[-1] < steps:
        checkpoints.append(steps)
    return checkpoints


def apgd(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    steps: int = 50,
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    random_start: bool = True,
    restarts: int = 1,
    rho: float = 0.75,
    momentum: float = 0.75,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
    attack_id: str = META.id,
) -> AttackResult:
    """Core Auto-PGD Algorithm 1 (Croce & Hein, ICML 2020) parameterized by loss."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
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
            attack_id=attack_id,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            extra={"restarts": restarts, "rho": rho, "momentum": momentum},
        )

    checkpoints = compute_apgd_checkpoints(steps)
    checkpoint_set = set(checkpoints)
    batch_size = x_clean.shape[0]
    view_shape = (batch_size,) + (1,) * (x_clean.ndim - 1)

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=x_clean.device)
        generator.manual_seed(seed)

    global_best_delta = torch.zeros_like(x_clean)
    global_best_obj = torch.full(
        (batch_size,), float("-inf"), device=x_clean.device, dtype=eps_budget.dtype
    )

    with eval_mode(model_or_fn):
        for _ in range(restarts):
            if random_start:
                delta = random_init_perturbation(
                    x_clean,
                    eps_budget,
                    norm=norm,
                    clip_bounds=clip_bounds,
                    generator=generator,
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
            eta = 2.0 * eps_budget.clone()
            prev_cp = 0
            best_obj_prev_cp = best_obj.clone()
            reduced_last_check = torch.zeros(batch_size, dtype=torch.bool, device=x_clean.device)
            improved_count = torch.zeros(batch_size, dtype=torch.long, device=x_clean.device)

            for k in range(steps):
                z_delta = project_perturbation(
                    delta + eta * step_direction(grad, norm=norm),
                    x_clean,
                    eps_budget,
                    norm=norm,
                    clip_bounds=clip_bounds,
                )
                # Blend projected step with previous displacement (a=1.0 on step 0, momentum=0.75 after).
                a = 1.0 if k == 0 else momentum
                delta_next = project_perturbation(
                    delta + a * (z_delta - delta) + (1.0 - a) * (delta - delta_old),
                    x_clean,
                    eps_budget,
                    norm=norm,
                    clip_bounds=clip_bounds,
                )
                delta_old = delta
                delta = delta_next

                is_last_step = k == steps - 1
                if is_last_step:
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
                if step_idx in checkpoint_set and not is_last_step:
                    window_len = max(step_idx - prev_cp, 1)
                    cond_oscillation = improved_count.float() < (rho * window_len)
                    cond_stalled = (~reduced_last_check) & (best_obj <= best_obj_prev_cp)
                    reduce_mask = cond_oscillation | cond_stalled

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
            global_view = improve_global.view(view_shape)
            global_best_delta = torch.where(global_view, best_delta, global_best_delta)

    return AttackResult(
        x_adv=x_clean + global_best_delta,
        perturbation=global_best_delta,
        eps=eps,
        norm=norm,
        steps_run=steps * restarts,
        attack_id=attack_id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "restarts": restarts,
            "steps_per_restart": steps,
            "checkpoints": checkpoints,
            "rho": rho,
            "momentum": momentum,
        },
    )


def apgd_ce(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    steps: int = 50,
    norm: str = "linf",
    targeted: bool = False,
    random_start: bool = True,
    restarts: int = 1,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run Auto-PGD with Cross-Entropy loss (APGD-CE)."""
    return apgd(
        model_or_fn=model_or_fn,
        x=x,
        y=y,
        eps=eps,
        steps=steps,
        norm=norm,
        loss="ce",
        targeted=targeted,
        random_start=random_start,
        restarts=restarts,
        clip_bounds=clip_bounds,
        seed=seed,
        attack_id=META.id,
    )
