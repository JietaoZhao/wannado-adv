"""AutoAttack: Parameter-Free Multi-Attack Ensemble (Croce & Hein, ICML 2020).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Gold-Standard Parameter-Free Robustness Evaluation Ensemble)
- Reference     : Croce, F., & Hein, M. (2020).
                  "Reliable evaluation of adversarial robustness with an ensemble
                  of diverse parameter-free attacks." ICML 2020, PMLR 119:2206-2216.
- URL           : https://proceedings.mlr.press/v119/croce20b.html
- Plain Intuition:
    Single attacks fail in blind spots: standard CE-PGD fails under Softmax
    saturation; untargeted losses fail when runner-up classes dominate narrowly;
    white-box methods fail when gradients are obfuscated; and boundary attacks
    sometimes struggle with flat loss plateaus. AutoAttack sequentially filters
    only the samples STILL surviving after each stage through four diverse,
    parameter-free attacks:
        1. APGD-CE (untargeted)
        2. APGD-DLR (untargeted + targeted against top runner-up classes)
        3. FAB (minimum-norm geometric projection capped at eps)
        4. Square Attack (gradient-free score-based random search)
- Diff vs. Standalone APGD / FAB / Square:
    Chains all four sequentially on the shrinking subset of un-fooled samples,
    combining scale-invariant white-box, targeted geometric, and gradient-free search.
- Why Chosen as a Baseline:
    The de-facto community standard for reporting reliable adversarial accuracy;
    a model is only as robust as its worst survival rate across the ensemble.
- Extension Directions (for further research):
    1. Pair with ``audit_robustness()`` to diagnose why individual stages failed.
    2. Use ``version="fast"`` (APGD-CE + APGD-DLR + Square) for rapid CI checks.
===============================================================================
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from wannado.adv.attacks.blackbox_score.square_andriushchenko2020 import square_attack
from wannado.adv.attacks.first_order.apgd_ce_croce2020 import apgd_ce
from wannado.adv.attacks.first_order.apgd_dlr_croce2020 import apgd_dlr
from wannado.adv.attacks.minimum_norm.fab_croce2020 import fab
from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    eval_mode,
    resolve_eps_budget,
)
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="autoattack",
    name="AutoAttack Ensemble",
    family="ensemble",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="high",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    parent_id="apgd_dlr",
    diff_from_parent="Sequentially chains APGD-CE, APGD-DLR (untargeted + targeted), FAB, and Square Attack on surviving samples for parameter-free worst-case robustness.",
    summary_plain="Sequential ensemble combining APGD-CE, APGD-DLR, FAB, and Square Attack to prevent single-attack blind spots.",
    when_to_use="Final robustness benchmarking of any classifier when reliable worst-case accuracy matters across white-box and black-box regimes.",
    when_not_to_use="For fast inner-loop adversarial training steps or non-classification regression losses (use PGD/Adam-PGD).",
    extension_hooks=(
        "Adjust apgd_steps / fab_steps / square_queries for heavy full-benchmark runs",
        "Use version='fast' to skip FAB when rapid screening is preferred",
    ),
    module_path="wannado.adv.attacks.ensemble.autoattack_croce2020",
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
        ieee='F. Croce and M. Hein, "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 119, 2020, pp. 2206-2216.',
        apa="Croce, F., & Hein, M. (2020). Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. Proceedings of the 37th International Conference on Machine Learning (ICML), PMLR 119, 2206-2216.",
    ),
)


def _run_whitebox_stage(
    stage_name: str,
    fn: Callable[..., AttackResult],
    notes: list[str],
    *args: Any,
    **kwargs: Any,
) -> AttackResult | None:
    """Execute a white-box AutoAttack stage, skipping cleanly only if autograd is disconnected."""
    try:
        return fn(*args, **kwargs)
    except RuntimeError as exc:
        msg = str(exc)
        if "does not require grad" in msg or "grad_fn" in msg:
            notes.append(f"{stage_name} skipped due to disconnected autograd")
            return None
        raise


def _apply_stage_result(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_adv: torch.Tensor,
    y: torch.Tensor,
    surviving: torch.Tensor,
    stage_x_adv: torch.Tensor,
) -> torch.Tensor:
    """Update ``x_adv`` for surviving samples fooled by the latest stage."""
    surv_idx = torch.nonzero(surviving, as_tuple=False).squeeze(1)
    with torch.no_grad():
        preds = model_or_fn(stage_x_adv).argmax(dim=1)
    fooled = preds != y[surv_idx]
    if fooled.any():
        fooled_idx = surv_idx[fooled]
        x_adv[fooled_idx] = stage_x_adv[fooled]
        surviving[fooled_idx] = False
    return surviving


def autoattack(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    norm: str = "linf",
    version: str = "standard",
    apgd_steps: int = 50,
    apgd_restarts: int = 2,
    n_target_classes: int = 3,
    fab_steps: int = 25,
    square_queries: int = 500,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run AutoAttack sequential ensemble on surviving un-fooled samples.

    Args:
        model_or_fn: Classifier ``nn.Module`` or callable mapping ``x -> (B, C)`` logits.
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Ground-truth labels (shape ``(B,)``).
        eps: Perturbation budget (or JSR in dB when ``norm="jsr_db"``).
        norm: Constraint norm: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        version: ``"standard"`` (APGD-CE, APGD-DLR, APGD-T, FAB, Square) or
            ``"fast"`` (APGD-CE, APGD-DLR, Square).
        apgd_steps: Iteration budget for APGD stages.
        apgd_restarts: Restarts for untargeted APGD stages.
        n_target_classes: Max runner-up target classes tried by targeted APGD-T when ``C >= 3``.
        fab_steps: Iteration budget for FAB stage in ``version="standard"``.
        square_queries: Query budget for Square Attack black-box fallback stage.
        clip_bounds: Optional valid value range ``(min, max)`` for real-valued inputs.
        seed: Optional base random seed propagated across stages.

    Returns:
        ``AttackResult`` containing worst-case ensemble ``x_adv`` and per-stage accuracy trace.
    """
    if version not in ("standard", "fast"):
        raise ValueError(f"Unsupported version '{version}'. Expected 'standard' or 'fast'.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    batch_size = x_clean.shape[0]
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
            extra={"version": version, "stage_accuracies": {}},
        )

    x_adv = x_clean.clone()
    stage_accuracies: dict[str, float] = {}
    autograd_notes: list[str] = []
    total_steps = 0

    with eval_mode(model_or_fn):
        with torch.no_grad():
            clean_logits = model_or_fn(x_clean)
            if clean_logits.ndim != 2:
                raise ValueError(
                    f"autoattack requires 2D classification logits (B, C), got shape {tuple(clean_logits.shape)}."
                )
            num_classes = clean_logits.shape[1]
            surviving = clean_logits.argmax(dim=1) == y
        stage_accuracies["clean"] = float(surviving.float().mean().item())

        # Stage 1: APGD-CE (untargeted)
        if surviving.any():
            idx = torch.nonzero(surviving, as_tuple=False).squeeze(1)
            res = _run_whitebox_stage(
                "apgd_ce",
                apgd_ce,
                autograd_notes,
                model_or_fn,
                x_clean[idx],
                y[idx],
                eps=eps,
                steps=apgd_steps,
                restarts=apgd_restarts,
                norm=norm,
                clip_bounds=clip_bounds,
                seed=seed,
            )
            if res is not None:
                total_steps += res.steps_run
                surviving = _apply_stage_result(model_or_fn, x_adv, y, surviving, res.x_adv)
        stage_accuracies["apgd_ce"] = float(surviving.float().mean().item())

        # Stage 2: APGD-DLR (untargeted; resistant to Softmax saturation)
        if surviving.any():
            idx = torch.nonzero(surviving, as_tuple=False).squeeze(1)
            res = _run_whitebox_stage(
                "apgd_dlr",
                apgd_dlr,
                autograd_notes,
                model_or_fn,
                x_clean[idx],
                y[idx],
                eps=eps,
                steps=apgd_steps,
                restarts=apgd_restarts,
                norm=norm,
                clip_bounds=clip_bounds,
                seed=None if seed is None else seed + 1,
            )
            if res is not None:
                total_steps += res.steps_run
                surviving = _apply_stage_result(model_or_fn, x_adv, y, surviving, res.x_adv)
        stage_accuracies["apgd_dlr"] = float(surviving.float().mean().item())

        # Stage 2b: Targeted APGD-DLR across top runner-up classes when multi-class
        if num_classes >= 3 and n_target_classes > 0 and surviving.any() and not autograd_notes:
            with torch.no_grad():
                logits_rem = model_or_fn(x_clean)
                sorted_cls = logits_rem.argsort(dim=1, descending=True)
            k_targets = min(n_target_classes, num_classes - 1)
            for t_rank in range(1, k_targets + 1):
                if not surviving.any():
                    break
                idx = torch.nonzero(surviving, as_tuple=False).squeeze(1)
                rank_targets = sorted_cls[idx, t_rank]
                # Ensure target class != true y for each item
                fallback = (y[idx] + t_rank) % num_classes
                y_target = torch.where(rank_targets != y[idx], rank_targets, fallback)
                res = _run_whitebox_stage(
                    f"apgd_t_{t_rank}",
                    apgd_dlr,
                    autograd_notes,
                    model_or_fn,
                    x_clean[idx],
                    y_target,
                    eps=eps,
                    steps=max(20, apgd_steps // 2),
                    restarts=1,
                    norm=norm,
                    targeted=True,
                    clip_bounds=clip_bounds,
                    seed=None if seed is None else seed + 10 + t_rank,
                )
                if res is not None:
                    total_steps += res.steps_run
                    surviving = _apply_stage_result(model_or_fn, x_adv, y, surviving, res.x_adv)
            stage_accuracies["apgd_t"] = float(surviving.float().mean().item())

        # Stage 3: FAB (geometric boundary fallback inside eps budget)
        if version == "standard" and surviving.any() and not autograd_notes:
            idx = torch.nonzero(surviving, as_tuple=False).squeeze(1)
            res = _run_whitebox_stage(
                "fab",
                fab,
                autograd_notes,
                model_or_fn,
                x_clean[idx],
                y[idx],
                eps=eps,
                steps=fab_steps,
                restarts=1,
                norm=norm,
                clip_bounds=clip_bounds,
                seed=None if seed is None else seed + 100,
            )
            if res is not None:
                total_steps += res.steps_run
                surviving = _apply_stage_result(model_or_fn, x_adv, y, surviving, res.x_adv)
        if version == "standard":
            stage_accuracies["fab"] = float(surviving.float().mean().item())

        # Stage 4: Square Attack (gradient-free score-based fallback)
        if surviving.any():
            idx = torch.nonzero(surviving, as_tuple=False).squeeze(1)
            res = square_attack(
                model_or_fn,
                x_clean[idx],
                y[idx],
                eps=eps,
                max_queries=square_queries,
                norm=norm,
                clip_bounds=clip_bounds,
                seed=None if seed is None else seed + 200,
            )
            total_steps += res.steps_run
            surviving = _apply_stage_result(model_or_fn, x_adv, y, surviving, res.x_adv)
        stage_accuracies["square_attack"] = float(surviving.float().mean().item())

    return AttackResult(
        x_adv=x_adv,
        perturbation=x_adv - x_clean,
        eps=eps,
        norm=norm,
        steps_run=total_steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "version": version,
            "robust_accuracy": float(surviving.float().mean().item()),
            "stage_accuracies": stage_accuracies,
            "autograd_notes": autograd_notes,
            "batch_size": batch_size,
        },
    )
