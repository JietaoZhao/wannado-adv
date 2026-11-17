"""UAP: Universal Adversarial Perturbations (Moosavi-Dezfooli et al., CVPR 2017).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Canonical Input-Agnostic Universal Perturbation Benchmark)
- Reference     : Moosavi-Dezfooli, S.-M., Fawzi, A., Fawzi, O., & Frossard, P. (2017).
                  "Universal Adversarial Perturbations." CVPR 2017 (pp. 1765-1773).
- URL           : https://arxiv.org/abs/1610.08401
- Plain Intuition:
    All per-sample attacks craft a bespoke perturbation for EACH individual input.
    UAP discovers ONE single, input-agnostic perturbation vector `v` that, when
    added to almost ANY input `x_i` from the distribution, causes misclassification.
    It iterates over a batch of samples: whenever `x_i + v` is still classified
    correctly, it calls DeepFool to find the smallest push across the decision
    boundary, adds it into `v`, and projects `v` back onto the global budget ball.
- Diff vs. DeepFool (2016):
    1. Aggregates minimal DeepFool boundary pushes across samples into ONE shared `v`.
    2. Projects the cumulative vector back onto a fixed global `eps` constraint ball.
    3. Measures empirical fooling rate across the evaluation batch per epoch.
- Why Chosen as a Baseline:
    The classic benchmark for black-box/streaming real-time threats where the
    attacker cannot optimize a fresh perturbation for every incoming frame or packet
    (e.g., continuous wireless jamming or physical universal patches).
- Extension Directions (for further research):
    1. Optimize universal `v` over random shifts/channels with mini-batch Adam/EoT.
    2. Train a neural generator G(z) -> universal/conditional jammer (GAP).
===============================================================================
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from wannado.adv.attacks.minimum_norm.deepfool_moosavi2016 import _deepfool_single_sample
from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    clamp_to_bounds,
    eval_mode,
    make_generator,
    predict_labels,
    project_perturbation,
    resolve_eps_budget,
)
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="uap",
    name="Universal Adversarial Perturbation (UAP)",
    family="universal",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="high",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    parent_id="deepfool",
    diff_from_parent="Aggregates per-sample minimal DeepFool boundary steps across a dataset into a single input-agnostic perturbation vector projected onto an eps ball.",
    summary_plain="Learns ONE single input-agnostic perturbation vector that fools a model across almost an entire batch simultaneously.",
    when_to_use="When evaluating streaming or real-time vulnerabilities (e.g., always-on wireless jammer or universal patch) where per-sample optimization is impossible.",
    when_not_to_use="When sample-specific worst-case robustness is needed (use APGD or FAB), or for continuous regression without class decision boundaries.",
    extension_hooks=(
        "Replace sequential DeepFool aggregation with mini-batch EoT-SGD/Adam optimization",
        "Condition universal vector on class labels for targeted UAP",
    ),
    module_path="wannado.adv.attacks.universal.uap_moosavi2017",
    citation=AttackCitation(
        key="moosavidezfooli2017universal",
        title="Universal Adversarial Perturbations",
        authors="Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, and Pascal Frossard",
        venue="IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
        year=2017,
        url="https://arxiv.org/abs/1610.08401",
        bibtex=(
            "@inproceedings{moosavidezfooli2017universal,\n"
            "  title={Universal Adversarial Perturbations},\n"
            "  author={Moosavi-Dezfooli, Seyed-Mohsen and Fawzi, Alhussein and Fawzi, Omar and Frossard, Pascal},\n"
            "  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},\n"
            "  pages={1765--1773},\n"
            "  year={2017},\n"
            "  doi={10.1109/CVPR.2017.17},\n"
            "  url={https://arxiv.org/abs/1610.08401},\n"
            "}"
        ),
        ieee='S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, and P. Frossard, "Universal adversarial perturbations," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 1765-1773, doi: 10.1109/CVPR.2017.17.',
        apa="Moosavi-Dezfooli, S.-M., Fawzi, A., Fawzi, O., & Frossard, P. (2017). Universal adversarial perturbations. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1765-1773.",
    ),
)


def _apply_universal_vector(
    x: torch.Tensor,
    v: torch.Tensor,
    clip_bounds: tuple[Any, Any] | None,
) -> torch.Tensor:
    """Apply universal vector ``v`` (shape ``(1, ...)``) across batch ``x``."""
    return clamp_to_bounds(x + v, clip_bounds)


def uap(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    eps: float = 0.05,
    delta: float = 0.2,
    max_iter_uni: int = 10,
    deepfool_steps: int = 10,
    overshoot: float = 0.02,
    max_classes: int = 10,
    norm: str = "linf",
    clip_bounds: tuple[Any, Any] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Compute a single universal adversarial perturbation ``v`` for batch ``x``.

    Args:
        model_or_fn: Differentiable ``nn.Module`` or callable mapping ``x -> (B, C)`` logits.
        x: Data batch used both to aggregate and evaluate the universal vector (shape ``(B, ...)``).
        y: Optional ground-truth labels. If ``None``, clean model predictions are fooled instead.
        eps: Perturbation budget (or JSR in dB when ``norm="jsr_db"``). For relative norms,
            the universal vector obeys the tightest per-sample radius in ``x`` so every item meets ``eps``.
        delta: Target upper bound on non-fooled fraction (stops early once ``fooling_rate >= 1 - delta``).
        max_iter_uni: Maximum passes over dataset batch ``x``.
        deepfool_steps: Maximum DeepFool iterations per un-fooled sample.
        overshoot: DeepFool boundary overshoot factor.
        max_classes: Maximum top candidate classes considered inside DeepFool.
        norm: Constraint norm: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        clip_bounds: Optional valid value range ``(min, max)`` for real-valued inputs.
        seed: Optional random seed controlling sample shuffling order across passes.

    Returns:
        ``AttackResult`` whose ``extra["universal_perturbation"]`` holds the shared ``(1, ...)`` vector.
    """
    if max_iter_uni < 1:
        raise ValueError(f"max_iter_uni must be >= 1, got {max_iter_uni}.")
    if deepfool_steps < 1:
        raise ValueError(f"deepfool_steps must be >= 1, got {deepfool_steps}.")
    if not 0.0 <= delta < 1.0:
        raise ValueError(f"delta must be in [0, 1), got {delta}.")
    if clip_bounds is not None and x.is_complex():
        raise ValueError("clip_bounds is only supported for real-valued tensors.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    batch_size = x_clean.shape[0]
    per_sample_budget = resolve_eps_budget(x_clean, eps=eps, norm=norm)
    # Using the tightest sample budget ensures the shared vector obeys eps for ALL items in x.
    uni_budget = per_sample_budget.amin(dim=0, keepdim=True)
    v = torch.zeros((1, *x_clean.shape[1:]), device=x_clean.device, dtype=x_clean.dtype)

    if (uni_budget == 0).all():
        return AttackResult(
            x_adv=x_clean.clone(),
            perturbation=torch.zeros_like(x_clean),
            eps=eps,
            norm=norm,
            steps_run=0,
            attack_id=META.id,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            extra={
                "universal_perturbation": v,
                "fooling_rate": 0.0,
                "epochs_run": 0,
            },
        )

    proj_norm = "l2" if norm in ("relative_l2", "jsr_db") else norm
    df_norm = "linf" if norm == "linf" else "l2"
    v_zero = torch.zeros_like(v)
    generator = make_generator(seed, x_clean.device)

    best_v = v.clone()
    best_fooling_rate = -1.0
    epochs_run = 0
    total_df_steps = 0

    with eval_mode(model_or_fn):
        with torch.no_grad():
            ref_labels = y if y is not None else predict_labels(model_or_fn(x_clean))

        for epoch in range(1, max_iter_uni + 1):
            epochs_run = epoch
            if generator is not None:
                order = torch.randperm(batch_size, device=x_clean.device, generator=generator).tolist()
            else:
                order = list(range(batch_size))

            for idx in order:
                x_cur = _apply_universal_vector(x_clean[idx : idx + 1], v, clip_bounds)
                with torch.no_grad():
                    cur_pred = int(predict_labels(model_or_fn(x_cur))[0].item())
                if cur_pred != int(ref_labels[idx].item()):
                    continue

                x_df, s_i = _deepfool_single_sample(
                    model_or_fn=model_or_fn,
                    x_single=x_cur,
                    norm=df_norm,
                    steps=deepfool_steps,
                    overshoot=overshoot,
                    max_classes=max_classes,
                    clip_bounds=clip_bounds,
                )
                total_df_steps += s_i
                dr = x_df - x_cur
                v = project_perturbation(
                    v + dr, v_zero, uni_budget, norm=proj_norm, clip_bounds=None
                )

            with torch.no_grad():
                x_eval = _apply_universal_vector(x_clean, v, clip_bounds)
                preds = predict_labels(model_or_fn(x_eval))
                fooling_rate = float((preds != ref_labels).float().mean().item())

            if fooling_rate > best_fooling_rate:
                best_fooling_rate = fooling_rate
                best_v = v.clone()

            if fooling_rate >= 1.0 - delta:
                break

    x_adv = _apply_universal_vector(x_clean, best_v, clip_bounds)
    return AttackResult(
        x_adv=x_adv,
        perturbation=x_adv - x_clean,
        eps=eps,
        norm=norm,
        steps_run=total_df_steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "universal_perturbation": best_v,
            "fooling_rate": float(max(best_fooling_rate, 0.0)),
            "epochs_run": epochs_run,
        },
    )
