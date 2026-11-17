"""DeepFool: Minimum-Norm Hyperplane Projection Attack (Moosavi-Dezfooli et al., CVPR 2016).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Classic Geometric Minimum-Norm Baseline)
- Reference     : Moosavi-Dezfooli, S.-M., Fawzi, A., & Frossard, P. (2016).
                  "DeepFool: A Simple and Accurate Method to Fool Deep Neural
                  Networks." CVPR 2016 (pp. 2574-2582).
- URL           : https://arxiv.org/abs/1511.04599
- Plain Intuition:
    Unlike FGSM/PGD which fix an energy budget ``eps`` and maximize error,
    DeepFool asks the reverse question: "What is the SMALLEST perturbation that
    crosses the nearest decision boundary?" At each step it pretends the neural
    network is locally linear around ``x_i``, computes the orthogonal distance
    to every other class hyperplane, and jumps straight across the closest one.
- Diff vs. First-Order Fixed-Budget Attacks (FGSM / PGD):
    Optimizes minimum perturbation distance rather than maximizing loss inside
    a fixed norm ball; stops early per sample as soon as the label flips.
- Why Chosen as a Baseline:
    Fast geometric estimator of minimal perturbation margins (and input-space
    robustness radius) under L2 or Linf geometry.
- Extension Directions (for further research):
    1. Add biased backward step & restarts -> FAB (Croce & Hein, ICML 2020).
    2. Promote sparsity -> SparseFool (Modas et al., CVPR 2019).
    3. Aggregate minimal DeepFool steps across batch -> UAP (Moosavi-Dezfooli 2017).
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
    compute_perturbation_norm,
    eval_mode,
    normalize_classification_logits,
    predict_labels,
    safe_l2_norm,
    step_direction,
)
from wannado.adv.core.projections import within_eps_budget
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="deepfool",
    name="DeepFool",
    family="minimum_norm",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="medium",
    supported_norms=("l2", "linf", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    parent_id=None,
    diff_from_parent="Origin of geometric minimum-norm boundary attacks: iteratively projects onto the nearest linearized class hyperplane until misclassification.",
    summary_plain="Finds a minimal perturbation to cross the nearest decision boundary via iterative hyperplane projection.",
    when_to_use="When measuring the minimum distortion/power margin needed to flip predictions rather than testing a fixed epsilon.",
    when_not_to_use="For non-classification regression losses (needs class logits) or high-class-count outputs without max_classes capping (use C&W or FAB).",
    extension_hooks=(
        "Add biased boundary tracking & restarts -> FAB (Croce & Hein, ICML 2020)",
        "Aggregate per-sample minimal vectors -> UAP (Moosavi-Dezfooli et al., CVPR 2017)",
    ),
    module_path="wannado.adv.attacks.minimum_norm.deepfool_moosavi2016",
    citation=AttackCitation(
        key="moosavidezfooli2016deepfool",
        title="DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks",
        authors="Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, and Pascal Frossard",
        venue="IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
        year=2016,
        url="https://arxiv.org/abs/1511.04599",
        bibtex=(
            "@inproceedings{moosavidezfooli2016deepfool,\n"
            "  title={DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks},\n"
            "  author={Moosavi-Dezfooli, Seyed-Mohsen and Fawzi, Alhussein and Frossard, Pascal},\n"
            "  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},\n"
            "  pages={2574--2582},\n"
            "  year={2016},\n"
            "  url={https://arxiv.org/abs/1511.04599},\n"
            "}"
        ),
        ieee='S.-M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard, "DeepFool: A simple and accurate method to fool deep neural networks," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 2574-2582.',
        apa="Moosavi-Dezfooli, S.-M., Fawzi, A., & Frossard, P. (2016). DeepFool: A simple and accurate method to fool deep neural networks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2574-2582.",
    ),
)


def _deepfool_single_sample(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_single: torch.Tensor,
    norm: str,
    steps: int,
    overshoot: float,
    max_classes: int,
    clip_bounds: tuple[Any, Any] | None,
) -> tuple[torch.Tensor, int]:
    """Run DeepFool hyperplane projection on a single sample ``(1, ...)``."""
    with torch.no_grad():
        logits_0 = normalize_classification_logits(model_or_fn(x_single))
        num_classes = logits_0.shape[1]
        sorted_classes = logits_0.argsort(dim=1, descending=True)[0]
        orig_label = int(sorted_classes[0].item())
        candidate_classes = [int(c.item()) for c in sorted_classes[1 : min(num_classes, max_classes)]]

    r_tot = torch.zeros_like(x_single)
    x_adv = x_single.clone()
    steps_taken = 0

    for step_idx in range(steps):
        steps_taken = step_idx + 1
        x_in = x_adv.detach().requires_grad_(True)
        logits = normalize_classification_logits(model_or_fn(x_in))
        if int(logits.argmax(dim=1)[0].item()) != orig_label:
            break

        (grad_orig,) = torch.autograd.grad(logits[0, orig_label], x_in, retain_graph=True)
        grad_orig = grad_orig.detach()

        min_dist = float("inf")
        best_r = torch.zeros_like(x_single)

        for idx, k in enumerate(candidate_classes):
            retain = idx < len(candidate_classes) - 1
            (grad_k,) = torch.autograd.grad(logits[0, k], x_in, retain_graph=retain)
            w_k = grad_k.detach() - grad_orig
            f_k = float((logits[0, k] - logits[0, orig_label]).detach().item())

            if norm == "linf":
                w_l1 = float(w_k.abs().sum().item())
                if w_l1 <= 1e-12:
                    continue
                dist = abs(f_k) / w_l1
                r_k = (abs(f_k) / w_l1 + 1e-4) * step_direction(w_k, norm="linf")
            else:
                w_l2 = float(safe_l2_norm(w_k.reshape(-1), dim=None).item())
                if w_l2 <= 1e-12:
                    continue
                dist = abs(f_k) / w_l2
                r_k = (abs(f_k) / (w_l2 * w_l2 + 1e-12)) * w_k

            if dist < min_dist:
                min_dist = dist
                best_r = r_k

        if min_dist == float("inf"):
            break

        r_tot = r_tot + (1.0 + overshoot) * best_r
        x_adv = clamp_to_bounds(x_single + r_tot, clip_bounds)

    with torch.no_grad():
        final_pred = int(predict_labels(model_or_fn(x_adv))[0].item())
    if final_pred == orig_label:
        # Try one gentle final overshoot scaling if still barely on the clean side.
        x_retry = clamp_to_bounds(x_single + (1.0 + overshoot * 2.0) * r_tot, clip_bounds)
        with torch.no_grad():
            if int(predict_labels(model_or_fn(x_retry))[0].item()) != orig_label:
                x_adv = x_retry

    return x_adv.detach(), steps_taken


def deepfool(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    eps: float | None = None,
    steps: int = 50,
    overshoot: float = 0.02,
    max_classes: int = 10,
    norm: str = "l2",
    clip_bounds: tuple[float, float] | None = None,
) -> AttackResult:
    """Run untargeted DeepFool to find minimal boundary-crossing perturbations.

    Args:
        model_or_fn: Differentiable ``nn.Module`` or callable mapping ``x -> (B, C)`` logits.
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Optional ground-truth labels. If supplied, samples already misclassified
            w.r.t. ``y`` are left unperturbed (zero distortion).
        eps: Optional maximum budget cutoff. When provided, perturbations exceeding
            ``eps_budget`` revert to clean ``x`` (useful for fixed-budget comparisons).
        steps: Maximum hyperplane projection iterations per sample.
        overshoot: Small factor ``> 0`` ensuring the final step crosses the boundary.
        max_classes: Maximum number of top candidate classes examined per step.
        norm: Geometry metric: ``"l2"``, ``"linf"``, ``"relative_l2"``, or ``"jsr_db"``.
        clip_bounds: Optional valid range ``(min, max)`` for real-valued inputs.

    Returns:
        ``AttackResult`` containing minimal adversarial examples and distortion statistics.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if clip_bounds is not None and x.is_complex():
        raise ValueError("clip_bounds is only supported for real-valued tensors.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    batch_size = x_clean.shape[0]
    x_adv = x_clean.clone()
    total_steps = 0

    with eval_mode(model_or_fn):
        with torch.no_grad():
            clean_preds = predict_labels(model_or_fn(x_clean))

        for idx in range(batch_size):
            if y is not None and int(clean_preds[idx].item()) != int(y[idx].item()):
                continue
            adv_i, s_i = _deepfool_single_sample(
                model_or_fn=model_or_fn,
                x_single=x_clean[idx : idx + 1],
                norm=norm,
                steps=steps,
                overshoot=overshoot,
                max_classes=max_classes,
                clip_bounds=clip_bounds,
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
            "overshoot": overshoot,
        },
    )
