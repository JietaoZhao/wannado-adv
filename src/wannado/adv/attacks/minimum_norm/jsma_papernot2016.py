"""JSMA: Jacobian-Based Saliency Map Attack (Papernot et al., IEEE EuroS&P 2016).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : A (Canonical Forward-Derivative Saliency L0 Attack)
- Reference     : Papernot, N., McDaniel, P., Jha, S., Fredrikson, M.,
                  Celik, Z. B., & Swami, A. (2016). "The Limitations of Deep
                  Learning in Adversarial Settings." 1st IEEE European Symposium
                  on Security and Privacy (EuroS&P), pp. 372-387.
- URL           : https://arxiv.org/abs/1511.07528
- Plain Intuition:
    Instead of updating all features at once, JSMA computes the Jacobian matrix
    (how EVERY input feature nudges EVERY output logit), constructs an adversarial
    saliency map scoring features that BOTH increase the target logit AND decrease
    all competing logits simultaneously, and greedily modifies only 1-2 features
    per iteration by ``theta`` until the target class wins or ``gamma`` fraction
    of features is exhausted.
- Diff vs. DeepFool / SparseFool:
    Uses full class-wise Jacobian forward sensitivity ``dZ_t/dx_i`` and
    ``sum_{j != t} dZ_j/dx_i`` (targeted by origin, with automatic untargeted
    runner-up target fallback) instead of iterative hyperplane projections.
- Why Chosen as a Baseline:
    The seminal white-box sparse L0 saliency attack in adversarial ML.
- Extension Directions / Prompt Hooks:
    1. Replace greedy single-feature update with pairwise feature search from
       Algorithm 2 of Papernot et al. (2016).
    2. Use ``theta > 0`` vs ``theta < 0`` to audit asymmetric monotonic features
       in tabular/healthcare classifiers.
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
)
from wannado.adv.core.projections import within_eps_budget
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="jsma",
    name="Jacobian Saliency Map Attack (JSMA)",
    family="minimum_norm",
    tier="A",
    threat_model="white_box",
    requires_training=False,
    compute_cost="high",
    supported_norms=("l0", "l1", "l2", "linf"),
    supported_domains=("vision", "tabular", "timeseries", "drl"),
    parent_id=None,
    diff_from_parent="Computes forward-derivative saliency S(x_i, t) combining target-logit sensitivity alpha_i and competing-logit sensitivity beta_i to perturb minimal feature counts.",
    summary_plain="Greedy Jacobian saliency L0 attack that alters only the top few highest-leverage input features toward a target class.",
    when_to_use="Low-to-medium dimensional tabular/sensor/image models when interpretable minimal feature-count (L0) tampering is studied.",
    when_not_to_use="High-resolution ImageNet tensors (O(C * iter) Jacobians are slow; use sparsefool or apgd_l1) or complex-valued waveforms.",
    extension_hooks=(
        "Swap single-feature saliency lookup with pair-wise search for ultra-tight L0",
        "Supply domain feature-modifiability masks (e.g. immutable tabular columns)",
    ),
    module_path="wannado.adv.attacks.minimum_norm.jsma_papernot2016",
    citation=AttackCitation(
        key="papernot2016limitations",
        title="The Limitations of Deep Learning in Adversarial Settings",
        authors="Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z. Berkay Celik, and Ananthram Swami",
        venue="2016 IEEE European Symposium on Security and Privacy (EuroS&P)",
        year=2016,
        url="https://arxiv.org/abs/1511.07528",
        bibtex=(
            "@inproceedings{papernot2016limitations,\n"
            "  title={The Limitations of Deep Learning in Adversarial Settings},\n"
            "  author={Papernot, Nicolas and McDaniel, Patrick and Jha, Somesh and Fredrikson, Matt and Celik, Z. Berkay and Swami, Ananthram},\n"
            "  booktitle={2016 IEEE European Symposium on Security and Privacy (EuroS&P)},\n"
            "  pages={372--387},\n"
            "  year={2016},\n"
            "  organization={IEEE},\n"
            "  url={https://arxiv.org/abs/1511.07528},\n"
            "}"
        ),
        ieee='N. Papernot et al., "The limitations of deep learning in adversarial settings," in Proc. IEEE Eur. Symp. Secur. Privacy (EuroS&P), 2016, pp. 372-387.',
        apa="Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z. B., & Swami, A. (2016). The limitations of deep learning in adversarial settings. 2016 IEEE European Symposium on Security and Privacy (EuroS&P), 372-387.",
    ),
)


def _jsma_single_sample(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_single: torch.Tensor,
    target_cls: int,
    targeted: bool,
    orig_label: int,
    theta: float,
    max_iters: int,
    clip_bounds: tuple[float, float] | None,
) -> tuple[torch.Tensor, int]:
    """Run greedy JSMA coordinate saliency updates on one sample ``(1, ...)``."""
    x_adv = x_single.clone()
    dim = x_single.numel()
    active_mask = torch.ones(dim, dtype=torch.bool, device=x_single.device)
    steps_taken = 0

    for step in range(max_iters):
        if not active_mask.any():
            break
        steps_taken = step + 1

        x_in = x_adv.detach().requires_grad_(True)
        logits = model_or_fn(x_in)
        pred = int(logits.argmax(dim=1)[0].item())
        if (targeted and pred == target_cls) or ((not targeted) and pred != orig_label):
            break

        num_classes = logits.shape[1]
        # Build Jacobian rows alpha = dZ_target/dx and beta = d(sum_{j != target} Z_j)/dx cleanly.
        target_logit = logits[0, target_cls]
        all_sum = logits[0].sum()
        (alpha_t,) = torch.autograd.grad(target_logit, x_in, retain_graph=True)
        (sum_grad,) = torch.autograd.grad(all_sum, x_in, retain_graph=False)
        alpha = alpha_t.detach().reshape(-1)
        beta = (sum_grad.detach().reshape(-1)) - alpha

        # [AI-Prompt Hook]: Papernot et al. (2016) Eq. (8)-(9) adversarial saliency condition.
        # Pairwise search can check top-K candidates in `alpha` + `beta` to tighten L0 further.
        if theta > 0.0:
            cond = (alpha > 0.0) & (beta < 0.0) & active_mask
            saliency = torch.where(cond, -alpha * beta, torch.zeros_like(alpha))
        else:
            cond = (alpha < 0.0) & (beta > 0.0) & active_mask
            saliency = torch.where(cond, -alpha * beta, torch.zeros_like(alpha))

        if float(saliency.max().item()) <= 1e-12:
            # Soft fallback to highest net margin increase rate among active coordinates.
            sign_pref = 1.0 if theta > 0.0 else -1.0
            score_soft = sign_pref * (alpha - beta / float(max(num_classes - 1, 1)))
            score_soft = torch.where(active_mask, score_soft, torch.full_like(score_soft, float("-inf")))
            best_idx = int(score_soft.argmax().item())
            if float(score_soft[best_idx].item()) == float("-inf"):
                break
        else:
            best_idx = int(saliency.argmax().item())

        flat_adv = x_adv.reshape(-1)
        new_val = float(flat_adv[best_idx].item()) + theta
        if clip_bounds is not None:
            low, high = clip_bounds
            if new_val <= low:
                new_val = float(low)
                active_mask[best_idx] = False
            elif new_val >= high:
                new_val = float(high)
                active_mask[best_idx] = False
        flat_adv[best_idx] = new_val
        x_adv = flat_adv.reshape_as(x_single)

    return x_adv.detach(), steps_taken


def jsma(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    eps: float | None = None,
    norm: str = "l0",
    theta: float = 0.25,
    gamma: float = 0.5,
    targeted: bool = True,
    clip_bounds: tuple[float, float] | None = None,
) -> AttackResult:
    """Run Jacobian-based Saliency Map Attack (JSMA)."""
    if x.is_complex():
        raise ValueError("JSMA forward-derivative saliency is defined only for real-valued inputs.")
    if theta == 0.0:
        raise ValueError("theta must be nonzero.")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    batch_size = x_clean.shape[0]
    dim = x_clean[0].numel()
    max_iters = max(1, round(dim * gamma))
    x_adv = x_clean.clone()
    total_steps = 0

    with eval_mode(model_or_fn):
        with torch.no_grad():
            clean_logits = model_or_fn(x_clean)
            clean_preds = clean_logits.argmax(dim=1)
            sorted_classes = clean_logits.argsort(dim=1, descending=True)

        for idx in range(batch_size):
            orig_pred = int(clean_preds[idx].item())
            if targeted:
                if y is not None:
                    tgt = int(y[idx].item())
                    if orig_pred == tgt:
                        continue
                else:
                    tgt = int(sorted_classes[idx, 1].item())
            else:
                if y is not None and orig_pred != int(y[idx].item()):
                    continue
                tgt = int(sorted_classes[idx, 1].item())

            adv_i, s_i = _jsma_single_sample(
                model_or_fn=model_or_fn,
                x_single=x_clean[idx : idx + 1],
                target_cls=tgt,
                targeted=targeted,
                orig_label=orig_pred,
                theta=theta,
                max_iters=max_iters,
                clip_bounds=clip_bounds,
            )
            x_adv[idx] = adv_i[0]
            total_steps += s_i

    delta = x_adv - x_clean
    if eps is not None:
        within = within_eps_budget(delta, x_clean, eps=eps, norm=norm).view(
            (batch_size,) + (1,) * (x_clean.ndim - 1)
        )
        x_adv = torch.where(within, x_adv, x_clean)
        delta = x_adv - x_clean

    l0_dist = compute_perturbation_norm(delta, norm="l0")
    return AttackResult(
        x_adv=x_adv,
        perturbation=delta,
        eps=float(eps) if eps is not None else float(l0_dist.median().item()),
        norm=norm,
        steps_run=total_steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "mean_l0": float(l0_dist.mean().item()),
            "median_l0": float(l0_dist.median().item()),
            "theta": theta,
            "gamma": gamma,
        },
    )
