"""ZOO: Zeroth Order Optimization Black-Box Attack (Chen et al., ACM AISec 2017).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : A (Canonical Coordinate Finite-Difference Black-Box C&W)
- Reference     : Chen, P.-Y., Zhang, H., Sharma, Y., Yi, J., & Hsieh, C.-J.
                  (2017). "ZOO: Zeroth Order Optimization Based Black-box Attacks
                  to Deep Neural Networks without Training Substitute Models."
                  ACM AISec 2017 (pp. 15-26).
- URL           : https://arxiv.org/abs/1708.03999
- Plain Intuition:
    White-box C&W optimizes ``||delta||_2^2 + c * f_CW(x + delta)`` using exact
    backpropagation gradients. ZOO replaces backpropagation with symmetric
    coordinate-wise finite differences ``(L(x + h*e_i) - L(x - h*e_i)) / (2h)``
    on random batches of coordinates, plugging those estimated gradients directly
    into an Adam coordinate solver using ONLY model output scores.
- Diff vs. C&W L2 (Carlini & Wagner, 2017):
    Replaces autograd with batched coordinate finite differences + coordinate Adam;
    needs zero gradients or substitute models.
- Why Chosen as a Baseline:
    The foundational bridge showing score-based black-box attacks match white-box
    C&W optimization without substitute-model training.
- Extension Directions / Prompt Hooks:
    1. Swap random coordinate batches for importance-weighted or hierarchical
       upsampled grid probes to lower query count on high-res inputs.
    2. Use ``norm='jsr_db'`` with complex coordinate probing on RF receivers.
===============================================================================
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from wannado.adv.attacks.minimum_norm.cw_l2_carlini2017 import _cw_margin_objective, _l2_sq_per_sample
from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    clamp_to_bounds,
    compute_perturbation_norm,
    eval_mode,
    make_generator,
    normalize_classification_logits,
    project_perturbation,
    resolve_eps_budget,
)
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="zoo",
    name="Zeroth Order Optimization Attack (ZOO-Adam)",
    family="blackbox_score",
    tier="A",
    threat_model="black_box_score",
    requires_training=False,
    compute_cost="high",
    supported_norms=("l2", "linf", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    parent_id="cw_l2",
    diff_from_parent="Replaces backpropagation in C&W with coordinate-wise symmetric finite differences and Adam updates using only output scores.",
    summary_plain="Score-based black-box version of C&W that estimates gradients coordinate-by-coordinate via finite differences.",
    when_to_use="When evaluating non-differentiable models that expose continuous logits/probabilities without gradient access.",
    when_not_to_use="When query budget is tight on high-dimensional images (prefer square_attack or nes).",
    extension_hooks=(
        "Add importance sampling over coordinate blocks for faster convergence",
        "Tune penalty constant `const` or finite-difference step `fd_eps` per domain",
    ),
    module_path="wannado.adv.attacks.blackbox_score.zoo_chen2017",
    citation=AttackCitation(
        key="chen2017zoo",
        title="ZOO: Zeroth Order Optimization Based Black-box Attacks to Deep Neural Networks without Training Substitute Models",
        authors="Pin-Yu Chen, Huan Zhang, Yash Sharma, Jinfeng Yi, and Cho-Jui Hsieh",
        venue="10th ACM Workshop on Artificial Intelligence and Security (AISec)",
        year=2017,
        url="https://arxiv.org/abs/1708.03999",
        bibtex=(
            "@inproceedings{chen2017zoo,\n"
            "  title={ZOO: Zeroth Order Optimization Based Black-box Attacks to Deep Neural Networks without Training Substitute Models},\n"
            "  author={Chen, Pin-Yu and Zhang, Huan and Sharma, Yash and Yi, Jinfeng and Hsieh, Cho-Jui},\n"
            "  booktitle={Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security (AISec)},\n"
            "  pages={15--26},\n"
            "  year={2017},\n"
            "  url={https://arxiv.org/abs/1708.03999},\n"
            "}"
        ),
        ieee='P.-Y. Chen, H. Zhang, Y. Sharma, J. Yi, and C.-J. Hsieh, "ZOO: Zeroth order optimization based black-box attacks to deep neural networks without training substitute models," in Proc. 10th ACM Workshop Artif. Intell. Secur. (AISec), 2017, pp. 15-26.',
        apa="Chen, P.-Y., Zhang, H., Sharma, Y., Yi, J., & Hsieh, C.-J. (2017). ZOO: Zeroth order optimization based black-box attacks to deep neural networks without training substitute models. Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security (AISec), 15-26.",
    ),
)


def _eval_zoo_loss(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_cand: torch.Tensor,
    x_clean_rep: torch.Tensor,
    y_rep: torch.Tensor,
    const_rep: torch.Tensor,
    confidence: float,
    targeted: bool,
    use_log_probs: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Evaluate per-sample ZOO objective ||delta||_2^2 + c * f_CW(outputs) without autograd."""
    outputs = normalize_classification_logits(model_or_fn(x_cand))
    scores = F.log_softmax(outputs, dim=1) if use_log_probs else outputs
    f_loss, succeeded = _cw_margin_objective(
        scores, y_rep, confidence=confidence, targeted=targeted
    )
    l2_sq = _l2_sq_per_sample(x_cand - x_clean_rep)
    return l2_sq + const_rep * f_loss, succeeded, int(x_cand.shape[0])


def zoo(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float | None = None,
    norm: str = "l2",
    steps: int = 80,
    coord_batch_size: int = 32,
    lr: float = 0.05,
    fd_eps: float = 1e-3,
    const: float = 10.0,
    confidence: float = 0.0,
    use_log_probs: bool = True,
    targeted: bool = False,
    clip_bounds: tuple[Any, Any] | None = None,
    max_queries: int | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run Zeroth Order Optimization (ZOO-Adam) score-based black-box attack."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if max_queries is not None and max_queries < 1:
        raise ValueError(f"max_queries must be >= 1, got {max_queries}.")
    if coord_batch_size < 1:
        raise ValueError(f"coord_batch_size must be >= 1, got {coord_batch_size}.")
    if fd_eps <= 0.0:
        raise ValueError(f"fd_eps must be > 0, got {fd_eps}.")
    if clip_bounds is not None and x.is_complex():
        raise ValueError("clip_bounds is only supported for real-valued tensors.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    batch_size = x_clean.shape[0]
    real_dtype = x_clean.real.dtype if x_clean.is_complex() else x_clean.dtype
    view_shape = (batch_size,) + (1,) * (x_clean.ndim - 1)

    eps_budget: torch.Tensor | None = None
    if eps is not None:
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
                extra={"queries": 0, "success_rate": 0.0},
            )

    # Represent perturbation as real vector (B, D_eff) where complex IQ becomes [real, imag].
    is_cplx = x_clean.is_complex()
    elem_dim = x_clean[0].numel()
    eff_dim = elem_dim * 2 if is_cplx else elem_dim
    k_coords = min(eff_dim, coord_batch_size)

    def _to_real_flat(t: torch.Tensor) -> torch.Tensor:
        if is_cplx:
            return torch.cat([t.real.reshape(batch_size, -1), t.imag.reshape(batch_size, -1)], dim=1)
        return t.reshape(batch_size, -1)

    def _from_real_flat(flat: torch.Tensor) -> torch.Tensor:
        target_shape = (flat.shape[0], *x_clean.shape[1:])
        if is_cplx:
            re = flat[:, :elem_dim].reshape(target_shape)
            im = flat[:, elem_dim:].reshape(target_shape)
            return torch.complex(re, im).to(x_clean.dtype)
        return flat.reshape(target_shape)

    delta_flat = torch.zeros((batch_size, eff_dim), device=x_clean.device, dtype=real_dtype)
    m_adam = torch.zeros_like(delta_flat)
    v_adam = torch.zeros_like(delta_flat)
    t_adam = torch.zeros_like(delta_flat)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    generator = make_generator(seed, x_clean.device)
    queries = 0
    best_l2 = torch.full((batch_size,), float("inf"), device=x_clean.device, dtype=real_dtype)
    best_delta = torch.zeros_like(x_clean)
    delta_cur = torch.zeros_like(x_clean)
    const_vec = torch.full((batch_size,), float(const), device=x_clean.device, dtype=real_dtype)

    with eval_mode(model_or_fn), torch.no_grad():
        for _ in range(steps):
            if max_queries is not None and queries >= max_queries:
                break
            perm = torch.randperm(eff_dim, device=x_clean.device, generator=generator)[:k_coords]
            eye = torch.zeros((k_coords, eff_dim), device=x_clean.device, dtype=real_dtype)
            eye[torch.arange(k_coords, device=x_clean.device), perm] = fd_eps

            # Broadcast probes -> (k_coords, batch_size, eff_dim)
            base = delta_flat.unsqueeze(0)
            plus_flat = (base + eye.unsqueeze(1)).reshape(k_coords * batch_size, eff_dim)
            minus_flat = (base - eye.unsqueeze(1)).reshape(k_coords * batch_size, eff_dim)

            x_clean_rep = x_clean.unsqueeze(0).expand(k_coords, *x_clean.shape).reshape(-1, *x_clean.shape[1:])
            y_rep = y.unsqueeze(0).expand(k_coords, batch_size).reshape(-1)
            c_rep = const_vec.unsqueeze(0).expand(k_coords, batch_size).reshape(-1)

            x_plus = clamp_to_bounds(x_clean_rep + _from_real_flat(plus_flat), clip_bounds)
            x_minus = clamp_to_bounds(x_clean_rep + _from_real_flat(minus_flat), clip_bounds)

            both = torch.cat([x_plus, x_minus], dim=0)
            y_both = torch.cat([y_rep, y_rep], dim=0)
            c_both = torch.cat([c_rep, c_rep], dim=0)
            x_ref_both = torch.cat([x_clean_rep, x_clean_rep], dim=0)

            losses, _, q = _eval_zoo_loss(
                model_or_fn,
                both,
                x_ref_both,
                y_both,
                c_both,
                confidence=confidence,
                targeted=targeted,
                use_log_probs=use_log_probs,
            )
            queries += q
            half_q = k_coords * batch_size
            l_plus = losses[:half_q].view(k_coords, batch_size)
            l_minus = losses[half_q:].view(k_coords, batch_size)

            # [AI-Prompt Hook]: ZOO-Adam symmetric finite difference on sampled coordinate batch.
            # Swap `perm` sampling strategy above for hierarchical upsampling on large spatial inputs.
            grad_k = ((l_plus - l_minus) / (2.0 * fd_eps)).transpose(0, 1)  # (B, k_coords)

            t_sel = t_adam[:, perm] + 1.0
            m_sel = beta1 * m_adam[:, perm] + (1.0 - beta1) * grad_k
            v_sel = beta2 * v_adam[:, perm] + (1.0 - beta2) * grad_k.pow(2)
            t_adam[:, perm] = t_sel
            m_adam[:, perm] = m_sel
            v_adam[:, perm] = v_sel

            m_hat = m_sel / (1.0 - beta1 ** t_sel)
            v_hat = v_sel / (1.0 - beta2 ** t_sel)
            delta_flat[:, perm] = delta_flat[:, perm] - lr * m_hat / (v_hat.sqrt() + eps_adam)

            delta_cur = _from_real_flat(delta_flat)
            if eps_budget is not None:
                delta_cur = project_perturbation(
                    delta_cur, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds
                )
            elif clip_bounds is not None:
                delta_cur = clamp_to_bounds(x_clean + delta_cur, clip_bounds) - x_clean
            delta_flat = _to_real_flat(delta_cur)

            logits_cur = normalize_classification_logits(model_or_fn(x_clean + delta_cur))
            queries += batch_size
            scores_cur = F.log_softmax(logits_cur, dim=1) if use_log_probs else logits_cur
            _, succeeded = _cw_margin_objective(
                scores_cur, y, confidence=confidence, targeted=targeted
            )
            l2_cur = _l2_sq_per_sample(delta_cur)
            improved = succeeded & (l2_cur < best_l2)
            best_l2 = torch.where(improved, l2_cur, best_l2)
            best_delta = torch.where(improved.view(view_shape), delta_cur, best_delta)

    # For samples that never crossed the decision boundary under an eps cap, fall back to the
    # final feasible iterate rather than discarding bounded progress.
    delta_final = (
        torch.where((best_l2 < float("inf")).view(view_shape), best_delta, delta_cur)
        if eps_budget is not None
        else best_delta
    )
    dist = compute_perturbation_norm(delta_final, norm=norm, x_ref=x_clean)
    return AttackResult(
        x_adv=x_clean + delta_final,
        perturbation=delta_final,
        eps=float(eps) if eps is not None else float(dist.median().item()),
        norm=norm,
        steps_run=steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "queries": queries,
            "success_rate": float((best_l2 < float("inf")).float().mean().item()),
            "mean_distortion": float(dist.mean().item()),
        },
    )
