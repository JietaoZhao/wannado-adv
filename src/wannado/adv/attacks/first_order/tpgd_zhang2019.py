"""TPGD: TRADES KL-Divergence Inner Maximization Attack (Zhang et al., ICML 2019).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : A (Canonical Self-Supervised / Unlabeled Logit Consistency Attack)
- Reference     : Zhang, H., Yu, Y., Jiao, J., Xing, E. P., Ghaoui, L. E., &
                  Jordan, M. I. (2019). "Theoretically Principled Trade-off between
                  Robustness and Accuracy." ICML 2019, PMLR 97:7472-7482.
- URL           : https://arxiv.org/abs/1901.08573
- Plain Intuition:
    Standard PGD needs ground-truth labels ``y`` so it can maximize Cross-Entropy
    loss. TPGD—the inner adversary inside TRADES—drops the label requirement
    altogether and maximizes KL divergence between the clean prediction distribution
    ``P_clean = softmax(f(x))`` and the perturbed prediction ``softmax(f(x + delta))``.
    It pushes the model as far away from its clean confidence distribution as
    possible inside the norm budget, working identically in unsupervised,
    semi-supervised, and TRADES adversarial-training pipelines.
- Diff vs. PGD (Madry et al., 2018):
    1. Maximizes ``D_KL(softmax(f(x_clean)) || softmax(f(x_adv)))`` instead of CE(y).
    2. Works cleanly even when ``y=None`` (unlabeled/self-supervised batches).
- Why Chosen as a Baseline:
    The gold-standard inner maximization for TRADES and label-free logit drift
    auditing.
- Extension Directions / Prompt Hooks:
    1. Swap ``D_KL`` for Jensen-Shannon / cosine feature divergence in latent encoders.
    2. Pair with ``audit_robustness()`` on unlabeled test streams.
===============================================================================
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    eval_mode,
    make_generator,
    project_perturbation,
    random_init_perturbation,
    resolve_eps_budget,
    step_direction,
    update_best_by_objective,
)
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="tpgd",
    name="TRADES KL-Divergence PGD (TPGD)",
    family="first_order",
    tier="A",
    threat_model="white_box",
    requires_training=False,
    compute_cost="low",
    supported_norms=("linf", "l2", "l1", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="pgd",
    diff_from_parent="Replaces supervised Cross-Entropy(y) loss with unsupervised KL(softmax(f(x)) || softmax(f(x_adv))) divergence maximization (TRADES inner adversary).",
    summary_plain="Label-free TRADES adversary that maximizes KL divergence away from the model's clean prediction distribution.",
    when_to_use="When ground-truth labels are unavailable, during TRADES robust training, or to test prediction stability on unlabeled streams.",
    when_not_to_use="For targeted class attacks or Softmax-saturated models where initial clean probabilities are one-hot (use apgd_dlr instead).",
    extension_hooks=(
        "Swap KL(p_clean || p_adv) with symmetric JSD or latent embedding cosine distance",
        "Plug directly into TRADES outer training loss L_clean + beta * KL",
    ),
    module_path="wannado.adv.attacks.first_order.tpgd_zhang2019",
    citation=AttackCitation(
        key="zhang2019theoretically",
        title="Theoretically Principled Trade-off between Robustness and Accuracy",
        authors="Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, and Michael I. Jordan",
        venue="36th International Conference on Machine Learning (ICML)",
        year=2019,
        url="https://arxiv.org/abs/1901.08573",
        bibtex=(
            "@inproceedings{zhang2019theoretically,\n"
            "  title={Theoretically Principled Trade-off between Robustness and Accuracy},\n"
            "  author={Zhang, Hongyang and Yu, Yaodong and Jiao, Jiantao and Xing, Eric P. and El Ghaoui, Laurent and Jordan, Michael I.},\n"
            "  booktitle={Proceedings of the 36th International Conference on Machine Learning},\n"
            "  pages={7472--7482},\n"
            "  year={2019},\n"
            "  volume={97},\n"
            "  series={Proceedings of Machine Learning Research},\n"
            "  publisher={PMLR},\n"
            "  url={https://arxiv.org/abs/1901.08573},\n"
            "}"
        ),
        ieee='H. Zhang et al., "Theoretically principled trade-off between robustness and accuracy," in Proc. 36th Int. Conf. Mach. Learn. (ICML), PMLR 97, 2019, pp. 7472-7482.',
        apa="Zhang, H., Yu, Y., Jiao, J., Xing, E. P., El Ghaoui, L., & Jordan, M. I. (2019). Theoretically principled trade-off between robustness and accuracy. Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97, 7472-7482.",
    ),
)


def tpgd(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    eps: float = 0.03,
    step_size: float | None = None,
    steps: int = 10,
    norm: str = "linf",
    random_start: bool = True,
    jitter_std: float = 1e-3,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run TRADES KL-divergence inner maximization PGD (unlabeled/self-supervised)."""
    del y  # TPGD maximizes KL drift from clean model predictions, independent of true labels.
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if jitter_std < 0.0:
        raise ValueError(f"jitter_std must be >= 0, got {jitter_std}.")

    t0 = time.perf_counter()
    x_clean = x.detach()
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
            extra={"kl_divergence_mean": 0.0},
        )

    alpha = (
        eps_budget * min(1.0, 2.5 / float(steps))
        if step_size is None
        else torch.full_like(eps_budget, float(step_size))
    )
    generator = make_generator(seed, x_clean.device)

    with eval_mode(model_or_fn):
        with torch.no_grad():
            clean_probs = F.softmax(model_or_fn(x_clean).detach(), dim=-1)

        if random_start:
            if jitter_std > 0.0 and not x_clean.is_complex():
                delta = (
                    torch.randn(
                        x_clean.shape,
                        device=x_clean.device,
                        dtype=x_clean.dtype,
                        generator=generator,
                    )
                    * jitter_std
                )
                delta = project_perturbation(
                    delta, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds
                )
            else:
                delta = random_init_perturbation(
                    x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds, generator=generator
                )
        else:
            delta = torch.zeros_like(x_clean)

        best_delta = delta.clone()
        best_kl = torch.full(
            (x_clean.shape[0],), float("-inf"), device=x_clean.device, dtype=eps_budget.dtype
        )

        for _ in range(steps):
            x_in = (x_clean + delta).detach().requires_grad_(True)
            adv_log_probs = F.log_softmax(model_or_fn(x_in), dim=-1)
            # [AI-Prompt Hook]: Zhang et al. (2019) TRADES Eq. (3) KL(p_clean || p_adv).
            # Swap `F.kl_div` for MSE/cosine distance between latent representations for SSL attacks.
            kl_per_sample = F.kl_div(adv_log_probs, clean_probs, reduction="none").sum(dim=-1)
            (grad,) = torch.autograd.grad(kl_per_sample.sum(), x_in, only_inputs=True)

            best_delta, best_kl = update_best_by_objective(
                best_delta, best_kl, delta, kl_per_sample.detach()
            )
            delta = delta + alpha * step_direction(grad.detach(), norm=norm)
            delta = project_perturbation(
                delta, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds
            )

        with torch.no_grad():
            final_log_probs = F.log_softmax(model_or_fn(x_clean + delta), dim=-1)
            final_kl = F.kl_div(final_log_probs, clean_probs, reduction="none").sum(dim=-1)
            best_delta, best_kl = update_best_by_objective(best_delta, best_kl, delta, final_kl)

    return AttackResult(
        x_adv=x_clean + best_delta,
        perturbation=best_delta,
        eps=eps,
        norm=norm,
        steps_run=steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={"kl_divergence_mean": float(best_kl.mean().item())},
    )
