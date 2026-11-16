"""MI-FGSM: Momentum Iterative Fast Gradient Sign Method (Dong et al., CVPR 2018).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Gold-Standard Transferable White-Box Attack)
- Reference     : Dong, Y., Liao, F., Pang, T., Su, H., Zhu, J., Hu, X., & Li, J.
                  (2018). "Boosting Adversarial Attacks With Momentum." CVPR 2018.
- URL           : https://arxiv.org/abs/1710.06081
- Plain Intuition:
    Standard BIM/PGD only follows the gradient at the current spot, so it can
    zig-zag or get trapped in tiny local bumps of one specific model. MI-FGSM
    accumulates an exponential moving average (momentum) of L1-normalized past
    gradients—like rolling a heavy ball downhill so it punches straight through
    bumps and transfers much better to unseen black-box surrogate models.
- Diff vs. BIM (2017) & PGD (2018):
    1. L1-normalizes the gradient at every iteration: ``g_t / ||g_t||_1``.
    2. Accumulates velocity ``m_{t+1} = decay * m_t + g_t / ||g_t||_1`` before stepping.
- Why Chosen as a Baseline:
    The foundational baseline whenever evaluating **cross-model transferability**
    (attacking a substitute model and testing against a victim model).
- Extension Directions (for further research):
    1. Add Nesterov look-ahead before gradient computation -> NI-FGSM (Lin et al., 2020).
    2. Combine with input diversity / translation invariance -> DI-FGSM / TI-FGSM.
    3. Use variance-reduced momentum -> VMI-FGSM (Wang & He, 2021).
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
    id="mi_fgsm",
    name="Momentum Iterative FGSM (MI-FGSM)",
    family="first_order",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="low",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="bim",
    diff_from_parent="Accumulates an L1-normalized gradient momentum buffer (decay=1.0) across steps to stabilize update directions and boost transferability.",
    summary_plain="Adds gradient inertia (momentum) across iterations to escape poor local optima and transfer better across unseen models.",
    when_to_use="Primary baseline for transfer-based black-box evaluations (surrogate -> victim) and noisy/non-smooth loss surfaces.",
    when_not_to_use="For pure white-box worst-case evaluation on a single model, APGD or multi-restart PGD is usually stronger.",
    extension_hooks=(
        "Look ahead with Nesterov momentum -> NI-FGSM (Lin et al., ICLR 2020)",
        "Combine with random input/channel transforms -> DI-FGSM / EoT-MI-FGSM",
    ),
    module_path="wannado.adv.attacks.first_order.mifgsm_dong2018",
    citation=AttackCitation(
        key="dong2018boosting",
        title="Boosting Adversarial Attacks With Momentum",
        authors="Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, and Jianguo Li",
        venue="IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
        year=2018,
        url="https://arxiv.org/abs/1710.06081",
        bibtex=(
            "@inproceedings{dong2018boosting,\n"
            "  title={Boosting Adversarial Attacks With Momentum},\n"
            "  author={Yinpeng Dong and Fangzhou Liao and Tianyu Pang and Hang Su and "
            "Jun Zhu and Xiaolin Hu and Jianguo Li},\n"
            "  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},\n"
            "  pages={9185--9193},\n"
            "  year={2018},\n"
            "  url={https://arxiv.org/abs/1710.06081},\n"
            "}"
        ),
        ieee='Y. Dong, F. Liao, T. Pang, H. Su, J. Zhu, X. Hu, and J. Li, "Boosting adversarial attacks with momentum," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 9185-9193.',
        apa="Dong, Y., Liao, F., Pang, T., Su, H., Zhu, J., Hu, X., & Li, J. (2018). Boosting adversarial attacks with momentum. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 9185-9193.",
    ),
)


def mifgsm(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    step_size: float | None = None,
    steps: int = 10,
    decay: float = 1.0,
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    random_start: bool = False,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run Momentum Iterative FGSM (MI-FGSM).

    Args:
        model_or_fn: Differentiable ``nn.Module`` or callable mapping ``x -> outputs``.
        x: Clean input batch (real or complex tensor, shape ``(B, ...)``).
        y: Ground-truth labels/targets (or target class when ``targeted=True``).
        eps: Perturbation budget (or JSR in dB when ``norm="jsr_db"``).
        step_size: Per-step magnitude. If ``None``, defaults to ``eps_budget / steps``.
        steps: Number of momentum-accelerated steps.
        decay: Momentum decay factor ``mu`` (Dong et al. default is ``1.0``).
        norm: Constraint norm: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        loss: Built-in loss name (``"ce"``, ``"margin"``, ``"dlr"``, ``"mse"``) or callable.
        targeted: If ``True``, minimizes loss toward ``y``.
        random_start: Optional uniform start inside the constraint ball (default ``False`` per paper).
        clip_bounds: Optional valid range ``(min, max)`` for real-valued inputs.
        seed: Optional seed used when ``random_start=True``.

    Returns:
        ``AttackResult`` containing ``x_adv``, ``perturbation``, and run metadata.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if decay < 0.0:
        raise ValueError(f"decay must be >= 0, got {decay}.")

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
            attack_id=META.id,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            extra={"decay": decay},
        )

    alpha = (
        eps_budget / float(steps)
        if step_size is None
        else torch.full_like(eps_budget, float(step_size))
    )

    if random_start:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=x_clean.device)
            generator.manual_seed(seed)
        delta = random_init_perturbation(
            x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds, generator=generator
        )
    else:
        delta = torch.zeros_like(x_clean)

    momentum = torch.zeros_like(x_clean)
    view_shape = (x_clean.shape[0],) + (1,) * (x_clean.ndim - 1)

    with eval_mode(model_or_fn):
        for _ in range(steps):
            x_in = (x_clean + delta).detach().requires_grad_(True)
            objective = compute_attack_objective(
                model_or_fn, x_in, y, loss_fn, targeted=targeted
            )
            (grad,) = torch.autograd.grad(objective.sum(), x_in, only_inputs=True)
            grad = grad.detach()

            # Dong et al. (2018) Eq. (7): normalize gradient by per-sample L1 norm before accumulating.
            grad_l1 = grad.abs().reshape(x_clean.shape[0], -1).sum(dim=1).view(view_shape)
            grad_normed = grad / grad_l1.clamp(min=1e-12)
            momentum = decay * momentum + grad_normed

            delta = delta + alpha * step_direction(momentum, norm=norm)
            delta = project_perturbation(
                delta, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds
            )

    return AttackResult(
        x_adv=x_clean + delta,
        perturbation=delta,
        eps=eps,
        norm=norm,
        steps_run=steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={"decay": decay},
    )
