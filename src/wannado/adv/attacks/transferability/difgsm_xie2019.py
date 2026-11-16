"""DI-FGSM: Diverse Inputs Iterative FGSM (Xie et al., CVPR 2019).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Classic Spatial Input-Diversity Transfer Benchmark)
- Reference     : Xie, C., Zhang, Z., Zhou, Y., Bai, S., Wang, J., Ren, Z.,
                  & Yuille, A. L. (2019). "Improving Transferability of Adversarial
                  Examples With Input Diversity." CVPR 2019 (pp. 2730-2739).
- URL           : https://arxiv.org/abs/1803.06978
- Plain Intuition:
    Iterative attacks easily overfit to the exact spatial pixel coordinates or
    sampling grid of a white-box surrogate model. DI-FGSM inserts a stochastic
    differentiable transform T(x; p) before computing gradients: with probability
    ``diversity_prob``, it randomly resizes the spatial tensor and pads it back,
    forcing the crafted perturbation to stay effective across geometric shifts.
- Diff vs. MI-FGSM (Dong et al., 2018):
    Applies stochastic resizing + zero-padding ``T(x_adv)`` before the forward
    and backward pass at probability ``diversity_prob=0.5`` (coupled with momentum).
- Shape-Safe Design in wannado.adv:
    1. On 4D ``(B, C, H, W)`` images (or 3D ``(B, C, L)`` sequences/signals), randomly
       resamples inside ``[round(dim * min_scale), dim]`` and randomly zero-pads back
       to the original shape so fixed-size MLP/Flatten heads never crash (or set
       ``preserve_shape=False`` with ``resize_rate > 1`` for variable-size CNNs).
    2. Accepts an optional ``diversity_transform`` callable hook for 2D/wireless/custom
       domain diversities (e.g., stochastic Doppler/sample-rate jitter).
- Extension Directions / Prompt Hooks:
    1. Pair with Gaussian gradient convolution -> DI-TI-MI-FGSM.
    2. Supply a domain-specific differentiable ``diversity_transform`` callable.
===============================================================================
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    compute_attack_objective,
    eval_mode,
    make_generator,
    project_perturbation,
    random_init_perturbation,
    resolve_eps_budget,
    resolve_loss_fn,
    step_direction,
)
from wannado.adv.core.losses import LossFn
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="di_fgsm",
    name="Diverse Inputs Iterative FGSM (DI-FGSM)",
    family="transferability",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="low",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "timeseries", "wireless"),
    parent_id="mi_fgsm",
    diff_from_parent="Applies stochastic differentiable resize + zero-pad input diversity T(x; p) prior to each surrogate forward/backward step.",
    summary_plain="Randomly resizes and zero-pads inputs with probability p each step so perturbations do not overfit surrogate spatial grids.",
    when_to_use="Cross-architecture transfer attacks on CNNs/ViTs (4D images) or temporal/IQ sequence classifiers (3D) and custom diversity hooks.",
    when_not_to_use="Purely tabular/unordered vector inputs without a meaningful spatial/temporal neighborhood or custom diversity_transform (use mi_fgsm or vmi_fgsm).",
    extension_hooks=(
        "Pass custom differentiable `diversity_transform` for 2D/wireless sample-rate or time-warping diversity",
        "Compose with TI-FGSM gradient smoothing kernel for defended CV targets",
    ),
    module_path="wannado.adv.attacks.transferability.difgsm_xie2019",
    citation=AttackCitation(
        key="xie2019improving",
        title="Improving Transferability of Adversarial Examples With Input Diversity",
        authors="Cihang Xie, Zhishuai Zhang, Yuyin Zhou, Song Bai, Jianyu Wang, Zhou Ren, and Alan L. Yuille",
        venue="IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
        year=2019,
        url="https://arxiv.org/abs/1803.06978",
        bibtex=(
            "@inproceedings{xie2019improving,\n"
            "  title={Improving Transferability of Adversarial Examples With Input Diversity},\n"
            "  author={Xie, Cihang and Zhang, Zhishuai and Zhou, Yuyin and Bai, Song and Wang, Jianyu and Ren, Zhou and Yuille, Alan L.},\n"
            "  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},\n"
            "  pages={2730--2739},\n"
            "  year={2019},\n"
            "  url={https://arxiv.org/abs/1803.06978},\n"
            "}"
        ),
        ieee='C. Xie et al., "Improving transferability of adversarial examples with input diversity," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 2730-2739.',
        apa="Xie, C., Zhang, Z., Zhou, Y., Bai, S., Wang, J., Ren, Z., & Yuille, A. L. (2019). Improving transferability of adversarial examples with input diversity. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2730-2739.",
    ),
)


def _interpolate_complex_safe(x: torch.Tensor, size: tuple[int, ...], mode: str) -> torch.Tensor:
    """Apply F.interpolate while preserving complex64/complex128 gradients cleanly."""
    align = False
    if x.is_complex():
        re = F.interpolate(x.real, size=size, mode=mode, align_corners=align)
        im = F.interpolate(x.imag, size=size, mode=mode, align_corners=align)
        return torch.complex(re, im)
    return F.interpolate(x, size=size, mode=mode, align_corners=align)


def _default_input_diversity(
    x: torch.Tensor,
    diversity_prob: float,
    resize_rate: float,
    preserve_shape: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Apply Xie et al. (CVPR 2019) stochastic resize + random zero-padding."""
    prob_draw = float(torch.rand((), device=x.device, generator=generator).item())
    if prob_draw >= diversity_prob:
        return x

    if x.ndim == 4:
        _, _, h, w = x.shape
        if preserve_shape:
            min_h = max(1, min(h - 1, int(round(h / resize_rate)))) if h > 1 else 1
            min_w = max(1, min(w - 1, int(round(w / resize_rate)))) if w > 1 else 1
            rnd_h = int(torch.randint(min_h, h + 1, (), device=x.device, generator=generator).item())
            rnd_w = int(torch.randint(min_w, w + 1, (), device=x.device, generator=generator).item())
            resized = _interpolate_complex_safe(x, size=(rnd_h, rnd_w), mode="bilinear")
            pad_h, pad_w = h - rnd_h, w - rnd_w
        else:
            out_h = max(h + 1, int(round(h * resize_rate)))
            out_w = max(w + 1, int(round(w * resize_rate)))
            rnd_h = int(torch.randint(h, out_h + 1, (), device=x.device, generator=generator).item())
            rnd_w = int(torch.randint(w, out_w + 1, (), device=x.device, generator=generator).item())
            resized = _interpolate_complex_safe(x, size=(rnd_h, rnd_w), mode="bilinear")
            pad_h, pad_w = out_h - rnd_h, out_w - rnd_w

        top = int(torch.randint(0, pad_h + 1, (), device=x.device, generator=generator).item())
        left = int(torch.randint(0, pad_w + 1, (), device=x.device, generator=generator).item())
        return F.pad(resized, (left, pad_w - left, top, pad_h - top))

    if x.ndim == 3:
        _, _, length = x.shape
        min_l = max(1, min(length - 1, int(round(length / resize_rate)))) if length > 1 else 1
        rnd_l = int(torch.randint(min_l, length + 1, (), device=x.device, generator=generator).item())
        resized = _interpolate_complex_safe(x, size=(rnd_l,), mode="linear")
        pad_l = length - rnd_l
        left = int(torch.randint(0, pad_l + 1, (), device=x.device, generator=generator).item())
        return F.pad(resized, (left, pad_l - left))

    # [AI-Prompt Hook]: For 2D feature/IQ tensors (B, D), fallback to random feature cutout diversity
    # unless a domain-specific `diversity_transform` callable is supplied.
    keep_prob = min(0.95, max(0.5, 1.0 / resize_rate))
    mask = torch.bernoulli(
        torch.full(x.shape, keep_prob, device=x.device, dtype=torch.float32), generator=generator
    )
    return x * (mask.to(x.dtype) / keep_prob)


def di_fgsm(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    step_size: float | None = None,
    steps: int = 10,
    decay: float = 1.0,
    diversity_prob: float = 0.5,
    resize_rate: float = 1.15,
    preserve_shape: bool = True,
    diversity_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    random_start: bool = False,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run Diverse Inputs Iterative FGSM (DI-FGSM / DI^2-FGSM)."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if decay < 0.0:
        raise ValueError(f"decay must be >= 0, got {decay}.")
    if not 0.0 <= diversity_prob <= 1.0:
        raise ValueError(f"diversity_prob must be in [0, 1], got {diversity_prob}.")
    if resize_rate < 1.0:
        raise ValueError(f"resize_rate must be >= 1.0, got {resize_rate}.")

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
            extra={"decay": decay, "diversity_prob": diversity_prob, "resize_rate": resize_rate},
        )

    alpha = (
        eps_budget / float(steps)
        if step_size is None
        else torch.full_like(eps_budget, float(step_size))
    )
    generator = make_generator(seed, x_clean.device)

    if random_start:
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
            if diversity_transform is not None:
                prob_draw = float(torch.rand((), device=x_clean.device, generator=generator).item())
                x_div = diversity_transform(x_in) if prob_draw < diversity_prob else x_in
            else:
                x_div = _default_input_diversity(
                    x_in,
                    diversity_prob=diversity_prob,
                    resize_rate=resize_rate,
                    preserve_shape=preserve_shape,
                    generator=generator,
                )

            objective = compute_attack_objective(
                model_or_fn, x_div, y, loss_fn, targeted=targeted
            )
            (grad,) = torch.autograd.grad(objective.sum(), x_in, only_inputs=True)
            grad = grad.detach()

            grad_l1 = grad.abs().reshape(x_clean.shape[0], -1).sum(dim=1).view(view_shape)
            momentum = decay * momentum + grad / grad_l1.clamp(min=1e-12)

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
        extra={"decay": decay, "diversity_prob": diversity_prob, "resize_rate": resize_rate},
    )
