"""TI-FGSM: Translation-Invariant Iterative FGSM (Dong et al., CVPR 2019).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Canonical Defended-Model Transfer Benchmark)
- Reference     : Dong, Y., Pang, T., Su, H., & Zhu, J. (2019). "Evading Defenses
                  to Transferable Adversarial Examples by Translation-Invariant
                  Attacks." CVPR 2019 (Oral, pp. 4312-4321).
- URL           : https://arxiv.org/abs/1904.02884
- Plain Intuition:
    Defended networks often focus on slightly shifted discriminative features or
    smooth out sharp single-pixel spikes. Instead of paying the compute cost to
    evaluate gradients over dozens of shifted copies of the input, Dong et al.
    prove that optimizing over an ensemble of translations reduces efficiently to
    **convolving the gradient with a symmetric smoothing kernel W** (e.g., Gaussian,
    linear, or uniform). This produces spatially/temporally smooth perturbations
    that bypass defenses such as adversarial training and random smoothing.
- Diff vs. MI-FGSM (Dong et al., 2018):
    Convolves the raw input gradient ``grad`` with smoothing kernel ``W`` along
    spatial dimensions (2D conv for ``(B, C, H, W)`` images) or sequence dimension
    (1D conv for ``(B, C, L)`` or ``(B, L)`` waveforms) before L1 normalization.
- Why Chosen as a Baseline:
    The de facto transfer attack baseline specifically targeting adversarially
    trained or denoised victim models.
- Extension Directions / Prompt Hooks:
    1. Swap ``kernel_name`` (``"gaussian"``, ``"linear"``, ``"uniform"``) or supply a
       domain-matched spectral filter kernel (e.g., wireless pulse-shaping RRC filter).
    2. Combine with NI-FGSM look-ahead + DI-FGSM diversity -> SI-NI-TI-DIM.
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
    id="ti_fgsm",
    name="Translation-Invariant FGSM (TI-FGSM)",
    family="transferability",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="low",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "timeseries", "wireless"),
    parent_id="mi_fgsm",
    diff_from_parent="Convolves surrogate gradients with a symmetric 2D/1D smoothing kernel W (Gaussian/linear/uniform) to approximate translation-ensemble optimization.",
    summary_plain="Convolves gradients with a Gaussian/smoothing kernel so perturbations are translation-invariant and evade defended victims.",
    when_to_use="When transferring adversarial attacks onto adversarially trained models, CNN/ViT image models, or shift-sensitive 1D IQ/waveform models.",
    when_not_to_use="On unordered tabular feature vectors where neighboring coordinates have no spatial or temporal adjacency.",
    extension_hooks=(
        "Swap smoothing kernel for a domain-specific bandpass/RRC filter in wireless RF jamming",
        "Compose with DI-FGSM diversity and NI-FGSM look-ahead for SOTA black-box transfer",
    ),
    module_path="wannado.adv.attacks.transferability.tifgsm_dong2019",
    citation=AttackCitation(
        key="dong2019evading",
        title="Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks",
        authors="Yinpeng Dong, Tianyu Pang, Hang Su, and Jun Zhu",
        venue="IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
        year=2019,
        url="https://arxiv.org/abs/1904.02884",
        bibtex=(
            "@inproceedings{dong2019evading,\n"
            "  title={Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks},\n"
            "  author={Dong, Yinpeng and Pang, Tianyu and Su, Hang and Zhu, Jun},\n"
            "  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},\n"
            "  pages={4312--4321},\n"
            "  year={2019},\n"
            "  url={https://arxiv.org/abs/1904.02884},\n"
            "}"
        ),
        ieee='Y. Dong, T. Pang, H. Su, and J. Zhu, "Evading defenses to transferable adversarial examples by translation-invariant attacks," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 4312-4321.',
        apa="Dong, Y., Pang, T., Su, H., & Zhu, J. (2019). Evading defenses to transferable adversarial examples by translation-invariant attacks. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 4312-4321.",
    ),
)


def _build_1d_kernel(
    kernlen: int,
    nsig: float,
    kernel_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build normalized 1D symmetric smoothing kernel matching Dong et al. (2019)."""
    if kernel_name == "gaussian":
        x = torch.linspace(-nsig, nsig, kernlen, device=device, dtype=dtype)
        # Approximate bin integration (matching scipy/Dong et al. Gaussian kernel construction).
        step = (2.0 * nsig / max(kernlen - 1, 1)) if kernlen > 1 else 1.0
        half = 0.5 * step
        cdf_hi = 0.5 * (1.0 + torch.erf((x + half) / (2.0**0.5)))
        cdf_lo = 0.5 * (1.0 + torch.erf((x - half) / (2.0**0.5)))
        kern1d = (cdf_hi - cdf_lo).clamp(min=0.0)
    elif kernel_name == "linear":
        center = kernlen // 2
        dist = (torch.arange(kernlen, device=device, dtype=dtype) - center).abs()
        kern1d = (center + 1.0 - dist).clamp(min=0.0)
    elif kernel_name == "uniform":
        kern1d = torch.ones(kernlen, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown kernel_name '{kernel_name}'. Expected 'gaussian', 'linear', or 'uniform'.")

    return kern1d / kern1d.sum().clamp(min=1e-12)


def _smooth_gradient(
    grad: torch.Tensor,
    kernlen: int,
    nsig: float,
    kernel_name: str,
) -> torch.Tensor:
    """Convolve input gradients with translation-invariant kernel W (2D/1D, complex-safe)."""
    real_dtype = grad.real.dtype if grad.is_complex() else grad.dtype
    k1 = _build_1d_kernel(kernlen, nsig, kernel_name, grad.device, real_dtype)
    pad = kernlen // 2

    def _conv_real(g: torch.Tensor) -> torch.Tensor:
        if g.ndim == 4:
            c = g.shape[1]
            k2 = torch.outer(k1, k1)
            k2 = (k2 / k2.sum().clamp(min=1e-12)).view(1, 1, kernlen, kernlen).expand(c, 1, kernlen, kernlen)
            return F.conv2d(g, k2, padding=pad, groups=c)
        if g.ndim == 3:
            c = g.shape[1]
            k = k1.view(1, 1, kernlen).expand(c, 1, kernlen)
            return F.conv1d(g, k, padding=pad, groups=c)
        if g.ndim == 2:
            # Treat 2D sequence/IQ tensor (B, L) as 1-channel 1D signal for temporal shift invariance.
            k = k1.view(1, 1, kernlen)
            return F.conv1d(g.unsqueeze(1), k, padding=pad).squeeze(1)
        return g

    if grad.is_complex():
        return torch.complex(_conv_real(grad.real), _conv_real(grad.imag))
    return _conv_real(grad)


def ti_fgsm(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    step_size: float | None = None,
    steps: int = 10,
    decay: float = 1.0,
    kernlen: int = 7,
    nsig: float = 3.0,
    kernel_name: str = "gaussian",
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    random_start: bool = False,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run Translation-Invariant Iterative FGSM (TI-FGSM / TI-MI-FGSM)."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if decay < 0.0:
        raise ValueError(f"decay must be >= 0, got {decay}.")
    if kernlen < 1 or kernlen % 2 == 0:
        raise ValueError(f"kernlen must be a positive odd integer, got {kernlen}.")

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
            extra={"decay": decay, "kernlen": kernlen, "nsig": nsig, "kernel_name": kernel_name},
        )

    alpha = (
        eps_budget / float(steps)
        if step_size is None
        else torch.full_like(eps_budget, float(step_size))
    )

    if random_start:
        generator = make_generator(seed, x_clean.device)
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

            # [AI-Prompt Hook]: Convolve gradients with translation kernel W (Dong et al., 2019, Eq. 9).
            # Swap `_smooth_gradient` with a spectral/matched filter for RF spectral shaping.
            grad = _smooth_gradient(grad, kernlen=kernlen, nsig=nsig, kernel_name=kernel_name)

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
        extra={"decay": decay, "kernlen": kernlen, "nsig": nsig, "kernel_name": kernel_name},
    )
