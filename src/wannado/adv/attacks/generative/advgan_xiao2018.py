"""AdvGAN: Generating Adversarial Examples with Adversarial Networks (Xiao et al., IJCAI 2018).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : A (Canonical GAN-Based Adversarial Generator + Discriminator)
- Reference     : Xiao, C., Li, B., Zhu, J.-Y., He, W., Liu, M., & Song, D.
                  (2018). "Generating Adversarial Examples with Adversarial
                  Networks." 27th International Joint Conference on Artificial
                  Intelligence (IJCAI-18), pp. 3905-3911.
- URL           : https://arxiv.org/abs/1801.02610 (DOI: 10.24963/ijcai.2018/543)
- Plain Intuition:
    GAP optimizes a perturbation generator solely for fooling loss. AdvGAN adds
    a discriminator ``D`` in a GAN game alongside ``G``:
      1. ``D`` learns to distinguish clean inputs ``x`` from perturbed inputs
         ``x + G(x)`` so adversarial outputs remain on the natural data manifold.
      2. ``G`` jointly minimizes (a) victim attack loss ``L_adv``, (b) GAN
         realism loss fooling ``D``, and (c) a soft hinge penalty shrinking ``||G(x)||_2``.
    Once trained, ``G`` outputs realistic norm-bounded adversarial examples in a
    single forward pass without further optimization!
- Diff vs. GAP (Poursaeed et al., 2018):
    Introduces adversarial discriminator ``D(x + G(x))`` + hinge perturbation
    regularization (``L_adv + lambda_gan * L_GAN + lambda_hinge * L_hinge``).
- Why Chosen as a Baseline:
    The canonical GAN formulation for stealthy, perceptual/distribution-matched
    single-pass adversarial generation.
- Extension Directions / Prompt Hooks:
    1. Supply domain-tailored ``generator`` and ``discriminator`` networks.
    2. Use semi-white-box distilled proxy ``model_or_fn`` for black-box AdvGAN.
===============================================================================
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from wannado.adv.attacks.generative.gap_poursaeed2018 import (
    _ShapeAdaptivePerturbGenerator,
    _pack_to_real,
    _smooth_budget_delta,
    _unpack_from_real,
    _victim_grad_guard,
)
from wannado.adv.attacks.minimum_norm.cw_l2_carlini2017 import _cw_margin_objective
from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    compute_attack_objective,
    eval_mode,
    make_generator,
    project_perturbation,
    resolve_eps_budget,
    resolve_loss_fn,
    safe_l2_norm,
)
from wannado.adv.core.losses import LossFn
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="advgan",
    name="AdvGAN",
    family="generative",
    tier="A",
    threat_model="generative_train",
    requires_training=True,
    compute_cost="very_high",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries"),
    parent_id="gap",
    diff_from_parent="Jointly trains generator G(x) and realism discriminator D(x + G(x)) with L_adv + lambda_gan * L_GAN + lambda_hinge * L_hinge.",
    summary_plain="Trains a GAN generator and discriminator together so synthesized adversarial perturbations look natural and fool victims in one forward pass.",
    when_to_use="When distribution/spectral realism matters alongside zero-step test-time attack inference.",
    when_not_to_use="When fast one-off white-box evaluation suffices (use apgd_ce or autoattack) or training sample count is tiny.",
    extension_hooks=(
        "Swap `_ShapeAdaptiveDiscriminator` with a multi-scale PatchGAN or RF PSD discriminator",
        "Train against a distilled surrogate classifier for black-box AdvGAN (Xiao et al. Section 3.3)",
    ),
    module_path="wannado.adv.attacks.generative.advgan_xiao2018",
    citation=AttackCitation(
        key="xiao2018generating",
        title="Generating Adversarial Examples with Adversarial Networks",
        authors="Chaowei Xiao, Bo Li, Jun-Yan Zhu, Warren He, Mingyan Liu, and Dawn Song",
        venue="27th International Joint Conference on Artificial Intelligence (IJCAI)",
        year=2018,
        url="https://arxiv.org/abs/1801.02610",
        bibtex=(
            "@inproceedings{xiao2018generating,\n"
            "  title={Generating Adversarial Examples with Adversarial Networks},\n"
            "  author={Xiao, Chaowei and Li, Bo and Zhu, Jun-Yan and He, Warren and Liu, Mingyan and Song, Dawn},\n"
            "  booktitle={Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence (IJCAI-18)},\n"
            "  pages={3905--3911},\n"
            "  year={2018},\n"
            "  doi={10.24963/ijcai.2018/543},\n"
            "  url={https://arxiv.org/abs/1801.02610},\n"
            "}"
        ),
        ieee='C. Xiao, B. Li, J.-Y. Zhu, W. He, M. Liu, and D. Song, "Generating adversarial examples with adversarial networks," in Proc. 27th Int. Joint Conf. Artif. Intell. (IJCAI), 2018, pp. 3905-3911, doi: 10.24963/ijcai.2018/543.',
        apa="Xiao, C., Li, B., Zhu, J.-Y., He, W., Liu, M., & Song, D. (2018). Generating adversarial examples with adversarial networks. Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence (IJCAI), 3905-3911.",
    ),
)


class _ShapeAdaptiveDiscriminator(nn.Module):
    """Lightweight discriminator outputting real/fake scalar logit per sample."""

    def __init__(self, sample_shape: tuple[int, ...], is_complex: bool, hidden: int = 64) -> None:
        super().__init__()
        mult = 2 if is_complex else 1
        if len(sample_shape) == 3:
            c = sample_shape[0] * mult
            h = max(16, min(hidden, 32))
            self.net = nn.Sequential(
                nn.Conv2d(c, h, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(h, 1),
            )
        elif len(sample_shape) == 2:
            c = sample_shape[0] * mult
            h = max(16, min(hidden, 32))
            self.net = nn.Sequential(
                nn.Conv1d(c, h, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(h, 1),
            )
        else:
            dim = math.prod(sample_shape) * mult
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(dim, hidden),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden, 1),
            )

    def forward(self, x_packed: torch.Tensor) -> torch.Tensor:
        return self.net(x_packed).squeeze(-1)


def advgan(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    epochs: int = 40,
    lr_g: float = 1e-2,
    lr_d: float = 5e-3,
    lambda_gan: float = 0.25,
    lambda_hinge: float = 0.1,
    hinge_bound: float | None = None,
    confidence: float = 0.0,
    norm: str = "linf",
    loss: str | LossFn | None = "cw",
    targeted: bool = False,
    generator: nn.Module | None = None,
    discriminator: nn.Module | None = None,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Train AdvGAN (generator + discriminator) and return bounded adversarial examples."""
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}.")
    if clip_bounds is not None and x.is_complex():
        raise ValueError("clip_bounds is only supported for real-valued tensors.")

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
            extra={"g_loss_history": [], "d_loss_history": []},
        )

    _ = make_generator(seed, x_clean.device)
    if seed is not None:
        torch.manual_seed(seed)

    real_dtype = x_clean.real.dtype if x_clean.is_complex() else x_clean.dtype
    gen_net = (
        generator
        if generator is not None
        else _ShapeAdaptivePerturbGenerator(tuple(x_clean.shape[1:]), x_clean.is_complex()).to(
            device=x_clean.device, dtype=real_dtype
        )
    )
    disc_net = (
        discriminator
        if discriminator is not None
        else _ShapeAdaptiveDiscriminator(tuple(x_clean.shape[1:]), x_clean.is_complex()).to(
            device=x_clean.device, dtype=real_dtype
        )
    )

    gen_net.train()
    disc_net.train()
    opt_g = torch.optim.Adam(gen_net.parameters(), lr=lr_g)
    opt_d = torch.optim.Adam(disc_net.parameters(), lr=lr_d)
    custom_loss_fn = None if loss == "cw" else resolve_loss_fn(loss, targeted=targeted)

    x_packed = _pack_to_real(x_clean)
    hinge_c = (
        float(hinge_bound)
        if hinge_bound is not None
        else float(eps_budget.mean().item()) * 0.5
    )

    g_history: list[float] = []
    d_history: list[float] = []
    best_fool_score = float("-inf")
    best_state = {k: v.detach().clone() for k, v in gen_net.state_dict().items()}

    with eval_mode(model_or_fn), _victim_grad_guard(model_or_fn):
        for _ in range(epochs):
            # 1. Discriminator update: distinguish clean x (real=1) vs x + G(x) (fake=0)
            with torch.no_grad():
                delta_d = _smooth_budget_delta(
                    _unpack_from_real(gen_net(x_packed), x_clean), eps_budget, norm=norm
                )
                x_fake_d = x_clean + delta_d
                if clip_bounds is not None:
                    x_fake_d = torch.clamp(x_fake_d, min=clip_bounds[0], max=clip_bounds[1])
                x_fake_packed = _pack_to_real(x_fake_d)

            opt_d.zero_grad()
            real_logits = disc_net(x_packed)
            fake_logits = disc_net(x_fake_packed)
            loss_d = 0.5 * (
                F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            )
            loss_d.backward()
            opt_d.step()
            d_history.append(float(loss_d.item()))

            # 2. Generator update: fool victim + fool discriminator + hinge perturbation penalty
            # [AI-Prompt Hook]: Xiao et al. (2018) Eq. (5) L = L_adv + alpha * L_GAN + beta * L_hinge.
            opt_g.zero_grad()
            raw_g = _unpack_from_real(gen_net(x_packed), x_clean)
            delta_g = _smooth_budget_delta(raw_g, eps_budget, norm=norm)
            x_adv_g = x_clean + delta_g
            if clip_bounds is not None:
                x_adv_g = torch.clamp(x_adv_g, min=clip_bounds[0], max=clip_bounds[1])

            if custom_loss_fn is None:
                logits_v = model_or_fn(x_adv_g)
                cw_loss, _ = _cw_margin_objective(
                    logits_v, y, confidence=confidence, targeted=targeted
                )
                loss_adv = cw_loss.mean()
                fool_score = float((-cw_loss).mean().item())
            else:
                obj_v = compute_attack_objective(
                    model_or_fn, x_adv_g, y, custom_loss_fn, targeted=targeted
                )
                loss_adv = -obj_v.mean()
                fool_score = float(obj_v.mean().item())

            d_adv_logits = disc_net(_pack_to_real(x_adv_g))
            loss_gan = F.binary_cross_entropy_with_logits(
                d_adv_logits, torch.ones_like(d_adv_logits)
            )
            delta_l2 = safe_l2_norm(delta_g.reshape(x_clean.shape[0], -1), dim=1)
            loss_hinge = torch.clamp(delta_l2 - hinge_c, min=0.0).mean()

            loss_g = loss_adv + lambda_gan * loss_gan + lambda_hinge * loss_hinge
            loss_g.backward()
            opt_g.step()
            g_history.append(float(loss_g.item()))

            if fool_score > best_fool_score:
                best_fool_score = fool_score
                best_state = {k: v.detach().clone() for k, v in gen_net.state_dict().items()}

    gen_net.load_state_dict(best_state)
    gen_net.eval()
    disc_net.eval()

    @torch.no_grad()
    def _generate_fn(x_new: torch.Tensor) -> torch.Tensor:
        x_n = x_new.detach()
        b_new = resolve_eps_budget(x_n, eps=eps, norm=norm)
        raw_d = _unpack_from_real(gen_net(_pack_to_real(x_n)), x_n)
        d = _smooth_budget_delta(raw_d, b_new, norm=norm)
        d = project_perturbation(d, x_n, b_new, norm=norm, clip_bounds=clip_bounds)
        return x_n + d

    x_adv = _generate_fn(x_clean)
    return AttackResult(
        x_adv=x_adv,
        perturbation=x_adv - x_clean,
        eps=eps,
        norm=norm,
        steps_run=epochs,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "generator": gen_net,
            "discriminator": disc_net,
            "generate_fn": _generate_fn,
            "g_loss_history": g_history,
            "d_loss_history": d_history,
        },
    )
