"""GAP: Generative Adversarial Perturbations (Poursaeed et al., CVPR 2018).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : A (Foundational Feed-Forward Perturbation Generator Attack)
- Reference     : Poursaeed, O., Katsman, I., Gao, B., & Belongie, S. (2018).
                  "Generative Adversarial Perturbations." CVPR 2018
                  (pp. 4422-4431).
- URL           : https://arxiv.org/abs/1712.02328
- Plain Intuition:
    Iterative attacks like PGD solve an optimization loop from scratch for
    EVERY test input—too slow for streaming or real-time applications. GAP
    trains a neural network generator ``G_theta`` once against the target model
    to synthesize norm-bounded perturbations directly in a single forward pass:
      - ``mode="input_dependent"``: ``G(x) -> delta(x)`` adapts per sample.
      - ``mode="universal"``: ``G(z) -> v`` maps fixed latent noise to one shared
        universal perturbation across all inputs.
    At test time, generating attacks takes zero iterative steps!
- Diff vs. UAP (Moosavi-Dezfooli et al., 2017):
    Replaces dataset-level geometric boundary accumulation with trainable
    feed-forward generator networks supporting both universal and conditional
    perturbation synthesis.
- Why Chosen as a Baseline:
    The canonical generative perturbation baseline for zero-step real-time attack
    synthesis (`threat_model="generative_train"`, `requires_training=True`).
- Extension Directions / Prompt Hooks:
    1. Supply custom ``generator=nn.Module`` (e.g. U-Net / ResNet / RF CNN).
    2. Add a PatchGAN discriminator for perceptual stealth -> AdvGAN (Xiao et al., 2018).
    3. Reuse ``result.extra["generate_fn"](x_unseen)`` for instant test-time evaluation.
===============================================================================
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from torch import nn

from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    clamp_to_bounds,
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
    id="gap",
    name="Generative Adversarial Perturbations (GAP)",
    family="generative",
    tier="A",
    threat_model="generative_train",
    requires_training=True,
    compute_cost="very_high",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl", "unsupervised"),
    parent_id="uap",
    diff_from_parent="Trains a neural generator G(x) or G(z) to emit norm-bounded perturbations in one feed-forward pass instead of iterative per-sample loops.",
    summary_plain="Trains a neural network generator to craft universal or input-dependent perturbations in a single inference forward pass.",
    when_to_use="When evaluating real-time/streaming vulnerability (amortized zero-step inference) or studying generative perturbation synthesis.",
    when_not_to_use="For fast single-batch evaluation or small sample counts without training overhead (use apgd_ce or uap).",
    extension_hooks=(
        "Pass a custom `generator` nn.Module (U-Net / Conv1D / Transformer)",
        "Combine with a discriminator network for distribution realism -> AdvGAN",
    ),
    module_path="wannado.adv.attacks.generative.gap_poursaeed2018",
    citation=AttackCitation(
        key="poursaeed2018generative",
        title="Generative Adversarial Perturbations",
        authors="Omid Poursaeed, Isay Katsman, Bicheng Gao, and Serge Belongie",
        venue="IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
        year=2018,
        url="https://arxiv.org/abs/1712.02328",
        bibtex=(
            "@inproceedings{poursaeed2018generative,\n"
            "  title={Generative Adversarial Perturbations},\n"
            "  author={Poursaeed, Omid and Katsman, Isay and Gao, Bicheng and Belongie, Serge},\n"
            "  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},\n"
            "  pages={4422--4431},\n"
            "  year={2018},\n"
            "  url={https://arxiv.org/abs/1712.02328},\n"
            "}"
        ),
        ieee='O. Poursaeed, I. Katsman, B. Gao, and S. Belongie, "Generative adversarial perturbations," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 4422-4431.',
        apa="Poursaeed, O., Katsman, I., Gao, B., & Belongie, S. (2018). Generative adversarial perturbations. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4422-4431.",
    ),
)


@contextmanager
def _victim_grad_guard(model_or_fn: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[None]:
    """Temporarily freeze victim weights so generator backward does not populate victim param.grad."""
    if not isinstance(model_or_fn, nn.Module):
        yield
        return
    params = list(model_or_fn.parameters())
    prev_flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(params, prev_flags):
            p.requires_grad_(flag)


def _pack_to_real(t: torch.Tensor) -> torch.Tensor:
    """Convert real or complex (B, ...) tensor to real channel-compatible tensor."""
    if not t.is_complex():
        return t
    if t.ndim == 2:
        return torch.cat([t.real, t.imag], dim=1)
    return torch.cat([t.real, t.imag], dim=1)


def _unpack_from_real(r: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Convert packed real tensor back to real or complex matching ``template``."""
    if not template.is_complex():
        return r.to(dtype=template.dtype)
    if template.ndim == 2:
        half = template.shape[1]
        return torch.complex(r[:, :half], r[:, half:]).to(dtype=template.dtype)
    c = template.shape[1]
    return torch.complex(r[:, :c], r[:, c:]).to(dtype=template.dtype)


class _ShapeAdaptivePerturbGenerator(nn.Module):
    """Lightweight residual perturbation generator adapting to 4D images, 3D waveforms, or 2D vectors."""

    def __init__(self, sample_shape: tuple[int, ...], is_complex: bool, hidden: int = 64) -> None:
        super().__init__()
        self.sample_shape = sample_shape
        self.is_complex = is_complex
        mult = 2 if is_complex else 1

        last: nn.Conv2d | nn.Conv1d | nn.Linear
        if len(sample_shape) == 3:
            # 4D batch input: (B, C, H, W)
            c = sample_shape[0] * mult
            h = max(16, min(hidden, 32))
            last = nn.Conv2d(h, c, kernel_size=3, padding=1)
            self.net = nn.Sequential(
                nn.Conv2d(c, h, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(h, h, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                last,
            )
        elif len(sample_shape) == 2:
            # 3D batch input: (B, C, L)
            c = sample_shape[0] * mult
            h = max(16, min(hidden, 32))
            last = nn.Conv1d(h, c, kernel_size=3, padding=1)
            self.net = nn.Sequential(
                nn.Conv1d(c, h, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(h, h, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                last,
            )
        else:
            # 2D batch (B, D) or arbitrary ND
            dim = math.prod(sample_shape) * mult
            last = nn.Linear(hidden, dim)
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                last,
            )

        nn.init.normal_(last.weight, mean=0.0, std=1e-2)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, z_real: torch.Tensor) -> torch.Tensor:
        out = self.net(z_real)
        if len(self.sample_shape) in (2, 3):
            return out
        mult = 2 if self.is_complex else 1
        target_shape = (z_real.shape[0], self.sample_shape[0] * mult, *self.sample_shape[1:])
        return out.view(target_shape)


def _smooth_budget_delta(
    raw_delta: torch.Tensor,
    eps_budget: torch.Tensor,
    norm: str,
) -> torch.Tensor:
    """Smoothly map unconstrained generator output into budget ball using polynomial saturation."""
    if norm == "linf":
        if raw_delta.is_complex():
            mag = raw_delta.abs()
            # d * eps / (eps + |d|) behaves like identity near 0 and saturates smoothly < eps.
            scale = eps_budget / (eps_budget + mag).clamp(min=1e-12)
            return raw_delta * scale.to(raw_delta.dtype)
        return eps_budget * (raw_delta / (1.0 + raw_delta.abs()))

    # Smooth differentiable radial squash inside L2 / relative_l2 / JSR_dB ball
    b = raw_delta.shape[0]
    d_norm = safe_l2_norm(raw_delta.reshape(b, -1), dim=1).view_as(eps_budget)
    scale = eps_budget / (eps_budget + d_norm).clamp(min=1e-12)
    return raw_delta * scale.to(raw_delta.dtype)


def gap(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    mode: str = "input_dependent",
    epochs: int = 40,
    lr: float = 1e-2,
    norm: str = "linf",
    loss: str | LossFn | None = "ce",
    targeted: bool = False,
    generator: nn.Module | None = None,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Train a Generative Adversarial Perturbations (GAP) generator and return attacks."""
    if mode not in ("input_dependent", "universal"):
        raise ValueError(f"mode must be 'input_dependent' or 'universal', got '{mode}'.")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}.")
    if lr <= 0.0:
        raise ValueError(f"lr must be > 0, got {lr}.")
    if clip_bounds is not None and x.is_complex():
        raise ValueError("clip_bounds is only supported for real-valued tensors.")

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
            extra={"mode": mode, "loss_history": []},
        )

    rng = make_generator(seed, x_clean.device)
    real_dtype = x_clean.real.dtype if x_clean.is_complex() else x_clean.dtype

    # [AI-Prompt Hook]: Swap `_ShapeAdaptivePerturbGenerator` by passing `generator=my_net`
    # expecting packed real channels `[real, imag]` for complex or native tensor shape for real inputs.
    gen_net = (
        generator
        if generator is not None
        else _ShapeAdaptivePerturbGenerator(tuple(x_clean.shape[1:]), x_clean.is_complex()).to(
            device=x_clean.device, dtype=real_dtype
        )
    )
    gen_net.train()
    optimizer = torch.optim.Adam(gen_net.parameters(), lr=lr)

    if mode == "universal":
        latent_z = torch.randn(
            (1, *x_clean.shape[1:]),
            device=x_clean.device,
            dtype=real_dtype,
            generator=rng,
        )
        if x_clean.is_complex():
            latent_z = torch.complex(latent_z, torch.randn_like(latent_z)).to(x_clean.dtype)
        latent_z_packed = _pack_to_real(latent_z)
        univ_budget = eps_budget.amin(dim=0, keepdim=True)
    else:
        x_packed = _pack_to_real(x_clean)

    loss_history: list[float] = []
    best_obj_mean = float("-inf")
    best_state = {k: v.detach().clone() for k, v in gen_net.state_dict().items()}

    with eval_mode(model_or_fn), _victim_grad_guard(model_or_fn):
        for _ in range(epochs):
            optimizer.zero_grad()
            if mode == "universal":
                raw_out = _unpack_from_real(gen_net(latent_z_packed), x_clean[:1])
                delta = _smooth_budget_delta(raw_out, univ_budget, norm=norm).expand_as(x_clean)
            else:
                raw_out = _unpack_from_real(gen_net(x_packed), x_clean)
                delta = _smooth_budget_delta(raw_out, eps_budget, norm=norm)

            x_adv_train = clamp_to_bounds(x_clean + delta, clip_bounds)

            obj = compute_attack_objective(
                model_or_fn, x_adv_train, y, loss_fn, targeted=targeted
            )
            train_loss = -obj.mean()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 1.0)
            optimizer.step()

            cur_obj = float(obj.mean().item())
            loss_history.append(float(train_loss.item()))
            if cur_obj > best_obj_mean:
                best_obj_mean = cur_obj
                best_state = {k: v.detach().clone() for k, v in gen_net.state_dict().items()}

    gen_net.load_state_dict(best_state)
    gen_net.eval()

    @torch.no_grad()
    def _generate_fn(x_new: torch.Tensor) -> torch.Tensor:
        x_n = x_new.detach()
        b_new = resolve_eps_budget(x_n, eps=eps, norm=norm)
        if mode == "universal":
            u_b = b_new.amin(dim=0, keepdim=True)
            raw_v = _unpack_from_real(gen_net(latent_z_packed.to(x_n.device)), x_n[:1])
            d = _smooth_budget_delta(raw_v, u_b, norm=norm).expand_as(x_n)
        else:
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
            "mode": mode,
            "generator": gen_net,
            "generate_fn": _generate_fn,
            "loss_history": loss_history,
        },
    )
