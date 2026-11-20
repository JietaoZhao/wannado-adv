"""LoRA-AWP: Parameter-Efficient LoRA + Adversarial Weight Perturbation Hardening (Hu et al., 2022; Wu et al., 2020).

===============================================================================
[Academic Provenance & Hardening Role]
- Tier      : S (Parameter-Efficient Production Backbone Hardening)
- Reference : Wu, D., Xia, S.-T., & Wang, Y. (2020).
              "Adversarial Weight Perturbation Helps Robust Generalization." NeurIPS 2020.
              Hu, E. J., et al. (2022).
              "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
- Plain Intuition:
    Retraining a large pretrained backbone from scratch with multi-step PGD-AT is
    prohibitively expensive. LoRA-AWP freezes base weights, injects lightweight
    low-rank trainable adapters (`LoRALinear` / `LoRAConv2d`, zero external `peft`
    dependency), applies TRADES adversarial regularization + Adversarial Weight
    Perturbation (AWP) to flatten the weight-loss landscape, and optionally merges
    adapters back into the original layers for ZERO inference latency overhead.
===============================================================================
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from wannado.adv.attacks.first_order.pgd_madry2018 import pgd
from wannado.adv.core.losses import normalize_classification_logits, predict_labels
from wannado.adv.core.types import AttackCitation
from wannado.adv.core.utils import (
    eval_mode,
    infer_model_device,
    iter_train_batches,
    make_generator,
    squeeze_class_targets,
)
from wannado.adv.defenses.types import DefenseMeta, DefenseTrainResult

META = DefenseMeta(
    id="lora_awp",
    name="LoRA + Adversarial Weight Perturbation Hardening (LoRA-AWP)",
    paradigm="parameter_efficient_hardening",
    tier="S",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    summary_plain="Freezes backbone weights, hardens low-rank LoRA adapters with TRADES + AWP weight flattening, and merges cleanly for zero inference overhead.",
    when_to_use="Post-training adversarial hardening of pretrained backbones (ViT, ResNet, ConvNeXt, CLIP, large MLPs) where full-weight adversarial retraining is too costly.",
    when_not_to_use="Tiny from-scratch models with minimal parameter count (standard trades or mart works directly).",
    module_path="wannado.adv.defenses.lora_awp_hardening",
    citation=AttackCitation(
        key="wu2020adversarial",
        title="Adversarial Weight Perturbation Helps Robust Generalization",
        authors="Dongxian Wu, Shu-Tao Xia, and Yisen Wang",
        venue="Advances in Neural Information Processing Systems (NeurIPS)",
        year=2020,
        url="https://arxiv.org/abs/2004.05884",
        bibtex=(
            "@inproceedings{wu2020adversarial,\n"
            "  title={Adversarial Weight Perturbation Helps Robust Generalization},\n"
            "  author={Wu, Dongxian and Xia, Shu-Tao and Wang, Yisen},\n"
            "  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},\n"
            "  volume={33},\n"
            "  pages={2958--2969},\n"
            "  year={2020},\n"
            "  url={https://arxiv.org/abs/2004.05884}\n"
            "}"
        ),
        ieee='D. Wu, S.-T. Xia, and Y. Wang, "Adversarial weight perturbation helps robust generalization," in Proc. NeurIPS, vol. 33, 2020, pp. 2958-2969.',
        apa="Wu, D., Xia, S.-T., & Wang, Y. (2020). Adversarial weight perturbation helps robust generalization. NeurIPS, 33, 2958-2969.",
    ),
)


class LoRALinear(nn.Module):
    """Zero-dependency low-rank adapter wrapping an existing ``nn.Linear`` layer."""

    def __init__(self, base: nn.Linear, *, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.base = base
        r = max(1, min(int(rank), int(base.in_features), int(base.out_features)))
        self.rank = r
        self.scaling = float(alpha) / float(r)
        dev = base.weight.device
        dt = base.weight.dtype
        self.lora_a = nn.Parameter(torch.empty((r, base.in_features), device=dev, dtype=dt))
        self.lora_b = nn.Parameter(torch.zeros((base.out_features, r), device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
        return base_out + delta

    @torch.no_grad()
    def merge_to_linear(self) -> nn.Linear:
        """Fold ``scaling * (B @ A)`` into ``self.base.weight`` and return the plain ``nn.Linear``."""
        delta_w = (self.lora_b @ self.lora_a) * self.scaling
        self.base.weight.add_(delta_w.to(dtype=self.base.weight.dtype, device=self.base.weight.device))
        return self.base


class LoRAConv2d(nn.Module):
    """Zero-dependency low-rank adapter wrapping an existing ``nn.Conv2d`` layer (groups=1)."""

    def __init__(self, base: nn.Conv2d, *, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.base = base
        kh, kw = base.kernel_size if isinstance(base.kernel_size, tuple) else (base.kernel_size, base.kernel_size)
        in_flat = int(base.in_channels) * int(kh) * int(kw)
        r = max(1, min(int(rank), in_flat, int(base.out_channels)))
        self.rank = r
        self.scaling = float(alpha) / float(r)
        dev = base.weight.device
        dt = base.weight.dtype
        self.lora_a = nn.Parameter(torch.empty((r, base.in_channels, kh, kw), device=dev, dtype=dt))
        self.lora_b = nn.Parameter(torch.zeros((base.out_channels, r, 1, 1), device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        hidden = F.conv2d(
            x,
            self.lora_a,
            bias=None,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=1,
        )
        delta = F.conv2d(hidden, self.lora_b, bias=None) * self.scaling
        return base_out + delta

    @torch.no_grad()
    def merge_to_conv2d(self) -> nn.Conv2d:
        """Fold ``scaling * (B * A)`` into ``self.base.weight`` and return the plain ``nn.Conv2d``."""
        b_mat = self.lora_b.squeeze(-1).squeeze(-1)  # (out_c, r)
        a_mat = self.lora_a.reshape(self.rank, -1)  # (r, in_c * kh * kw)
        delta_w = (b_mat @ a_mat).reshape_as(self.base.weight) * self.scaling
        self.base.weight.add_(delta_w.to(dtype=self.base.weight.dtype, device=self.base.weight.device))
        return self.base


def inject_lora_adapters(
    model: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 8.0,
    freeze_base: bool = True,
) -> tuple[nn.Module, list[nn.Parameter]]:
    """Replace eligible ``nn.Linear`` and ``nn.Conv2d`` (groups=1) layers with LoRA adapters in-place."""
    if freeze_base:
        for p in model.parameters():
            p.requires_grad_(False)

    lora_params: list[nn.Parameter] = []

    def _walk(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, (LoRALinear, LoRAConv2d)):
                child.lora_a.requires_grad_(True)
                child.lora_b.requires_grad_(True)
                lora_params.extend([child.lora_a, child.lora_b])
            elif isinstance(child, nn.Linear):
                wrapped = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(parent, name, wrapped)
                lora_params.extend([wrapped.lora_a, wrapped.lora_b])
            elif isinstance(child, nn.Conv2d) and child.groups == 1:
                wrapped_c = LoRAConv2d(child, rank=rank, alpha=alpha)
                setattr(parent, name, wrapped_c)
                lora_params.extend([wrapped_c.lora_a, wrapped_c.lora_b])
            else:
                _walk(child)

    _walk(model)
    return model, lora_params


def merge_lora_weights(model: nn.Module, *, unfreeze_base: bool = True) -> nn.Module:
    """Merge all ``LoRALinear`` / ``LoRAConv2d`` layers back into standard PyTorch layers (zero inference overhead)."""

    def _walk(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                plain = child.merge_to_linear()
                setattr(parent, name, plain)
            elif isinstance(child, LoRAConv2d):
                plain_c = child.merge_to_conv2d()
                setattr(parent, name, plain_c)
            else:
                _walk(child)

    _walk(model)
    if unfreeze_base:
        for p in model.parameters():
            p.requires_grad_(True)
    return model


def _apply_awp_perturbation(params: list[nn.Parameter], gamma: float) -> list[torch.Tensor]:
    """Perturb trainable weights along normalized gradient ascent direction and return saved deltas."""
    saved_deltas: list[torch.Tensor] = []
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                saved_deltas.append(torch.zeros_like(p))
                continue
            g_norm = p.grad.norm(p=2).clamp(min=1e-12)
            w_norm = p.norm(p=2).clamp(min=1e-3)
            delta = (float(gamma) * w_norm / g_norm) * p.grad
            p.add_(delta)
            saved_deltas.append(delta)
    return saved_deltas


def _restore_awp_perturbation(params: list[nn.Parameter], saved_deltas: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, d in zip(params, saved_deltas):
            p.sub_(d)


def lora_adversarial_train(
    model: nn.Module,
    train_data: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    *,
    rank: int = 4,
    lora_alpha: float = 8.0,
    eps: float = 8.0 / 255.0,
    norm: str = "linf",
    attack_steps: int = 5,
    step_size: float | None = None,
    beta: float = 3.0,
    awp_gamma: float = 0.005,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    clip_bounds: tuple[Any, Any] | None = (0.0, 1.0),
    merge_after_train: bool = True,
    eval_data: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
    eval_y: torch.Tensor | None = None,
    device: str | torch.device | None = None,
    seed: int = 42,
) -> DefenseTrainResult:
    """Parameter-efficient LoRA + TRADES + AWP adversarial hardening with optional zero-overhead weight merge."""
    t0 = time.perf_counter()
    dev = torch.device(device) if device is not None else infer_model_device(model, torch.device("cpu"))
    model.to(dev)

    total_base_params = sum(int(p.numel()) for p in model.parameters())
    _, lora_params = inject_lora_adapters(model, rank=rank, alpha=lora_alpha, freeze_base=True)
    if not lora_params:
        raise ValueError("lora_adversarial_train found no eligible nn.Linear or nn.Conv2d layers to adapt.")

    trainable_params = sum(int(p.numel()) for p in lora_params)
    trainable_fraction = float(trainable_params / max(total_base_params, 1))

    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    gen = make_generator(seed, dev)

    history: list[dict[str, float]] = []
    for ep in range(1, max(1, int(epochs)) + 1):
        model.train()
        total_loss = 0.0
        total_clean_ok = 0
        total_adv_ok = 0
        total_n = 0

        for xb, yb_raw in iter_train_batches(
            train_data, y, batch_size=batch_size, device=dev, shuffle=True, generator=gen
        ):
            yb = squeeze_class_targets(yb_raw, batch_size=xb.shape[0])

            # 1. Generate inner adversarial examples cleanly under eval_mode
            with eval_mode(model):
                adv_res = pgd(
                    model,
                    xb,
                    yb,
                    eps=eps,
                    norm=norm,
                    steps=max(1, int(attack_steps)),
                    step_size=step_size,
                    clip_bounds=clip_bounds,
                    seed=seed + ep,
                )
            x_adv = adv_res.x_adv.detach()

            def _trades_loss(
                x_in: torch.Tensor,
                x_adv_in: torch.Tensor,
                y_in: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                logits_c = normalize_classification_logits(model(x_in))
                logits_a = normalize_classification_logits(model(x_adv_in))
                ce = F.cross_entropy(logits_c, y_in)
                kl = F.kl_div(
                    F.log_softmax(logits_a, dim=1),
                    F.softmax(logits_c.detach(), dim=1),
                    reduction="batchmean",
                )
                return ce + float(beta) * kl, logits_c, logits_a

            # 2. Adversarial Weight Perturbation (AWP) ascent on LoRA parameters
            saved_awp: list[torch.Tensor] | None = None
            if awp_gamma > 0.0:
                optimizer.zero_grad()
                probe_loss, _, _ = _trades_loss(xb, x_adv, yb)
                probe_loss.backward()
                saved_awp = _apply_awp_perturbation(lora_params, float(awp_gamma))

            # 3. Outer gradient update at worst-case perturbed weights, then restore AWP delta
            optimizer.zero_grad()
            loss, logits_clean, logits_adv = _trades_loss(xb, x_adv, yb)
            loss.backward()
            if saved_awp is not None:
                _restore_awp_perturbation(lora_params, saved_awp)
            torch.nn.utils.clip_grad_norm_(lora_params, 5.0)
            optimizer.step()

            n_b = int(xb.shape[0])
            total_loss += float(loss.item()) * n_b
            total_clean_ok += int((predict_labels(logits_clean.detach()) == yb).sum().item())
            total_adv_ok += int((predict_labels(logits_adv.detach()) == yb).sum().item())
            total_n += n_b

        scheduler.step()
        ep_rec: dict[str, float] = {
            "epoch": float(ep),
            "train_loss": round(total_loss / max(total_n, 1), 5),
            "train_clean_acc": round(total_clean_ok / max(total_n, 1), 4),
            "train_adv_acc": round(total_adv_ok / max(total_n, 1), 4),
        }
        history.append(ep_rec)

    if merge_after_train:
        merge_lora_weights(model, unfreeze_base=True)

    final_clean = history[-1]["train_clean_acc"] if history else None
    final_adv = history[-1]["train_adv_acc"] if history else None
    if eval_data is not None:
        with eval_mode(model):
            c_ok = 0
            a_ok = 0
            ev_n = 0
            for ex, ey_raw in iter_train_batches(eval_data, eval_y, batch_size=batch_size, device=dev, shuffle=False):
                ey = squeeze_class_targets(ey_raw, batch_size=ex.shape[0])
                with torch.no_grad():
                    c_ok += int((predict_labels(model(ex)) == ey).sum().item())
                ev_res = pgd(model, ex, ey, eps=eps, norm=norm, steps=max(1, int(attack_steps)), clip_bounds=clip_bounds, seed=seed)
                with torch.no_grad():
                    a_ok += int((predict_labels(model(ev_res.x_adv)) == ey).sum().item())
                ev_n += int(ex.shape[0])
            if ev_n > 0:
                final_clean = round(c_ok / ev_n, 4)
                final_adv = round(a_ok / ev_n, 4)

    return DefenseTrainResult(
        defense_id=META.id,
        model=model,
        epochs_run=max(1, int(epochs)),
        elapsed_seconds=round(time.perf_counter() - t0, 4),
        history=history,
        final_clean_acc=final_clean,
        final_adv_acc=final_adv,
        extra={
            "rank": int(rank),
            "lora_alpha": float(lora_alpha),
            "awp_gamma": float(awp_gamma),
            "base_param_count": total_base_params,
            "trainable_lora_params": trainable_params,
            "trainable_param_ratio": round(trainable_fraction, 4),
            "merged_to_base": bool(merge_after_train),
        },
        citation=META.to_dict()["citation"],
    )


__all__ = [
    "META",
    "LoRAConv2d",
    "LoRALinear",
    "inject_lora_adapters",
    "lora_adversarial_train",
    "merge_lora_weights",
]
