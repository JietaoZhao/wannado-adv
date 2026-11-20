"""Fast-AT: Fast Adversarial Training with Catastrophic Overfitting Guard (Wong et al., ICLR 2020).

===============================================================================
[Academic Provenance & Hardening Role]
- Tier      : S (Fastest Production-Scale Single-Step Adversarial Training Baseline)
- Reference : Wong, E., Rice, L., & Kolter, J. Z. (2020).
              "Fast is better than free: Revisiting adversarial training." ICLR 2020.
- URL       : https://arxiv.org/abs/2001.03994
- Plain Intuition:
    Standard 7-10 step PGD-AT costs 8x-11x normal training time. Fast-AT achieves
    comparable robustness at ~2x clean cost by pairing a random initialization
    inside the constraint ball with a single large-step FGSM update (alpha = 1.25 * eps).
    Because unconstrained single-step training can suffer from Catastrophic
    Overfitting (sudden collapse of multi-step PGD accuracy despite high 1-step
    FGSM accuracy), this implementation embeds an automatic Catastrophic
    Overfitting (CO) Guard that monitors multi-step PGD vs. FGSM accuracy each
    epoch, rolls back to the best robust checkpoint, and shrinks step size + LR
    automatically on collapse.
===============================================================================
"""

from __future__ import annotations

import copy
import time
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from wannado.adv.attacks.first_order.pgd_madry2018 import pgd
from wannado.adv.core.losses import normalize_classification_logits, predict_labels
from wannado.adv.core.projections import (
    project_perturbation,
    random_init_perturbation,
    resolve_eps_budget,
    step_direction,
)
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
    id="fast_at",
    name="Fast Adversarial Training with CO Guard (Fast-AT)",
    paradigm="adversarial_training",
    tier="S",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    summary_plain="Single-step random-init FGSM adversarial training (~5-10x faster than PGD-AT) with automatic Catastrophic Overfitting detection and rollback.",
    when_to_use="Large-scale datasets or tight GPU budgets where 7-10 step PGD-AT/TRADES is too slow, while still protecting against catastrophic overfitting.",
    when_not_to_use="Small datasets where full multi-step TRADES/MART fits easily in compute budget, or when certified L2 radii are required.",
    module_path="wannado.adv.defenses.fast_at_wong2020",
    citation=AttackCitation(
        key="wong2020fast",
        title="Fast is better than free: Revisiting adversarial training",
        authors="Eric Wong, Leslie Rice, and J. Zico Kolter",
        venue="8th International Conference on Learning Representations (ICLR)",
        year=2020,
        url="https://arxiv.org/abs/2001.03994",
        bibtex=(
            "@inproceedings{wong2020fast,\n"
            "  title={Fast is better than free: Revisiting adversarial training},\n"
            "  author={Wong, Eric and Rice, Leslie and Kolter, J. Zico},\n"
            "  booktitle={International Conference on Learning Representations (ICLR)},\n"
            "  year={2020},\n"
            "  url={https://arxiv.org/abs/2001.03994}\n"
            "}"
        ),
        ieee='E. Wong, L. Rice, and J. Z. Kolter, "Fast is better than free: Revisiting adversarial training," in Proc. Int. Conf. Learn. Represent. (ICLR), 2020.',
        apa="Wong, E., Rice, L., & Kolter, J. Z. (2020). Fast is better than free: Revisiting adversarial training. International Conference on Learning Representations (ICLR).",
    ),
)


def _fast_fgsm_batch(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    eps: float,
    norm: str,
    alpha_scale: float,
    clip_bounds: tuple[Any, Any] | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Generate random-start single-step FGSM adversarial batch under ``eval_mode``."""
    eps_budget = resolve_eps_budget(xb, eps, norm)
    step_scale = eps_budget * float(alpha_scale)

    delta = random_init_perturbation(
        xb,
        eps_budget,
        norm=norm,
        clip_bounds=clip_bounds,
        generator=generator,
    )
    delta = delta.detach().requires_grad_(True)
    saved_req: list[tuple[nn.Parameter, bool]] = []
    for p in model.parameters():
        saved_req.append((p, p.requires_grad))
        p.requires_grad_(False)
    try:
        with eval_mode(model):
            logits = normalize_classification_logits(model(xb + delta))
            loss = F.cross_entropy(logits, yb)
            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
    finally:
        for p, req in saved_req:
            p.requires_grad_(req)

    direction = step_direction(grad.detach(), norm)
    delta_next = project_perturbation(
        delta.detach() + step_scale * direction,
        xb,
        eps_budget,
        norm=norm,
        clip_bounds=clip_bounds,
    )
    return (xb + delta_next).detach()


def _extract_probe_batch(
    train_data: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None,
    eval_data: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    max_probe: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Collect a small deterministic batch for the per-epoch Catastrophic Overfitting probe."""
    if eval_data is not None:
        xe = eval_data[0][:max_probe].to(device)
        ye = squeeze_class_targets(eval_data[1][:max_probe].to(device), batch_size=xe.shape[0])
        return xe, ye
    if isinstance(train_data, torch.Tensor) and y is not None:
        n = min(max_probe, int(train_data.shape[0]))
        xe = train_data[:n].to(device)
        ye = squeeze_class_targets(y[:n].to(device), batch_size=n)
        return xe, ye
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    seen = 0
    for bx, by in iter_train_batches(train_data, y, batch_size=max_probe, device=device, shuffle=False):
        xs.append(bx)
        ys.append(squeeze_class_targets(by, batch_size=bx.shape[0]))
        seen += int(bx.shape[0])
        if seen >= max_probe:
            break
    if not xs:
        return None
    xe_cat = torch.cat(xs, dim=0)[:max_probe]
    ye_cat = torch.cat(ys, dim=0)[:max_probe]
    return xe_cat, ye_cat


def fast_adversarial_train(
    model: nn.Module,
    train_data: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    *,
    eps: float = 8.0 / 255.0,
    norm: str = "linf",
    alpha_scale: float = 1.25,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    clip_bounds: tuple[Any, Any] | None = None,
    eval_data: tuple[torch.Tensor, torch.Tensor] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    shuffle: bool = True,
    lr_scheduler: str | None = "cosine",
    co_guard: bool = True,
    co_pgd_steps: int = 5,
    co_probe_samples: int = 64,
    co_drop_threshold: float = 0.20,
    co_gap_threshold: float = 0.25,
    save_best: bool = True,
    seed: int = 42,
) -> DefenseTrainResult:
    """Harden a PyTorch classifier via Wong et al. (ICLR 2020) Fast-AT with Catastrophic Overfitting Guard."""
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}.")
    if alpha_scale <= 0.0:
        raise ValueError(f"alpha_scale must be positive, got {alpha_scale}.")

    t0 = time.perf_counter()
    torch.manual_seed(seed)

    dev = infer_model_device(model, torch.device("cpu"))
    opt = (
        optimizer
        if optimizer is not None
        else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    )
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        if lr_scheduler == "cosine"
        else None
    )
    shuf_gen = make_generator(seed, train_data.device) if isinstance(train_data, torch.Tensor) else None
    atk_gen = make_generator(seed + 13, dev)

    probe_pair = (
        _extract_probe_batch(train_data, y, eval_data, max_probe=max(4, int(co_probe_samples)), device=dev)
        if co_guard or eval_data is not None
        else None
    )

    cur_alpha_scale = float(alpha_scale)
    co_events = 0
    best_pgd_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for ep in range(epochs):
        ep_loss = 0.0
        n_seen = 0
        for xb, yb in iter_train_batches(
            train_data,
            y,
            batch_size=batch_size,
            device=dev,
            shuffle=shuffle,
            generator=shuf_gen,
        ):
            yb_l = squeeze_class_targets(yb, batch_size=xb.shape[0])
            x_adv = _fast_fgsm_batch(
                model,
                xb,
                yb_l,
                eps=eps,
                norm=norm,
                alpha_scale=cur_alpha_scale,
                clip_bounds=clip_bounds,
                generator=atk_gen,
            )

            model.train()
            opt.zero_grad()
            logits = normalize_classification_logits(model(x_adv))
            loss = F.cross_entropy(logits, yb_l)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bs = int(xb.shape[0])
            ep_loss += float(loss.item()) * bs
            n_seen += bs

        if sched is not None:
            sched.step()

        rec: dict[str, float] = {
            "epoch": float(ep + 1),
            "train_adv_loss": ep_loss / max(n_seen, 1),
            "alpha_scale": cur_alpha_scale,
        }

        if probe_pair is not None:
            xp, yp = probe_pair
            with eval_mode(model):
                with torch.no_grad():
                    clean_acc = float((predict_labels(model(xp)) == yp).float().mean().item())
                x_fgsm = _fast_fgsm_batch(
                    model,
                    xp,
                    yp,
                    eps=eps,
                    norm=norm,
                    alpha_scale=cur_alpha_scale,
                    clip_bounds=clip_bounds,
                    generator=make_generator(seed + 500 + ep, dev),
                )
                with torch.no_grad():
                    fgsm_acc = float((predict_labels(model(x_fgsm)) == yp).float().mean().item())
                pgd_res = pgd(
                    model,
                    xp,
                    yp,
                    eps=eps,
                    norm=norm,
                    steps=max(2, int(co_pgd_steps)),
                    clip_bounds=clip_bounds,
                    random_start=True,
                    seed=seed + 1000 + ep,
                )
                with torch.no_grad():
                    pgd_acc = float((predict_labels(model(pgd_res.x_adv)) == yp).float().mean().item())

            rec["probe_clean_acc"] = clean_acc
            rec["probe_fgsm_acc"] = fgsm_acc
            rec["probe_pgd_acc"] = pgd_acc

            # Detect catastrophic overfitting: sharp multi-step PGD collapse or wide FGSM-vs-PGD gap
            co_triggered = False
            if co_guard and best_state is not None and ep >= 1:
                sharp_drop = (best_pgd_acc - pgd_acc) >= float(co_drop_threshold)
                wide_gap = (fgsm_acc - pgd_acc) >= float(co_gap_threshold) and pgd_acc < best_pgd_acc
                if sharp_drop or wide_gap:
                    co_triggered = True
                    co_events += 1
                    model.load_state_dict(best_state)
                    cur_alpha_scale = max(0.5, cur_alpha_scale * 0.75)
                    for pg in opt.param_groups:
                        pg["lr"] = float(pg["lr"]) * 0.5
                    rec["co_rollback"] = 1.0

            if not co_triggered and pgd_acc >= best_pgd_acc:
                best_pgd_acc = pgd_acc
                best_state = copy.deepcopy(model.state_dict())

        history.append(rec)

    if save_best and best_state is not None:
        model.load_state_dict(best_state)

    final_clean: float | None = None
    final_adv: float | None = None
    if eval_data is not None:
        xe = eval_data[0].to(dev)
        ye = squeeze_class_targets(eval_data[1].to(dev), batch_size=xe.shape[0])
        with eval_mode(model):
            with torch.no_grad():
                final_clean = float((predict_labels(model(xe)) == ye).float().mean().item())
            res_eval = pgd(
                model,
                xe,
                ye,
                eps=eps,
                norm=norm,
                steps=max(5, int(co_pgd_steps)),
                clip_bounds=clip_bounds,
                seed=seed + 9999,
            )
            with torch.no_grad():
                final_adv = float((predict_labels(model(res_eval.x_adv)) == ye).float().mean().item())
    elif history and "probe_clean_acc" in history[-1]:
        final_clean = history[-1]["probe_clean_acc"]
        final_adv = best_pgd_acc if best_pgd_acc >= 0.0 else history[-1].get("probe_pgd_acc")

    return DefenseTrainResult(
        defense_id=META.id,
        model=model,
        epochs_run=epochs,
        elapsed_seconds=round(time.perf_counter() - t0, 4),
        history=history,
        final_clean_acc=final_clean,
        final_adv_acc=final_adv,
        extra={
            "eps": eps,
            "norm": norm,
            "initial_alpha_scale": float(alpha_scale),
            "final_alpha_scale": cur_alpha_scale,
            "co_guard_enabled": bool(co_guard),
            "co_events": co_events,
            "best_probe_pgd_acc": best_pgd_acc if best_pgd_acc >= 0.0 else None,
        },
        citation=asdict(META.citation),
    )


__all__ = ["META", "fast_adversarial_train"]
