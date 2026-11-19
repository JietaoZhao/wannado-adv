"""TRADES: Theoretically Principled Trade-off Between Robustness and Accuracy (Zhang et al., ICML 2019).

===============================================================================
[Academic Provenance & Hardening Role]
- Tier      : S (Gold-Standard Accuracy–Robustness Trade-Off Hardening)
- Reference : Zhang, H., Yu, Y., Jiao, J., Xing, E. P., El Ghaoui, L., & Jordan, M. I. (2019).
              "Theoretically Principled Trade-off between Robustness and Accuracy."
              ICML 2019 (PMLR 97:7472-7482).
- URL       : https://proceedings.mlr.press/v97/zhang19p.html
- Plain Intuition:
    Separates natural classification accuracy from boundary smoothness:
      L_TRADES = CE(f(x), y) + beta * max_{||delta|| <= eps} KL(f(x) || f(x + delta)).
    The inner loop finds the perturbation inside the budget that shifts the predicted
    probability distribution most strongly (without needing labels), while the outer
    update balances clean accuracy against local prediction smoothness via ``beta``.
===============================================================================
"""

from __future__ import annotations

import time
from collections.abc import Iterable
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
from wannado.adv.core.utils import eval_mode, iter_train_batches, make_generator, squeeze_class_targets
from wannado.adv.defenses.types import DefenseMeta, DefenseTrainResult

META = DefenseMeta(
    id="trades",
    name="TRADES Adversarial Training",
    paradigm="adversarial_training",
    tier="S",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    summary_plain="Balances clean accuracy and adversarial robustness via natural CE loss plus beta-weighted KL boundary smoothing.",
    when_to_use="Default hardening choice when you want strong empirical robustness with controlled clean-accuracy trade-off via beta.",
    when_not_to_use="When provable L2 certificates are required instead of empirical robustness (use randomized_smoothing).",
    module_path="wannado.adv.defenses.trades_zhang2019",
    citation=AttackCitation(
        key="zhang2019theoretically",
        title="Theoretically Principled Trade-off between Robustness and Accuracy",
        authors="Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, and Michael I. Jordan",
        venue="36th International Conference on Machine Learning (ICML)",
        year=2019,
        url="https://proceedings.mlr.press/v97/zhang19p.html",
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
            "  url={https://proceedings.mlr.press/v97/zhang19p.html},\n"
            "}"
        ),
        ieee='H. Zhang, Y. Yu, J. Jiao, E. P. Xing, L. El Ghaoui, and M. I. Jordan, "Theoretically principled trade-off between robustness and accuracy," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 97, 2019, pp. 7472-7482.',
        apa="Zhang, H., Yu, Y., Jiao, J., Xing, E. P., El Ghaoui, L., & Jordan, M. I. (2019). Theoretically principled trade-off between robustness and accuracy. Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97, 7472-7482.",
    ),
)


def _trades_inner_kl_attack(
    model: nn.Module,
    x_clean: torch.Tensor,
    *,
    eps: float,
    norm: str,
    steps: int,
    step_size: float | None,
    clip_bounds: tuple[Any, Any] | None,
    seed: int,
) -> torch.Tensor:
    """Maximize KL(softmax(f(x_clean)) || softmax(f(x_adv))) inside the constraint ball."""
    eps_budget = resolve_eps_budget(x_clean, eps=eps, norm=norm)
    if (eps_budget == 0).all():
        return x_clean.clone()
    alpha = (
        eps_budget * min(1.0, 2.5 / float(max(steps, 1)))
        if step_size is None
        else torch.full_like(eps_budget, float(step_size))
    )
    gen = make_generator(seed, x_clean.device)

    with eval_mode(model):
        with torch.no_grad():
            p_clean = F.softmax(normalize_classification_logits(model(x_clean)), dim=1)
        delta = 0.01 * random_init_perturbation(
            x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds, generator=gen
        )
        for _ in range(max(1, steps)):
            x_in = (x_clean + delta).detach().requires_grad_(True)
            log_p_adv = F.log_softmax(normalize_classification_logits(model(x_in)), dim=1)
            kl_sum = F.kl_div(log_p_adv, p_clean, reduction="sum")
            (grad,) = torch.autograd.grad(kl_sum, x_in, only_inputs=True)
            delta = delta + alpha * step_direction(grad.detach(), norm=norm)
            delta = project_perturbation(delta, x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds)
    return (x_clean + delta).detach()


def trades_train(
    model: nn.Module,
    train_data: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    *,
    eps: float = 8.0 / 255.0,
    beta: float = 6.0,
    norm: str = "linf",
    attack_steps: int = 7,
    step_size: float | None = None,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    clip_bounds: tuple[Any, Any] | None = None,
    eval_data: tuple[torch.Tensor, torch.Tensor] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    shuffle: bool = True,
    lr_scheduler: str | None = None,
    save_best: bool = False,
    seed: int = 42,
) -> DefenseTrainResult:
    """Harden a PyTorch classifier with Zhang et al. (ICML 2019) TRADES regularization."""
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}.")
    t0 = time.perf_counter()
    torch.manual_seed(seed)

    dev = next((p.device for p in model.parameters()), torch.device("cpu"))
    opt = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1)) if lr_scheduler == "cosine" else None
    shuf_gen = make_generator(seed, train_data.device) if isinstance(train_data, torch.Tensor) else None

    history: list[dict[str, float]] = []
    best_metric = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    for ep in range(epochs):
        ep_loss = 0.0
        ep_nat = 0.0
        ep_rob = 0.0
        n_seen = 0
        for b_idx, (xb, yb) in enumerate(
            iter_train_batches(train_data, y, batch_size=batch_size, device=dev, shuffle=shuffle, generator=shuf_gen)
        ):
            yb_l = squeeze_class_targets(yb, batch_size=xb.shape[0])
            x_adv = _trades_inner_kl_attack(
                model,
                xb,
                eps=eps,
                norm=norm,
                steps=attack_steps,
                step_size=step_size,
                clip_bounds=clip_bounds,
                seed=seed + ep * 1009 + b_idx,
            )

            model.train()
            opt.zero_grad()
            logits_nat = normalize_classification_logits(model(xb))
            logits_adv = normalize_classification_logits(model(x_adv))
            loss_nat = F.cross_entropy(logits_nat, yb_l)
            loss_rob = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_nat, dim=1),
                reduction="batchmean",
            )
            loss = loss_nat + float(beta) * loss_rob
            loss.backward()
            opt.step()

            bs = int(xb.shape[0])
            ep_loss += float(loss.item()) * bs
            ep_nat += float(loss_nat.item()) * bs
            ep_rob += float(loss_rob.item()) * bs
            n_seen += bs
        if sched is not None:
            sched.step()

        rec: dict[str, float] = {
            "epoch": float(ep + 1),
            "train_loss": ep_loss / max(n_seen, 1),
            "train_nat_loss": ep_nat / max(n_seen, 1),
            "train_kl_loss": ep_rob / max(n_seen, 1),
        }
        if eval_data is not None:
            xe = eval_data[0].to(dev)
            ye = squeeze_class_targets(eval_data[1].to(dev), batch_size=xe.shape[0])
            with eval_mode(model):
                with torch.no_grad():
                    rec["eval_clean_acc"] = float((predict_labels(model(xe)) == ye).float().mean().item())
                xe_adv = pgd(
                    model, xe, ye, eps=eps, steps=max(attack_steps, 10), norm=norm, clip_bounds=clip_bounds, seed=seed + ep
                ).x_adv
                with torch.no_grad():
                    rec["eval_adv_acc"] = float((predict_labels(model(xe_adv)) == ye).float().mean().item())
            if save_best and rec["eval_adv_acc"] >= best_metric:
                best_metric = rec["eval_adv_acc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(rec)

    if save_best and best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    model.eval()
    return DefenseTrainResult(
        defense_id=META.id,
        model=model,
        epochs_run=epochs,
        elapsed_seconds=time.perf_counter() - t0,
        history=history,
        final_clean_acc=history[-1].get("eval_clean_acc") if history else None,
        final_adv_acc=history[-1].get("eval_adv_acc") if history else None,
        extra={"eps": float(eps), "beta": float(beta), "norm": norm, "attack_steps": attack_steps, "lr": lr},
        citation=META.to_dict()["citation"],
    )


__all__ = ["META", "trades_train"]
