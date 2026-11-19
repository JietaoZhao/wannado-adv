"""MART: Misclassification-Aware Adversarial Training (Wang et al., ICLR 2020).

===============================================================================
[Academic Provenance & Hardening Role]
- Tier      : S (Margin-Boosted Misclassification-Aware Adversarial Training)
- Reference : Wang, Y., Zou, D., Yi, J., Bailey, J., Ma, X., & Gu, Q. (2020).
              "Improving Adversarial Robustness Requires Revisiting Misclassified Examples."
              ICLR 2020.
- URL       : https://openreview.net/forum?id=rklOg6EFwS
- Plain Intuition:
    Strengthens TRADES by explicitly distinguishing clean-misclassified examples from
    clean-correct ones:
      L_MART = BCE(f(x_adv), y) + beta * KL(p(x) || p(x_adv)) * (1 - p_y(x)),
    where BCE boosts decision margin by penalizing the strongest runner-up probability
    ``-log(1 - max_{k != y} p_k(x_adv))``, and ``(1 - p_y(x))`` focuses KL smoothing
    heavily on uncertain or misclassified boundary samples.
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
from wannado.adv.core.types import AttackCitation
from wannado.adv.core.utils import eval_mode, iter_eval_batches
from wannado.adv.defenses.types import DefenseMeta, DefenseTrainResult

META = DefenseMeta(
    id="mart",
    name="Misclassification-Aware Adversarial Training (MART)",
    paradigm="adversarial_training",
    tier="S",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    summary_plain="Boosts adversarial decision margins using Boosted Cross-Entropy plus (1 - p_y(x))-weighted KL smoothing on hard examples.",
    when_to_use="When maximizing worst-case adversarial accuracy on hard boundary samples beyond standard PGD-AT or TRADES.",
    when_not_to_use="When strict clean-accuracy preservation dominates over worst-case boundary hardening (prefer trades with small beta).",
    module_path="wannado.adv.defenses.mart_wang2020",
    citation=AttackCitation(
        key="wang2020improving",
        title="Improving Adversarial Robustness Requires Revisiting Misclassified Examples",
        authors="Yisen Wang, Difan Zou, Jinfeng Yi, James Bailey, Xingjun Ma, and Quanquan Gu",
        venue="8th International Conference on Learning Representations (ICLR)",
        year=2020,
        url="https://openreview.net/forum?id=rklOg6EFwS",
        bibtex=(
            "@inproceedings{wang2020improving,\n"
            "  title={Improving Adversarial Robustness Requires Revisiting Misclassified Examples},\n"
            "  author={Wang, Yisen and Zou, Difan and Yi, Jinfeng and Bailey, James and Ma, Xingjun and Gu, Quanquan},\n"
            "  booktitle={International Conference on Learning Representations (ICLR)},\n"
            "  year={2020},\n"
            "  url={https://openreview.net/forum?id=rklOg6EFwS},\n"
            "}"
        ),
        ieee='Y. Wang, D. Zou, J. Yi, J. Bailey, X. Ma, and Q. Gu, "Improving adversarial robustness requires revisiting misclassified examples," in Proc. Int. Conf. Learn. Represent. (ICLR), 2020.',
        apa="Wang, Y., Zou, D., Yi, J., Bailey, J., Ma, X., & Gu, Q. (2020). Improving adversarial robustness requires revisiting misclassified examples. International Conference on Learning Representations (ICLR).",
    ),
)


def mart_train(
    model: nn.Module,
    train_data: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    *,
    eps: float = 8.0 / 255.0,
    beta: float = 5.0,
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
    seed: int = 42,
) -> DefenseTrainResult:
    """Harden a PyTorch classifier via Wang et al. (ICLR 2020) MART."""
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}.")
    t0 = time.perf_counter()
    torch.manual_seed(seed)

    dev = next((p.device for p in model.parameters()), torch.device("cpu"))
    opt = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict[str, float]] = []
    for ep in range(epochs):
        batches = list(iter_eval_batches(train_data, y, batch_size=batch_size, device=dev))
        ep_loss = 0.0
        n_seen = 0
        for b_idx, (xb, yb) in enumerate(batches):
            yb_l = yb.to(torch.long)
            with eval_mode(model):
                x_adv = pgd(
                    model,
                    xb,
                    yb_l,
                    eps=eps,
                    step_size=step_size,
                    steps=attack_steps,
                    norm=norm,
                    clip_bounds=clip_bounds,
                    random_start=True,
                    seed=seed + ep * 1009 + b_idx,
                ).x_adv.detach()

            model.train()
            opt.zero_grad()
            logits_nat = normalize_classification_logits(model(xb))
            logits_adv = normalize_classification_logits(model(x_adv))

            adv_probs = F.softmax(logits_adv, dim=1)
            nat_probs = F.softmax(logits_nat, dim=1)
            ar = torch.arange(xb.shape[0], device=dev)

            # Boosted Cross-Entropy (BCE): CE(f(x_adv), y) - log(1 - max_{k != y} p_k(x_adv))
            top2_probs, top2_idx = torch.topk(adv_probs, k=2, dim=1)
            max_other_prob = torch.where(top2_idx[:, 0] == yb_l, top2_probs[:, 1], top2_probs[:, 0])
            loss_bce = F.cross_entropy(logits_adv, yb_l) + torch.mean(
                -torch.log((1.0 - max_other_prob).clamp(min=1e-6))
            )

            # Misclassification-weighted KL: sum_c p_nat(c) * log(p_nat(c) / p_adv(c)) * (1 - p_y(x))
            true_nat_prob = nat_probs[ar, yb_l]
            kl_per_sample = torch.sum(
                nat_probs * (torch.log(nat_probs.clamp(min=1e-8)) - F.log_softmax(logits_adv, dim=1)),
                dim=1,
            )
            loss_rob = torch.mean(kl_per_sample * (1.0 - true_nat_prob))
            loss = loss_bce + float(beta) * loss_rob
            loss.backward()
            opt.step()

            bs = int(xb.shape[0])
            ep_loss += float(loss.item()) * bs
            n_seen += bs

        rec: dict[str, float] = {"epoch": float(ep + 1), "train_mart_loss": ep_loss / max(n_seen, 1)}
        if eval_data is not None:
            xe, ye = eval_data[0].to(dev), eval_data[1].to(dev, dtype=torch.long)
            with eval_mode(model):
                with torch.no_grad():
                    rec["eval_clean_acc"] = float((predict_labels(model(xe)) == ye).float().mean().item())
                xe_adv = pgd(
                    model, xe, ye, eps=eps, steps=max(attack_steps, 10), norm=norm, clip_bounds=clip_bounds, seed=seed + ep
                ).x_adv
                with torch.no_grad():
                    rec["eval_adv_acc"] = float((predict_labels(model(xe_adv)) == ye).float().mean().item())
        history.append(rec)

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


__all__ = ["META", "mart_train"]
