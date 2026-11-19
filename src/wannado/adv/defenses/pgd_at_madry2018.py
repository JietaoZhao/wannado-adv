"""PGD-AT: Min-Max PGD Adversarial Training (Madry et al., ICLR 2018).

===============================================================================
[Academic Provenance & Hardening Role]
- Tier      : S (Canonical Empirical Adversarial Training Baseline)
- Reference : Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018).
              "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR 2018.
- URL       : https://openreview.net/forum?id=rJzIBfZAb
- Plain Intuition:
    Trains the model against worst-case inner-loop PGD examples generated on the
    fly during each mini-batch (min_theta E[max_{||delta|| <= eps} L(f_theta(x+delta), y)]).
    Inner PGD runs under ``eval_mode(model)`` so BatchNorm running stats are never
    corrupted by adversarial inner steps before outer parameter updates.
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
    id="pgd_at",
    name="PGD Adversarial Training (PGD-AT)",
    paradigm="adversarial_training",
    tier="S",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    summary_plain="Canonical min-max adversarial training that hardens a classifier on worst-case PGD perturbations each batch.",
    when_to_use="Primary empirical hardening baseline when red-team evaluation exposes vulnerability under white-box PGD/AutoAttack.",
    when_not_to_use="When clean accuracy trade-off must be tuned explicitly via KL regularization (prefer trades) or when certified L2 radii are required (prefer randomized_smoothing).",
    module_path="wannado.adv.defenses.pgd_at_madry2018",
    citation=AttackCitation(
        key="madry2018towards",
        title="Towards Deep Learning Models Resistant to Adversarial Attacks",
        authors="Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu",
        venue="6th International Conference on Learning Representations (ICLR)",
        year=2018,
        url="https://openreview.net/forum?id=rJzIBfZAb",
        bibtex=(
            "@inproceedings{madry2018towards,\n"
            "  title={Towards Deep Learning Models Resistant to Adversarial Attacks},\n"
            "  author={Aleksander Madry and Aleksandar Makelov and Ludwig Schmidt and Dimitris Tsipras and Adrian Vladu},\n"
            "  booktitle={International Conference on Learning Representations (ICLR)},\n"
            "  year={2018},\n"
            "  url={https://openreview.net/forum?id=rJzIBfZAb},\n"
            "}"
        ),
        ieee='A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, "Towards deep learning models resistant to adversarial attacks," in Proc. Int. Conf. Learn. Represent. (ICLR), 2018.',
        apa="Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. International Conference on Learning Representations (ICLR).",
    ),
)


def pgd_adversarial_train(
    model: nn.Module,
    train_data: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    *,
    eps: float = 8.0 / 255.0,
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
    """Harden a PyTorch classifier via Madry et al. (ICLR 2018) min-max PGD Adversarial Training."""
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
                adv_res = pgd(
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
                )
                x_adv = adv_res.x_adv.detach()

            model.train()
            opt.zero_grad()
            logits = normalize_classification_logits(model(x_adv))
            loss = F.cross_entropy(logits, yb_l)
            loss.backward()
            opt.step()

            bs = int(xb.shape[0])
            ep_loss += float(loss.item()) * bs
            n_seen += bs

        rec: dict[str, float] = {"epoch": float(ep + 1), "train_adv_loss": ep_loss / max(n_seen, 1)}
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
    final_clean = history[-1].get("eval_clean_acc") if history else None
    final_adv = history[-1].get("eval_adv_acc") if history else None
    return DefenseTrainResult(
        defense_id=META.id,
        model=model,
        epochs_run=epochs,
        elapsed_seconds=time.perf_counter() - t0,
        history=history,
        final_clean_acc=final_clean,
        final_adv_acc=final_adv,
        extra={"eps": float(eps), "norm": norm, "attack_steps": attack_steps, "lr": lr},
        citation=META.citation.to_dict() if hasattr(META.citation, "to_dict") else META.to_dict()["citation"],
    )


__all__ = ["META", "pgd_adversarial_train"]
