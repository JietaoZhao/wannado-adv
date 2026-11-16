"""APGD-DLR: Auto-PGD with Difference of Logits Ratio Loss (Croce & Hein, ICML 2020).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Anti-Gradient-Saturation White-Box SOTA Benchmark)
- Reference     : Croce, F., & Hein, M. (2020). "Reliable evaluation of
                  adversarial robustness with an ensemble of diverse
                  parameter-free attacks." ICML 2020 (PMLR 119:2206-2216).
- URL           : https://proceedings.mlr.press/v119/croce20b.html
- Plain Intuition:
    Standard Cross-Entropy passes logits through Softmax, so multiplying all
    logits by 100 (or high model confidence) makes Softmax saturate to ~1.0
    and turns Cross-Entropy gradients into near-zero—creating fake robustness!
    DLR replaces Cross-Entropy with a shift- and scale-invariant ratio:
        L_DLR(z, y) = -(z_y - max_{i != y} z_i) / (z_{(1)} - z_{(3)})
    Multiplying or shifting logits by any constant leaves DLR gradients intact.
- Diff vs. APGD-CE (2020):
    Uses the identical parameter-free APGD Algorithm 1 optimizer, swapping
    Cross-Entropy for the scale/shift-invariant DLR loss (in ``core/losses.py``).
- Why Chosen as a Baseline:
    Essential companion to APGD-CE for catching false robustness caused by
    logit scaling or Softmax saturation.
- Extension Directions (for further research):
    1. Pair with APGD-CE + FAB + Square Attack -> AutoAttack ensemble.
    2. For binary classifiers (<3 classes), automatically falls back to
       range-normalized logit margin.
===============================================================================
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from wannado.adv.attacks.first_order.apgd_ce_croce2020 import apgd
from wannado.adv.core import AttackMeta, AttackResult
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="apgd_dlr",
    name="Auto-PGD Difference of Logits Ratio (APGD-DLR)",
    family="first_order",
    tier="S",
    threat_model="white_box",
    requires_training=False,
    compute_cost="medium",
    supported_norms=("linf", "l2", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    parent_id="apgd_ce",
    diff_from_parent="Uses APGD Algorithm 1 with shift- and scale-invariant Difference of Logits Ratio (DLR) loss to eliminate Softmax gradient saturation.",
    summary_plain="Scale-invariant Auto-PGD that defeats Softmax saturation and logit-scaling gradient obfuscation.",
    when_to_use="Alongside APGD-CE on classifiers, especially if CE-based attacks unexpectedly stall with near-zero gradients.",
    when_not_to_use="Non-classification regression/MSE tasks without class logits (use APGD with a custom regression loss instead).",
    extension_hooks=(
        "Ensemble with APGD-CE, FAB, and Square Attack -> AutoAttack",
        "Add EoT stochastic transforms for random wireless fading channels",
    ),
    module_path="wannado.adv.attacks.first_order.apgd_dlr_croce2020",
    citation=AttackCitation(
        key="croce2020reliable",
        title="Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks",
        authors="Francesco Croce and Matthias Hein",
        venue="37th International Conference on Machine Learning (ICML)",
        year=2020,
        url="https://proceedings.mlr.press/v119/croce20b.html",
        bibtex=(
            "@inproceedings{croce2020reliable,\n"
            "  title={Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks},\n"
            "  author={Croce, Francesco and Hein, Matthias},\n"
            "  booktitle={Proceedings of the 37th International Conference on Machine Learning},\n"
            "  pages={2206--2216},\n"
            "  year={2020},\n"
            "  volume={119},\n"
            "  series={Proceedings of Machine Learning Research},\n"
            "  publisher={PMLR},\n"
            "  url={https://proceedings.mlr.press/v119/croce20b.html},\n"
            "}"
        ),
        ieee='F. Croce and M. Hein, "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks," in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR, 2020, pp. 2206-2216.',
        apa="Croce, F., & Hein, M. (2020). Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. Proceedings of the 37th International Conference on Machine Learning (ICML), 2206-2216.",
    ),
)


def apgd_dlr(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    steps: int = 50,
    norm: str = "linf",
    targeted: bool = False,
    random_start: bool = True,
    restarts: int = 1,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run Auto-PGD with Difference of Logits Ratio (DLR) loss (APGD-DLR)."""
    return apgd(
        model_or_fn=model_or_fn,
        x=x,
        y=y,
        eps=eps,
        steps=steps,
        norm=norm,
        loss="dlr",
        targeted=targeted,
        random_start=random_start,
        restarts=restarts,
        clip_bounds=clip_bounds,
        seed=seed,
        attack_id=META.id,
    )
