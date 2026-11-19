"""Single-file modular adversarial defenses & certified L2 robustness (`PGD-AT`, `TRADES`, `MART`, `Randomized Smoothing`)."""

from __future__ import annotations

from wannado.adv.defenses.mart_wang2020 import META as MART_META
from wannado.adv.defenses.mart_wang2020 import mart_train
from wannado.adv.defenses.pgd_at_madry2018 import META as PGD_AT_META
from wannado.adv.defenses.pgd_at_madry2018 import pgd_adversarial_train
from wannado.adv.defenses.randomized_smoothing_cohen2019 import META as SMOOTHING_META
from wannado.adv.defenses.randomized_smoothing_cohen2019 import (
    certify_randomized_smoothing,
    clopper_pearson_lower,
    gaussian_augmented_train,
    norm_ppf,
    smooth_predict,
)
from wannado.adv.defenses.trades_zhang2019 import META as TRADES_META
from wannado.adv.defenses.trades_zhang2019 import trades_train
from wannado.adv.defenses.types import DefenseMeta, DefenseTrainResult, SmoothingCertificationResult

DEFENSE_METAS: dict[str, DefenseMeta] = {
    PGD_AT_META.id: PGD_AT_META,
    TRADES_META.id: TRADES_META,
    MART_META.id: MART_META,
    SMOOTHING_META.id: SMOOTHING_META,
}

DEFENSE_EXPORTS: dict[str, tuple[str, str]] = {
    "pgd_adversarial_train": ("wannado.adv.defenses.pgd_at_madry2018", "pgd_adversarial_train"),
    "trades_train": ("wannado.adv.defenses.trades_zhang2019", "trades_train"),
    "mart_train": ("wannado.adv.defenses.mart_wang2020", "mart_train"),
    "certify_randomized_smoothing": (
        "wannado.adv.defenses.randomized_smoothing_cohen2019",
        "certify_randomized_smoothing",
    ),
    "gaussian_augmented_train": (
        "wannado.adv.defenses.randomized_smoothing_cohen2019",
        "gaussian_augmented_train",
    ),
    "smooth_predict": (
        "wannado.adv.defenses.randomized_smoothing_cohen2019",
        "smooth_predict",
    ),
}


def list_defenses() -> list[dict[str, object]]:
    """Return metadata cards for all built-in single-file adversarial defenses & certifiers."""
    return [m.to_dict() for m in DEFENSE_METAS.values()]


__all__ = [
    "DEFENSE_EXPORTS",
    "DEFENSE_METAS",
    "DefenseMeta",
    "DefenseTrainResult",
    "SmoothingCertificationResult",
    "certify_randomized_smoothing",
    "clopper_pearson_lower",
    "gaussian_augmented_train",
    "list_defenses",
    "mart_train",
    "norm_ppf",
    "pgd_adversarial_train",
    "smooth_predict",
    "trades_train",
]
