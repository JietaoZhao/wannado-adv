"""CleanRL-style single-file adversarial attack implementations organized by family.

Supports lazy single-file imports such as:
    from wannado.adv.attacks import fgsm, pgd
"""

from __future__ import annotations

import importlib
from typing import Any

ATTACK_EXPORTS: dict[str, tuple[str, str]] = {
    # First-order white-box family
    "fgsm": ("wannado.adv.attacks.first_order.fgsm_goodfellow2015", "fgsm"),
    "rfgsm": ("wannado.adv.attacks.first_order.rfgsm_tramer2018", "rfgsm"),
    "bim": ("wannado.adv.attacks.first_order.bim_kurakin2017", "bim"),
    "pgd": ("wannado.adv.attacks.first_order.pgd_madry2018", "pgd"),
    "mifgsm": ("wannado.adv.attacks.first_order.mifgsm_dong2018", "mifgsm"),
    "mi_fgsm": ("wannado.adv.attacks.first_order.mifgsm_dong2018", "mifgsm"),
    "adam_pgd": ("wannado.adv.attacks.first_order.adam_pgd", "adam_pgd"),
    "tpgd": ("wannado.adv.attacks.first_order.tpgd_zhang2019", "tpgd"),
    "apgd": ("wannado.adv.attacks.first_order.apgd_ce_croce2020", "apgd"),
    "apgd_ce": ("wannado.adv.attacks.first_order.apgd_ce_croce2020", "apgd_ce"),
    "apgd_dlr": ("wannado.adv.attacks.first_order.apgd_dlr_croce2020", "apgd_dlr"),
    "apgd_l1": ("wannado.adv.attacks.first_order.apgd_l1_croce2021", "apgd_l1"),
    # Transferability & input-transformation white-box family
    "ni_fgsm": ("wannado.adv.attacks.transferability.nifgsm_lin2020", "ni_fgsm"),
    "nifgsm": ("wannado.adv.attacks.transferability.nifgsm_lin2020", "ni_fgsm"),
    "sini_fgsm": ("wannado.adv.attacks.transferability.sinifgsm_lin2020", "sini_fgsm"),
    "sinifgsm": ("wannado.adv.attacks.transferability.sinifgsm_lin2020", "sini_fgsm"),
    "di_fgsm": ("wannado.adv.attacks.transferability.difgsm_xie2019", "di_fgsm"),
    "difgsm": ("wannado.adv.attacks.transferability.difgsm_xie2019", "di_fgsm"),
    "ti_fgsm": ("wannado.adv.attacks.transferability.tifgsm_dong2019", "ti_fgsm"),
    "tifgsm": ("wannado.adv.attacks.transferability.tifgsm_dong2019", "ti_fgsm"),
    "vmi_fgsm": ("wannado.adv.attacks.transferability.vmifgsm_wang2021", "vmi_fgsm"),
    "vmifgsm": ("wannado.adv.attacks.transferability.vmifgsm_wang2021", "vmi_fgsm"),
    # Minimum-norm & sparse geometric boundary family
    "deepfool": ("wannado.adv.attacks.minimum_norm.deepfool_moosavi2016", "deepfool"),
    "cw_l2": ("wannado.adv.attacks.minimum_norm.cw_l2_carlini2017", "cw_l2"),
    "ead": ("wannado.adv.attacks.minimum_norm.ead_chen2018", "ead"),
    "sparsefool": ("wannado.adv.attacks.minimum_norm.sparsefool_modas2019", "sparsefool"),
    "jsma": ("wannado.adv.attacks.minimum_norm.jsma_papernot2016", "jsma"),
    "fab": ("wannado.adv.attacks.minimum_norm.fab_croce2020", "fab"),
    # Black-box score-based family
    "zoo": ("wannado.adv.attacks.blackbox_score.zoo_chen2017", "zoo"),
    "nes": ("wannado.adv.attacks.blackbox_score.nes_ilyas2018", "nes"),
    "spsa": ("wannado.adv.attacks.blackbox_score.spsa_uesato2018", "spsa"),
    "simba": ("wannado.adv.attacks.blackbox_score.simba_guo2019", "simba"),
    "square_attack": ("wannado.adv.attacks.blackbox_score.square_andriushchenko2020", "square_attack"),
    # Black-box decision-based family
    "boundary": ("wannado.adv.attacks.blackbox_decision.boundary_brendel2018", "boundary"),
    "hopskipjump": ("wannado.adv.attacks.blackbox_decision.hopskipjump_chen2020", "hopskipjump"),
    # Physical / stochastic EoT & BPDA family
    "eot_pgd": ("wannado.adv.attacks.physical_and_eot.eot_pgd_athalye2018", "eot_pgd"),
    "bpda_eot": ("wannado.adv.attacks.physical_and_eot.bpda_eot_athalye2018", "bpda_eot"),
    "adv_patch": ("wannado.adv.attacks.physical_and_eot.patch_brown2017", "adv_patch"),
    # Universal perturbations
    "uap": ("wannado.adv.attacks.universal.uap_moosavi2017", "uap"),
    # Generative train-based family
    "gap": ("wannado.adv.attacks.generative.gap_poursaeed2018", "gap"),
    "advgan": ("wannado.adv.attacks.generative.advgan_xiao2018", "advgan"),
    # Multi-attack evaluation ensemble
    "autoattack": ("wannado.adv.attacks.ensemble.autoattack_croce2020", "autoattack"),
}

__all__ = [
    "adam_pgd",
    "adv_patch",
    "advgan",
    "apgd",
    "apgd_ce",
    "apgd_dlr",
    "apgd_l1",
    "apgd_t",
    "autoattack",
    "bim",
    "boundary",
    "bpda_eot",
    "cw_l2",
    "deepfool",
    "di_fgsm",
    "difgsm",
    "ead",
    "eot_pgd",
    "fab",
    "fab_t",
    "fgsm",
    "gap",
    "hopskipjump",
    "jsma",
    "mi_fgsm",
    "mifgsm",
    "nes",
    "ni_fgsm",
    "nifgsm",
    "pgd",
    "rfgsm",
    "simba",
    "sini_fgsm",
    "sinifgsm",
    "sparsefool",
    "spsa",
    "square_attack",
    "square_t",
    "ti_fgsm",
    "tifgsm",
    "tpgd",
    "uap",
    "vmi_fgsm",
    "vmifgsm",
    "zoo",
]


def lazy_resolve(
    name: str,
    exports: dict[str, tuple[str, str]],
    namespace: dict[str, Any],
    package_name: str,
) -> Any:
    """Resolve and cache a lazily exported symbol without eager package imports."""
    if name not in exports:
        raise AttributeError(f"module {package_name!r} has no attribute {name!r}")
    module_name, attr_name = exports[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    namespace[name] = value
    return value


def __getattr__(name: str) -> Any:
    return lazy_resolve(name, ATTACK_EXPORTS, globals(), __name__)


def __dir__() -> list[str]:
    return sorted(__all__)
