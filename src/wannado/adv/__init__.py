"""wannado.adv — AI-native, CleanRL-style adversarial robustness library.

Uses PEP 562 lazy loading so importing a single attack (e.g. ``from wannado.adv import fgsm``)
only loads that standalone implementation without importing the rest of the catalog.
"""

from __future__ import annotations

import importlib
from typing import Any

from wannado.adv.attacks import ATTACK_EXPORTS, lazy_resolve

__version__ = "1.2.0"

_SUBMODULES = frozenset(
    {
        "agent_kit",
        "attacks",
        "audit",
        "catalog",
        "copilot",
        "core",
        "defenses",
        "reporter",
        "workspace",
    }
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    **ATTACK_EXPORTS,
    # Single-file modular adversarial defenses & certified L2 smoothing
    "DefenseMeta": ("wannado.adv.defenses.types", "DefenseMeta"),
    "DefenseTrainResult": ("wannado.adv.defenses.types", "DefenseTrainResult"),
    "SmoothingCertificationResult": ("wannado.adv.defenses.types", "SmoothingCertificationResult"),
    "certify_randomized_smoothing": (
        "wannado.adv.defenses.randomized_smoothing_cohen2019",
        "certify_randomized_smoothing",
    ),
    "clopper_pearson_lower": (
        "wannado.adv.defenses.randomized_smoothing_cohen2019",
        "clopper_pearson_lower",
    ),
    "gaussian_augmented_train": (
        "wannado.adv.defenses.randomized_smoothing_cohen2019",
        "gaussian_augmented_train",
    ),
    "list_defenses": ("wannado.adv.defenses", "list_defenses"),
    "mart_train": ("wannado.adv.defenses.mart_wang2020", "mart_train"),
    "norm_ppf": ("wannado.adv.defenses.randomized_smoothing_cohen2019", "norm_ppf"),
    "pgd_adversarial_train": ("wannado.adv.defenses.pgd_at_madry2018", "pgd_adversarial_train"),
    "smooth_predict": ("wannado.adv.defenses.randomized_smoothing_cohen2019", "smooth_predict"),
    "trades_train": ("wannado.adv.defenses.trades_zhang2019", "trades_train"),
    # AI-native preflight, Carlini 2.0 auditing, autonomous red-teaming & verify gate
    "AdvVerifyReport": ("wannado.adv.audit.verify_gate", "AdvVerifyReport"),
    "AuditReport": ("wannado.adv.audit.carlini_auditor", "AuditReport"),
    "PreflightReport": ("wannado.adv.audit.preflight", "PreflightReport"),
    "RedTeamReport": ("wannado.adv.audit.red_team", "RedTeamReport"),
    "audit_robustness": ("wannado.adv.audit.carlini_auditor", "audit_robustness"),
    "auto_red_team": ("wannado.adv.audit.red_team", "auto_red_team"),
    "compute_aurc_and_eps50": ("wannado.adv.audit.red_team", "compute_aurc_and_eps50"),
    "preflight": ("wannado.adv.audit.preflight", "preflight"),
    "verify_robustness_gate": ("wannado.adv.audit.verify_gate", "verify_robustness_gate"),
    "wilson_ci": ("wannado.adv.audit.red_team", "wilson_ci"),
    # AI Co-Pilot project inspection, SOTA planning, and ETA estimation
    "ModelContextProbe": ("wannado.adv.copilot", "ModelContextProbe"),
    "ProjectScan": ("wannado.adv.copilot", "ProjectScan"),
    "RecommendedSuite": ("wannado.adv.copilot", "RecommendedSuite"),
    "RuntimeEstimate": ("wannado.adv.copilot", "RuntimeEstimate"),
    "build_interactive_brief": ("wannado.adv.copilot", "build_interactive_brief"),
    "estimate_runtime": ("wannado.adv.copilot", "estimate_runtime"),
    "explain_parameters": ("wannado.adv.copilot", "explain_parameters"),
    "get_sota_lineage": ("wannado.adv.copilot", "get_sota_lineage"),
    "probe_model_context": ("wannado.adv.copilot", "probe_model_context"),
    "recommend_suite": ("wannado.adv.copilot", "recommend_suite"),
    "scan_project": ("wannado.adv.copilot", "scan_project"),
    # Non-intrusive interface scaffolding & single-file attack/defense ejection
    "VictimInterface": ("wannado.adv.workspace", "VictimInterface"),
    "eject_attack": ("wannado.adv.workspace", "eject_attack"),
    "eject_defense": ("wannado.adv.workspace", "eject_defense"),
    "scaffold_interface": ("wannado.adv.workspace", "scaffold_interface"),
    # Publication-ready reporting & zero-hallucination citation export
    "AttackRunRecord": ("wannado.adv.reporter", "AttackRunRecord"),
    "EvaluationReport": ("wannado.adv.reporter", "EvaluationReport"),
    "evaluate_suite": ("wannado.adv.reporter", "evaluate_suite"),
    "export_citations": ("wannado.adv.reporter", "export_citations"),
    # Multi-IDE agent skill & instruction installer
    "install_agent_kit": ("wannado.adv.agent_kit", "install_agent_kit"),
    # Knowledge graph catalog helpers
    "build_registry_json": ("wannado.adv.catalog.build_catalog", "build_registry_json"),
    "load_catalog": ("wannado.adv.catalog.build_catalog", "load_catalog"),
    "query_catalog": ("wannado.adv.catalog.build_catalog", "query_catalog"),
    # Core primitives
    "AttackMeta": ("wannado.adv.core.types", "AttackMeta"),
    "AttackResult": ("wannado.adv.core.types", "AttackResult"),
    "DefendedModelWrapper": ("wannado.adv.core.utils", "DefendedModelWrapper"),
    "SklearnModelAdapter": ("wannado.adv.core.utils", "SklearnModelAdapter"),
    "clamp_to_bounds": ("wannado.adv.core.projections", "clamp_to_bounds"),
    "coerce_clip_bounds": ("wannado.adv.core.projections", "coerce_clip_bounds"),
    "compute_perturbation_norm": ("wannado.adv.core.projections", "compute_perturbation_norm"),
    "iter_eval_batches": ("wannado.adv.core.utils", "iter_eval_batches"),
    "materialize_eval_tensors": ("wannado.adv.core.utils", "materialize_eval_tensors"),
    "normalize_classification_logits": ("wannado.adv.core.losses", "normalize_classification_logits"),
    "predict_labels": ("wannado.adv.core.losses", "predict_labels"),
    "project_discrete_features": ("wannado.adv.core.projections", "project_discrete_features"),
    "project_perturbation": ("wannado.adv.core.projections", "project_perturbation"),
    "random_init_perturbation": ("wannado.adv.core.projections", "random_init_perturbation"),
    "resolve_eps_budget": ("wannado.adv.core.projections", "resolve_eps_budget"),
    "unwrap_model_outputs": ("wannado.adv.core.losses", "unwrap_model_outputs"),
    "wrap_sklearn_model": ("wannado.adv.core.utils", "wrap_sklearn_model"),
}

__all__ = [
    "AdvVerifyReport",
    "AttackMeta",
    "AttackResult",
    "AttackRunRecord",
    "AuditReport",
    "DefendedModelWrapper",
    "DefenseMeta",
    "DefenseTrainResult",
    "EvaluationReport",
    "ModelContextProbe",
    "PreflightReport",
    "ProjectScan",
    "RecommendedSuite",
    "RedTeamReport",
    "RuntimeEstimate",
    "SklearnModelAdapter",
    "SmoothingCertificationResult",
    "VictimInterface",
    "__version__",
    "adam_pgd",
    "adv_patch",
    "advgan",
    "apgd",
    "apgd_ce",
    "apgd_dlr",
    "apgd_l1",
    "apgd_t",
    "audit_robustness",
    "auto_red_team",
    "autoattack",
    "bim",
    "boundary",
    "bpda_eot",
    "build_interactive_brief",
    "build_registry_json",
    "certify_randomized_smoothing",
    "clamp_to_bounds",
    "clopper_pearson_lower",
    "coerce_clip_bounds",
    "compute_aurc_and_eps50",
    "compute_perturbation_norm",
    "cw_l2",
    "deepfool",
    "di_fgsm",
    "difgsm",
    "ead",
    "eject_attack",
    "eject_defense",
    "eot_pgd",
    "estimate_runtime",
    "evaluate_suite",
    "explain_parameters",
    "export_citations",
    "fab",
    "fab_t",
    "fgsm",
    "gap",
    "gaussian_augmented_train",
    "get_sota_lineage",
    "hopskipjump",
    "install_agent_kit",
    "iter_eval_batches",
    "jsma",
    "list_defenses",
    "load_catalog",
    "mart_train",
    "materialize_eval_tensors",
    "mi_fgsm",
    "mifgsm",
    "nes",
    "ni_fgsm",
    "nifgsm",
    "norm_ppf",
    "normalize_classification_logits",
    "pgd",
    "pgd_adversarial_train",
    "predict_labels",
    "preflight",
    "probe_model_context",
    "project_discrete_features",
    "project_perturbation",
    "query_catalog",
    "random_init_perturbation",
    "recommend_suite",
    "resolve_eps_budget",
    "rfgsm",
    "scaffold_interface",
    "scan_project",
    "simba",
    "sini_fgsm",
    "sinifgsm",
    "smooth_predict",
    "sparsefool",
    "spsa",
    "square_attack",
    "square_t",
    "ti_fgsm",
    "tifgsm",
    "tpgd",
    "trades_train",
    "uap",
    "unwrap_model_outputs",
    "verify_robustness_gate",
    "vmi_fgsm",
    "vmifgsm",
    "wilson_ci",
    "wrap_sklearn_model",
    "zoo",
]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    return lazy_resolve(name, _LAZY_EXPORTS, globals(), __name__)


def __dir__() -> list[str]:
    return sorted(__all__)
