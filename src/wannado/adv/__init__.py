"""wannado.adv — AI-native, CleanRL-style adversarial robustness library.

Uses PEP 562 lazy loading so importing a single attack (e.g. ``from wannado.adv import fgsm``)
only loads that standalone implementation without importing the rest of the catalog.
"""

from __future__ import annotations

from typing import Any

from wannado.adv.attacks import ATTACK_EXPORTS, lazy_resolve

__version__ = "1.0.3"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    **ATTACK_EXPORTS,
    # AI-native preflight & Carlini robustness auditing
    "AuditReport": ("wannado.adv.audit.carlini_auditor", "AuditReport"),
    "PreflightReport": ("wannado.adv.audit.preflight", "PreflightReport"),
    "audit_robustness": ("wannado.adv.audit.carlini_auditor", "audit_robustness"),
    "preflight": ("wannado.adv.audit.preflight", "preflight"),
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
    # Non-intrusive interface scaffolding & single-file attack ejection
    "VictimInterface": ("wannado.adv.workspace", "VictimInterface"),
    "eject_attack": ("wannado.adv.workspace", "eject_attack"),
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
    "compute_perturbation_norm": ("wannado.adv.core.projections", "compute_perturbation_norm"),
    "project_perturbation": ("wannado.adv.core.projections", "project_perturbation"),
    "resolve_eps_budget": ("wannado.adv.core.projections", "resolve_eps_budget"),
}

__all__ = ["__version__", *_LAZY_EXPORTS.keys()]


def __getattr__(name: str) -> Any:
    return lazy_resolve(name, _LAZY_EXPORTS, globals(), __name__)


def __dir__() -> list[str]:
    return sorted(__all__)
