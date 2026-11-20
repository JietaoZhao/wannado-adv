"""AI-native preflight safety checks, Carlini 2.0 auditing, autonomous red-teaming, and verify gates."""

from __future__ import annotations

from typing import Any

from wannado.adv.attacks import lazy_resolve
from wannado.adv.audit.carlini_auditor import AuditReport, audit_robustness
from wannado.adv.audit.preflight import PreflightReport, preflight

# Keep red_team & verify_gate lazy so importing audit.carlini_auditor / audit.preflight
# from reporter.paper_reporter can never trigger a circular import while preserving
# callable `wannado.adv.audit.preflight` and `wannado.adv.audit.audit_robustness`.
_LAZY_AUDIT_EXPORTS: dict[str, tuple[str, str]] = {
    "AdvVerifyReport": ("wannado.adv.audit.verify_gate", "AdvVerifyReport"),
    "BackdoorAuditReport": ("wannado.adv.audit.backdoor_auditor", "BackdoorAuditReport"),
    "PrivacyAuditReport": ("wannado.adv.audit.privacy_auditor", "PrivacyAuditReport"),
    "RedTeamReport": ("wannado.adv.audit.red_team", "RedTeamReport"),
    "audit_backdoor": ("wannado.adv.audit.backdoor_auditor", "audit_backdoor"),
    "audit_privacy": ("wannado.adv.audit.privacy_auditor", "audit_privacy"),
    "auto_red_team": ("wannado.adv.audit.red_team", "auto_red_team"),
    "compute_aurc_and_eps50": ("wannado.adv.audit.red_team", "compute_aurc_and_eps50"),
    "verify_robustness_gate": ("wannado.adv.audit.verify_gate", "verify_robustness_gate"),
    "wilson_ci": ("wannado.adv.audit.red_team", "wilson_ci"),
}

__all__ = [
    "AdvVerifyReport",
    "AuditReport",
    "BackdoorAuditReport",
    "PreflightReport",
    "PrivacyAuditReport",
    "RedTeamReport",
    "audit_backdoor",
    "audit_privacy",
    "audit_robustness",
    "auto_red_team",
    "compute_aurc_and_eps50",
    "preflight",
    "verify_robustness_gate",
    "wilson_ci",
]


def __getattr__(name: str) -> Any:
    return lazy_resolve(name, _LAZY_AUDIT_EXPORTS, globals(), __name__)


def __dir__() -> list[str]:
    return sorted(__all__)
