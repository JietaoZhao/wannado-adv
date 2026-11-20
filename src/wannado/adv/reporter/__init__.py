"""Academic reporting, citation verification, compliance conformity dossiers, and SARIF 2.1.0 export."""

from wannado.adv.reporter.conformity import (
    STANDARDS_MAPPING,
    ConformityDossier,
    build_conformity_dossier,
    compare_verdicts,
    compute_dataset_fingerprint,
    compute_model_fingerprint,
    export_sarif,
    get_git_provenance,
    map_flag_to_standards,
    verify_dossier_integrity,
)
from wannado.adv.reporter.paper_reporter import (
    AttackRunRecord,
    EvaluationReport,
    evaluate_suite,
    export_citations,
)

__all__ = [
    "STANDARDS_MAPPING",
    "AttackRunRecord",
    "ConformityDossier",
    "EvaluationReport",
    "build_conformity_dossier",
    "compare_verdicts",
    "compute_dataset_fingerprint",
    "compute_model_fingerprint",
    "evaluate_suite",
    "export_citations",
    "export_sarif",
    "get_git_provenance",
    "map_flag_to_standards",
    "verify_dossier_integrity",
]
