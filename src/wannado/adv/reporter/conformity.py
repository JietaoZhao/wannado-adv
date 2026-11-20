"""Enterprise Conformity Dossier, Regulatory Mapping, Cryptographic Integrity & SARIF 2.1.0 Export.

Bridges adversarial robustness results (Evasion + Carlini 2.0, Backdoor/Poisoning, Privacy/MIA,
and Quantization-Shift Differential Audits) to verifiable compliance and CI/CD artifacts:
  - NIST AI 100-2 (E2025) Adversarial ML Taxonomy
  - MITRE ATLAS Technique IDs
  - EU AI Act Article 15 (Accuracy, Robustness & Cybersecurity) Clauses
  - Tamper-evident SHA-256 / HMAC-SHA256 model, dataset, and verdict fingerprints
  - Stateful baseline PR regression gating (`compare_verdicts`)
  - Native SARIF 2.1.0 output for GitHub Code Scanning / GitLab Security tabs
"""

from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from wannado.adv.core.utils import iter_eval_batches

if TYPE_CHECKING:
    from wannado.adv.audit.verify_gate import AdvVerifyReport

# Grounded mappings aligned with NIST AI 100-2 (E2025), MITRE ATLAS, and EU AI Act Article 15.
STANDARDS_MAPPING: dict[str, dict[str, Any]] = {
    "evasion_whitebox": {
        "nist_ai_100_2": "AML.Evasion.WhiteBox",
        "mitre_atlas": ["AML.T0043", "AML.T0015"],
        "eu_ai_act": ["Article 15(1)", "Article 15(5)"],
        "pillar": "evasion",
    },
    "evasion_blackbox": {
        "nist_ai_100_2": "AML.Evasion.BlackBox",
        "mitre_atlas": ["AML.T0043.002", "AML.T0015"],
        "eu_ai_act": ["Article 15(1)", "Article 15(5)"],
        "pillar": "evasion",
    },
    "evasion_adaptive_anti_masking": {
        "nist_ai_100_2": "AML.Evasion.AdaptiveEvaluation",
        "mitre_atlas": ["AML.T0043", "AML.T0015"],
        "eu_ai_act": ["Article 15(4)", "Article 15(5)"],
        "pillar": "evasion",
    },
    "quantization_shift": {
        "nist_ai_100_2": "AML.Evasion.DeploymentShift",
        "mitre_atlas": ["AML.T0015"],
        "eu_ai_act": ["Article 15(1)", "Article 15(4)"],
        "pillar": "evasion",
    },
    "backdoor_poisoning": {
        "nist_ai_100_2": "AML.Poisoning.Backdoor",
        "mitre_atlas": ["AML.T0020", "AML.T0018"],
        "eu_ai_act": ["Article 15(5)"],
        "pillar": "poisoning_backdoor",
    },
    "privacy_mia": {
        "nist_ai_100_2": "AML.Privacy.MembershipInference",
        "mitre_atlas": ["AML.T0024.000"],
        "eu_ai_act": ["Article 15(5)"],
        "pillar": "privacy_confidentiality",
    },
    "privacy_extraction": {
        "nist_ai_100_2": "AML.Privacy.ModelExtraction",
        "mitre_atlas": ["AML.T0024.002"],
        "eu_ai_act": ["Article 15(5)"],
        "pillar": "privacy_confidentiality",
    },
    "regression_gate": {
        "nist_ai_100_2": "AML.Lifecycle.ContinuousVerification",
        "mitre_atlas": ["AML.T0015"],
        "eu_ai_act": ["Article 15(1)", "Article 15(4)"],
        "pillar": "lifecycle_governance",
    },
}


def map_flag_to_standards(flag: str) -> dict[str, Any]:
    """Map any wannado-adv gate red flag to NIST AI 100-2, MITRE ATLAS, and EU AI Act Art. 15."""
    lower = flag.lower()
    if lower.startswith("backdoor:") or "trojan" in lower or "strip" in lower:
        return dict(STANDARDS_MAPPING["backdoor_poisoning"])
    if lower.startswith("privacy:extraction") or "extraction" in lower:
        return dict(STANDARDS_MAPPING["privacy_extraction"])
    if lower.startswith("privacy:") or "mia" in lower:
        return dict(STANDARDS_MAPPING["privacy_mia"])
    if lower.startswith("quant:") or "quantization" in lower:
        return dict(STANDARDS_MAPPING["quantization_shift"])
    if lower.startswith("regression:"):
        return dict(STANDARDS_MAPPING["regression_gate"])
    if lower.startswith("carlini:") or "masking" in lower or "obfuscat" in lower:
        return dict(STANDARDS_MAPPING["evasion_adaptive_anti_masking"])
    if "blackbox" in lower or "hard_label" in lower:
        return dict(STANDARDS_MAPPING["evasion_blackbox"])
    return dict(STANDARDS_MAPPING["evasion_whitebox"])


def compute_model_fingerprint(model_or_fn: Any) -> dict[str, Any]:
    """Compute a deterministic SHA-256 fingerprint of model weights or runtime artifact."""
    hasher = hashlib.sha256()
    if isinstance(model_or_fn, nn.Module):
        state = model_or_fn.state_dict()
        param_count = 0
        for key in sorted(state.keys()):
            tensor = state[key].detach().cpu().contiguous()
            hasher.update(key.encode("utf-8"))
            hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
            hasher.update(str(tensor.dtype).encode("utf-8"))
            hasher.update(tensor.numpy().tobytes())
            param_count += int(tensor.numel())
        return {
            "kind": "torch_module",
            "class_name": type(model_or_fn).__name__,
            "num_tensors": len(state),
            "num_elements": param_count,
            "sha256": hasher.hexdigest(),
        }

    artifact_path = getattr(model_or_fn, "model_path", None)
    if artifact_path is not None and Path(str(artifact_path)).is_file():
        p = Path(str(artifact_path))
        hasher.update(p.read_bytes())
        return {
            "kind": "file_artifact",
            "class_name": type(model_or_fn).__name__,
            "file_name": p.name,
            "sha256": hasher.hexdigest(),
        }

    # Deterministic fallback for sklearn / custom callable wrappers
    descriptor = f"{type(model_or_fn).__module__}.{type(model_or_fn).__qualname__}"
    est = getattr(model_or_fn, "estimator", None)
    if est is not None and hasattr(est, "get_params"):
        try:
            descriptor += ":" + json.dumps(est.get_params(deep=False), sort_keys=True, default=str)
        except Exception:  # noqa: BLE001
            descriptor += ":" + repr(est)
    hasher.update(descriptor.encode("utf-8"))
    return {
        "kind": "callable_adapter",
        "class_name": type(model_or_fn).__name__,
        "sha256": hasher.hexdigest(),
    }


def compute_dataset_fingerprint(
    x_or_loader: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    *,
    max_samples: int = 256,
) -> dict[str, Any]:
    """Compute a deterministic SHA-256 fingerprint over the evaluated data slice."""
    hasher = hashlib.sha256()
    n_total = 0
    x_shape: tuple[int, ...] = ()
    x_dtype = ""
    y_dtype = ""
    for xb, yb in iter_eval_batches(x_or_loader, y, batch_size=64, max_samples=max_samples):
        xc = xb.detach().cpu().contiguous()
        yc = yb.detach().cpu().contiguous()
        if n_total == 0:
            x_shape = tuple(int(s) for s in xc.shape[1:])
            x_dtype = str(xc.dtype)
            y_dtype = str(yc.dtype)
        n_total += int(xc.shape[0])
        hasher.update(xc.numpy().tobytes())
        hasher.update(yc.numpy().tobytes())
    return {
        "num_samples_hashed": n_total,
        "sample_shape": list(x_shape),
        "x_dtype": x_dtype,
        "y_dtype": y_dtype,
        "sha256": hasher.hexdigest(),
    }


def get_git_provenance(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """Collect best-effort non-blocking git commit / branch / dirty metadata."""
    cwd = str(Path(repo_dir).resolve()) if repo_dir is not None else None
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return {"commit_sha": sha, "branch": branch, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"commit_sha": None, "branch": None, "dirty": None}


def compare_verdicts(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    max_robust_regression: float = 0.02,
    max_clean_regression: float = 0.02,
    max_drift_increase: float = 0.10,
) -> dict[str, Any]:
    """Compare current verification summary against a prior baseline verdict to catch PR regressions."""
    cur_s = current.get("summary", current)
    base_s = baseline.get("summary", baseline)
    regressions: list[str] = []
    deltas: dict[str, float | None] = {}

    cur_rob = cur_s.get("worst_case_robust_acc")
    base_rob = base_s.get("worst_case_robust_acc")
    if cur_rob is not None and base_rob is not None:
        d_rob = float(cur_rob) - float(base_rob)
        deltas["worst_case_robust_acc_delta"] = round(d_rob, 6)
        if d_rob < -abs(float(max_robust_regression)) - 1e-9:
            regressions.append(
                f"regression:worst_case_robust_acc_dropped:{d_rob:+.4f}<-{abs(float(max_robust_regression)):.4f}"
            )

    cur_clean = cur_s.get("clean_acc")
    base_clean = base_s.get("clean_acc")
    if cur_clean is not None and base_clean is not None:
        d_clean = float(cur_clean) - float(base_clean)
        deltas["clean_acc_delta"] = round(d_clean, 6)
        if d_clean < -abs(float(max_clean_regression)) - 1e-9:
            regressions.append(
                f"regression:clean_acc_dropped:{d_clean:+.4f}<-{abs(float(max_clean_regression)):.4f}"
            )

    cur_aurc = cur_s.get("aurc")
    base_aurc = base_s.get("aurc")
    if cur_aurc is not None and base_aurc is not None:
        deltas["aurc_delta"] = round(float(cur_aurc) - float(base_aurc), 6)

    cur_drift = cur_s.get("worst_case_output_l2_drift")
    base_drift = base_s.get("worst_case_output_l2_drift")
    if cur_drift is not None and base_drift is not None:
        d_drift = float(cur_drift) - float(base_drift)
        deltas["output_l2_drift_delta"] = round(d_drift, 6)
        if d_drift > abs(float(max_drift_increase)) + 1e-9:
            regressions.append(
                f"regression:output_l2_drift_increased:{d_drift:+.4f}>+{abs(float(max_drift_increase)):.4f}"
            )

    return {
        "passed": len(regressions) == 0,
        "deltas": deltas,
        "regressions": regressions,
        "tolerances": {
            "max_robust_regression": float(max_robust_regression),
            "max_clean_regression": float(max_clean_regression),
            "max_drift_increase": float(max_drift_increase),
        },
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass
class ConformityDossier:
    """Tamper-evident regulatory & engineering conformity evidence dossier."""

    schema_version: str
    generated_at_utc: str
    target_name: str
    overall_passed: bool
    model_fingerprint: dict[str, Any] | None
    dataset_fingerprint: dict[str, Any] | None
    git_provenance: dict[str, Any]
    pillar_coverage: dict[str, dict[str, Any]]
    findings: list[dict[str, Any]] = field(default_factory=list)
    evasion_summary: dict[str, Any] = field(default_factory=dict)
    backdoor_summary: dict[str, Any] | None = None
    privacy_summary: dict[str, Any] | None = None
    quantization_summary: dict[str, Any] | None = None
    regression_diff: dict[str, Any] | None = None
    dossier_sha256: str = ""
    hmac_sha256: str | None = None

    def _unsigned_payload(self) -> dict[str, Any]:
        raw = asdict(self)
        raw.pop("dossier_sha256", None)
        raw.pop("hmac_sha256", None)
        return raw

    def seal(self, signing_key: str | bytes | None = None) -> ConformityDossier:
        """Compute canonical SHA-256 digest and optional HMAC-SHA256 signature."""
        canon = _canonical_json_bytes(self._unsigned_payload())
        self.dossier_sha256 = hashlib.sha256(canon).hexdigest()
        if signing_key is not None:
            key_b = signing_key.encode("utf-8") if isinstance(signing_key, str) else bytes(signing_key)
            self.hmac_sha256 = hmac.new(key_b, canon, hashlib.sha256).hexdigest()
        else:
            self.hmac_sha256 = None
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False) + "\n"

    def format_markdown(self) -> str:
        verdict = "PASS" if self.overall_passed else "FAIL"
        lines = [
            f"# Model Security & Conformity Dossier — `{self.target_name}` ({verdict})",
            "",
            "## 1. Cryptographic Provenance & Integrity",
            f"- **Generated (UTC)**: `{self.generated_at_utc}`",
            f"- **Dossier SHA-256**: `{self.dossier_sha256}`",
            f"- **HMAC Signature**: `{'SIGNED (' + self.hmac_sha256[:16] + '...)' if self.hmac_sha256 else 'unsigned (SHA-256 digest only)'}`",
        ]
        if self.model_fingerprint is not None:
            lines.append(
                f"- **Model Fingerprint**: `{self.model_fingerprint.get('class_name')}` "
                f"(SHA-256 `{str(self.model_fingerprint.get('sha256', ''))[:16]}...`)"
            )
        if self.dataset_fingerprint is not None:
            lines.append(
                f"- **Eval Slice Fingerprint**: `N={self.dataset_fingerprint.get('num_samples_hashed')}` "
                f"(SHA-256 `{str(self.dataset_fingerprint.get('sha256', ''))[:16]}...`)"
            )
        if self.git_provenance.get("commit_sha"):
            lines.append(
                f"- **Git Revision**: `{self.git_provenance['commit_sha'][:12]}` "
                f"(branch `{self.git_provenance.get('branch')}`, dirty=`{self.git_provenance.get('dirty')}`)"
            )

        lines.extend(
            [
                "",
                "## 2. EU AI Act Article 15 / NIST AI 100-2 / MITRE ATLAS Coverage Matrix",
                "| Threat Pillar | Status | NIST AI 100-2 | MITRE ATLAS | EU AI Act |",
                "| --- | :---: | --- | --- | --- |",
            ]
        )
        for pillar_name, info in self.pillar_coverage.items():
            st = str(info.get("status", "NOT_RUN"))
            nist = str(info.get("nist_ai_100_2", "-"))
            atlas = ", ".join(info.get("mitre_atlas", []))
            eu = ", ".join(info.get("eu_ai_act", []))
            lines.append(f"| `{pillar_name}` | **{st}** | `{nist}` | `{atlas}` | {eu} |")

        if self.regression_diff is not None:
            lines.extend(["", "## 3. Baseline PR Regression Diff"])
            for k, v in self.regression_diff.get("deltas", {}).items():
                lines.append(f"- `{k}`: `{v}`")
            for reg in self.regression_diff.get("regressions", []):
                lines.append(f"- **REGRESSION**: `{reg}`")

        lines.extend(["", "## 4. Structured Security Findings"])
        if not self.findings:
            lines.append("- Zero blocking red flags detected across executed pillars.")
        else:
            for f in self.findings:
                lines.append(
                    f"- **[{f['severity'].upper()}]** `{f['rule_id']}` "
                    f"(NIST: `{f['standards']['nist_ai_100_2']}`, ATLAS: `{','.join(f['standards']['mitre_atlas'])}`) — {f['message']}"
                )
        lines.append("")
        return "\n".join(lines)


def build_conformity_dossier(
    verify_report: AdvVerifyReport,
    *,
    backdoor_report: Any | None = None,
    privacy_report: Any | None = None,
    quant_report: Any | None = None,
    model_or_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    x_eval: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
    y_eval: torch.Tensor | None = None,
    regression_diff: dict[str, Any] | None = None,
    signing_key: str | bytes | None = None,
    repo_dir: str | Path | None = None,
) -> ConformityDossier:
    """Assemble and cryptographically seal a NIST AI 100-2 / EU AI Act Art. 15 conformity dossier."""
    findings: list[dict[str, Any]] = []

    def _add_findings(flags: Iterable[str], severity: str, source: str) -> None:
        for flag in flags:
            std = map_flag_to_standards(flag)
            findings.append(
                {
                    "rule_id": flag.split(":")[0] + ":" + flag.split(":")[1] if ":" in flag else flag,
                    "source": source,
                    "severity": severity,
                    "message": flag,
                    "standards": std,
                }
            )

    _add_findings(verify_report.red_flags, "error", "evasion_verify_gate")
    _add_findings(verify_report.warnings, "warning", "evasion_verify_gate")

    evasion_passed = bool(verify_report.passed)
    backdoor_summary: dict[str, Any] | None = None
    backdoor_status = "NOT_RUN"
    backdoor_passed = True
    if backdoor_report is not None:
        backdoor_summary = backdoor_report.to_dict() if hasattr(backdoor_report, "to_dict") else dict(backdoor_report)
        backdoor_passed = bool(backdoor_summary.get("passed", False))
        backdoor_status = "PASS" if backdoor_passed else "FAIL"
        _add_findings(backdoor_summary.get("red_flags", []), "error", "backdoor_auditor")

    privacy_summary: dict[str, Any] | None = None
    privacy_status = "NOT_RUN"
    privacy_passed = True
    if privacy_report is not None:
        privacy_summary = privacy_report.to_dict() if hasattr(privacy_report, "to_dict") else dict(privacy_report)
        privacy_passed = bool(privacy_summary.get("passed", False))
        privacy_status = "PASS" if privacy_passed else "FAIL"
        _add_findings(privacy_summary.get("red_flags", []), "error", "privacy_auditor")

    quant_summary: dict[str, Any] | None = None
    quant_status = "NOT_RUN"
    quant_passed = True
    if quant_report is not None:
        quant_summary = quant_report.to_dict() if hasattr(quant_report, "to_dict") else dict(quant_report)
        quant_passed = bool(quant_summary.get("passed", False))
        quant_status = "PASS" if quant_passed else "FAIL"
        _add_findings(quant_summary.get("red_flags", []), "error", "quantization_shift_auditor")

    reg_passed = True if regression_diff is None else bool(regression_diff.get("passed", True))
    overall_passed = evasion_passed and backdoor_passed and privacy_passed and quant_passed and reg_passed

    pillar_coverage = {
        "evasion_and_anti_masking": {
            "status": "PASS" if evasion_passed else "FAIL",
            **STANDARDS_MAPPING["evasion_adaptive_anti_masking"],
        },
        "poisoning_and_backdoor": {
            "status": backdoor_status,
            **STANDARDS_MAPPING["backdoor_poisoning"],
        },
        "privacy_mia_and_extraction": {
            "status": privacy_status,
            **STANDARDS_MAPPING["privacy_mia"],
        },
        "deployment_quantization_shift": {
            "status": quant_status,
            **STANDARDS_MAPPING["quantization_shift"],
        },
    }

    model_fp = compute_model_fingerprint(model_or_fn) if model_or_fn is not None else None
    data_fp = compute_dataset_fingerprint(x_eval, y_eval) if x_eval is not None else None

    dossier = ConformityDossier(
        schema_version="1.3.0",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        target_name=verify_report.target_name,
        overall_passed=overall_passed,
        model_fingerprint=model_fp,
        dataset_fingerprint=data_fp,
        git_provenance=get_git_provenance(repo_dir),
        pillar_coverage=pillar_coverage,
        findings=findings,
        evasion_summary=dict(verify_report.summary),
        backdoor_summary=backdoor_summary,
        privacy_summary=privacy_summary,
        quantization_summary=quant_summary,
        regression_diff=regression_diff,
    )
    return dossier.seal(signing_key=signing_key)


def verify_dossier_integrity(
    dossier_or_dict: ConformityDossier | Mapping[str, Any],
    *,
    signing_key: str | bytes | None = None,
) -> tuple[bool, list[str]]:
    """Verify that a serialized ConformityDossier has not been tampered with after sealing."""
    data = dossier_or_dict.to_dict() if isinstance(dossier_or_dict, ConformityDossier) else dict(dossier_or_dict)
    recorded_sha = str(data.get("dossier_sha256", ""))
    recorded_hmac = data.get("hmac_sha256")
    unsigned = {k: v for k, v in data.items() if k not in ("dossier_sha256", "hmac_sha256")}
    canon = _canonical_json_bytes(unsigned)
    expected_sha = hashlib.sha256(canon).hexdigest()

    errors: list[str] = []
    if not hmac.compare_digest(recorded_sha, expected_sha):
        errors.append(f"SHA-256 digest mismatch: expected {expected_sha}, recorded {recorded_sha}")

    if signing_key is not None:
        if not recorded_hmac:
            errors.append("Signing key supplied, but dossier contains no HMAC-SHA256 signature.")
        else:
            key_b = signing_key.encode("utf-8") if isinstance(signing_key, str) else bytes(signing_key)
            expected_hmac = hmac.new(key_b, canon, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(str(recorded_hmac), expected_hmac):
                errors.append("HMAC-SHA256 signature verification failed (invalid key or tampered payload).")
    return len(errors) == 0, errors


def export_sarif(
    verify_report: AdvVerifyReport,
    *,
    backdoor_report: Any | None = None,
    privacy_report: Any | None = None,
    quant_report: Any | None = None,
    artifact_uri: str = "adv_eval/interface.py",
) -> dict[str, Any]:
    """Export findings as a valid SARIF 2.1.0 JSON object for GitHub Security / GitLab CI tabs."""
    from wannado.adv import __version__

    dossier = build_conformity_dossier(
        verify_report,
        backdoor_report=backdoor_report,
        privacy_report=privacy_report,
        quant_report=quant_report,
    )
    rules_by_id: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []

    for finding in dossier.findings:
        rule_id = str(finding["rule_id"])
        std = finding["standards"]
        if rule_id not in rules_by_id:
            rules_by_id[rule_id] = {
                "id": rule_id,
                "name": rule_id.replace(":", "_"),
                "shortDescription": {"text": f"Adversarial ML security check: {rule_id}"},
                "properties": {
                    "tags": [
                        "security",
                        "adversarial-ml",
                        std["nist_ai_100_2"],
                        *std["mitre_atlas"],
                        *std["eu_ai_act"],
                    ]
                },
            }
        results.append(
            {
                "ruleId": rule_id,
                "level": "error" if finding["severity"] == "error" else "warning",
                "message": {
                    "text": (
                        f"{finding['message']} "
                        f"[NIST={std['nist_ai_100_2']}; ATLAS={','.join(std['mitre_atlas'])}; "
                        f"EU_AI_Act={','.join(std['eu_ai_act'])}]"
                    )
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": artifact_uri},
                        }
                    }
                ],
            }
        )

    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "wannado-adv",
                        "version": __version__,
                        "informationUri": "https://github.com/JietaoZhao/wannado-adv",
                        "rules": list(rules_by_id.values()),
                    }
                },
                "results": results,
                "properties": {
                    "target_name": verify_report.target_name,
                    "overall_passed": dossier.overall_passed,
                    "dossier_sha256": dossier.dossier_sha256,
                },
            }
        ],
    }


__all__ = [
    "STANDARDS_MAPPING",
    "ConformityDossier",
    "build_conformity_dossier",
    "compare_verdicts",
    "compute_dataset_fingerprint",
    "compute_model_fingerprint",
    "export_sarif",
    "get_git_provenance",
    "map_flag_to_standards",
    "verify_dossier_integrity",
]
