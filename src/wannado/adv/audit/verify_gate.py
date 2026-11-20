"""Machine-Verifiable Adversarial Robustness CI & Agent Hard Gate (`wannado-adv verify`)."""

from __future__ import annotations

import importlib.util
import inspect
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from wannado.adv.audit.red_team import RedTeamReport, auto_red_team
from wannado.adv.reporter.conformity import (
    build_conformity_dossier,
    compare_verdicts,
    export_sarif,
)
from wannado.adv.workspace.interface_manager import VictimInterface


@dataclass
class AdvVerifyReport:
    """Structured machine-verifiable pass/fail verdict emitted by `verify_robustness_gate`."""

    passed: bool
    exit_code: int
    target_name: str
    checks: dict[str, bool] = field(default_factory=dict)
    red_flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    saved_files: dict[str, str] = field(default_factory=dict)
    red_team: RedTeamReport | None = None
    backdoor: Any | None = None
    privacy: Any | None = None
    quantization: Any | None = None
    regression_diff: dict[str, Any] | None = None
    dossier: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["red_team"] = self.red_team.to_dict() if self.red_team is not None else None
        d["backdoor"] = self.backdoor.to_dict() if (self.backdoor is not None and hasattr(self.backdoor, "to_dict")) else self.backdoor
        d["privacy"] = self.privacy.to_dict() if (self.privacy is not None and hasattr(self.privacy, "to_dict")) else self.privacy
        d["quantization"] = (
            self.quantization.to_dict()
            if (self.quantization is not None and hasattr(self.quantization, "to_dict"))
            else self.quantization
        )
        return d

    def format_report(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"[wannado-adv verify: {verdict}] target={self.target_name}",
            "  Gate Checks:",
        ]
        for k, v in self.checks.items():
            lines.append(f"    - {k}: {'OK' if v else 'FAIL'}")
        if self.red_team is not None:
            rt = self.red_team
            if rt.clean_acc is not None and rt.worst_case_robust_acc is not None:
                lines.append(
                    f"  Summary: clean={rt.clean_acc * 100:.2f}%, worst_case_robust={rt.worst_case_robust_acc * 100:.2f}%, "
                    f"strongest={rt.strongest_attack_id}, modes={rt.detected_defenses or ['clean']}"
                )
            else:
                lines.append(
                    f"  Summary: clean_mse={rt.clean_mse}, worst_adv_mse={rt.worst_case_adv_mse}, "
                    f"worst_l2_drift={rt.worst_case_output_l2_drift}, strongest={rt.strongest_attack_id}"
                )
        if self.red_flags:
            lines.append("  Red Flags:")
            lines.extend(f"    - {rf}" for rf in self.red_flags)
        if self.warnings:
            lines.append("  Warnings:")
            lines.extend(f"    - {w}" for w in self.warnings)
        return "\n".join(lines)


def _load_victim_from_interface(
    interface_path: Path,
    *,
    device: str = "cpu",
    num_samples: int = 100,
) -> VictimInterface:
    p = interface_path.resolve()
    mod_spec = importlib.util.spec_from_file_location("_wannado_adv_verify_iface", p)
    if mod_spec is None or mod_spec.loader is None:
        raise FileNotFoundError(f"Cannot load VictimInterface module at {p}")
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    loader_fn = mod.load_victim_interface
    sig_params = inspect.signature(loader_fn).parameters
    kwargs: dict[str, Any] = {}
    if "device" in sig_params:
        kwargs["device"] = device
    if "num_samples" in sig_params:
        kwargs["num_samples"] = num_samples
    return loader_fn(**kwargs)


def verify_robustness_gate(
    *,
    interface_path: str | Path | None = None,
    victim: VictimInterface | None = None,
    model_or_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    x: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
    y: torch.Tensor | None = None,
    eps: float | None = None,
    norm: str | None = None,
    clip_bounds: tuple[Any, Any] | None = None,
    defense_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    surrogate_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    goal: str = "standard",
    device: str = "cpu",
    samples: int = 100,
    batch_size: int | None = None,
    min_clean_acc: float | None = None,
    min_robust_acc: float | None = None,
    max_output_drift: float | None = None,
    fail_on_red_flag: bool = True,
    sweep_curve: bool = True,
    seed: int = 42,
    out_dir: str | Path | None = None,
    baseline_verdict_path: str | Path | None = None,
    max_robust_regression: float = 0.02,
    max_clean_regression: float = 0.02,
    sarif_path: str | Path | None = None,
    emit_conformity_dossier: bool = True,
    signing_key: str | bytes | None = None,
    run_backdoor_audit: bool = False,
    backdoor_steps: int = 40,
    member_data: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
    member_y: torch.Tensor | None = None,
    quantized_model_or_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_quant_robust_drop: float = 0.05,
    verbose: bool = False,
) -> AdvVerifyReport:
    """Execute automated red-team verification gate and emit machine-readable pass/fail verdict."""
    target_name = "custom_victim"
    if interface_path is not None:
        p = Path(interface_path).resolve()
        target_name = str(p)
        victim = _load_victim_from_interface(p, device=device, num_samples=samples)

    if victim is not None:
        model_or_fn = victim.model_or_fn
        x = victim.x_eval
        y = victim.y_eval
        eps = victim.eps if eps is None else eps
        norm = victim.norm if norm is None else norm
        clip_bounds = victim.clip_bounds if clip_bounds is None else clip_bounds
        defense_fn = victim.defense_fn if defense_fn is None else defense_fn
        surrogate_fn = victim.surrogate_fn if surrogate_fn is None else surrogate_fn
        domain: str | None = victim.domain
        model_name = victim.model_name
        dataset_name = victim.dataset_name
        if interface_path is None:
            target_name = f"{model_name}@{dataset_name}"
    else:
        domain = None
        model_name = "VictimModel"
        dataset_name = "EvalSplit"

    if model_or_fn is None or x is None:
        raise ValueError("Provide interface_path, victim (VictimInterface), or (model_or_fn, x[, y]).")

    resolved_norm = norm or "linf"
    resolved_eps = float(eps if eps is not None else (-10.0 if resolved_norm == "jsr_db" else 8.0 / 255.0))

    rt = auto_red_team(
        model_or_fn,
        x,
        y,
        eps=resolved_eps,
        norm=resolved_norm,
        clip_bounds=clip_bounds,
        defense_fn=defense_fn,
        surrogate_fn=surrogate_fn,
        goal=goal,
        batch_size=batch_size,
        max_samples=samples,
        sweep_curve=sweep_curve,
        seed=seed,
        domain=domain,
        model_name=model_name,
        dataset_name=dataset_name,
        verbose=verbose,
    )

    checks: dict[str, bool] = {}
    red_flags: list[str] = []
    warnings: list[str] = []

    pf_ok = bool(rt.preflight is None or rt.preflight.ok)
    checks["preflight_ok"] = pf_ok
    if not pf_ok and rt.preflight is not None:
        red_flags.extend(f"preflight:{e}" for e in rt.preflight.errors)
    if rt.preflight is not None:
        warnings.extend(rt.preflight.warnings)

    budget_ok = len(rt.records) > 0 and all(r.budget_respected and r.error is None for r in rt.records)
    checks["budget_and_execution_ok"] = budget_ok
    if not budget_ok:
        for r in rt.records:
            if r.error is not None:
                red_flags.append(f"attack_error:{r.attack_id}:{r.error}")
            elif not r.budget_respected:
                red_flags.append(f"budget_exceeded:{r.attack_id}")

    audit_ok = True
    if rt.audit is not None and not rt.audit.passed:
        audit_ok = False
        if fail_on_red_flag:
            red_flags.extend(f"carlini:{rf}" for rf in rt.audit.red_flags)
        else:
            warnings.extend(f"carlini:{rf}" for rf in rt.audit.red_flags)
    elif rt.audit is None and "hard_label_only" in rt.detected_defenses:
        audit_ok = False
        if fail_on_red_flag:
            red_flags.append("carlini:hard_label_only_unverifiable_whitebox")
        else:
            warnings.append("carlini:hard_label_only_unverifiable_whitebox")
    checks["carlini_audit_ok"] = audit_ok if fail_on_red_flag else True

    if min_clean_acc is not None and rt.clean_acc is not None:
        clean_ok = rt.clean_acc >= float(min_clean_acc)
        checks["min_clean_acc_ok"] = clean_ok
        if not clean_ok:
            red_flags.append(f"clean_acc_below_threshold:{rt.clean_acc:.4f}<{min_clean_acc:.4f}")

    if min_robust_acc is not None and rt.worst_case_robust_acc is not None:
        rob_ok = rt.worst_case_robust_acc >= float(min_robust_acc)
        checks["min_robust_acc_ok"] = rob_ok
        if not rob_ok:
            red_flags.append(
                f"worst_case_robust_acc_below_threshold:{rt.worst_case_robust_acc:.4f}<{min_robust_acc:.4f}"
            )

    if max_output_drift is not None and rt.worst_case_output_l2_drift is not None:
        drift_ok = rt.worst_case_output_l2_drift <= float(max_output_drift)
        checks["max_output_drift_ok"] = drift_ok
        if not drift_ok:
            red_flags.append(
                f"output_l2_drift_exceeded:{rt.worst_case_output_l2_drift:.4f}>{max_output_drift:.4f}"
            )

    # Optional Article 15 multi-pillar checks (Backdoor/Trojan, Privacy/MIA, Quantization-Shift)
    backdoor_rep: Any | None = None
    if run_backdoor_audit and rt.task_type == "classification" and not (
        isinstance(x, torch.Tensor) and x.is_complex()
    ):
        from wannado.adv.audit.backdoor_auditor import audit_backdoor

        backdoor_rep = audit_backdoor(
            model_or_fn,
            x,
            y,
            max_samples=min(samples, 64),
            steps=backdoor_steps,
            clip_bounds=clip_bounds,
            seed=seed,
        )
        checks["backdoor_audit_ok"] = backdoor_rep.passed
        if not backdoor_rep.passed:
            red_flags.extend(backdoor_rep.red_flags)

    privacy_rep: Any | None = None
    if member_data is not None and rt.task_type == "classification":
        from wannado.adv.audit.privacy_auditor import audit_privacy

        privacy_rep = audit_privacy(
            model_or_fn,
            member_data,
            x,
            member_y=member_y,
            non_member_y=y,
            max_samples=min(samples, 256),
            seed=seed,
        )
        checks["privacy_audit_ok"] = privacy_rep.passed
        if not privacy_rep.passed:
            red_flags.extend(privacy_rep.red_flags)

    quant_rep: Any | None = None
    if quantized_model_or_fn is not None and rt.task_type == "classification":
        from wannado.adv.adapters.deployment import audit_quantization_shift

        quant_rep = audit_quantization_shift(
            model_or_fn,
            quantized_model_or_fn,
            x,
            y,
            eps=resolved_eps,
            norm=resolved_norm,
            clip_bounds=clip_bounds,
            max_samples=min(samples, 64),
            max_robust_drop=max_quant_robust_drop,
            seed=seed,
        )
        checks["quantization_shift_ok"] = quant_rep.passed
        if not quant_rep.passed:
            red_flags.extend(quant_rep.red_flags)

    summary = {
        "task_type": rt.task_type,
        "clean_acc": rt.clean_acc,
        "clean_ci_95": list(rt.clean_ci_95) if rt.clean_ci_95 is not None else None,
        "worst_case_robust_acc": rt.worst_case_robust_acc,
        "worst_case_ci_95": list(rt.worst_case_ci_95) if rt.worst_case_ci_95 is not None else None,
        "ensemble_cond_asr": rt.ensemble_cond_asr,
        "adaptive_bypass_gap": rt.adaptive_bypass_gap,
        "strongest_attack_id": rt.strongest_attack_id,
        "detected_defenses": rt.detected_defenses,
        "aurc": rt.aurc,
        "eps_50": rt.eps_50,
        "clean_mse": rt.clean_mse,
        "worst_case_adv_mse": rt.worst_case_adv_mse,
        "worst_case_output_l2_drift": rt.worst_case_output_l2_drift,
    }

    regression_diff: dict[str, Any] | None = None
    if baseline_verdict_path is not None:
        base_p = Path(baseline_verdict_path).resolve()
        baseline_payload = json.loads(base_p.read_text(encoding="utf-8"))
        regression_diff = compare_verdicts(
            {"summary": summary},
            baseline_payload,
            max_robust_regression=max_robust_regression,
            max_clean_regression=max_clean_regression,
        )
        checks["no_robustness_regression_ok"] = bool(regression_diff["passed"])
        if not regression_diff["passed"]:
            red_flags.extend(regression_diff["regressions"])

    passed = all(checks.values()) and len(red_flags) == 0

    saved_files: dict[str, str] = {}
    verdict_obj = AdvVerifyReport(
        passed=passed,
        exit_code=0 if passed else 1,
        target_name=target_name,
        checks=checks,
        red_flags=red_flags,
        warnings=warnings,
        summary=summary,
        saved_files=saved_files,
        red_team=rt,
        backdoor=backdoor_rep,
        privacy=privacy_rep,
        quantization=quant_rep,
        regression_diff=regression_diff,
    )

    dossier_obj = (
        build_conformity_dossier(
            verdict_obj,
            backdoor_report=backdoor_rep,
            privacy_report=privacy_rep,
            quant_report=quant_rep,
            model_or_fn=model_or_fn,
            x_eval=x,
            y_eval=y,
            regression_diff=regression_diff,
            signing_key=signing_key,
        )
        if emit_conformity_dossier
        else None
    )
    if dossier_obj is not None:
        verdict_obj.dossier = dossier_obj.to_dict()

    if out_dir is not None:
        out_root = Path(out_dir).resolve()
        saved = rt.save(out_root)
        saved_files = {k: str(v) for k, v in saved.items()}
        if dossier_obj is not None:
            d_json = out_root / "conformity_dossier.json"
            d_md = out_root / "conformity_dossier.md"
            d_json.write_text(dossier_obj.to_json(), encoding="utf-8")
            d_md.write_text(dossier_obj.format_markdown(), encoding="utf-8")
            saved_files["conformity_dossier_json"] = str(d_json)
            saved_files["conformity_dossier_md"] = str(d_md)
        sarif_out = Path(sarif_path).resolve() if sarif_path is not None else (out_root / "verification_report.sarif")
        sarif_out.parent.mkdir(parents=True, exist_ok=True)
        sarif_payload = export_sarif(
            verdict_obj,
            backdoor_report=backdoor_rep,
            privacy_report=privacy_rep,
            quant_report=quant_rep,
            artifact_uri=target_name,
        )
        sarif_out.write_text(json.dumps(sarif_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        saved_files["sarif"] = str(sarif_out)
        verdict_path = out_root / "verification_verdict.json"
        saved_files["verification_verdict"] = str(verdict_path)
        verdict_obj.saved_files = saved_files
        verdict_path.write_text(
            json.dumps(verdict_obj.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    elif sarif_path is not None:
        sarif_out = Path(sarif_path).resolve()
        sarif_out.parent.mkdir(parents=True, exist_ok=True)
        sarif_payload = export_sarif(
            verdict_obj,
            backdoor_report=backdoor_rep,
            privacy_report=privacy_rep,
            quant_report=quant_rep,
            artifact_uri=target_name,
        )
        sarif_out.write_text(json.dumps(sarif_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        verdict_obj.saved_files = {"sarif": str(sarif_out)}

    return verdict_obj


__all__ = ["AdvVerifyReport", "verify_robustness_gate"]
