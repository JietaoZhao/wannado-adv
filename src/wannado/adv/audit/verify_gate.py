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

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["red_team"] = self.red_team.to_dict() if self.red_team is not None else None
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

    passed = all(checks.values()) and len(red_flags) == 0
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
    )

    if out_dir is not None:
        out_root = Path(out_dir).resolve()
        saved = rt.save(out_root)
        saved_files = {k: str(v) for k, v in saved.items()}
        verdict_path = out_root / "verification_verdict.json"
        saved_files["verification_verdict"] = str(verdict_path)
        verdict_obj.saved_files = saved_files
        verdict_path.write_text(
            json.dumps(verdict_obj.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return verdict_obj


__all__ = ["AdvVerifyReport", "verify_robustness_gate"]
