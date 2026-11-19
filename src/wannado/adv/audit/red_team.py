"""Closed-loop Autonomous Adversarial Red-Teaming Engine (`auto_red_team`).

Automates end-to-end robustness testing without manual attack guessing:
  1. Diagnoses victim modality, determinism/stochasticity, autograd health, and Carlini 2.0 traps.
  2. Adaptively escalates attacks matched to the detected failure mode
     (APGD-DLR for Softmax saturation, EoT-PGD for stochasticity, BPDA+EoT for defense_fn,
     APGD-L1 for L1 geometry, Square / HopSkipJump for broken/hard-label black-box victims).
  3. Tracks true per-sample worst-case ensemble survival accuracy with Wilson 95% CIs
     and true conditional ASR on originally correct predictions.
  4. Sweeps a perturbation robustness curve (AURC + eps_50 critical failure budget) and
     exports publication-ready JSON, Markdown, CSV, LaTeX, BibTeX, and plots.
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from wannado.adv.audit.carlini_auditor import AuditReport, audit_robustness
from wannado.adv.audit.preflight import PreflightReport, preflight
from wannado.adv.catalog.build_catalog import load_catalog
from wannado.adv.core.losses import predict_labels, unwrap_model_outputs
from wannado.adv.core.projections import compute_perturbation_norm, safe_l2_norm
from wannado.adv.core.utils import DefendedModelWrapper, eval_mode, iter_eval_batches

if TYPE_CHECKING:
    from wannado.adv.reporter.paper_reporter import AttackRunRecord, EvaluationReport


def wilson_ci(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Compute exact 95% Wilson score confidence interval for a binomial proportion (zero SciPy deps)."""
    if total <= 0:
        return 0.0, 0.0
    p = max(0.0, min(1.0, float(successes) / float(total)))
    z2 = z * z
    denom = 1.0 + z2 / float(total)
    center = (p + z2 / (2.0 * float(total))) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / float(total)) + (z2 / (4.0 * float(total) * float(total))))
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_aurc_and_eps50(
    curve_points: Sequence[dict[str, Any]],
    *,
    clean_acc: float | None,
) -> tuple[float | None, float | None]:
    """Compute normalized Area Under Robustness Curve (AURC) and critical 50% drop budget `eps_50`."""
    if len(curve_points) < 2 or clean_acc is None:
        return None, None
    pts = sorted(curve_points, key=lambda d: float(d["eps"]))
    xs = [float(p["eps"]) for p in pts]
    ys = [float(p["robust_acc"]) for p in pts]
    span = xs[-1] - xs[0]
    if span <= 1e-12:
        return float(ys[0]), None
    area = 0.0
    for i in range(len(xs) - 1):
        area += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
    aurc = float(area / span)

    target = 0.5 * float(clean_acc)
    eps_50: float | None = None
    if target > 0.0:
        for i in range(len(xs) - 1):
            y0, y1 = ys[i], ys[i + 1]
            if y0 >= target >= y1:
                if abs(y0 - y1) <= 1e-12:
                    eps_50 = xs[i + 1]
                else:
                    frac = (y0 - target) / (y0 - y1)
                    eps_50 = xs[i] + frac * (xs[i + 1] - xs[i])
                break
        if eps_50 is None and ys[0] <= target:
            eps_50 = xs[0]
    return aurc, eps_50


@dataclass
class RedTeamReport:
    """Complete closed-loop autonomous red-team dossier emitted by ``auto_red_team()``."""

    model_name: str
    dataset_name: str
    domain: str
    task_type: str
    norm: str
    eps: float
    clip_bounds: tuple[Any, Any] | None
    num_samples: int
    seed: int
    device: str
    dtype: str
    timestamp_utc: str
    clean_acc: float | None
    clean_ci_95: tuple[float, float] | None
    clean_mse: float | None
    worst_case_robust_acc: float | None
    worst_case_ci_95: tuple[float, float] | None
    ensemble_cond_asr: float | None
    worst_case_adv_mse: float | None
    worst_case_output_l2_drift: float | None
    strongest_attack_id: str | None
    adaptive_bypass_gap: float
    detected_defenses: list[str]
    escalation_plan: list[dict[str, Any]]
    aurc: float | None
    eps_50: float | None
    curve_attack_id: str | None
    curve_points: list[dict[str, Any]] = field(default_factory=list)
    preflight: PreflightReport | None = None
    audit: AuditReport | None = None
    records: list[AttackRunRecord] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_evaluation_report(self) -> EvaluationReport:
        """Convert into a standard ``EvaluationReport`` for LaTeX/BibTeX export compatibility."""
        from wannado.adv.reporter.paper_reporter import EvaluationReport

        return EvaluationReport(
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            domain=self.domain,
            norm=self.norm,
            eps=self.eps,
            clip_bounds=self.clip_bounds,
            num_samples=self.num_samples,
            seed=self.seed,
            device=self.device,
            dtype=self.dtype,
            torch_version=torch.__version__,
            clean_acc=self.clean_acc,
            timestamp_utc=self.timestamp_utc,
            notes=f"Autonomous red-team (detected_defenses={self.detected_defenses}, strongest={self.strongest_attack_id})",
            preflight=self.preflight,
            audit=self.audit,
            records=list(self.records),
            clean_mse=self.clean_mse,
        )

    def to_dict(self) -> dict[str, Any]:
        from wannado.adv.reporter.paper_reporter import _serialize_bounds

        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "domain": self.domain,
            "task_type": self.task_type,
            "norm": self.norm,
            "eps": self.eps,
            "clip_bounds": _serialize_bounds(self.clip_bounds),
            "num_samples": self.num_samples,
            "seed": self.seed,
            "device": self.device,
            "dtype": self.dtype,
            "timestamp_utc": self.timestamp_utc,
            "clean_acc": self.clean_acc,
            "clean_ci_95": list(self.clean_ci_95) if self.clean_ci_95 is not None else None,
            "clean_mse": self.clean_mse,
            "worst_case_robust_acc": self.worst_case_robust_acc,
            "worst_case_ci_95": list(self.worst_case_ci_95) if self.worst_case_ci_95 is not None else None,
            "ensemble_cond_asr": self.ensemble_cond_asr,
            "worst_case_adv_mse": self.worst_case_adv_mse,
            "worst_case_output_l2_drift": self.worst_case_output_l2_drift,
            "strongest_attack_id": self.strongest_attack_id,
            "adaptive_bypass_gap": self.adaptive_bypass_gap,
            "detected_defenses": list(self.detected_defenses),
            "escalation_plan": list(self.escalation_plan),
            "aurc": self.aurc,
            "eps_50": self.eps_50,
            "curve_attack_id": self.curve_attack_id,
            "curve_points": list(self.curve_points),
            "elapsed_seconds": self.elapsed_seconds,
            "preflight": self.preflight.to_dict() if self.preflight is not None else None,
            "audit": self.audit.to_dict() if self.audit is not None else None,
            "attacks": [asdict(r) for r in self.records],
        }

    def format_summary(self) -> str:
        lines = [
            f"# Autonomous Red-Team Robustness Report — {self.model_name} on {self.dataset_name}",
            "",
            "## 1. Executive Verdict & Adaptive Escalation",
            f"- **Task / Domain / Budget**: `{self.task_type}` | `{self.domain}` | `norm='{self.norm}'`, `eps={self.eps:.6g}` | `N={self.num_samples}`",
            f"- **Detected Failure / Defense Modes**: `{self.detected_defenses or ['clean_differentiable']}`",
            f"- **Strongest Escalated Attack**: `{self.strongest_attack_id}`",
        ]
        if self.clean_acc is not None and self.worst_case_robust_acc is not None:
            c_lo, c_hi = self.clean_ci_95 or (self.clean_acc, self.clean_acc)
            w_lo, w_hi = self.worst_case_ci_95 or (self.worst_case_robust_acc, self.worst_case_robust_acc)
            lines.append(
                f"- **Clean Accuracy**: `{self.clean_acc * 100:.2f}%` (95% Wilson CI `[{c_lo * 100:.1f}%, {c_hi * 100:.1f}%]`)"
            )
            lines.append(
                f"- **Worst-Case Ensemble Robust Accuracy**: `{self.worst_case_robust_acc * 100:.2f}%` "
                f"(95% Wilson CI `[{w_lo * 100:.1f}%, {w_hi * 100:.1f}%]`, Ensemble Cond. ASR `{(self.ensemble_cond_asr or 0.0) * 100:.2f}%`)"
            )
            if self.adaptive_bypass_gap > 1e-3:
                lines.append(
                    f"- **Adaptive Bypass Gap (False Robustness Exposed)**: `+{self.adaptive_bypass_gap * 100:.2f} pp` "
                    "(naive white-box overestimated robustness relative to escalated worst-case)."
                )
            if self.aurc is not None:
                e50_str = f"`{self.eps_50:.4g}`" if self.eps_50 is not None else "`> sweep max`"
                lines.append(
                    f"- **Robustness Curve (`{self.curve_attack_id}`)**: `AURC = {self.aurc:.4f}` | `eps_50 = {e50_str}`"
                )
        else:
            lines.append(
                f"- **Clean MSE**: `{self.clean_mse}` | **Worst-Case Adv MSE**: `{self.worst_case_adv_mse}` | "
                f"**Worst-Case Output L2 Drift**: `{self.worst_case_output_l2_drift}`"
            )

        lines.extend(["", "## 2. Escalation Trace"])
        for step in self.escalation_plan:
            lines.append(f"- `{step['attack_id']}` — {step['reason']}")

        lines.extend(["", self.to_evaluation_report().format_paper_summary()])
        return "\n".join(lines)

    def _save_curve_plot(self, out_path: Path) -> Path | None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # noqa: BLE001
            return None

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=160)

        # Panel A: Attack ladder bar chart
        ax0 = axes[0]
        if self.clean_acc is not None:
            labels = ["Clean"] + [r.attack_id for r in self.records if r.robust_acc is not None] + ["Ensemble\nWorst"]
            vals = (
                [self.clean_acc * 100.0]
                + [(r.robust_acc or 0.0) * 100.0 for r in self.records if r.robust_acc is not None]
                + [(self.worst_case_robust_acc or 0.0) * 100.0]
            )
            colors = ["#0f766e"] + ["#2563eb"] * (len(vals) - 2) + ["#dc2626"]
            ax0.bar(range(len(labels)), vals, color=colors, alpha=0.88)
            ax0.set_xticks(range(len(labels)))
            ax0.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
            ax0.set_ylabel("Accuracy (%)")
            ax0.set_ylim(0, 105)
            ax0.set_title("Adaptive Red-Team Attack Ladder")
        else:
            labels = [r.attack_id for r in self.records if r.output_l2_drift is not None]
            vals = [r.output_l2_drift or 0.0 for r in self.records if r.output_l2_drift is not None]
            ax0.bar(range(len(labels)), vals, color="#2563eb", alpha=0.88)
            ax0.set_xticks(range(len(labels)))
            ax0.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
            ax0.set_ylabel("Output L2 Drift")
            ax0.set_title("Continuous Model Drift Under Attack")
        ax0.grid(axis="y", alpha=0.3)

        # Panel B: Budget decay curve
        ax1 = axes[1]
        if self.curve_points:
            xs = [float(p["eps"]) for p in self.curve_points]
            if self.clean_acc is not None:
                ys = [float(p["robust_acc"]) * 100.0 for p in self.curve_points]
                lo = [float(p.get("ci_low", p["robust_acc"])) * 100.0 for p in self.curve_points]
                hi = [float(p.get("ci_high", p["robust_acc"])) * 100.0 for p in self.curve_points]
                ax1.plot(xs, ys, "-o", color="#dc2626", linewidth=2.0, label=f"{self.curve_attack_id}")
                ax1.fill_between(xs, lo, hi, color="#dc2626", alpha=0.16, label="95% Wilson CI")
                if self.eps_50 is not None:
                    ax1.axvline(self.eps_50, color="#475569", linestyle="--", linewidth=1.2, label=f"eps_50={self.eps_50:.3g}")
                ax1.set_ylim(0, 105)
                ax1.set_ylabel("Robust Accuracy (%)")
                aurc_s = f" (AURC={self.aurc:.3f})" if self.aurc is not None else ""
                ax1.set_title(f"Robustness Decay Curve{aurc_s}")
            else:
                ys = [float(p.get("output_l2_drift", 0.0)) for p in self.curve_points]
                ax1.plot(xs, ys, "-o", color="#dc2626", linewidth=2.0, label=f"{self.curve_attack_id}")
                ax1.set_ylabel("Output L2 Drift")
                ax1.set_title("Continuous Drift vs Budget")
            ax1.set_xlabel("JSR (dB)" if self.norm == "jsr_db" else f"Perturbation Budget ({self.norm})")
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "Curve sweep skipped", ha="center", va="center")
            ax1.axis("off")

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        """Export complete red-team JSON, Markdown, LaTeX, BibTeX, CSV curve, and optional PNG chart."""
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        ev = self.to_evaluation_report()

        paths: dict[str, Path] = {
            "red_team_json": out / "red_team_report.json",
            "red_team_markdown": out / "red_team_summary.md",
            "latex": out / "paper_table.tex",
            "bibtex": out / "references.bib",
            "curve_csv": out / "robustness_curve.csv",
        }
        paths["red_team_json"].write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        paths["red_team_markdown"].write_text(self.format_summary() + "\n", encoding="utf-8")
        paths["latex"].write_text(ev.to_latex(), encoding="utf-8")
        paths["bibtex"].write_text(ev.to_bibtex(), encoding="utf-8")

        with paths["curve_csv"].open("w", newline="", encoding="utf-8") as fh:
            if self.curve_points:
                keys = list(self.curve_points[0].keys())
                writer = csv.DictWriter(fh, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.curve_points)
            else:
                fh.write("eps,robust_acc\n")

        plot_p = self._save_curve_plot(out / "robustness_curve.png")
        if plot_p is not None:
            paths["curve_png"] = plot_p
        return paths


def _build_adaptive_escalation_plan(
    *,
    is_classification: bool,
    norm: str,
    goal: str,
    detected_defenses: Sequence[str],
    has_defense_fn: bool,
) -> list[dict[str, Any]]:
    """Construct an ordered attack escalation roster responding directly to diagnosed victim properties."""
    plan: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(aid: str, reason: str, kwargs: dict[str, Any]) -> None:
        if aid not in seen:
            seen.add(aid)
            plan.append({"attack_id": aid, "reason": reason, "kwargs": kwargs})

    steps_scale = {"sanity": 0.5, "standard": 1.0, "paper_sota": 1.6}.get(goal, 1.0)
    is_hard = "hard_label_only" in detected_defenses
    is_stoch = "stochastic_randomization" in detected_defenses
    grad_broken = "broken_or_obfuscated_gradients" in detected_defenses
    sat = "softmax_saturation" in detected_defenses

    if not is_classification:
        if has_defense_fn:
            _add("bpda_eot", "Wired defense_fn detected; bypassing preprocessor with BPDA+EoT (MSE).", {"steps": max(8, int(20 * steps_scale)), "eot_samples": 4 if is_stoch else 2, "loss": "mse"})
        if not grad_broken:
            _add("fgsm", "Single-step sensitivity baseline on continuous outputs.", {"loss": "mse"})
            _add("pgd", "Iterative white-box worst-case continuous drift.", {"steps": max(8, int(20 * steps_scale)), "loss": "mse"})
            if is_stoch:
                _add("eot_pgd", "Test-time stochasticity detected; averaging gradients over EoT draws.", {"steps": max(8, int(20 * steps_scale)), "eot_samples": 6, "loss": "mse"})
        elif norm != "l1":
            _add("square_attack", "Broken continuous gradients; gradient-free score search.", {"max_queries": max(100, int(250 * steps_scale)), "loss": "mse"})
        return plan

    if is_hard:
        _add("boundary", "Hard-label-only oracle detected (no logits/gradients); boundary random walk.", {"steps": max(30, int(60 * steps_scale))})
        _add("hopskipjump", "Hard-label SOTA escalation (HopSkipJumpAttack).", {"steps": max(10, int(20 * steps_scale)), "initial_num_evals": 30})
        return plan

    if sat and not is_hard:
        if norm == "l1":
            _add("apgd_l1", "Softmax saturation detected; escalating to scale-invariant APGD-L1 (DLR).", {"steps": max(20, int(35 * steps_scale)), "restarts": 2, "loss": "dlr"})
        else:
            _add("apgd_dlr", "Softmax saturation detected; escalating to multi-restart scale-invariant APGD-DLR to defeat logit rescaling.", {"steps": max(25, int(40 * steps_scale)), "restarts": 2})

    if not grad_broken:
        _add("fgsm", "Fast 1-step linear sensitivity baseline.", {})
        if sat:
            _add("pgd", "Softmax saturation detected; running margin-loss PGD immune to Softmax gradient vanishing.", {"steps": max(15, int(25 * steps_scale)), "loss": "margin"})
        else:
            _add("pgd", "Standard iterative first-order white-box baseline.", {"steps": max(10, int(20 * steps_scale))})
        if norm == "l1":
            _add("apgd_l1", "L1 geometry SOTA (Mind-the-Box APGD-L1).", {"steps": max(15, int(35 * steps_scale)), "loss": "dlr"})
        else:
            dlr_reason = (
                "Softmax saturation detected; scale-invariant APGD-DLR defeats logit rescaling."
                if sat
                else "Parameter-free scale-invariant APGD-DLR verification."
            )
            _add("apgd_dlr", dlr_reason, {"steps": max(15, int(30 * steps_scale)), "restarts": 1 if goal == "sanity" else 2})
            if goal in ("standard", "paper_sota"):
                _add("fab", "Vectorized geometric boundary check (FAB).", {"steps": max(10, int(18 * steps_scale)), "restarts": 1})
                _add("square_attack", "Gradient-free score-based black-box sanity check (Square Attack).", {"max_queries": max(150, int(300 * steps_scale))})
            if goal == "paper_sota":
                _add(
                    "autoattack",
                    "Full AutoAttack ensemble verification.",
                    {"version": "standard", "apgd_steps": 40, "fab_steps": 20, "square_queries": 400},
                )

    if is_stoch:
        if has_defense_fn:
            _add("bpda_eot", "Stochastic + defense_fn detected; bypassing with BPDA+EoT.", {"steps": max(12, int(25 * steps_scale)), "eot_samples": max(4, int(6 * steps_scale))})
        elif not grad_broken:
            _add("eot_pgd", "Test-time randomization detected; escalating to EoT-PGD expectation gradients.", {"steps": max(12, int(25 * steps_scale)), "eot_samples": max(4, int(6 * steps_scale))})
        if norm != "l1":
            _add("square_attack", "Score-based black-box confirmation under stochasticity.", {"max_queries": max(180, int(350 * steps_scale))})

    if grad_broken or has_defense_fn:
        if has_defense_fn:
            _add("bpda_eot", "Non-differentiable/custom defense_fn wired; bypassing with BPDA+EoT.", {"steps": max(15, int(25 * steps_scale)), "eot_samples": 4 if is_stoch else 2})
        if norm != "l1":
            _add("square_attack", "Broken/obfuscated gradients detected; escalating to gradient-free Square Attack.", {"max_queries": max(200, int(400 * steps_scale))})
        elif not has_defense_fn:
            _add("hopskipjump", "Broken L1 gradients; escalating to label-only HopSkipJump.", {"steps": max(10, int(18 * steps_scale)), "initial_num_evals": 24})

    return plan


def auto_red_team(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    *,
    eps: float = 0.03,
    norm: str = "linf",
    clip_bounds: tuple[Any, Any] | None = None,
    defense_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    surrogate_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    goal: str = "standard",
    batch_size: int | None = None,
    max_samples: int | None = None,
    audit_samples: int = 48,
    curve_samples: int = 48,
    sweep_curve: bool = True,
    curve_budgets: Sequence[float] | None = None,
    seed: int = 42,
    domain: str | None = None,
    model_name: str = "VictimModel",
    dataset_name: str = "EvalSplit",
    verbose: bool = True,
) -> RedTeamReport:
    """Run one-click closed-loop autonomous adversarial red-teaming on any victim model or DataLoader."""
    from wannado.adv.reporter.paper_reporter import (
        AttackRunRecord,
        _build_call_kwargs,
        _per_sample_cosine,
        _per_sample_mse,
        _resolve_attack_entry,
    )

    if goal not in ("sanity", "standard", "paper_sota"):
        raise ValueError(f"goal must be 'sanity', 'standard', or 'paper_sota', got '{goal}'.")

    t_start = time.perf_counter()
    from_loader = not isinstance(x, torch.Tensor)
    raw_batches = list(iter_eval_batches(x, y, batch_size=batch_size, max_samples=max_samples))
    if not raw_batches:
        raise ValueError("Evaluation input yielded zero samples.")

    first_x, _first_y = raw_batches[0]
    eval_device = first_x.device
    if from_loader and isinstance(model_or_fn, nn.Module):
        for p in model_or_fn.parameters():
            eval_device = p.device
            break

    is_complex = bool(first_x.is_complex())
    resolved_domain = domain or ("wireless" if is_complex else ("vision" if first_x.ndim == 4 else "tabular"))
    deployed_victim: Callable[[torch.Tensor], torch.Tensor] = (
        DefendedModelWrapper(model_or_fn, defense_fn) if defense_fn is not None else model_or_fn
    )
    total_n = sum(int(xb.shape[0]) for xb, _ in raw_batches)

    # 1. Clean pass across all chunks
    is_classification = False
    is_hard_label_only = False
    clean_correct_chunks: list[torch.Tensor] = []
    clean_outs_chunks: list[torch.Tensor] = []
    clean_mse_sum = 0.0
    clean_mse_applicable = True

    with eval_mode(deployed_victim), torch.no_grad():
        for b_idx, (xb_raw, yb_raw) in enumerate(raw_batches):
            xb = xb_raw.to(eval_device)
            yb = yb_raw.to(eval_device)
            out = unwrap_model_outputs(deployed_victim(xb))
            if b_idx == 0:
                is_hard_label_only = out.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)
                is_classification = bool(
                    out.ndim >= 1
                    and yb.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.long)
                )
            if is_classification:
                c_preds = predict_labels(out)
                y_cmp = yb.squeeze(1) if c_preds.ndim == 1 and yb.ndim == 2 and yb.shape[1] == 1 else yb
                clean_correct_chunks.append((c_preds == y_cmp).detach().cpu())
            else:
                clean_outs_chunks.append(out.detach().cpu())
                if out.shape == yb.shape:
                    clean_mse_sum += float(_per_sample_mse(out, yb).sum().item())
                else:
                    clean_mse_applicable = False

    clean_correct_all = torch.cat(clean_correct_chunks, dim=0) if is_classification else None
    clean_correct_count = int(clean_correct_all.sum().item()) if clean_correct_all is not None else 0
    clean_acc: float | None = (clean_correct_count / max(total_n, 1)) if is_classification else None
    clean_ci_95 = wilson_ci(clean_correct_count, total_n) if is_classification else None
    clean_mse: float | None = (clean_mse_sum / max(total_n, 1)) if (not is_classification and clean_mse_applicable) else None

    # 2. Preflight & Carlini 2.0 diagnosis on bounded prefix slice
    k_audit = min(total_n, max(2, audit_samples))
    ax_list: list[torch.Tensor] = []
    ay_list: list[torch.Tensor] = []
    collected = 0
    for xb_raw, yb_raw in raw_batches:
        take = min(xb_raw.shape[0], k_audit - collected)
        if take <= 0:
            break
        ax_list.append(xb_raw[:take])
        ay_list.append(yb_raw[:take])
        collected += take
    x_audit = torch.cat(ax_list, dim=0).to(eval_device)
    y_audit = torch.cat(ay_list, dim=0).to(eval_device)

    pf_loss = "ce" if is_classification else ("mse" if clean_mse_applicable else None)
    pf_rep = preflight(
        model_or_fn,
        x_audit[: min(x_audit.shape[0], 16)],
        y_audit[: min(y_audit.shape[0], 16)],
        eps=eps,
        norm=norm,
        clip_bounds=clip_bounds,
        loss=pf_loss,
        defense_fn=defense_fn,
    )

    audit_rep: AuditReport | None = None
    if total_n >= 2 and not is_hard_label_only:
        audit_rep = audit_robustness(
            model_or_fn,
            x_audit,
            y_audit,
            eps=eps,
            norm=norm,
            clip_bounds=clip_bounds,
            pgd_steps=12 if goal == "sanity" else 18,
            apgd_steps=15 if goal == "sanity" else 25,
            square_queries=150 if goal == "sanity" else 280,
            defense_fn=defense_fn,
            surrogate_fn=surrogate_fn,
            seed=seed,
        )

    detected_defenses: list[str] = []
    if is_hard_label_only:
        detected_defenses.append("hard_label_only")
    if defense_fn is not None:
        detected_defenses.append("custom_defense_fn")
    if audit_rep is not None:
        if audit_rep.metrics.get("is_stochastic") or "stochastic_defense_requires_eot" in audit_rep.red_flags:
            detected_defenses.append("stochastic_randomization")
        if "softmax_saturation_masking" in audit_rep.red_flags:
            detected_defenses.append("softmax_saturation")
        if any(
            rf in audit_rep.red_flags
            for rf in (
                "broken_or_vanishing_gradients",
                "gradient_obfuscation_blackbox_beats_whitebox",
                "obfuscated_defense_broken_by_bpda",
            )
        ):
            detected_defenses.append("broken_or_obfuscated_gradients")
    elif not pf_rep.summary.get("gradient_probe", {}).get("autograd_connected", False) and not is_hard_label_only:
        detected_defenses.append("broken_or_obfuscated_gradients")

    plan = _build_adaptive_escalation_plan(
        is_classification=is_classification,
        norm=norm,
        goal=goal,
        detected_defenses=detected_defenses,
        has_defense_fn=defense_fn is not None,
    )

    if verbose:
        print(
            f"[wannado-adv red-team] Detected modes={detected_defenses or ['clean_differentiable']} -> "
            f"escalating {[p['attack_id'] for p in plan]}",
            flush=True,
        )

    # 3. Execute escalation plan while tracking per-sample ensemble survival
    by_id = {c["id"]: c for c in load_catalog()}
    tol = 1e-2 if norm == "jsr_db" else max(1e-4, abs(eps) * 1e-3)
    ensemble_surviving = clean_correct_all.clone() if clean_correct_all is not None else None
    records: list[AttackRunRecord] = []
    naive_wb_accs: list[float] = []

    for step_idx, step_spec in enumerate(plan, start=1):
        aid = str(step_spec["attack_id"])
        overrides = dict(step_spec["kwargs"])
        resolved_id, fn, card = _resolve_attack_entry(aid, by_id)
        sig = inspect.signature(fn)
        uses_bpda = defense_fn is not None and "defense_fn" in sig.parameters
        if uses_bpda:
            overrides.setdefault("defense_fn", defense_fn)
            if surrogate_fn is not None and "surrogate_fn" in sig.parameters:
                overrides.setdefault("surrogate_fn", surrogate_fn)
        call_kwargs, disclosed_hp = _build_call_kwargs(
            fn, eps=eps, norm=norm, clip_bounds=clip_bounds, seed=seed + step_idx * 17, overrides=overrides
        )
        if batch_size is not None and "batch_size" in sig.parameters and "batch_size" not in call_kwargs:
            call_kwargs["batch_size"] = int(batch_size)
        attack_victim = model_or_fn if uses_bpda else deployed_victim

        t0 = time.perf_counter()
        try:
            norm_sum = 0.0
            max_norm = float("-inf")
            steps_max = 0
            queries_total: int | None = None
            elapsed_ms_sum = 0.0
            adv_correct_chunks: list[torch.Tensor] = []
            adv_mse_sum = 0.0
            l2_drift_sum = 0.0
            cos_sum = 0.0
            last_extra: dict[str, Any] = {"escalation_reason": step_spec["reason"]}

            for b_idx, (xb_raw, yb_raw) in enumerate(raw_batches):
                xb = xb_raw.to(eval_device)
                yb = yb_raw.to(eval_device)
                ck = dict(call_kwargs)
                if "seed" in ck and ck["seed"] is not None and b_idx > 0:
                    ck["seed"] = int(ck["seed"]) + b_idx * 1009

                tc = time.perf_counter()
                res = fn(attack_victim, xb, yb, **ck) if "y" in sig.parameters else fn(attack_victim, xb, **ck)
                elapsed_ms_sum += float(res.elapsed_ms) if getattr(res, "elapsed_ms", None) else (time.perf_counter() - tc) * 1000.0
                steps_max = max(steps_max, int(res.steps_run))

                norms = compute_perturbation_norm(res.perturbation, norm=norm, x_ref=xb)
                norm_sum += float(norms.sum().item())
                max_norm = max(max_norm, float(norms.max().item()))
                if isinstance(res.extra, dict):
                    q_val = res.extra.get("queries")
                    if q_val is not None:
                        queries_total = (queries_total or 0) + int(q_val)
                    last_extra.update(res.extra)

                with eval_mode(deployed_victim), torch.no_grad():
                    adv_out = unwrap_model_outputs(deployed_victim(res.x_adv))
                    if is_classification:
                        adv_preds = predict_labels(adv_out)
                        y_cmp = yb.squeeze(1) if adv_preds.ndim == 1 and yb.ndim == 2 and yb.shape[1] == 1 else yb
                        adv_correct_chunks.append((adv_preds == y_cmp).detach().cpu())
                    else:
                        c_out_b = clean_outs_chunks[b_idx].to(eval_device)
                        diff_out = (adv_out - c_out_b).reshape(xb.shape[0], -1)
                        l2_drift_sum += float(safe_l2_norm(diff_out, dim=1).sum().item())
                        cos_sum += float(_per_sample_cosine(adv_out, c_out_b).sum().item())
                        if clean_mse_applicable and adv_out.shape == yb.shape:
                            adv_mse_sum += float(_per_sample_mse(adv_out, yb).sum().item())

            mean_norm = norm_sum / max(total_n, 1)
            budget_ok = (max_norm <= eps + tol) if math.isfinite(max_norm) else True

            rob_acc: float | None = None
            cond_asr: float | None = None
            adv_mse_val: float | None = None
            l2_drift_val: float | None = None
            cos_val: float | None = None

            if is_classification and clean_correct_all is not None and ensemble_surviving is not None:
                adv_ok_all = torch.cat(adv_correct_chunks, dim=0)
                rob_acc = float(adv_ok_all.float().mean().item())
                fooled_clean = int((clean_correct_all & (~adv_ok_all)).sum().item())
                cond_asr = float(fooled_clean / clean_correct_count) if clean_correct_count > 0 else 0.0
                ci_lo, ci_hi = wilson_ci(int(adv_ok_all.sum().item()), total_n)
                last_extra["wilson_ci_95"] = [ci_lo, ci_hi]
                last_extra["misclassification_rate"] = 1.0 - rob_acc
                ensemble_surviving = ensemble_surviving & adv_ok_all
                if resolved_id in ("fgsm", "pgd"):
                    naive_wb_accs.append(rob_acc)
            else:
                l2_drift_val = float(l2_drift_sum / max(total_n, 1))
                cos_val = float(cos_sum / max(total_n, 1))
                if clean_mse_applicable:
                    adv_mse_val = float(adv_mse_sum / max(total_n, 1))

            records.append(
                AttackRunRecord(
                    attack_id=resolved_id,
                    display_name=card.get("name", resolved_id),
                    tier=card.get("tier", "custom"),
                    threat_model=card.get("threat_model", "custom"),
                    robust_acc=rob_acc,
                    attack_success_rate=cond_asr,
                    mean_perturbation=mean_norm,
                    max_perturbation=max_norm,
                    budget_respected=budget_ok,
                    steps_run=steps_max,
                    queries_used=queries_total,
                    elapsed_ms=float(elapsed_ms_sum),
                    hyperparameters=disclosed_hp,
                    citation=dict(card.get("citation", {})),
                    extra=last_extra,
                    adv_mse=adv_mse_val,
                    output_l2_drift=l2_drift_val,
                    cosine_similarity=cos_val,
                )
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                AttackRunRecord(
                    attack_id=resolved_id,
                    display_name=card.get("name", resolved_id),
                    tier=card.get("tier", "custom"),
                    threat_model=card.get("threat_model", "custom"),
                    robust_acc=None,
                    attack_success_rate=None,
                    mean_perturbation=0.0,
                    max_perturbation=0.0,
                    budget_respected=False,
                    steps_run=0,
                    queries_used=None,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    hyperparameters=disclosed_hp,
                    citation=dict(card.get("citation", {})),
                    extra={"escalation_reason": step_spec["reason"]},
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    # Compute worst-case ensemble & strongest escalated attack
    worst_case_robust_acc: float | None = None
    worst_case_ci_95: tuple[float, float] | None = None
    ensemble_cond_asr: float | None = None
    worst_case_adv_mse: float | None = None
    worst_case_output_l2_drift: float | None = None
    strongest_attack_id: str | None = None
    adaptive_bypass_gap = 0.0

    if is_classification and ensemble_surviving is not None and clean_correct_all is not None:
        surv_cnt = int(ensemble_surviving.sum().item())
        worst_case_robust_acc = float(surv_cnt / max(total_n, 1))
        worst_case_ci_95 = wilson_ci(surv_cnt, total_n)
        ens_fooled = int((clean_correct_all & (~ensemble_surviving)).sum().item())
        ensemble_cond_asr = float(ens_fooled / clean_correct_count) if clean_correct_count > 0 else 0.0
        valid_cls = [r for r in records if r.robust_acc is not None and r.error is None]
        if valid_cls:
            best_rec = min(valid_cls, key=lambda r: float(r.robust_acc if r.robust_acc is not None else 1.0))
            strongest_attack_id = best_rec.attack_id
        naive_ref = (
            min(naive_wb_accs)
            if naive_wb_accs
            else (
                float(audit_rep.metrics["pgd_acc"])
                if audit_rep is not None and "pgd_acc" in audit_rep.metrics
                else worst_case_robust_acc
            )
        )
        adaptive_bypass_gap = max(0.0, float(naive_ref) - worst_case_robust_acc)
    else:
        valid_mse = [r for r in records if r.adv_mse is not None and r.error is None]
        valid_drift = [r for r in records if r.output_l2_drift is not None and r.error is None]
        if valid_mse:
            top_mse = max(valid_mse, key=lambda r: float(r.adv_mse or 0.0))
            worst_case_adv_mse = top_mse.adv_mse
            strongest_attack_id = top_mse.attack_id
        if valid_drift:
            top_drift = max(valid_drift, key=lambda r: float(r.output_l2_drift or 0.0))
            worst_case_output_l2_drift = top_drift.output_l2_drift
            if strongest_attack_id is None:
                strongest_attack_id = top_drift.attack_id

    # 4. Robustness Decay Curve Sweep (AURC + eps_50) using strongest escalated attack
    curve_points: list[dict[str, Any]] = []
    curve_attack_id = strongest_attack_id
    if sweep_curve and curve_attack_id is not None:
        if curve_budgets is not None:
            budgets = [float(b) for b in curve_budgets]
        elif norm == "jsr_db":
            budgets = [eps - 12.0, eps - 6.0, eps - 3.0, eps, eps + 3.0, eps + 6.0]
        elif eps > 0.0:
            budgets = [0.0, 0.25 * eps, 0.5 * eps, 1.0 * eps, 1.5 * eps, 2.0 * eps]
        else:
            budgets = [0.0]

        k_curve = min(total_n, max(2, curve_samples))
        cx_list: list[torch.Tensor] = []
        cy_list: list[torch.Tensor] = []
        c_got = 0
        for xb_raw, yb_raw in raw_batches:
            take = min(xb_raw.shape[0], k_curve - c_got)
            if take <= 0:
                break
            cx_list.append(xb_raw[:take])
            cy_list.append(yb_raw[:take])
            c_got += take
        x_curve = torch.cat(cx_list, dim=0).to(eval_device)
        y_curve = torch.cat(cy_list, dim=0).to(eval_device)

        # Avoid nested full AutoAttack inside the 6-point curve; step down to its white-box driver if needed
        if curve_attack_id == "autoattack":
            curve_attack_id = "apgd_dlr" if "broken_or_obfuscated_gradients" not in detected_defenses else "square_attack"

        _, c_fn, _ = _resolve_attack_entry(curve_attack_id, by_id)
        c_sig = inspect.signature(c_fn)
        c_uses_bpda = defense_fn is not None and "defense_fn" in c_sig.parameters
        c_victim = model_or_fn if c_uses_bpda else deployed_victim

        c_overrides: dict[str, Any] = {}
        if c_uses_bpda:
            c_overrides["defense_fn"] = defense_fn
            if surrogate_fn is not None and "surrogate_fn" in c_sig.parameters:
                c_overrides["surrogate_fn"] = surrogate_fn
        if "steps" in c_sig.parameters:
            c_overrides["steps"] = 12 if curve_attack_id in ("hopskipjump", "fab") else 15
        if "max_queries" in c_sig.parameters:
            c_overrides["max_queries"] = 180
        if "restarts" in c_sig.parameters:
            c_overrides["restarts"] = 1
        if "eot_samples" in c_sig.parameters:
            c_overrides["eot_samples"] = 4 if "stochastic_randomization" in detected_defenses else 2
        if not is_classification and "loss" in c_sig.parameters and clean_mse_applicable:
            c_overrides["loss"] = "mse"

        with eval_mode(deployed_victim), torch.no_grad():
            c_clean_out = unwrap_model_outputs(deployed_victim(x_curve))
            if is_classification:
                c_clean_preds = predict_labels(c_clean_out)
                y_curve_cmp = y_curve.squeeze(1) if c_clean_preds.ndim == 1 and y_curve.ndim == 2 and y_curve.shape[1] == 1 else y_curve
                c_clean_ok = c_clean_preds == y_curve_cmp
                c_clean_cnt = int(c_clean_ok.sum().item())
            else:
                c_clean_ok = None
                c_clean_cnt = 0

        for b_idx, b_eps in enumerate(budgets):
            if b_eps == 0.0 and norm != "jsr_db":
                if is_classification and c_clean_ok is not None:
                    acc_b = float(c_clean_ok.float().mean().item())
                    lo_b, hi_b = wilson_ci(c_clean_cnt, x_curve.shape[0])
                    curve_points.append({"eps": 0.0, "robust_acc": acc_b, "cond_asr": 0.0, "ci_low": lo_b, "ci_high": hi_b})
                else:
                    mse0 = float(_per_sample_mse(c_clean_out, y_curve).mean().item()) if clean_mse_applicable else 0.0
                    curve_points.append({"eps": 0.0, "adv_mse": mse0, "output_l2_drift": 0.0})
                continue

            ck_b, _ = _build_call_kwargs(
                c_fn, eps=b_eps, norm=norm, clip_bounds=clip_bounds, seed=seed + 500 + b_idx, overrides=c_overrides
            )
            try:
                c_res = c_fn(c_victim, x_curve, y_curve, **ck_b) if "y" in c_sig.parameters else c_fn(c_victim, x_curve, **ck_b)
                with eval_mode(deployed_victim), torch.no_grad():
                    out_b = unwrap_model_outputs(deployed_victim(c_res.x_adv))
                    if is_classification and c_clean_ok is not None:
                        preds_b = predict_labels(out_b)
                        ok_b = preds_b == y_curve_cmp
                        ok_cnt = int(ok_b.sum().item())
                        acc_b = float(ok_cnt / x_curve.shape[0])
                        asr_b = float((c_clean_ok & (~ok_b)).sum().item() / max(c_clean_cnt, 1)) if c_clean_cnt > 0 else 0.0
                        lo_b, hi_b = wilson_ci(ok_cnt, x_curve.shape[0])
                        curve_points.append({"eps": float(b_eps), "robust_acc": acc_b, "cond_asr": asr_b, "ci_low": lo_b, "ci_high": hi_b})
                    else:
                        drift_b = float(safe_l2_norm((out_b - c_clean_out).reshape(x_curve.shape[0], -1), dim=1).mean().item())
                        mse_b = float(_per_sample_mse(out_b, y_curve).mean().item()) if clean_mse_applicable else 0.0
                        curve_points.append({"eps": float(b_eps), "adv_mse": mse_b, "output_l2_drift": drift_b})
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[wannado-adv red-team] Curve budget eps={b_eps} skipped: {exc}", flush=True)

    aurc, eps_50 = compute_aurc_and_eps50(curve_points, clean_acc=clean_acc)

    return RedTeamReport(
        model_name=model_name,
        dataset_name=dataset_name,
        domain=resolved_domain,
        task_type="classification" if is_classification else "continuous",
        norm=norm,
        eps=float(eps),
        clip_bounds=clip_bounds,
        num_samples=int(total_n),
        seed=seed,
        device=str(eval_device),
        dtype=str(first_x.dtype),
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        clean_acc=clean_acc,
        clean_ci_95=clean_ci_95,
        clean_mse=clean_mse,
        worst_case_robust_acc=worst_case_robust_acc,
        worst_case_ci_95=worst_case_ci_95,
        ensemble_cond_asr=ensemble_cond_asr,
        worst_case_adv_mse=worst_case_adv_mse,
        worst_case_output_l2_drift=worst_case_output_l2_drift,
        strongest_attack_id=strongest_attack_id,
        adaptive_bypass_gap=adaptive_bypass_gap,
        detected_defenses=detected_defenses,
        escalation_plan=plan,
        aurc=aurc,
        eps_50=eps_50,
        curve_attack_id=curve_attack_id,
        curve_points=curve_points,
        preflight=pf_rep,
        audit=audit_rep,
        records=records,
        elapsed_seconds=time.perf_counter() - t_start,
    )


__all__ = [
    "RedTeamReport",
    "auto_red_team",
    "compute_aurc_and_eps50",
    "wilson_ci",
]
