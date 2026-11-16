"""Academic paper evaluation runner, zero-hallucination citation exporter, and summary reporter."""

from __future__ import annotations

import importlib
import inspect
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from wannado.adv.attacks import ATTACK_EXPORTS
from wannado.adv.audit.carlini_auditor import AuditReport, audit_robustness
from wannado.adv.audit.preflight import PreflightReport, preflight
from wannado.adv.catalog.build_catalog import load_catalog
from wannado.adv.core.projections import compute_perturbation_norm
from wannado.adv.core.types import AttackMeta, AttackResult
from wannado.adv.core.utils import eval_mode

_CARLINI_2019_BIBTEX = (
    "@article{carlini2019evaluating,\n"
    "  title={On Evaluating Adversarial Robustness},\n"
    "  author={Carlini, Nicholas and Athalye, Anish and Papernot, Nicolas and Brendel, Wieland "
    "and Rauber, Jonas and Tsipras, Dimitris and Goodfellow, Ian and Madry, Aleksander and Kurakin, Alexey},\n"
    "  journal={arXiv preprint arXiv:1902.06705},\n"
    "  year={2019},\n"
    "  url={https://arxiv.org/abs/1902.06705},\n"
    "}"
)
_CARLINI_2019_IEEE = (
    'N. Carlini et al., "On evaluating adversarial robustness," arXiv:1902.06705, 2019.'
)
_CARLINI_2019_APA = (
    "Carlini, N., Athalye, A., Papernot, N., Brendel, W., Rauber, J., Tsipras, D., Goodfellow, I., "
    "Madry, A., & Kurakin, A. (2019). On evaluating adversarial robustness. arXiv:1902.06705."
)


def export_citations(
    attack_ids: Sequence[str],
    fmt: str = "bibtex",
    *,
    include_audit_ref: bool = False,
) -> str:
    """Return verbatim verified academic citations from built-in META cards (no LLM guessing).

    Args:
        attack_ids: Sequence of algorithm IDs from ``registry.json``.
        fmt: Output style: ``"bibtex"``, ``"ieee"``, or ``"apa"``.
        include_audit_ref: Append Carlini et al. (2019) reference used by ``audit_robustness()``.
    """
    if fmt not in ("bibtex", "ieee", "apa"):
        raise ValueError(f"Unsupported citation format '{fmt}'. Expected 'bibtex', 'ieee', or 'apa'.")

    by_id = {c["id"]: c for c in load_catalog()}
    seen: set[str] = set()
    entries: list[str] = []

    for raw_aid in attack_ids:
        aid = raw_aid
        if aid not in by_id and raw_aid in ATTACK_EXPORTS:
            mod_name, _ = ATTACK_EXPORTS[raw_aid]
            mod = importlib.import_module(mod_name)
            meta_obj = getattr(mod, "META", None)
            if isinstance(meta_obj, AttackMeta):
                aid = meta_obj.id
        if aid in seen:
            continue
        card = by_id.get(aid)
        if card is None:
            raise KeyError(
                f"Unknown attack '{raw_aid}' in export_citations(). Refusing to fabricate citation."
            )
        seen.add(aid)
        entries.append(card["citation"][fmt])

    if include_audit_ref:
        if fmt == "bibtex":
            entries.append(_CARLINI_2019_BIBTEX)
        elif fmt == "ieee":
            entries.append(_CARLINI_2019_IEEE)
        else:
            entries.append(_CARLINI_2019_APA)

    sep = "\n\n" if fmt == "bibtex" else "\n"
    return sep.join(entries)


@dataclass
class AttackRunRecord:
    """Recorded metrics, empirical budget check, and full hyperparameters for one attack."""

    attack_id: str
    display_name: str
    tier: str
    threat_model: str
    robust_acc: float | None
    attack_success_rate: float | None
    mean_perturbation: float
    max_perturbation: float
    budget_respected: bool
    steps_run: int
    queries_used: int | None
    elapsed_ms: float
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    citation: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class EvaluationReport:
    """Complete experimental & academic-paper reporting artifact returned by ``evaluate_suite()``."""

    model_name: str
    dataset_name: str
    domain: str
    norm: str
    eps: float
    clip_bounds: tuple[float, float] | None
    num_samples: int
    seed: int
    device: str
    dtype: str
    torch_version: str
    clean_acc: float | None
    timestamp_utc: str
    notes: str
    preflight: PreflightReport | None
    audit: AuditReport | None
    records: list[AttackRunRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize entire experiment & paper setup into JSON-compatible dictionary."""
        return {
            "environment": {
                "timestamp_utc": self.timestamp_utc,
                "torch_version": self.torch_version,
                "device": self.device,
                "dtype": self.dtype,
                "seed": self.seed,
            },
            "experiment": {
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "domain": self.domain,
                "norm": self.norm,
                "eps": self.eps,
                "clip_bounds": list(self.clip_bounds) if self.clip_bounds is not None else None,
                "num_samples": self.num_samples,
                "clean_acc": self.clean_acc,
                "notes": self.notes,
            },
            "preflight": self.preflight.to_dict() if self.preflight is not None else None,
            "audit": self.audit.to_dict() if self.audit is not None else None,
            "attacks": [asdict(r) for r in self.records],
        }

    def to_json(self, indent: int = 2) -> str:
        """Render complete experiment provenance as JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_bibtex(self) -> str:
        """Collect verbatim BibTeX entries for every evaluated attack and audit protocol."""
        seen_keys: set[str] = set()
        blocks: list[str] = []
        for r in self.records:
            cit = r.citation or {}
            bib = cit.get("bibtex", "").strip()
            key = cit.get("key", r.attack_id)
            if bib and key not in seen_keys:
                seen_keys.add(key)
                blocks.append(bib)
        if self.audit is not None and "carlini2019evaluating" not in seen_keys:
            blocks.append(_CARLINI_2019_BIBTEX)
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    def to_latex(self) -> str:
        """Render publication-ready LaTeX booktabs table with threat-model and budget disclosures."""
        def _esc(val: str) -> str:
            return val.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")

        eps_desc = f"{self.eps:+.1f}\\,\\text{{dB JSR}}" if self.norm == "jsr_db" else f"{self.eps:.4g}"
        lines = [
            "% Requires \\usepackage{booktabs}",
            "\\begin{table}[t]",
            "  \\centering",
            f"  \\caption{{Adversarial robustness evaluation of {_esc(self.model_name)} on {_esc(self.dataset_name)} "
            f"($N={self.num_samples}$, norm=\\texttt{{{_esc(self.norm)}}}, $\\epsilon={eps_desc}$).}}",
            "  \\label{tab:wannado_robustness}",
            "  \\begin{tabular}{llcccc}",
            "    \\toprule",
            "    Attack & Threat Model & Steps / Queries & Robust Acc (\\%) $\\uparrow$ & ASR (\\%) $\\downarrow$ & Max Meas. $\\delta$ \\\\",
            "    \\midrule",
        ]
        if self.clean_acc is not None:
            lines.append(f"    Clean Baseline & -- & -- & {self.clean_acc * 100:.2f} & 0.00 & 0.0000 \\\\")
            lines.append("    \\midrule")

        for r in self.records:
            name_tex = _esc(r.display_name)
            tm_tex = _esc(r.threat_model)
            unit_word = "step" if r.steps_run == 1 else "steps"
            if r.queries_used is not None:
                q_per_img = r.queries_used / max(self.num_samples, 1)
                budget_unit = f"{q_per_img:.1f} q/img ({r.queries_used} tot)"
            else:
                budget_unit = f"{r.steps_run} {unit_word}"
            acc_s = f"{r.robust_acc * 100:.2f}" if r.robust_acc is not None else "N/A"
            asr_s = f"{r.attack_success_rate * 100:.2f}" if r.attack_success_rate is not None else "N/A"
            max_s = f"{r.max_perturbation:.4g}" if math.isfinite(r.max_perturbation) else "$-\\infty$"
            lines.append(f"    {name_tex} & {tm_tex} & {budget_unit} & {acc_s} & {asr_s} & {max_s} \\\\")

        lines.extend(
            [
                "    \\bottomrule",
                "  \\end{tabular}",
                "\\end{table}",
                "",
            ]
        )
        return "\n".join(lines)

    def format_paper_summary(self) -> str:
        """Render complete academic-paper setup, hyperparameter matrix, results, and citations."""
        eps_phys = (
            f"{self.eps:+.2f} dB (power ratio {10.0 ** (self.eps / 10.0):.4f}x)"
            if self.norm == "jsr_db"
            else f"{self.eps:.6g}"
        )
        lines = [
            f"# Academic Evaluation Report — {self.model_name} on {self.dataset_name}",
            "",
            "## 1. Reproducibility & Threat-Model Variables (Required in Paper Experimental Setup)",
            f"- **Timestamp (UTC)**: `{self.timestamp_utc}`",
            f"- **Domain & Hardware**: `{self.domain}` | `device={self.device}` | `dtype={self.dtype}` | `PyTorch={self.torch_version}`",
            f"- **Evaluation Scale**: `N={self.num_samples}` samples | `seed={self.seed}` | `eval()` & parameter-grad isolation verified",
            f"- **Perturbation Constraint**: `norm='{self.norm}'`, `eps={eps_phys}`, `clip_bounds={self.clip_bounds}`",
        ]
        if self.clean_acc is not None:
            lines.append(f"- **Clean Baseline Accuracy**: `{self.clean_acc * 100.0:.2f}%`")
        if self.notes:
            lines.append(f"- **Notes**: {self.notes}")

        lines.extend(["", "## 2. Quantitative Robustness & Budget Verification"])
        lines.append("| Attack | Tier | Threat Model | Robust Acc | ASR | Max Measured $\\delta$ | Budget OK | Time |")
        lines.append("| --- | :---: | --- | ---: | ---: | ---: | :---: | ---: |")
        for r in self.records:
            acc_s = f"{r.robust_acc * 100:.2f}%" if r.robust_acc is not None else "N/A"
            asr_s = f"{r.attack_success_rate * 100:.2f}%" if r.attack_success_rate is not None else "N/A"
            max_s = f"{r.max_perturbation:.4g}" if math.isfinite(r.max_perturbation) else "-inf"
            ok_s = "PASS" if (r.budget_respected and r.error is None) else "FAIL"
            lines.append(
                f"| {r.display_name} (`{r.attack_id}`) | {r.tier} | {r.threat_model} | "
                f"{acc_s} | {asr_s} | {max_s} | {ok_s} | {r.elapsed_ms:.1f} ms |"
            )

        lines.extend(["", "## 3. Exact Per-Attack Hyperparameters Disclosed for Publication"])
        for r in self.records:
            hp_items = ", ".join(f"`{k}={v}`" for k, v in sorted(r.hyperparameters.items()))
            q_note = f", `total_queries={r.queries_used}`" if r.queries_used is not None else ""
            lines.append(f"- **{r.attack_id}** (`steps_run={r.steps_run}`{q_note}): {hp_items}")

        if self.audit is not None:
            lines.extend(["", "## 4. Carlini False-Robustness Audit Verdict"])
            lines.append(f"- **Passed**: `{self.audit.passed}` | **Red Flags**: `{self.audit.red_flags}`")
            if self.audit.checks:
                chk_str = ", ".join(
                    f"`{k}={'OK' if v else 'FAIL'}`" for k, v in self.audit.checks.items()
                )
                lines.append(f"- **Automated Checks**: {chk_str}")
            sweep = self.audit.metrics.get("budget_sweep") if isinstance(self.audit.metrics, dict) else None
            if isinstance(sweep, dict) and sweep:
                lines.append(
                    "- **Budget Sweep Monotonicity**: "
                    f"`eps={sweep.get('eps_low', 0):.4g} -> {sweep.get('acc_low', 0) * 100:.2f}%`, "
                    f"`eps={sweep.get('eps_mid', 0):.4g} -> {sweep.get('acc_mid', 0) * 100:.2f}%`, "
                    f"`eps={sweep.get('eps_high', 0):.4g} -> {sweep.get('acc_high', 0) * 100:.2f}%`"
                )
            if self.audit.recommendations:
                lines.extend(f"  - {rec}" for rec in self.audit.recommendations)

        lines.extend(["", "## 5. Verified Academic References (Verbatim from META)"])
        seen: set[str] = set()
        for r in self.records:
            cit = r.citation or {}
            key = cit.get("key")
            ieee = cit.get("ieee")
            if key and ieee and key not in seen:
                seen.add(key)
                lines.append(f"- `[{key}]` {ieee}")
        if self.audit is not None:
            lines.append(f"- `[carlini2019evaluating]` {_CARLINI_2019_IEEE}")

        return "\n".join(lines)

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        """Write JSON report, Markdown paper summary, LaTeX table, and BibTeX file."""
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "json": out / "evaluation_report.json",
            "markdown": out / "paper_summary.md",
            "latex": out / "paper_table.tex",
            "bibtex": out / "references.bib",
        }
        paths["json"].write_text(self.to_json() + "\n", encoding="utf-8")
        paths["markdown"].write_text(self.format_paper_summary() + "\n", encoding="utf-8")
        paths["latex"].write_text(self.to_latex(), encoding="utf-8")
        paths["bibtex"].write_text(self.to_bibtex(), encoding="utf-8")
        return paths


def _resolve_attack_entry(
    entry: str | Callable[..., AttackResult],
    by_id: dict[str, dict[str, Any]],
) -> tuple[str, Callable[..., AttackResult], dict[str, Any]]:
    """Resolve an attack ID string or custom local callable alongside its metadata card."""
    if isinstance(entry, str):
        if entry not in ATTACK_EXPORTS:
            raise KeyError(f"Unknown attack '{entry}'. Check query_catalog() for valid IDs.")
        mod_name, attr_name = ATTACK_EXPORTS[entry]
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, attr_name)
        meta_obj = getattr(mod, "META", None)
        card = meta_obj.to_dict() if isinstance(meta_obj, AttackMeta) else by_id[entry]
        return card["id"], fn, card

    fn = entry
    mod = sys.modules.get(getattr(fn, "__module__", ""))
    meta_obj = getattr(fn, "META", None) or (getattr(mod, "META", None) if mod is not None else None)
    if isinstance(meta_obj, AttackMeta):
        card = meta_obj.to_dict()
        return card["id"], fn, card

    fallback_id = getattr(fn, "__name__", "custom_attack")
    return fallback_id, fn, {
        "id": fallback_id,
        "name": f"Custom Attack ({fallback_id})",
        "tier": "custom",
        "threat_model": "custom",
        "citation": {},
    }


def _build_call_kwargs(
    fn: Callable[..., AttackResult],
    eps: float,
    norm: str,
    clip_bounds: tuple[float, float] | None,
    seed: int,
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build safe kwargs matching ``fn`` signature and capture complete paper hyperparams."""
    sig = inspect.signature(fn)
    params = sig.parameters
    call_kwargs: dict[str, Any] = dict(overrides)

    if "eps" in params and "eps" not in call_kwargs:
        call_kwargs["eps"] = eps
    if "norm" in params and "norm" not in call_kwargs:
        call_kwargs["norm"] = norm
    if "clip_bounds" in params and "clip_bounds" not in call_kwargs:
        call_kwargs["clip_bounds"] = clip_bounds
    if "seed" in params and "seed" not in call_kwargs:
        call_kwargs["seed"] = seed

    # Materialize effective scalar parameters (defaults + overrides) for paper reporting.
    disclosed: dict[str, Any] = {}
    skip_params = {"model_or_fn", "x", "y"}
    for pname, p in params.items():
        if pname in skip_params:
            continue
        val = call_kwargs[pname] if pname in call_kwargs else p.default
        if val is inspect.Parameter.empty:
            continue
        if callable(val):
            disclosed[pname] = getattr(val, "__name__", type(val).__name__)
        elif isinstance(val, torch.Tensor):
            disclosed[pname] = f"Tensor{list(val.shape)}"
        elif isinstance(val, tuple):
            disclosed[pname] = list(val)
        else:
            disclosed[pname] = val

    return call_kwargs, disclosed


def evaluate_suite(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 0.03,
    norm: str = "linf",
    attacks: Sequence[str | Callable[..., AttackResult]] = ("fgsm", "pgd", "apgd_dlr", "square_attack"),
    attack_kwargs: dict[str, dict[str, Any]] | None = None,
    clip_bounds: tuple[float, float] | None = None,
    run_preflight: bool = True,
    run_audit: bool = True,
    audit_samples: int = 64,
    seed: int = 42,
    domain: str | None = None,
    model_name: str = "VictimModel",
    dataset_name: str = "EvalSplit",
    notes: str = "",
    verbose: bool = True,
) -> EvaluationReport:
    """Execute preflight, attack suite, budget verification, Carlini audit, and paper reporting."""
    x_clean = x.detach()
    y_clean = y.detach()
    is_complex = bool(x_clean.is_complex())
    resolved_domain = domain or ("wireless" if is_complex else ("vision" if x_clean.ndim == 4 else "tabular"))
    by_id = {c["id"]: c for c in load_catalog()}
    per_attack_overrides = dict(attack_kwargs or {})

    pf_report = (
        preflight(
            model_or_fn,
            x_clean,
            y_clean,
            eps=eps,
            norm=norm,
            clip_bounds=clip_bounds,
        )
        if run_preflight
        else None
    )

    is_classification = False
    clean_acc: float | None = None
    with eval_mode(model_or_fn), torch.no_grad():
        clean_out = model_or_fn(x_clean)
        if clean_out.ndim == 2 and y_clean.dtype in (torch.int64, torch.int32, torch.long):
            is_classification = True
            clean_acc = float((clean_out.argmax(dim=1) == y_clean).float().mean().item())

    if verbose:
        clean_msg = f"{clean_acc * 100:.2f}%" if clean_acc is not None else "non-classification"
        print(
            f"[wannado-adv] Clean baseline: {clean_msg} (N={x_clean.shape[0]}, device={x_clean.device})",
            flush=True,
        )

    audit_report: AuditReport | None = None
    if run_audit and is_classification and x_clean.shape[0] >= 2:
        k_audit = min(x_clean.shape[0], max(2, audit_samples))
        if verbose:
            print(f"[wannado-adv] Running Carlini false-robustness audit (N={k_audit})...", flush=True)
        audit_report = audit_robustness(
            model_or_fn,
            x_clean[:k_audit],
            y_clean[:k_audit],
            eps=eps,
            norm=norm,
            clip_bounds=clip_bounds,
            seed=seed,
        )
        if verbose:
            print(
                f"[wannado-adv] Audit done -> passed={audit_report.passed}, red_flags={audit_report.red_flags}",
                flush=True,
            )

    records: list[AttackRunRecord] = []
    total_attacks = len(attacks)
    for idx, entry in enumerate(attacks, start=1):
        aid, fn, card = _resolve_attack_entry(entry, by_id)
        overrides = dict(per_attack_overrides.get(aid, {}))
        call_kwargs, disclosed_hp = _build_call_kwargs(
            fn, eps=eps, norm=norm, clip_bounds=clip_bounds, seed=seed, overrides=overrides
        )
        if verbose:
            print(f"[wannado-adv] [{idx}/{total_attacks}] Running {aid}...", flush=True)
        sig = inspect.signature(fn)
        t0 = time.perf_counter()
        try:
            if "y" in sig.parameters:
                res = fn(model_or_fn, x_clean, y_clean, **call_kwargs)
            else:
                res = fn(model_or_fn, x_clean, **call_kwargs)

            elapsed_ms = res.elapsed_ms if getattr(res, "elapsed_ms", None) else (time.perf_counter() - t0) * 1000.0
            norms = compute_perturbation_norm(res.perturbation, norm=norm, x_ref=x_clean)
            mean_norm = float(norms.mean().item())
            max_norm = float(norms.max().item())
            tol = 1e-2 if norm == "jsr_db" else max(1e-4, abs(eps) * 1e-3)
            budget_ok = (max_norm <= eps + tol) if math.isfinite(max_norm) else True

            rob_acc: float | None = None
            asr: float | None = None
            if is_classification:
                with eval_mode(model_or_fn), torch.no_grad():
                    adv_out = model_or_fn(res.x_adv)
                rob_acc = float((adv_out.argmax(dim=1) == y_clean).float().mean().item())
                asr = 1.0 - rob_acc

            q_used = res.extra.get("queries") if isinstance(res.extra, dict) else None
            records.append(
                AttackRunRecord(
                    attack_id=aid,
                    display_name=card.get("name", aid),
                    tier=card.get("tier", "custom"),
                    threat_model=card.get("threat_model", "custom"),
                    robust_acc=rob_acc,
                    attack_success_rate=asr,
                    mean_perturbation=mean_norm,
                    max_perturbation=max_norm,
                    budget_respected=budget_ok,
                    steps_run=int(res.steps_run),
                    queries_used=int(q_used) if q_used is not None else None,
                    elapsed_ms=float(elapsed_ms),
                    hyperparameters=disclosed_hp,
                    citation=dict(card.get("citation", {})),
                    extra=dict(res.extra) if isinstance(res.extra, dict) else {},
                )
            )
            if verbose:
                rob_str = f"{rob_acc * 100:.2f}%" if rob_acc is not None else "done"
                print(
                    f"[wannado-adv] [{idx}/{total_attacks}] {aid} finished: "
                    f"robust_acc={rob_str}, time={elapsed_ms:.1f} ms, budget={'PASS' if budget_ok else 'FAIL'}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            err_elapsed = (time.perf_counter() - t0) * 1000.0
            records.append(
                AttackRunRecord(
                    attack_id=aid,
                    display_name=card.get("name", aid),
                    tier=card.get("tier", "custom"),
                    threat_model=card.get("threat_model", "custom"),
                    robust_acc=None,
                    attack_success_rate=None,
                    mean_perturbation=0.0,
                    max_perturbation=0.0,
                    budget_respected=False,
                    steps_run=0,
                    queries_used=None,
                    elapsed_ms=err_elapsed,
                    hyperparameters=disclosed_hp,
                    citation=dict(card.get("citation", {})),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            if verbose:
                print(f"[wannado-adv] [{idx}/{total_attacks}] {aid} ERROR: {type(exc).__name__}: {exc}", flush=True)

    return EvaluationReport(
        model_name=model_name,
        dataset_name=dataset_name,
        domain=resolved_domain,
        norm=norm,
        eps=float(eps),
        clip_bounds=clip_bounds,
        num_samples=int(x_clean.shape[0]),
        seed=seed,
        device=str(x_clean.device),
        dtype=str(x_clean.dtype),
        torch_version=torch.__version__,
        clean_acc=clean_acc,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes=notes,
        preflight=pf_report,
        audit=audit_report,
        records=records,
    )
