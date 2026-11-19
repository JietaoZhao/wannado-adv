"""Academic paper evaluation runner, zero-hallucination citation exporter, and summary reporter."""

from __future__ import annotations

import importlib
import inspect
import json
import math
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from wannado.adv.attacks import ATTACK_EXPORTS
from wannado.adv.audit.carlini_auditor import AuditReport, audit_robustness
from wannado.adv.audit.preflight import PreflightReport, preflight
from wannado.adv.catalog.build_catalog import load_catalog
from wannado.adv.core.losses import predict_labels, unwrap_model_outputs
from wannado.adv.core.projections import compute_perturbation_norm, safe_l2_norm
from wannado.adv.core.types import AttackMeta, AttackResult
from wannado.adv.core.utils import DefendedModelWrapper, eval_mode, iter_eval_batches


def _serialize_bounds(bounds: Any) -> Any:
    if bounds is None:
        return None
    out: list[Any] = []
    for val in bounds:
        if isinstance(val, torch.Tensor):
            out.append(val.detach().cpu().tolist())
        elif hasattr(val, "tolist"):
            out.append(val.tolist())
        else:
            out.append(float(val))
    return out


def _format_bounds_brief(bounds: Any) -> str:
    if bounds is None:
        return "None"
    ser = _serialize_bounds(bounds)
    if isinstance(ser, list) and len(ser) == 2 and all(isinstance(v, (int, float)) for v in ser):
        return f"({ser[0]}, {ser[1]})"
    return "per-feature bounds"

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
    from wannado.adv.defenses import DEFENSE_EXPORTS, DEFENSE_METAS

    for d_id, d_meta in DEFENSE_METAS.items():
        if d_id not in by_id:
            by_id[d_id] = d_meta.to_dict()

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
        elif aid not in by_id and raw_aid in DEFENSE_EXPORTS:
            mod_name, _ = DEFENSE_EXPORTS[raw_aid]
            mod = importlib.import_module(mod_name)
            meta_obj = getattr(mod, "META", None)
            if meta_obj is not None and hasattr(meta_obj, "id"):
                aid = str(meta_obj.id)
        if aid in seen:
            continue
        card = by_id.get(aid)
        if card is None:
            raise KeyError(
                f"Unknown attack/defense '{raw_aid}' in export_citations(). Refusing to fabricate citation."
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
    adv_mse: float | None = None
    output_l2_drift: float | None = None
    cosine_similarity: float | None = None


@dataclass
class EvaluationReport:
    """Complete experimental & academic-paper reporting artifact returned by ``evaluate_suite()``."""

    model_name: str
    dataset_name: str
    domain: str
    norm: str
    eps: float
    clip_bounds: tuple[Any, Any] | None
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
    clean_mse: float | None = None

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
                "clip_bounds": _serialize_bounds(self.clip_bounds),
                "num_samples": self.num_samples,
                "clean_acc": self.clean_acc,
                "clean_mse": self.clean_mse,
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
            return (
                val.replace("&", "\\&")
                .replace("%", "\\%")
                .replace("$", "\\$")
                .replace("#", "\\#")
                .replace("_", "\\_")
                .replace("^", "\\^{}")
                .replace("~", "\\~{}")
            )

        eps_desc = f"{self.eps:+.1f}\\,\\text{{dB JSR}}" if self.norm == "jsr_db" else f"{self.eps:.4g}"
        is_cls_report = self.clean_acc is not None
        header_cols = (
            "    Attack & Threat Model & Steps / Queries & Robust Acc (\\%) $\\uparrow$ & Cond. ASR (\\%) $\\downarrow$ & Max Meas. $\\delta$ \\\\"
            if is_cls_report
            else "    Attack & Threat Model & Steps / Queries & Adv MSE $\\downarrow$ & Out L2 Drift $\\downarrow$ & Max Meas. $\\delta$ \\\\"
        )
        lines = [
            "% Requires \\usepackage{booktabs}",
            "\\begin{table}[t]",
            "  \\centering",
            f"  \\caption{{Adversarial robustness evaluation of {_esc(self.model_name)} on {_esc(self.dataset_name)} ($N={self.num_samples}$, norm=\\texttt{{{_esc(self.norm)}}}, $\\epsilon={eps_desc}$).}}",
            "  \\label{tab:wannado_robustness}",
            "  \\begin{tabular}{llcccc}",
            "    \\toprule",
            header_cols,
            "    \\midrule",
        ]
        if is_cls_report and self.clean_acc is not None:
            lines.append(f"    Clean Baseline & -- & -- & {self.clean_acc * 100:.2f} & 0.00 & 0.0000 \\\\")
            lines.append("    \\midrule")
        elif self.clean_mse is not None:
            lines.append(f"    Clean Baseline & -- & -- & {self.clean_mse:.4g} & 0.0000 & 0.0000 \\\\")
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
            max_s = f"{r.max_perturbation:.4g}" if math.isfinite(r.max_perturbation) else "$-\\infty$"
            if is_cls_report:
                c1 = f"{r.robust_acc * 100:.2f}" if r.robust_acc is not None else "N/A"
                c2 = f"{r.attack_success_rate * 100:.2f}" if r.attack_success_rate is not None else "N/A"
            else:
                c1 = f"{r.adv_mse:.4g}" if r.adv_mse is not None else "N/A"
                c2 = f"{r.output_l2_drift:.4g}" if r.output_l2_drift is not None else "N/A"
            lines.append(f"    {name_tex} & {tm_tex} & {budget_unit} & {c1} & {c2} & {max_s} \\\\")

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
        is_cls_report = self.clean_acc is not None
        lines = [
            f"# Academic Evaluation Report — {self.model_name} on {self.dataset_name}",
            "",
            "## 1. Reproducibility & Threat-Model Variables (Required in Paper Experimental Setup)",
            f"- **Timestamp (UTC)**: `{self.timestamp_utc}`",
            f"- **Domain & Hardware**: `{self.domain}` | `device={self.device}` | `dtype={self.dtype}` | `PyTorch={self.torch_version}`",
            f"- **Evaluation Scale**: `N={self.num_samples}` samples | `seed={self.seed}` | `eval()` & parameter-grad isolation verified",
            f"- **Perturbation Constraint**: `norm='{self.norm}'`, `eps={eps_phys}`, `clip_bounds={_format_bounds_brief(self.clip_bounds)}`",
        ]
        if self.clean_acc is not None:
            lines.append(f"- **Clean Baseline Accuracy**: `{self.clean_acc * 100.0:.2f}%`")
        elif self.clean_mse is not None:
            lines.append(f"- **Clean Baseline MSE**: `{self.clean_mse:.6g}`")
        if self.notes:
            lines.append(f"- **Notes**: {self.notes}")

        lines.extend(["", "## 2. Quantitative Robustness & Budget Verification"])
        if is_cls_report:
            lines.append("| Attack | Tier | Threat Model | Robust Acc | Cond. ASR | Max Measured $\\delta$ | Budget OK | Time |")
            lines.append("| --- | :---: | --- | ---: | ---: | ---: | :---: | ---: |")
        else:
            lines.append("| Attack | Tier | Threat Model | Adv MSE | Output L2 Drift | Cosine Sim | Max Measured $\\delta$ | Budget OK | Time |")
            lines.append("| --- | :---: | --- | ---: | ---: | ---: | ---: | :---: | ---: |")
        for r in self.records:
            max_s = f"{r.max_perturbation:.4g}" if math.isfinite(r.max_perturbation) else "-inf"
            ok_s = "PASS" if (r.budget_respected and r.error is None) else "FAIL"
            if is_cls_report:
                acc_s = f"{r.robust_acc * 100:.2f}%" if r.robust_acc is not None else "N/A"
                asr_s = f"{r.attack_success_rate * 100:.2f}%" if r.attack_success_rate is not None else "N/A"
                lines.append(
                    f"| {r.display_name} (`{r.attack_id}`) | {r.tier} | {r.threat_model} | "
                    f"{acc_s} | {asr_s} | {max_s} | {ok_s} | {r.elapsed_ms:.1f} ms |"
                )
            else:
                mse_s = f"{r.adv_mse:.4g}" if r.adv_mse is not None else "N/A"
                drift_s = f"{r.output_l2_drift:.4g}" if r.output_l2_drift is not None else "N/A"
                cos_s = f"{r.cosine_similarity:.4f}" if r.cosine_similarity is not None else "N/A"
                lines.append(
                    f"| {r.display_name} (`{r.attack_id}`) | {r.tier} | {r.threat_model} | "
                    f"{mse_s} | {drift_s} | {cos_s} | {max_s} | {ok_s} | {r.elapsed_ms:.1f} ms |"
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
    fn_mod = sys.modules.get(getattr(fn, "__module__", ""))
    meta_obj = getattr(fn, "META", None) or (getattr(fn_mod, "META", None) if fn_mod is not None else None)
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
    clip_bounds: tuple[Any, Any] | None,
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
        val = call_kwargs.get(pname, p.default)
        if val is inspect.Parameter.empty:
            continue
        if callable(val):
            disclosed[pname] = getattr(val, "__name__", type(val).__name__)
        elif isinstance(val, torch.Tensor):
            disclosed[pname] = f"Tensor{list(val.shape)}"
        elif isinstance(val, tuple):
            disclosed[pname] = _serialize_bounds(val) if len(val) == 2 and any(isinstance(v, torch.Tensor) or hasattr(v, "tolist") for v in val) else list(val)
        else:
            disclosed[pname] = val

    return call_kwargs, disclosed


_DATASET_LEVEL_ATTACKS = frozenset({"uap", "adv_patch", "gap", "advgan"})


def _infer_target_device(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    first_x: torch.Tensor,
    from_loader: bool,
) -> torch.device:
    if not from_loader:
        return first_x.device
    if isinstance(model_or_fn, nn.Module):
        for p in model_or_fn.parameters():
            return p.device
    return first_x.device


def _per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).reshape(pred.shape[0], -1)
    if diff.is_complex():
        return (diff.real.pow(2) + diff.imag.pow(2)).mean(dim=1)
    return diff.pow(2).mean(dim=1)


def _per_sample_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    af = a.reshape(a.shape[0], -1)
    bf = b.reshape(b.shape[0], -1)
    if af.is_complex() or bf.is_complex():
        af = torch.cat([af.real, af.imag], dim=1)
        bf = torch.cat([bf.real, bf.imag], dim=1)
    return F.cosine_similarity(af.float(), bf.float(), dim=1, eps=1e-8)


def evaluate_suite(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    *,
    eps: float = 0.03,
    norm: str = "linf",
    attacks: Sequence[str | Callable[..., AttackResult]] = ("fgsm", "pgd", "apgd_dlr", "square_attack"),
    attack_kwargs: dict[str, dict[str, Any]] | None = None,
    clip_bounds: tuple[Any, Any] | None = None,
    batch_size: int | None = None,
    max_samples: int | None = None,
    defense_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    surrogate_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
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
    """Execute preflight, mini-batch/DataLoader attack suite, budget verification, Carlini audit, and paper reporting."""
    from_loader = not isinstance(x, torch.Tensor)
    raw_batches = list(iter_eval_batches(x, y, batch_size=batch_size, max_samples=max_samples))
    if not raw_batches:
        raise ValueError("Evaluation input yielded zero samples.")

    first_x, first_y = raw_batches[0]
    eval_device = _infer_target_device(model_or_fn, first_x, from_loader=from_loader)
    is_complex = bool(first_x.is_complex())
    resolved_domain = domain or ("wireless" if is_complex else ("vision" if first_x.ndim == 4 else "tabular"))
    by_id = {c["id"]: c for c in load_catalog()}
    per_attack_overrides = dict(attack_kwargs or {})

    deployed_victim: Callable[[torch.Tensor], torch.Tensor] = (
        DefendedModelWrapper(model_or_fn, defense_fn) if defense_fn is not None else model_or_fn
    )

    total_n = sum(int(xb.shape[0]) for xb, _ in raw_batches)

    # 1. Clean baseline pass (chunked to keep peak memory O(batch_size))
    is_classification = False
    clean_correct_total = 0
    clean_preds_per_batch: list[torch.Tensor] = []
    clean_outs_per_batch: list[torch.Tensor] = []
    clean_mse_sum = 0.0
    clean_mse_applicable = True

    with eval_mode(deployed_victim), torch.no_grad():
        for b_idx, (xb_raw, yb_raw) in enumerate(raw_batches):
            xb = xb_raw.to(eval_device)
            yb = yb_raw.to(eval_device)
            clean_out = unwrap_model_outputs(deployed_victim(xb))
            if b_idx == 0:
                is_classification = bool(
                    clean_out.ndim >= 1
                    and yb.dtype
                    in (
                        torch.int64,
                        torch.int32,
                        torch.int16,
                        torch.int8,
                        torch.uint8,
                        torch.long,
                    )
                )
            if is_classification:
                c_preds = predict_labels(clean_out)
                y_cmp = (
                    yb.squeeze(1)
                    if c_preds.ndim == 1 and yb.ndim == 2 and yb.shape[1] == 1
                    else yb
                )
                clean_correct_total += int((c_preds == y_cmp).sum().item())
                clean_preds_per_batch.append(c_preds.detach().cpu())
            else:
                clean_outs_per_batch.append(clean_out.detach().cpu())
                if clean_out.shape == yb.shape:
                    clean_mse_sum += float(_per_sample_mse(clean_out, yb).sum().item())
                else:
                    clean_mse_applicable = False

    clean_acc: float | None = (clean_correct_total / max(total_n, 1)) if is_classification else None
    clean_mse: float | None = (
        (clean_mse_sum / max(total_n, 1)) if (not is_classification and clean_mse_applicable) else None
    )

    # 2. Preflight on first slice (continuous-safe loss selection + defense_fn aware)
    pf_loss = "ce" if is_classification else ("mse" if clean_mse_applicable else None)
    k_pf = min(first_x.shape[0], 16)
    pf_report = (
        preflight(
            model_or_fn,
            first_x[:k_pf].to(eval_device),
            first_y[:k_pf].to(eval_device),
            eps=eps,
            norm=norm,
            clip_bounds=clip_bounds,
            loss=pf_loss,
            defense_fn=defense_fn,
        )
        if run_preflight
        else None
    )

    if verbose:
        if clean_acc is not None:
            clean_msg = f"{clean_acc * 100:.2f}%"
        elif clean_mse is not None:
            clean_msg = f"MSE={clean_mse:.4g}"
        else:
            clean_msg = "non-classification"
        chunk_note = f", batch_size={batch_size}" if batch_size is not None else ""
        print(
            f"[wannado-adv] Clean baseline: {clean_msg} (N={total_n}{chunk_note}, device={eval_device})",
            flush=True,
        )

    # 3. Carlini 2.0 false-robustness audit on bounded prefix slice (classification AND continuous!)
    audit_report: AuditReport | None = None
    if run_audit and total_n >= 2:
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
        if verbose:
            print(f"[wannado-adv] Running Carlini 2.0 false-robustness audit (N={collected})...", flush=True)
        audit_report = audit_robustness(
            model_or_fn,
            x_audit,
            y_audit,
            eps=eps,
            norm=norm,
            clip_bounds=clip_bounds,
            defense_fn=defense_fn,
            surrogate_fn=surrogate_fn,
            seed=seed,
        )
        if verbose:
            print(
                f"[wannado-adv] Audit done -> passed={audit_report.passed}, red_flags={audit_report.red_flags}",
                flush=True,
            )

    # 4. Run each attack across chunks (or materialized dataset for universal/generative attacks)
    records: list[AttackRunRecord] = []
    total_attacks = len(attacks)
    tol = 1e-2 if norm == "jsr_db" else max(1e-4, abs(eps) * 1e-3)

    for idx, entry in enumerate(attacks, start=1):
        aid, fn, card = _resolve_attack_entry(entry, by_id)
        overrides = dict(per_attack_overrides.get(aid, {}))
        sig = inspect.signature(fn)
        uses_bpda_defense = defense_fn is not None and "defense_fn" in sig.parameters
        if uses_bpda_defense:
            overrides.setdefault("defense_fn", defense_fn)
            if surrogate_fn is not None and "surrogate_fn" in sig.parameters:
                overrides.setdefault("surrogate_fn", surrogate_fn)
        if not is_classification and "loss" in sig.parameters and "loss" not in overrides and clean_mse_applicable:
            overrides["loss"] = "mse"

        call_kwargs, disclosed_hp = _build_call_kwargs(
            fn, eps=eps, norm=norm, clip_bounds=clip_bounds, seed=seed, overrides=overrides
        )
        if batch_size is not None and "batch_size" in sig.parameters and "batch_size" not in call_kwargs:
            call_kwargs["batch_size"] = int(batch_size)

        attack_victim = model_or_fn if uses_bpda_defense else deployed_victim
        is_targeted = bool(call_kwargs.get("targeted", False))
        if verbose:
            print(f"[wannado-adv] [{idx}/{total_attacks}] Running {aid}...", flush=True)

        t0 = time.perf_counter()
        try:
            if aid in _DATASET_LEVEL_ATTACKS and len(raw_batches) > 1:
                exec_batches = [
                    (
                        torch.cat([xb for xb, _ in raw_batches], dim=0),
                        torch.cat([yb for _, yb in raw_batches], dim=0),
                    )
                ]
                clean_preds_exec = [torch.cat(clean_preds_per_batch, dim=0)] if is_classification else []
                clean_outs_exec = [torch.cat(clean_outs_per_batch, dim=0)] if not is_classification else []
            else:
                exec_batches = raw_batches
                clean_preds_exec = clean_preds_per_batch
                clean_outs_exec = clean_outs_per_batch

            norm_sum = 0.0
            max_norm = float("-inf")
            steps_max = 0
            queries_total: int | None = None
            elapsed_sum_ms = 0.0
            adv_correct_total = 0
            fooled_clean_correct_total = 0
            targeted_hits_total = 0
            adv_mse_sum = 0.0
            l2_drift_sum = 0.0
            cos_sum = 0.0
            last_extra: dict[str, Any] = {}

            for b_idx, (xb_raw, yb_raw) in enumerate(exec_batches):
                xb = xb_raw.to(eval_device)
                yb = yb_raw.to(eval_device)
                chunk_kwargs = dict(call_kwargs)
                if "seed" in chunk_kwargs and chunk_kwargs["seed"] is not None and b_idx > 0:
                    chunk_kwargs["seed"] = int(chunk_kwargs["seed"]) + b_idx * 1009

                t_chunk = time.perf_counter()
                if "y" in sig.parameters:
                    res = fn(attack_victim, xb, yb, **chunk_kwargs)
                else:
                    res = fn(attack_victim, xb, **chunk_kwargs)
                chunk_ms = (
                    float(res.elapsed_ms)
                    if getattr(res, "elapsed_ms", None)
                    else (time.perf_counter() - t_chunk) * 1000.0
                )
                elapsed_sum_ms += chunk_ms
                steps_max = max(steps_max, int(res.steps_run))

                norms = compute_perturbation_norm(res.perturbation, norm=norm, x_ref=xb)
                norm_sum += float(norms.sum().item())
                chunk_max = float(norms.max().item())
                max_norm = max(max_norm, chunk_max)

                if isinstance(res.extra, dict):
                    q_val = res.extra.get("queries")
                    if q_val is not None:
                        queries_total = (queries_total or 0) + int(q_val)
                    last_extra = dict(res.extra)

                with eval_mode(deployed_victim), torch.no_grad():
                    adv_out = unwrap_model_outputs(deployed_victim(res.x_adv))
                    if is_classification:
                        adv_preds = predict_labels(adv_out)
                        y_cmp = (
                            yb.squeeze(1)
                            if adv_preds.ndim == 1 and yb.ndim == 2 and yb.shape[1] == 1
                            else yb
                        )
                        adv_correct = adv_preds == y_cmp
                        adv_correct_total += int(adv_correct.sum().item())
                        if is_targeted:
                            targeted_hits_total += int(adv_correct.sum().item())
                        else:
                            c_preds_b = clean_preds_exec[b_idx].to(eval_device)
                            clean_ok_b = c_preds_b == y_cmp
                            fooled_clean_correct_total += int((clean_ok_b & (~adv_correct)).sum().item())
                    else:
                        c_out_b = clean_outs_exec[b_idx].to(eval_device)
                        diff_out = (adv_out - c_out_b).reshape(xb.shape[0], -1)
                        l2_drift_sum += float(safe_l2_norm(diff_out, dim=1).sum().item())
                        cos_sum += float(_per_sample_cosine(adv_out, c_out_b).sum().item())
                        if clean_mse_applicable and adv_out.shape == yb.shape:
                            adv_mse_sum += float(_per_sample_mse(adv_out, yb).sum().item())

            mean_norm = norm_sum / max(total_n, 1)
            budget_ok = (max_norm <= eps + tol) if math.isfinite(max_norm) else True

            rob_acc: float | None = None
            asr: float | None = None
            adv_mse_val: float | None = None
            l2_drift_val: float | None = None
            cos_val: float | None = None

            if is_classification:
                rob_acc = float(adv_correct_total / max(total_n, 1))
                if is_targeted:
                    asr = float(targeted_hits_total / max(total_n, 1))
                else:
                    asr = (
                        float(fooled_clean_correct_total / clean_correct_total)
                        if clean_correct_total > 0
                        else 0.0
                    )
                last_extra["misclassification_rate"] = 1.0 - rob_acc
                last_extra["clean_correct_count"] = clean_correct_total
                last_extra["conditional_asr"] = asr
            else:
                l2_drift_val = float(l2_drift_sum / max(total_n, 1))
                cos_val = float(cos_sum / max(total_n, 1))
                if clean_mse_applicable:
                    adv_mse_val = float(adv_mse_sum / max(total_n, 1))
                last_extra["output_l2_drift"] = l2_drift_val
                last_extra["cosine_similarity"] = cos_val
                if adv_mse_val is not None:
                    last_extra["adv_mse"] = adv_mse_val

            if len(exec_batches) > 1:
                last_extra["num_batches"] = len(exec_batches)

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
                    steps_run=steps_max,
                    queries_used=queries_total,
                    elapsed_ms=float(elapsed_sum_ms),
                    hyperparameters=disclosed_hp,
                    citation=dict(card.get("citation", {})),
                    extra=last_extra,
                    adv_mse=adv_mse_val,
                    output_l2_drift=l2_drift_val,
                    cosine_similarity=cos_val,
                )
            )
            if verbose:
                if rob_acc is not None and asr is not None:
                    metric_str = f"robust_acc={rob_acc * 100:.2f}%, cond_asr={asr * 100:.2f}%"
                elif adv_mse_val is not None and l2_drift_val is not None:
                    metric_str = f"adv_mse={adv_mse_val:.4g}, l2_drift={l2_drift_val:.4g}"
                elif l2_drift_val is not None:
                    metric_str = f"l2_drift={l2_drift_val:.4g}"
                else:
                    metric_str = "done"
                print(
                    f"[wannado-adv] [{idx}/{total_attacks}] {aid} finished: "
                    f"{metric_str}, time={elapsed_sum_ms:.1f} ms, budget={'PASS' if budget_ok else 'FAIL'}",
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
        num_samples=int(total_n),
        seed=seed,
        device=str(eval_device),
        dtype=str(first_x.dtype),
        torch_version=torch.__version__,
        clean_acc=clean_acc,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes=notes,
        preflight=pf_report,
        audit=audit_report,
        records=records,
        clean_mse=clean_mse,
    )
