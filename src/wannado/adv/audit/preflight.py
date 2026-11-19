"""AI-native preflight safety checks before running adversarial attacks.

Validates model eval/BatchNorm/Dropout state, autograd connectivity, input value ranges,
complex baseband / JSR dB signal baselines, catalog compatibility, and compute warnings
without mutating caller model state or parameter gradients.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from wannado.adv.catalog.build_catalog import load_catalog
from wannado.adv.core.losses import resolve_loss_fn, unwrap_model_outputs
from wannado.adv.core.projections import (
    SUPPORTED_NORMS,
    coerce_clip_bounds,
    resolve_eps_budget,
    safe_l2_norm,
)
from wannado.adv.core.utils import eval_mode


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


@dataclass
class PreflightReport:
    """Structured diagnostic report returned by ``preflight()``."""

    ok: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable report dictionary."""
        return {
            "ok": self.ok,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "recommendations": list(self.recommendations),
            "summary": dict(self.summary),
        }

    def format_report(self) -> str:
        """Render a concise human- and AI-readable diagnostic summary."""
        status = "PASS" if self.ok else "FAIL"
        lines = [f"[wannado.adv Preflight: {status}]"]
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  - {err}" for err in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {warn}" for warn in self.warnings)
        if self.recommendations:
            lines.append("Recommendations:")
            lines.extend(f"  - {rec}" for rec in self.recommendations)
        return "\n".join(lines)


def _inspect_module(model_or_fn: Callable[[torch.Tensor], torch.Tensor]) -> dict[str, Any]:
    if not isinstance(model_or_fn, nn.Module):
        return {
            "is_nn_module": False,
            "was_training": None,
            "has_batchnorm": False,
            "has_dropout": False,
            "params_with_existing_grads": 0,
        }

    has_bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model_or_fn.modules())
    has_dropout = any(isinstance(m, nn.modules.dropout._DropoutNd) for m in model_or_fn.modules())
    existing_grads = sum(1 for p in model_or_fn.parameters() if p.grad is not None)
    return {
        "is_nn_module": True,
        "was_training": bool(model_or_fn.training),
        "has_batchnorm": has_bn,
        "has_dropout": has_dropout,
        "params_with_existing_grads": existing_grads,
    }


def _probe_gradients(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_probe: torch.Tensor,
    y_probe: torch.Tensor | None,
    loss_name: str | None,
) -> dict[str, Any]:
    """Probe forward output health and input gradient flow without mutating model param grads."""
    result: dict[str, Any] = {
        "outputs_shape": None,
        "outputs_finite": True,
        "autograd_connected": False,
        "grad_norm_mean": 0.0,
        "zero_grad_fraction": 1.0,
        "margin_grad_norm_mean": None,
        "error": None,
    }

    with eval_mode(model_or_fn):
        x_in = x_probe.detach().requires_grad_(True)
        try:
            outputs = unwrap_model_outputs(model_or_fn(x_in))
        except Exception as exc:  # noqa: BLE001 — surface any user model failure in preflight report
            result["error"] = f"Forward pass failed during preflight: {type(exc).__name__}: {exc}"
            return result

        result["outputs_shape"] = list(outputs.shape)
        if outputs.is_complex():
            finite = bool(torch.isfinite(outputs.real).all() and torch.isfinite(outputs.imag).all())
        else:
            finite = bool(torch.isfinite(outputs).all())
        result["outputs_finite"] = finite
        if not finite:
            result["error"] = "Model produced NaN or Inf outputs on clean inputs."
            return result

        result["autograd_connected"] = bool(outputs.requires_grad)
        if not outputs.requires_grad:
            return result

        is_cls_target = y_probe is not None and y_probe.dtype in (
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.long,
        )
        effective_loss_name = loss_name
        if y_probe is not None and not is_cls_target and loss_name in (None, "ce", "margin", "dlr"):
            effective_loss_name = "mse" if outputs.shape == y_probe.shape else None

        result["probe_loss_used"] = effective_loss_name or ("l2_output" if y_probe is None or not is_cls_target else "ce")
        if y_probe is not None and (is_cls_target or effective_loss_name is not None):
            loss_fn = resolve_loss_fn(effective_loss_name)
            scalar_vec = loss_fn(outputs, y_probe)
        else:
            scalar_vec = outputs.abs().pow(2) if outputs.is_complex() else outputs.pow(2)
        if scalar_vec.is_complex():
            scalar_vec = scalar_vec.real

        will_probe_margin = is_cls_target and outputs.ndim >= 1 and (effective_loss_name in (None, "ce"))
        try:
            (grad,) = torch.autograd.grad(
                scalar_vec.sum(),
                x_in,
                retain_graph=will_probe_margin,
                only_inputs=True,
                allow_unused=True,
            )
        except RuntimeError as exc:
            result["error"] = f"Autograd failed: {exc}"
            result["autograd_connected"] = False
            return result

        if grad is None:
            result["autograd_connected"] = False
            return result

        b = x_in.shape[0]
        g_norms = safe_l2_norm(grad.detach().reshape(b, -1), dim=1)
        result["grad_norm_mean"] = float(g_norms.mean().item())
        result["zero_grad_fraction"] = float((g_norms < 1e-10).float().mean().item())

        if will_probe_margin and y_probe is not None:
            margin_vec = resolve_loss_fn("margin")(outputs, y_probe)
            (m_grad,) = torch.autograd.grad(
                margin_vec.sum(),
                x_in,
                only_inputs=True,
                allow_unused=True,
            )
            if m_grad is not None:
                m_norms = safe_l2_norm(m_grad.detach().reshape(b, -1), dim=1)
                result["margin_grad_norm_mean"] = float(m_norms.mean().item())

    return result


def preflight(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    eps: float = 0.03,
    norm: str = "linf",
    clip_bounds: tuple[Any, Any] | None = None,
    attack: str | None = None,
    loss: str | None = "ce",
    defense_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    check_gradients: bool = True,
    probe_batch_size: int = 8,
    strict: bool = False,
) -> PreflightReport:
    """Run automated domain, model-state, gradient, energy, and compute preflight checks.

    Args:
        model_or_fn: Victim ``nn.Module`` or callable mapping ``x -> outputs``.
        x: Clean sample batch (real or complex tensor, shape ``(B, ...)``).
        y: Optional ground-truth labels/targets corresponding to ``x``.
        eps: Intended perturbation budget (or JSR in dB when ``norm="jsr_db"``).
        norm: Target geometry: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        clip_bounds: Optional valid input box ``(min, max)`` for real-valued tensors.
        attack: Optional attack ID (from ``registry.json``) to cross-check requirements & cost.
        loss: Loss function name used for gradient probe (default ``"ce"``).
        check_gradients: Whether to run an input-gradient connectivity probe.
        probe_batch_size: Max batch slice evaluated during the gradient probe.
        strict: If ``True``, raise ``ValueError`` when any hard error is detected.

    Returns:
        ``PreflightReport`` containing ``ok``, ``warnings``, ``errors``, ``recommendations``, and ``summary``.
    """
    warnings: list[str] = []
    errors: list[str] = []
    recommendations: list[str] = []

    module_info = _inspect_module(model_or_fn)
    module_info["has_defense_fn"] = defense_fn is not None
    effective_model: Callable[[torch.Tensor], torch.Tensor] = model_or_fn
    if defense_fn is not None:
        from wannado.adv.core.utils import DefendedModelWrapper

        effective_model = DefendedModelWrapper(model_or_fn, defense_fn)
    summary: dict[str, Any] = {
        "model": module_info,
        "input": {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "is_complex": bool(x.is_complex()),
        },
        "budget": {
            "eps": float(eps),
            "norm": norm,
            "clip_bounds": _serialize_bounds(clip_bounds),
        },
    }

    if not callable(model_or_fn):
        cls_name = type(model_or_fn).__name__
        errors.append(
            f"model_or_fn ({cls_name}) is not directly callable on PyTorch tensors."
        )
        recommendations.append(
            "For scikit-learn estimators, wrap with wrap_sklearn_model(estimator) "
            "and use black-box attacks (square_attack, nes, spsa, hopskipjump, boundary)."
        )

    if module_info["was_training"]:
        layer_note = []
        if module_info["has_batchnorm"]:
            layer_note.append("BatchNorm")
        if module_info["has_dropout"]:
            layer_note.append("Dropout")
        extra_str = f" ({', '.join(layer_note)} detected)" if layer_note else ""
        warnings.append(
            f"Model is currently in model.training=True mode{extra_str}. Attacks in wannado.adv "
            "automatically isolate eval() and restore mode afterward, but remember to call model.eval() before outer evaluations."
        )

    if x.ndim < 1 or x.shape[0] < 1:
        errors.append(f"Input batch must have shape (B, ...) with B >= 1, got {tuple(x.shape)}.")
        report = PreflightReport(False, warnings, errors, recommendations, summary)
        if strict:
            raise ValueError("; ".join(errors))
        return report

    if y is not None and y.shape[0] != x.shape[0]:
        errors.append(f"Batch mismatch: x.shape[0]={x.shape[0]} vs y.shape[0]={y.shape[0]}.")

    # Input finiteness & domain bounds check
    if x.is_complex():
        x_finite = bool(torch.isfinite(x.real).all() and torch.isfinite(x.imag).all())
    else:
        x_finite = bool(torch.isfinite(x).all())
    if not x_finite:
        errors.append("Clean input tensor contains NaN or Inf values.")

    if clip_bounds is not None and x.is_complex():
        errors.append(
            "clip_bounds=(min, max) cannot be applied to complex-valued tensors; pass clip_bounds=None "
            "and constrain energy via norm='jsr_db' or norm='relative_l2'."
        )
        recommendations.append("Use clip_bounds=None and norm='jsr_db' for complex IQ signals.")
    elif clip_bounds is not None:
        try:
            lo_t, hi_t = coerce_clip_bounds(clip_bounds, x)
            if x_finite and bool(((x < lo_t - 1e-5) | (x > hi_t + 1e-5)).any().item()):
                x_min = float(x.min().item())
                x_max = float(x.max().item())
                warnings.append(
                    f"Clean input values range [{x_min:.4g}, {x_max:.4g}], exceeding supplied clip_bounds."
                )
        except ValueError as exc:
            errors.append(str(exc))
    elif not x.is_complex() and x_finite:
        x_min = float(x.min().item())
        x_max = float(x.max().item())
        summary["input"]["min"] = x_min
        summary["input"]["max"] = x_max
        if 0.0 <= x_min and x_max <= 1.0 and loss in (None, "ce", "margin", "dlr"):
            recommendations.append(
                "Inputs lie inside [0, 1]; pass clip_bounds=(0.0, 1.0) if valid sensor/pixel range is bounded."
            )

    # Energy & budget diagnostics
    if norm not in SUPPORTED_NORMS:
        errors.append(f"Unsupported norm '{norm}'. Supported: {sorted(SUPPORTED_NORMS)}.")
    else:
        b = x.shape[0]
        x_l2 = safe_l2_norm(x.reshape(b, -1), dim=1)
        x_l2_min = float(x_l2.min().item())
        x_l2_mean = float(x_l2.mean().item())
        summary["input"]["l2_norm_min"] = x_l2_min
        summary["input"]["l2_norm_mean"] = x_l2_mean

        if norm in ("relative_l2", "jsr_db") and x_l2_min < 1e-9:
            errors.append(
                f"Input contains near-zero-energy sample (min ||x||_2={x_l2_min:.2e}), making norm='{norm}' ill-defined."
            )

        if x.is_complex() and norm == "linf":
            recommendations.append(
                "For complex wireless IQ signals, prefer norm='jsr_db' or norm='relative_l2' over elementwise norm='linf'."
            )

        if norm != "jsr_db" and eps < 0:
            errors.append(f"eps must be >= 0 for norm='{norm}', got {eps}.")
        else:
            budget = resolve_eps_budget(x, eps=eps, norm=norm)
            summary["budget"]["resolved_min"] = float(budget.min().item())
            summary["budget"]["resolved_mean"] = float(budget.mean().item())
            if norm == "jsr_db" and eps > 0.0:
                warnings.append(
                    f"JSR={eps} dB > 0 dB means adversarial jammer power ({10 ** (eps / 10):.2f}x) exceeds clean signal power."
                )

    # Gradient connectivity & Softmax saturation probe
    grad_info: dict[str, Any] | None = None
    if check_gradients and not errors:
        k = min(x.shape[0], max(1, probe_batch_size))
        y_k = y[:k] if y is not None else None
        grad_info = _probe_gradients(effective_model, x[:k], y_k, loss_name=loss)
        summary["gradient_probe"] = grad_info

        if grad_info["error"] is not None:
            warnings.append(str(grad_info["error"]))
        if not grad_info["outputs_finite"]:
            errors.append("Model output contains NaN/Inf on clean inputs.")
        elif not grad_info["autograd_connected"]:
            warnings.append(
                "Autograd input gradients are disconnected (outputs have no grad_fn). Gradient-based white-box attacks will fail."
            )
            recommendations.append("Use gradient-free attacks: square_attack, spsa, or hopskipjump.")
        else:
            ce_grad_mean = grad_info["grad_norm_mean"]
            margin_g = grad_info["margin_grad_norm_mean"]
            if grad_info["zero_grad_fraction"] > 0.5:
                if margin_g is not None and margin_g > 1e-4 and ce_grad_mean < 1e-7:
                    warnings.append(
                        f"Cross-Entropy input gradients nearly vanished (mean={ce_grad_mean:.2e}) while margin gradients "
                        f"remain strong (mean={margin_g:.2e}) — classic Softmax saturation masking."
                    )
                    recommendations.append("Switch to apgd_dlr or loss='margin' / 'dlr' to defeat Softmax saturation.")
                else:
                    warnings.append(
                        f"{grad_info['zero_grad_fraction']:.0%} of probed samples had near-zero gradients (<1e-10)."
                    )
                    recommendations.append("Run audit_robustness() or verify with square_attack / hopskipjump.")

    # Catalog & compute cost pre-check
    if attack is not None:
        cards = {c["id"]: c for c in load_catalog()}
        card = cards.get(attack)
        if card is None:
            errors.append(f"Unknown attack '{attack}' not found in wannado.adv catalog.")
        else:
            summary["attack"] = {
                "id": card["id"],
                "tier": card["tier"],
                "threat_model": card["threat_model"],
                "requires_training": card["requires_training"],
                "compute_cost": card["compute_cost"],
            }
            if norm not in card["supported_norms"]:
                errors.append(
                    f"Attack '{attack}' does not support norm='{norm}'. Supported: {card['supported_norms']}."
                )
            if card["requires_training"] or card["compute_cost"] in ("high", "very_high"):
                warnings.append(
                    f"Attack '{attack}' has compute_cost='{card['compute_cost']}', "
                    f"requires_training={card['requires_training']}."
                )
            if (
                grad_info is not None
                and not grad_info["autograd_connected"]
                and card["threat_model"] in ("white_box", "generative_train")
            ):
                errors.append(
                    f"White-box attack '{attack}' requires input gradients, but autograd is disconnected on this model."
                )
                recommendations.append("Switch attack to 'square_attack', 'spsa', or 'hopskipjump'.")

    ok = len(errors) == 0
    report = PreflightReport(
        ok=ok,
        warnings=warnings,
        errors=errors,
        recommendations=recommendations,
        summary=summary,
    )
    if strict and not ok:
        raise ValueError("; ".join(errors))
    return report
