"""Carlini-style False-Robustness Auditor (Carlini et al., arXiv:1902.06705).

===============================================================================
[Academic Provenance]
- Reference : Carlini, N., Athalye, A., Papernot, N., Brendel, W., Rauber, J.,
              Tsipras, D., Goodfellow, I., Madry, A., & Kurakin, A. (2019).
              "On Evaluating Adversarial Robustness." arXiv:1902.06705.
- URL       : https://arxiv.org/abs/1902.06705
- Automated Sanity Checks Implemented:
    1. Autograd connectivity & vanishing/shattered gradient check (CE vs Margin).
    2. Iterative optimization dominance: iterative white-box <= single-step FGSM.
    3. Random-noise baseline: worst-case white-box <= uniform random search in ball.
    4. Softmax saturation detector: compares APGD-CE against scale-invariant APGD-DLR.
    5. Gradient obfuscation trap: verifies score-based black-box (Square Attack)
       does NOT materially outperform white-box gradient attacks.
    6. Perturbation-budget monotonicity sweep: larger eps / JSR cannot improve accuracy.
===============================================================================
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from wannado.adv.attacks.blackbox_score.square_andriushchenko2020 import square_attack
from wannado.adv.attacks.first_order.apgd_ce_croce2020 import apgd_ce
from wannado.adv.attacks.first_order.apgd_dlr_croce2020 import apgd_dlr
from wannado.adv.attacks.first_order.fgsm_goodfellow2015 import fgsm
from wannado.adv.attacks.first_order.pgd_madry2018 import pgd
from wannado.adv.audit.preflight import preflight
from wannado.adv.core.projections import random_init_perturbation, resolve_eps_budget
from wannado.adv.core.utils import eval_mode, make_generator


@dataclass
class AuditReport:
    """Structured false-robustness diagnostic card from ``audit_robustness()``."""

    passed: bool
    red_flags: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable audit dictionary."""
        return {
            "passed": self.passed,
            "red_flags": list(self.red_flags),
            "checks": dict(self.checks),
            "metrics": dict(self.metrics),
            "recommendations": list(self.recommendations),
        }

    def format_report(self) -> str:
        """Render human- and AI-readable audit report."""
        verdict = "PASS (No Gradient Masking Detected)" if self.passed else "WARNING (Potential False Robustness)"
        lines = [f"[wannado.adv Carlini Robustness Audit: {verdict}]"]
        if self.red_flags:
            lines.append("Red Flags:")
            lines.extend(f"  - {flag}" for flag in self.red_flags)
        lines.append("Checks:")
        for k, v in self.checks.items():
            lines.append(f"  - {k}: {'OK' if v else 'FAIL'}")
        lines.append(
            f"Accuracies -> clean: {self.metrics.get('clean_acc', 0.0):.3f}, "
            f"random: {self.metrics.get('random_acc', 0.0):.3f}, "
            f"fgsm: {self.metrics.get('fgsm_acc', 0.0):.3f}, "
            f"pgd: {self.metrics.get('pgd_acc', 0.0):.3f}, "
            f"apgd_ce: {self.metrics.get('apgd_ce_acc', 0.0):.3f}, "
            f"apgd_dlr: {self.metrics.get('apgd_dlr_acc', 0.0):.3f}, "
            f"square: {self.metrics.get('square_acc', 0.0):.3f}"
        )
        if self.recommendations:
            lines.append("Recommendations:")
            lines.extend(f"  - {rec}" for rec in self.recommendations)
        return "\n".join(lines)


def _eval_acc(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_adv: torch.Tensor,
    y: torch.Tensor,
) -> float:
    with torch.no_grad():
        return float((model_or_fn(x_adv).argmax(dim=1) == y).float().mean().item())


def _safe_attack_acc(
    attack_fn: Callable[..., Any],
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    clean_acc: float,
    **kwargs: Any,
) -> tuple[float, str | None]:
    """Run an attack while safely catching disconnected-autograd RuntimeErrors."""
    try:
        res = attack_fn(model_or_fn, x, y, **kwargs)
        return _eval_acc(model_or_fn, res.x_adv, y), None
    except RuntimeError as exc:
        return clean_acc, str(exc)


def audit_robustness(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    norm: str = "linf",
    clip_bounds: tuple[float, float] | None = None,
    pgd_steps: int = 20,
    apgd_steps: int = 35,
    square_queries: int = 350,
    random_trials: int = 5,
    seed: int = 0,
) -> AuditReport:
    """Run Carlini-style false-robustness checks (gradient masking, Softmax saturation, budget monotonicity).

    Args:
        model_or_fn: Classifier ``nn.Module`` or callable mapping ``x -> (B, C)`` logits.
        x: Clean evaluation batch (real or complex tensor, shape ``(B, ...)``).
        y: Ground-truth integer labels (shape ``(B,)``).
        eps: Perturbation budget (or JSR in dB when ``norm="jsr_db"``).
        norm: Constraint norm: ``"linf"``, ``"l2"``, ``"relative_l2"``, or ``"jsr_db"``.
        clip_bounds: Optional valid input box ``(min, max)`` for real-valued tensors.
        pgd_steps: Steps used for the PGD probe.
        apgd_steps: Steps used for APGD-CE / APGD-DLR probes.
        square_queries: Black-box query budget for Square Attack sanity check.
        random_trials: Number of uniform random perturbation draws inside the norm ball.
        seed: Random seed ensuring deterministic reproducibility.

    Returns:
        ``AuditReport`` detailing checks, ``red_flags``, accuracy breakdown, and next-step guidance.
    """
    if x.shape[0] < 2:
        raise ValueError(f"audit_robustness expects batch size >= 2, got {x.shape[0]}.")

    x_clean = x.detach()
    batch_size = x_clean.shape[0]
    pre = preflight(
        model_or_fn,
        x_clean,
        y,
        eps=eps,
        norm=norm,
        clip_bounds=clip_bounds,
        loss="ce",
        check_gradients=True,
        probe_batch_size=min(batch_size, 16),
    )
    if not pre.ok:
        return AuditReport(
            passed=False,
            red_flags=["preflight_failed"],
            checks={"preflight_ok": False},
            metrics={"preflight": pre.to_dict()},
            recommendations=list(pre.errors) + list(pre.recommendations),
        )

    red_flags: list[str] = []
    recommendations: list[str] = []
    grad_probe = pre.summary.get("gradient_probe", {})
    autograd_ok = bool(grad_probe.get("autograd_connected", False))
    zero_grad_frac = float(grad_probe.get("zero_grad_fraction", 1.0))
    ce_grad_mean = float(grad_probe.get("grad_norm_mean", 0.0))
    margin_grad_mean = grad_probe.get("margin_grad_norm_mean")

    tol_single = max(0.05, 1.5 / batch_size)
    tol_bb = max(0.10, 1.5 / batch_size)

    with eval_mode(model_or_fn):
        clean_acc = _eval_acc(model_or_fn, x_clean, y)

        # Random uniform baseline inside exact norm ball
        gen = make_generator(seed, x_clean.device)
        eps_budget = resolve_eps_budget(x_clean, eps=eps, norm=norm)
        with torch.no_grad():
            still_correct = model_or_fn(x_clean).argmax(dim=1) == y
            for _ in range(max(1, random_trials)):
                rand_delta = random_init_perturbation(
                    x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds, generator=gen
                )
                rand_correct = model_or_fn(x_clean + rand_delta).argmax(dim=1) == y
                still_correct = still_correct & rand_correct
        random_acc = float(still_correct.float().mean().item())

        # White-box ladder + black-box Square Attack
        fgsm_acc, err_fgsm = _safe_attack_acc(
            fgsm, model_or_fn, x_clean, y, clean_acc, eps=eps, norm=norm, clip_bounds=clip_bounds
        )
        pgd_acc, err_pgd = _safe_attack_acc(
            pgd,
            model_or_fn,
            x_clean,
            y,
            clean_acc,
            eps=eps,
            steps=pgd_steps,
            norm=norm,
            clip_bounds=clip_bounds,
            seed=seed,
        )
        apgd_ce_acc, err_ce = _safe_attack_acc(
            apgd_ce,
            model_or_fn,
            x_clean,
            y,
            clean_acc,
            eps=eps,
            steps=apgd_steps,
            restarts=1,
            norm=norm,
            clip_bounds=clip_bounds,
            seed=seed + 1,
        )
        apgd_dlr_acc, err_dlr = _safe_attack_acc(
            apgd_dlr,
            model_or_fn,
            x_clean,
            y,
            clean_acc,
            eps=eps,
            steps=apgd_steps,
            restarts=1,
            norm=norm,
            clip_bounds=clip_bounds,
            seed=seed + 2,
        )
        square_acc, _ = _safe_attack_acc(
            square_attack,
            model_or_fn,
            x_clean,
            y,
            clean_acc,
            eps=eps,
            max_queries=square_queries,
            norm=norm,
            clip_bounds=clip_bounds,
            seed=seed + 3,
        )

        # Budget monotonicity sweep
        if norm == "jsr_db":
            eps_low, eps_high = eps - 6.0, eps + 6.0
        else:
            eps_low, eps_high = 0.5 * eps, 1.5 * eps

        sweep_fn = apgd_dlr if (autograd_ok and err_dlr is None) else square_attack
        sweep_kwargs = (
            {"steps": apgd_steps, "restarts": 1, "norm": norm, "clip_bounds": clip_bounds}
            if sweep_fn is apgd_dlr
            else {"max_queries": square_queries, "norm": norm, "clip_bounds": clip_bounds}
        )
        acc_low, _ = _safe_attack_acc(
            sweep_fn, model_or_fn, x_clean, y, clean_acc, eps=eps_low, seed=seed + 4, **sweep_kwargs
        )
        acc_mid = apgd_dlr_acc if sweep_fn is apgd_dlr else square_acc
        acc_high, _ = _safe_attack_acc(
            sweep_fn, model_or_fn, x_clean, y, clean_acc, eps=eps_high, seed=seed + 5, **sweep_kwargs
        )

    best_whitebox_acc = min(pgd_acc, apgd_ce_acc, apgd_dlr_acc)
    worst_case_acc = min(best_whitebox_acc, fgsm_acc, square_acc, random_acc)

    # Evaluate Carlini check flags
    ce_vanished_vs_margin = (
        margin_grad_mean is not None and margin_grad_mean > 1e-4 and ce_grad_mean < 1e-7
    )
    softmax_saturated = (apgd_ce_acc - apgd_dlr_acc > tol_bb) or ce_vanished_vs_margin
    no_softmax_saturation = not softmax_saturated
    autograd_healthy = (
        autograd_ok and ((zero_grad_frac <= 0.5) or ce_vanished_vs_margin) and err_dlr is None
    )

    iterative_ran = (err_pgd is None) or (err_ce is None) or (err_dlr is None)
    if autograd_ok and iterative_ran:
        iterative_beats_single_step = (err_fgsm is None) and (best_whitebox_acc <= fgsm_acc + tol_single)
        whitebox_beats_random = best_whitebox_acc <= random_acc + tol_single
    else:
        iterative_beats_single_step = False
        whitebox_beats_random = False

    whitebox_beats_blackbox = square_acc >= best_whitebox_acc - tol_bb
    monotonic_over_budget = (
        (acc_mid <= acc_low + tol_bb)
        and (acc_high <= acc_mid + tol_bb)
        and (acc_high <= acc_low + tol_bb)
    )

    if not autograd_healthy:
        red_flags.append("broken_or_vanishing_gradients")
        recommendations.append(
            "Autograd gradients are broken or vanished on >50% of samples; rely on square_attack or hopskipjump instead of PGD."
        )
    if softmax_saturated:
        red_flags.append("softmax_saturation_masking")
        recommendations.append(
            "Softmax saturation detected (APGD-DLR / Margin outperforms CE). Do NOT report CE-PGD accuracy as robustness."
        )
    if not whitebox_beats_blackbox:
        red_flags.append("gradient_obfuscation_blackbox_beats_whitebox")
        recommendations.append(
            f"Black-box Square Attack ({square_acc:.3f}) outperformed white-box best ({best_whitebox_acc:.3f}), "
            "indicating gradient obfuscation or shattered gradients."
        )
    if autograd_ok and iterative_ran and not iterative_beats_single_step:
        red_flags.append("iterative_weaker_than_single_step")
        recommendations.append(
            "Single-step FGSM beat iterative white-box attacks; inspect step size or gradient shattering."
        )
    if autograd_ok and iterative_ran and not whitebox_beats_random:
        red_flags.append("whitebox_no_better_than_random")
        recommendations.append(
            "Random perturbations inside the ball matched or beat white-box gradients; gradients are uninformative."
        )
    if not monotonic_over_budget:
        red_flags.append("non_monotonic_budget_response")
        recommendations.append(
            "Increasing perturbation budget increased robust accuracy; verify gradient stability and step sizes."
        )

    checks = {
        "preflight_ok": True,
        "autograd_healthy": autograd_healthy,
        "no_softmax_saturation": no_softmax_saturation,
        "iterative_beats_single_step": iterative_beats_single_step,
        "whitebox_beats_blackbox": whitebox_beats_blackbox,
        "whitebox_beats_random": whitebox_beats_random,
        "monotonic_over_budget": monotonic_over_budget,
    }

    passed = len(red_flags) == 0 and all(checks.values())
    metrics = {
        "clean_acc": clean_acc,
        "random_acc": random_acc,
        "fgsm_acc": fgsm_acc,
        "pgd_acc": pgd_acc,
        "apgd_ce_acc": apgd_ce_acc,
        "apgd_dlr_acc": apgd_dlr_acc,
        "square_acc": square_acc,
        "best_whitebox_acc": best_whitebox_acc,
        "worst_case_acc": worst_case_acc,
        "budget_sweep": {
            "eps_low": float(eps_low),
            "acc_low": acc_low,
            "eps_mid": float(eps),
            "acc_mid": acc_mid,
            "eps_high": float(eps_high),
            "acc_high": acc_high,
        },
        "gradient_probe": grad_probe,
        "attack_errors": {
            k: v
            for k, v in (
                ("fgsm", err_fgsm),
                ("pgd", err_pgd),
                ("apgd_ce", err_ce),
                ("apgd_dlr", err_dlr),
            )
            if v is not None
        },
    }
    return AuditReport(
        passed=passed,
        red_flags=red_flags,
        checks=checks,
        metrics=metrics,
        recommendations=recommendations,
    )
