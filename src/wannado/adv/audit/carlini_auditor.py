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

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from wannado.adv.attacks.blackbox_score.square_andriushchenko2020 import square_attack
from wannado.adv.attacks.first_order.apgd_ce_croce2020 import apgd_ce
from wannado.adv.attacks.first_order.apgd_dlr_croce2020 import apgd_dlr
from wannado.adv.attacks.first_order.apgd_l1_croce2021 import apgd_l1
from wannado.adv.attacks.first_order.fgsm_goodfellow2015 import fgsm
from wannado.adv.attacks.first_order.pgd_madry2018 import pgd
from wannado.adv.attacks.physical_and_eot.bpda_eot_athalye2018 import bpda_eot
from wannado.adv.attacks.physical_and_eot.eot_pgd_athalye2018 import eot_pgd
from wannado.adv.audit.preflight import preflight
from wannado.adv.core.losses import predict_labels, unwrap_model_outputs
from wannado.adv.core.projections import random_init_perturbation, resolve_eps_budget
from wannado.adv.core.utils import DefendedModelWrapper, eval_mode, make_generator, materialize_eval_tensors


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
    *,
    mc_draws: int = 1,
) -> float:
    with torch.no_grad():
        total = 0.0
        for _ in range(max(1, mc_draws)):
            preds = predict_labels(model_or_fn(x_adv))
            y_cmp = y.squeeze(1) if preds.ndim == 1 and y.ndim == 2 and y.shape[1] == 1 else y
            total += float((preds == y_cmp).float().mean().item())
        return total / float(max(1, mc_draws))


def _eval_continuous_error(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_adv: torch.Tensor,
    y_ref: torch.Tensor,
    clean_out: torch.Tensor,
    *,
    use_mse: bool,
    mc_draws: int = 1,
) -> float:
    with torch.no_grad():
        total = 0.0
        for _ in range(max(1, mc_draws)):
            out = unwrap_model_outputs(model_or_fn(x_adv))
            diff = (out - (y_ref if use_mse else clean_out)).reshape(x_adv.shape[0], -1)
            val = (diff.real.pow(2) + diff.imag.pow(2)).mean() if diff.is_complex() else diff.pow(2).mean()
            total += float(val.item())
        return total / float(max(1, mc_draws))


def _safe_attack_acc(
    attack_fn: Callable[..., Any],
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    clean_acc: float,
    *,
    eval_victim: Callable[[torch.Tensor], torch.Tensor] | None = None,
    mc_draws: int = 1,
    **kwargs: Any,
) -> tuple[float, str | None]:
    """Run an attack while safely catching disconnected-autograd RuntimeErrors."""
    victim = eval_victim if eval_victim is not None else model_or_fn
    try:
        res = attack_fn(model_or_fn, x, y, **kwargs)
        return _eval_acc(victim, res.x_adv, y, mc_draws=mc_draws), None
    except RuntimeError as exc:
        return clean_acc, str(exc)


def _safe_attack_continuous(
    attack_fn: Callable[..., Any],
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    clean_out: torch.Tensor,
    clean_err: float,
    *,
    use_mse: bool,
    eval_victim: Callable[[torch.Tensor], torch.Tensor] | None = None,
    mc_draws: int = 1,
    **kwargs: Any,
) -> tuple[float, str | None]:
    victim = eval_victim if eval_victim is not None else model_or_fn
    try:
        res = attack_fn(model_or_fn, x, y, **kwargs)
        return _eval_continuous_error(victim, res.x_adv, y, clean_out, use_mse=use_mse, mc_draws=mc_draws), None
    except RuntimeError as exc:
        return clean_err, str(exc)


def _probe_stochasticity(
    victim: Callable[[torch.Tensor], torch.Tensor],
    x_probe: torch.Tensor,
) -> tuple[bool, float, bool]:
    """Run two forward passes on identical clean inputs to detect test-time stochasticity & hard-label outputs."""
    with eval_mode(victim), torch.no_grad():
        out1 = unwrap_model_outputs(victim(x_probe))
        out2 = unwrap_model_outputs(victim(x_probe))
    is_hard_label = out1.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)
    diff = (out1.float() - out2.float()).abs() if is_hard_label else (out1 - out2).abs()
    max_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
    return max_diff > 1e-5, max_diff, is_hard_label


def audit_robustness(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    eps: float = 0.03,
    norm: str = "linf",
    clip_bounds: tuple[Any, Any] | None = None,
    pgd_steps: int = 20,
    apgd_steps: int = 35,
    square_queries: int = 350,
    random_trials: int = 5,
    max_samples: int | None = None,
    defense_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    surrogate_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    eot_samples: int = 4,
    seed: int = 0,
) -> AuditReport:
    """Run Carlini 2.0 false-robustness checks across classification, L1, stochastic, defended, and continuous victims.

    Checks:
        1. Preflight & perturbation scale sanity (`eps_scale_meaningful`).
        2. Autograd health (`autograd_healthy`) and Softmax saturation (`no_softmax_saturation`).
        3. Iterative dominance (`iterative_beats_single_step`) and random-noise floor (`whitebox_beats_random`).
        4. Black-box vs white-box obfuscation check (`whitebox_beats_blackbox`).
        5. Perturbation budget monotonicity (`monotonic_over_budget`).
        6. Test-time stochasticity / randomization masking check (`deterministic_or_eot_verified`).
        7. Optional `defense_fn` BPDA+EoT cross-check (`defense_bpda_consistent`).
    """
    dev: torch.device | None = None
    if not isinstance(x, torch.Tensor) and isinstance(model_or_fn, torch.nn.Module):
        for p in model_or_fn.parameters():
            dev = p.device
            break
    x_clean, y_clean = materialize_eval_tensors(x, y, max_samples=max_samples, device=dev)
    if x_clean.shape[0] < 2:
        raise ValueError(f"audit_robustness expects batch size >= 2, got {x_clean.shape[0]}.")

    y = y_clean
    batch_size = x_clean.shape[0]
    deployed_victim: Callable[[torch.Tensor], torch.Tensor] = (
        DefendedModelWrapper(model_or_fn, defense_fn) if defense_fn is not None else model_or_fn
    )

    is_classification = y.dtype in (
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.long,
    )
    pf_loss: str | None = "ce" if is_classification else "mse"
    pre = preflight(
        model_or_fn,
        x_clean,
        y,
        eps=eps,
        norm=norm,
        clip_bounds=clip_bounds,
        loss=pf_loss,
        defense_fn=defense_fn,
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
    eps_scale_meaningful = True
    if not x_clean.is_complex() and norm in ("linf", "l2", "l1") and eps > 0:
        span = max(float((x_clean.max() - x_clean.min()).item()), 1e-6)
        eff_eps = eps / math.sqrt(max(1, x_clean[0].numel())) if norm == "l2" else eps
        if eff_eps / span < 5e-4:
            eps_scale_meaningful = False
            red_flags.append("eps_scale_mismatch_trivial_budget")
            recommendations.append(
                f"Perturbation budget eps={eps:.4g} under norm='{norm}' is <0.05% of the input dynamic range "
                f"(span={span:.4g}). Rescale eps to the domain's physical units or normalize inputs to [0, 1]."
            )

    grad_probe = pre.summary.get("gradient_probe", {})
    autograd_ok = bool(grad_probe.get("autograd_connected", False))
    zero_grad_frac = float(grad_probe.get("zero_grad_fraction", 1.0))
    ce_grad_mean = float(grad_probe.get("grad_norm_mean", 0.0))
    margin_grad_mean = grad_probe.get("margin_grad_norm_mean")

    is_stochastic, stoch_diff, is_hard_label_only = _probe_stochasticity(
        deployed_victim, x_clean[: min(batch_size, 16)]
    )
    mc_draws = 3 if is_stochastic else 1

    # Continuous / Regression / DRL-Box Audit Mode
    if not is_classification:
        with eval_mode(deployed_victim):
            with torch.no_grad():
                clean_out = unwrap_model_outputs(deployed_victim(x_clean))
            use_mse = clean_out.shape == y.shape
            clean_err = _eval_continuous_error(deployed_victim, x_clean, y, clean_out, use_mse=use_mse, mc_draws=mc_draws)
            gen = make_generator(seed, x_clean.device)
            eps_budget = resolve_eps_budget(x_clean, eps=eps, norm=norm)
            rand_err = clean_err
            for _ in range(max(1, random_trials)):
                rd = random_init_perturbation(x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds, generator=gen)
                rand_err = max(
                    rand_err,
                    _eval_continuous_error(deployed_victim, x_clean + rd, y, clean_out, use_mse=use_mse, mc_draws=mc_draws),
                )
            loss_kw = "mse" if use_mse else (lambda out, _y: (out - clean_out[: out.shape[0]]).abs().pow(2).reshape(out.shape[0], -1).mean(dim=1))
            fgsm_err, err_fgsm = _safe_attack_continuous(
                fgsm, deployed_victim, x_clean, y, clean_out, clean_err, use_mse=use_mse, mc_draws=mc_draws,
                eps=eps, norm=norm, loss=loss_kw, clip_bounds=clip_bounds,
            )
            pgd_err, err_pgd = _safe_attack_continuous(
                pgd, deployed_victim, x_clean, y, clean_out, clean_err, use_mse=use_mse, mc_draws=mc_draws,
                eps=eps, steps=pgd_steps, norm=norm, loss=loss_kw, clip_bounds=clip_bounds, seed=seed,
            )
            eot_err: float | None = None
            if is_stochastic and autograd_ok:
                eot_err, _ = _safe_attack_continuous(
                    eot_pgd, deployed_victim, x_clean, y, clean_out, clean_err, use_mse=use_mse, mc_draws=mc_draws,
                    eps=eps, steps=pgd_steps, eot_samples=eot_samples, norm=norm, loss=loss_kw, clip_bounds=clip_bounds, seed=seed + 6,
                )
            if norm == "jsr_db":
                eps_low, eps_high = eps - 6.0, eps + 6.0
            else:
                eps_low, eps_high = 0.5 * eps, 1.5 * eps
            err_low, _ = _safe_attack_continuous(
                pgd, deployed_victim, x_clean, y, clean_out, clean_err, use_mse=use_mse, mc_draws=mc_draws,
                eps=eps_low, steps=pgd_steps, norm=norm, loss=loss_kw, clip_bounds=clip_bounds, seed=seed + 4,
            )
            err_high, _ = _safe_attack_continuous(
                pgd, deployed_victim, x_clean, y, clean_out, clean_err, use_mse=use_mse, mc_draws=mc_draws,
                eps=eps_high, steps=pgd_steps, norm=norm, loss=loss_kw, clip_bounds=clip_bounds, seed=seed + 5,
            )

        best_wb_err = max(pgd_err, eot_err if eot_err is not None else pgd_err)
        scale_tol = max(1e-4, 0.05 * max(best_wb_err, clean_err, 1.0))
        autograd_healthy = autograd_ok and zero_grad_frac <= 0.5 and err_pgd is None
        iterative_beats_single_step = autograd_healthy and (best_wb_err >= fgsm_err - scale_tol)
        whitebox_beats_random = autograd_healthy and (best_wb_err >= rand_err - scale_tol)
        monotonic_over_budget = (pgd_err >= err_low - scale_tol) and (err_high >= pgd_err - scale_tol)
        deterministic_or_eot_verified = not is_stochastic
        if not autograd_healthy:
            red_flags.append("broken_or_vanishing_gradients")
            recommendations.append("Continuous victim gradients are disconnected or vanished; use square_attack/nes with MSE objective.")
        if is_stochastic:
            red_flags.append("stochastic_defense_requires_eot")
            recommendations.append("Model output is stochastic across identical eval() passes; evaluate with eot_pgd.")
        if autograd_healthy and not iterative_beats_single_step:
            red_flags.append("iterative_weaker_than_single_step")
        if autograd_healthy and not whitebox_beats_random:
            red_flags.append("whitebox_no_better_than_random")
        if not monotonic_over_budget:
            red_flags.append("non_monotonic_budget_response")

        checks_c = {
            "preflight_ok": True,
            "eps_scale_meaningful": eps_scale_meaningful,
            "autograd_healthy": autograd_healthy,
            "deterministic_or_eot_verified": deterministic_or_eot_verified,
            "iterative_beats_single_step": iterative_beats_single_step,
            "whitebox_beats_random": whitebox_beats_random,
            "monotonic_over_budget": monotonic_over_budget,
        }
        return AuditReport(
            passed=len(red_flags) == 0 and all(checks_c.values()),
            red_flags=red_flags,
            checks=checks_c,
            metrics={
                "task_type": "continuous",
                "is_stochastic": is_stochastic,
                "stochastic_max_output_diff": stoch_diff,
                "clean_mse": clean_err,
                "random_mse": rand_err,
                "fgsm_mse": fgsm_err,
                "pgd_mse": pgd_err,
                "eot_pgd_mse": eot_err,
                "budget_sweep": {
                    "eps_low": float(eps_low),
                    "err_low": err_low,
                    "eps_mid": float(eps),
                    "err_mid": pgd_err,
                    "eps_high": float(eps_high),
                    "err_high": err_high,
                },
                "gradient_probe": grad_probe,
            },
            recommendations=recommendations,
        )

    tol_single = max(0.05, 1.5 / batch_size)
    tol_bb = max(0.10, 1.5 / batch_size)

    with eval_mode(deployed_victim):
        clean_acc = _eval_acc(deployed_victim, x_clean, y, mc_draws=mc_draws)

        # Random uniform baseline inside exact norm ball
        gen = make_generator(seed, x_clean.device)
        eps_budget = resolve_eps_budget(x_clean, eps=eps, norm=norm)
        with torch.no_grad():
            still_correct = predict_labels(deployed_victim(x_clean)) == y
            for _ in range(max(1, random_trials)):
                rand_delta = random_init_perturbation(
                    x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds, generator=gen
                )
                rand_correct = predict_labels(deployed_victim(x_clean + rand_delta)) == y
                still_correct = still_correct & rand_correct
        random_acc = float(still_correct.float().mean().item())

        # White-box ladder (L1-aware via apgd_l1) + black-box Square Attack
        fgsm_acc, err_fgsm = _safe_attack_acc(
            fgsm, deployed_victim, x_clean, y, clean_acc, mc_draws=mc_draws,
            eps=eps, norm=norm, clip_bounds=clip_bounds,
        )
        pgd_acc, err_pgd = _safe_attack_acc(
            pgd, deployed_victim, x_clean, y, clean_acc, mc_draws=mc_draws,
            eps=eps, steps=pgd_steps, norm=norm, clip_bounds=clip_bounds, seed=seed,
        )
        if norm == "l1":
            apgd_ce_acc, err_ce = _safe_attack_acc(
                apgd_l1, deployed_victim, x_clean, y, clean_acc, mc_draws=mc_draws,
                eps=eps, steps=apgd_steps, restarts=1, loss="ce", clip_bounds=clip_bounds, seed=seed + 1,
            )
            apgd_dlr_acc, err_dlr = _safe_attack_acc(
                apgd_l1, deployed_victim, x_clean, y, clean_acc, mc_draws=mc_draws,
                eps=eps, steps=apgd_steps, restarts=1, loss="dlr", clip_bounds=clip_bounds, seed=seed + 2,
            )
            square_acc = clean_acc
        else:
            apgd_ce_acc, err_ce = _safe_attack_acc(
                apgd_ce, deployed_victim, x_clean, y, clean_acc, mc_draws=mc_draws,
                eps=eps, steps=apgd_steps, restarts=1, norm=norm, clip_bounds=clip_bounds, seed=seed + 1,
            )
            apgd_dlr_acc, err_dlr = _safe_attack_acc(
                apgd_dlr, deployed_victim, x_clean, y, clean_acc, mc_draws=mc_draws,
                eps=eps, steps=apgd_steps, restarts=1, norm=norm, clip_bounds=clip_bounds, seed=seed + 2,
            )
            square_acc, _ = _safe_attack_acc(
                square_attack, deployed_victim, x_clean, y, clean_acc, mc_draws=mc_draws,
                eps=eps, max_queries=square_queries, norm=norm, clip_bounds=clip_bounds, seed=seed + 3,
            )

        # Stochastic randomization EoT cross-check
        eot_acc: float | None = None
        if is_stochastic and autograd_ok:
            eot_acc, _ = _safe_attack_acc(
                eot_pgd, deployed_victim, x_clean, y, clean_acc, mc_draws=mc_draws,
                eps=eps, steps=pgd_steps, eot_samples=eot_samples, norm=norm, clip_bounds=clip_bounds, seed=seed + 6,
            )

        # Defense BPDA+EoT cross-check when defense_fn is supplied
        bpda_acc: float | None = None
        if defense_fn is not None:
            bpda_acc, _ = _safe_attack_acc(
                bpda_eot, model_or_fn, x_clean, y, clean_acc, eval_victim=deployed_victim, mc_draws=mc_draws,
                eps=eps, steps=pgd_steps, eot_samples=eot_samples if is_stochastic else 2,
                defense_fn=defense_fn, surrogate_fn=surrogate_fn, norm=norm, clip_bounds=clip_bounds, seed=seed + 7,
            )

        # Budget monotonicity sweep
        if norm == "jsr_db":
            eps_low, eps_high = eps - 6.0, eps + 6.0
        else:
            eps_low, eps_high = 0.5 * eps, 1.5 * eps

        if norm == "l1":
            sweep_fn: Callable[..., Any] = apgd_l1
            sweep_kwargs: dict[str, Any] = {"steps": apgd_steps, "restarts": 1, "loss": "dlr", "clip_bounds": clip_bounds}
            acc_mid = apgd_dlr_acc
        elif autograd_ok and err_dlr is None:
            sweep_fn = apgd_dlr
            sweep_kwargs = {"steps": apgd_steps, "restarts": 1, "norm": norm, "clip_bounds": clip_bounds}
            acc_mid = apgd_dlr_acc
        else:
            sweep_fn = square_attack
            sweep_kwargs = {"max_queries": square_queries, "norm": norm, "clip_bounds": clip_bounds}
            acc_mid = square_acc

        acc_low, _ = _safe_attack_acc(
            sweep_fn, deployed_victim, x_clean, y, clean_acc, mc_draws=mc_draws, eps=eps_low, seed=seed + 4, **sweep_kwargs
        )
        acc_high, _ = _safe_attack_acc(
            sweep_fn, deployed_victim, x_clean, y, clean_acc, mc_draws=mc_draws, eps=eps_high, seed=seed + 5, **sweep_kwargs
        )

    naive_whitebox_acc = min(pgd_acc, apgd_ce_acc, apgd_dlr_acc)
    best_whitebox_acc = naive_whitebox_acc
    if eot_acc is not None:
        best_whitebox_acc = min(best_whitebox_acc, eot_acc)
    worst_case_candidates = [best_whitebox_acc, fgsm_acc, square_acc, random_acc]
    if bpda_acc is not None:
        worst_case_candidates.append(bpda_acc)
    worst_case_acc = min(worst_case_candidates)

    # Evaluate Carlini 2.0 check flags
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

    whitebox_beats_blackbox = True if norm == "l1" else (square_acc >= best_whitebox_acc - tol_bb)
    monotonic_over_budget = (
        (acc_mid <= acc_low + tol_bb)
        and (acc_high <= acc_mid + tol_bb)
        and (acc_high <= acc_low + tol_bb)
    )
    deterministic_or_eot_verified = not (
        is_stochastic and (stoch_diff > 1e-2 or (eot_acc is not None and eot_acc < pgd_acc - tol_single))
    )
    defense_bpda_consistent = True if bpda_acc is None else (bpda_acc >= naive_whitebox_acc - tol_bb)

    if not autograd_healthy:
        red_flags.append("broken_or_vanishing_gradients")
        recommendations.append(
            "Autograd gradients are broken or vanished on >50% of samples; rely on bpda_eot, square_attack, or hopskipjump instead of PGD."
        )
    if softmax_saturated:
        red_flags.append("softmax_saturation_masking")
        recommendations.append(
            "Softmax saturation detected (APGD-DLR / Margin outperforms CE). Do NOT report CE-PGD accuracy as robustness."
        )
    if not deterministic_or_eot_verified:
        red_flags.append("stochastic_defense_requires_eot")
        recommendations.append(
            f"Test-time stochasticity detected (max output diff={stoch_diff:.3g}); single-pass PGD is unreliable — escalate to eot_pgd / bpda_eot."
        )
    if not defense_bpda_consistent:
        red_flags.append("obfuscated_defense_broken_by_bpda")
        recommendations.append(
            f"BPDA+EoT ({bpda_acc:.3f}) bypassed defense_fn far below naive white-box ({naive_whitebox_acc:.3f}); report BPDA+EoT accuracy."
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

    checks: dict[str, bool] = {
        "preflight_ok": True,
        "eps_scale_meaningful": eps_scale_meaningful,
        "autograd_healthy": autograd_healthy,
        "no_softmax_saturation": no_softmax_saturation,
        "deterministic_or_eot_verified": deterministic_or_eot_verified,
        "iterative_beats_single_step": iterative_beats_single_step,
        "whitebox_beats_blackbox": whitebox_beats_blackbox,
        "whitebox_beats_random": whitebox_beats_random,
        "monotonic_over_budget": monotonic_over_budget,
    }
    if defense_fn is not None:
        checks["defense_bpda_consistent"] = defense_bpda_consistent

    passed = len(red_flags) == 0 and all(checks.values())
    metrics = {
        "task_type": "classification",
        "is_stochastic": is_stochastic,
        "stochastic_max_output_diff": stoch_diff,
        "is_hard_label_only": is_hard_label_only,
        "clean_acc": clean_acc,
        "random_acc": random_acc,
        "fgsm_acc": fgsm_acc,
        "pgd_acc": pgd_acc,
        "apgd_ce_acc": apgd_ce_acc,
        "apgd_dlr_acc": apgd_dlr_acc,
        "square_acc": square_acc,
        "eot_pgd_acc": eot_acc,
        "bpda_eot_acc": bpda_acc,
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
