"""Randomized Smoothing: Certified L2 Adversarial Robustness (Cohen et al., ICML 2019).

===============================================================================
[Academic Provenance & Certification Role]
- Tier      : S (Canonical Model-Agnostic Certified L2 Robustness Defense)
- Reference : Cohen, J., Rosenfeld, E., & Kolter, Z. (2019).
              "Certified Adversarial Robustness via Randomized Smoothing."
              ICML 2019 (PMLR 97:1310-1320).
- URL       : https://proceedings.mlr.press/v97/cohen19c.html
- Plain Intuition:
    Any classifier ``f`` can be converted into a provably L2-robust smoothed
    classifier ``g(x) = argmax_c P(f(x + eps) = c)`` with ``eps ~ N(0, sigma^2 I)``.
    Using Monte Carlo draws and an exact one-sided Clopper-Pearson binomial lower
    bound ``p_A_low`` (implemented purely in Python via regularized incomplete beta
    continued fractions + Acklam's inverse normal CDF — ZERO SciPy dependency!),
    if ``p_A_low > 0.5``, no adversary within L2 radius ``R = sigma * Phi^{-1}(p_A_low)``
    can change ``g(x)``'s prediction with confidence >= 1 - alpha.
===============================================================================
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from wannado.adv.core.losses import normalize_classification_logits, predict_labels
from wannado.adv.core.projections import clamp_to_bounds
from wannado.adv.core.types import AttackCitation
from wannado.adv.core.utils import (
    eval_mode,
    infer_model_device,
    iter_train_batches,
    make_generator,
    squeeze_class_targets,
)
from wannado.adv.defenses.types import DefenseMeta, DefenseTrainResult, SmoothingCertificationResult

META = DefenseMeta(
    id="randomized_smoothing",
    name="Randomized Smoothing Certified L2 Defense",
    paradigm="certified_defense",
    tier="S",
    supported_norms=("l2",),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    summary_plain="Certifies provable L2 robustness radii R = sigma * Phi^{-1}(p_A_low) via Monte Carlo Gaussian smoothing and exact Clopper-Pearson bounds.",
    when_to_use="When mathematical L2 robustness certificates (not just empirical attack survival) are required at arbitrary model scale.",
    when_not_to_use="When strict Linf-only robustness is required at high dimension without L2-to-Linf radius conversion.",
    module_path="wannado.adv.defenses.randomized_smoothing_cohen2019",
    citation=AttackCitation(
        key="cohen2019certified",
        title="Certified Adversarial Robustness via Randomized Smoothing",
        authors="Jeremy Cohen, Elan Rosenfeld, and Zico Kolter",
        venue="36th International Conference on Machine Learning (ICML)",
        year=2019,
        url="https://proceedings.mlr.press/v97/cohen19c.html",
        bibtex=(
            "@inproceedings{cohen2019certified,\n"
            "  title={Certified Adversarial Robustness via Randomized Smoothing},\n"
            "  author={Cohen, Jeremy and Rosenfeld, Elan and Kolter, Zico},\n"
            "  booktitle={Proceedings of the 36th International Conference on Machine Learning},\n"
            "  pages={1310--1320},\n"
            "  year={2019},\n"
            "  volume={97},\n"
            "  series={Proceedings of Machine Learning Research},\n"
            "  publisher={PMLR},\n"
            "  url={https://proceedings.mlr.press/v97/cohen19c.html},\n"
            "}"
        ),
        ieee='J. Cohen, E. Rosenfeld, and Z. Kolter, "Certified adversarial robustness via randomized smoothing," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 97, 2019, pp. 1310-1320.',
        apa="Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). Certified adversarial robustness via randomized smoothing. Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97, 1310-1320.",
    ),
)


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF Phi^{-1}(p) via Acklam's rational approximation (<1.15e-9 error)."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def _betacf(a: float, b: float, x: float) -> float:
    """Modified Lentz's algorithm for regularized incomplete beta continued fraction."""
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _reg_incbeta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) = P(Bin(a+b-1, x) >= a)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """Exact one-sided Clopper-Pearson lower confidence bound p_low s.t. P(Bin(n, p_low) >= k) = alpha."""
    if n <= 0 or k <= 0:
        return 0.0
    if k >= n:
        return float(alpha ** (1.0 / float(n)))
    lo = 0.0
    hi = float(k) / float(n)
    a = float(k)
    b = float(n - k + 1)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _reg_incbeta(mid, a, b) < alpha:
            lo = mid
        else:
            hi = mid
    return lo


def _add_isotropic_gaussian(
    x: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None,
    clip_bounds: tuple[Any, Any] | None,
) -> torch.Tensor:
    if x.is_complex():
        real_dtype = x.real.dtype
        std_c = sigma / math.sqrt(2.0)
        re = torch.randn(x.shape, device=x.device, dtype=real_dtype, generator=generator) * std_c
        im = torch.randn(x.shape, device=x.device, dtype=real_dtype, generator=generator) * std_c
        return x + torch.complex(re, im).to(x.dtype)
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * sigma
    out = x + noise
    return clamp_to_bounds(out, clip_bounds) if clip_bounds is not None else out


def gaussian_augmented_train(
    model: nn.Module,
    train_data: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    *,
    sigma: float = 0.25,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    clip_bounds: tuple[Any, Any] | None = None,
    eval_data: tuple[torch.Tensor, torch.Tensor] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    shuffle: bool = True,
    lr_scheduler: str | None = None,
    save_best: bool = False,
    seed: int = 42,
) -> DefenseTrainResult:
    """Train a base classifier under N(0, sigma^2 I) noise augmentation for Randomized Smoothing."""
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}.")
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    dev = next((p.device for p in model.parameters()), torch.device("cpu"))
    opt = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1)) if lr_scheduler == "cosine" else None
    gen = make_generator(seed, dev)
    shuf_gen = make_generator(seed + 1, train_data.device) if isinstance(train_data, torch.Tensor) else None

    history: list[dict[str, float]] = []
    best_metric = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    for ep in range(epochs):
        ep_loss = 0.0
        n_seen = 0
        for xb, yb in iter_train_batches(
            train_data, y, batch_size=batch_size, device=dev, shuffle=shuffle, generator=shuf_gen
        ):
            yb_l = squeeze_class_targets(yb, batch_size=xb.shape[0])
            x_noisy = _add_isotropic_gaussian(xb, sigma, gen, clip_bounds)
            model.train()
            opt.zero_grad()
            logits = normalize_classification_logits(model(x_noisy))
            loss = F.cross_entropy(logits, yb_l)
            loss.backward()
            opt.step()
            bs = int(xb.shape[0])
            ep_loss += float(loss.item()) * bs
            n_seen += bs
        if sched is not None:
            sched.step()
        rec: dict[str, float] = {"epoch": float(ep + 1), "train_loss": ep_loss / max(n_seen, 1)}
        if eval_data is not None:
            xe = eval_data[0].to(dev)
            ye = squeeze_class_targets(eval_data[1].to(dev), batch_size=xe.shape[0])
            with eval_mode(model), torch.no_grad():
                rec["eval_clean_acc"] = float((predict_labels(model(xe)) == ye).float().mean().item())
                xe_noisy = _add_isotropic_gaussian(xe, sigma, gen, clip_bounds)
                rec["eval_noisy_acc"] = float((predict_labels(model(xe_noisy)) == ye).float().mean().item())
            if save_best and rec["eval_noisy_acc"] >= best_metric:
                best_metric = rec["eval_noisy_acc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(rec)

    if save_best and best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    model.eval()
    return DefenseTrainResult(
        defense_id=META.id,
        model=model,
        epochs_run=epochs,
        elapsed_seconds=time.perf_counter() - t0,
        history=history,
        final_clean_acc=history[-1].get("eval_clean_acc") if history else None,
        final_adv_acc=history[-1].get("eval_noisy_acc") if history else None,
        extra={"sigma": float(sigma), "lr": lr},
        citation=META.to_dict()["citation"],
    )


def _binomial_two_sided_pvalue_p05(k: int, m: int) -> float:
    """Exact two-sided binomial p-value for H0: p = 0.5 with observed k >= m/2 out of m trials."""
    if m <= 0:
        return 1.0
    hi = max(k, m - k)
    # P(Bin(m, 0.5) >= hi) = I_{0.5}(hi, m - hi + 1); two-sided p-value is min(1, 2 * upper_tail).
    upper = _reg_incbeta(0.5, float(hi), float(m - hi + 1)) if hi > 0 else 1.0
    return min(1.0, 2.0 * upper)


def _sample_class_counts(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_single: torch.Tensor,
    *,
    num_samples: int,
    sigma: float,
    batch_size: int,
    clip_bounds: tuple[Any, Any] | None,
    generator: torch.Generator | None,
) -> dict[int, int]:
    counts: dict[int, int] = {}
    rem = int(num_samples)
    while rem > 0:
        cur = min(rem, max(1, int(batch_size)))
        reps = x_single.expand((cur,) + x_single.shape[1:])
        noisy = _add_isotropic_gaussian(reps, sigma, generator, clip_bounds)
        preds = predict_labels(model_or_fn(noisy)).detach().cpu().tolist()
        for p in preds:
            pi = int(p)
            counts[pi] = counts.get(pi, 0) + 1
        rem -= cur
    return counts


def smooth_predict(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    sigma: float = 0.25,
    n_samples: int = 100,
    alpha: float | None = None,
    batch_size: int = 256,
    clip_bounds: tuple[Any, Any] | None = None,
    seed: int = 42,
) -> torch.Tensor:
    """Return majority-vote predictions of the Gaussian-smoothed classifier ``g(x)``.

    When ``alpha`` is provided (e.g. ``0.001``), applies Cohen et al. (ICML 2019) Algorithm 2
    two-sided binomial hypothesis test and returns ``-1`` (ABSTAIN) when the top-2 margin is not
    statistically significant at level ``alpha``.
    """
    orig_dev = x.device
    eval_dev = infer_model_device(model_or_fn, orig_dev)
    x_clean = x.detach().to(eval_dev)
    gen = make_generator(seed, eval_dev)
    preds: list[int] = []
    with eval_mode(model_or_fn), torch.no_grad():
        for i in range(x_clean.shape[0]):
            counts = _sample_class_counts(
                model_or_fn,
                x_clean[i : i + 1],
                num_samples=n_samples,
                sigma=sigma,
                batch_size=batch_size,
                clip_bounds=clip_bounds,
                generator=gen,
            )
            ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            best_c, n_a = ranked[0]
            if alpha is not None and len(ranked) >= 2:
                n_b = ranked[1][1]
                if _binomial_two_sided_pvalue_p05(n_a, n_a + n_b) > float(alpha):
                    preds.append(-1)
                    continue
            preds.append(best_c)
    return torch.tensor(preds, device=orig_dev, dtype=torch.long)


def certify_randomized_smoothing(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    sigma: float = 0.25,
    n0: int = 50,
    n: int = 500,
    alpha: float = 0.001,
    batch_size: int = 256,
    radii_grid: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    clip_bounds: tuple[Any, Any] | None = None,
    seed: int = 42,
) -> SmoothingCertificationResult:
    """Certify exact L2 robustness radii via Cohen et al. (ICML 2019) Algorithm 1 (CERTIFY)."""
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")
    if n0 < 1 or n < 1:
        raise ValueError(f"n0 and n must be >= 1, got n0={n0}, n={n}.")
    if not (0.0 < alpha < 0.5):
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}.")

    t0 = time.perf_counter()
    eval_dev = infer_model_device(model_or_fn, x.device)
    x_clean = x.detach().to(eval_dev)
    gen = make_generator(seed, eval_dev)
    # Complex inputs inject N(0, (sigma/sqrt(2))^2) independently per I/Q axis; the isomorphic
    # Euclidean L2 noise scale in R^{2d} is sigma / sqrt(2), so Cohen Theorem 1 certifies R = (sigma/sqrt(2))*Phi^{-1}(p_A).
    eff_sigma = float(sigma) / math.sqrt(2.0) if x_clean.is_complex() else float(sigma)

    preds: list[int] = []
    radii: list[float] = []
    with eval_mode(model_or_fn), torch.no_grad():
        for i in range(x_clean.shape[0]):
            c0 = _sample_class_counts(
                model_or_fn,
                x_clean[i : i + 1],
                num_samples=n0,
                sigma=sigma,
                batch_size=batch_size,
                clip_bounds=clip_bounds,
                generator=gen,
            )
            c_hat = max(c0.items(), key=lambda kv: kv[1])[0]
            cn = _sample_class_counts(
                model_or_fn,
                x_clean[i : i + 1],
                num_samples=n,
                sigma=sigma,
                batch_size=batch_size,
                clip_bounds=clip_bounds,
                generator=gen,
            )
            n_a = cn.get(c_hat, 0)
            p_low = clopper_pearson_lower(n_a, n, alpha)
            if p_low > 0.5:
                preds.append(c_hat)
                radii.append(float(eff_sigma * norm_ppf(p_low)))
            else:
                preds.append(-1)
                radii.append(0.0)

    n_total = max(len(preds), 1)
    abstain_rate = float(sum(1 for p in preds if p == -1) / n_total)
    sorted_r = sorted(radii)
    mid = len(sorted_r) // 2
    median_r = (
        float(sorted_r[mid])
        if len(sorted_r) % 2 == 1
        else float(0.5 * (sorted_r[mid - 1] + sorted_r[mid]))
    )
    mean_r = float(sum(radii) / n_total)

    clean_smoothed_acc: float | None = None
    cert_at_radii: dict[str, float] = {}
    if y is not None:
        y_list = [int(v) for v in y.detach().cpu().reshape(-1).tolist()]
        correct = [p != -1 and p == yt for p, yt in zip(preds, y_list, strict=True)]
        clean_smoothed_acc = float(sum(correct) / n_total)
        for rad in radii_grid:
            cert_at_radii[f"{float(rad):.3g}"] = float(
                sum(1 for ok, r in zip(correct, radii, strict=True) if ok and r >= float(rad)) / n_total
            )
    else:
        for rad in radii_grid:
            cert_at_radii[f"{float(rad):.3g}"] = float(
                sum(1 for p, r in zip(preds, radii, strict=True) if p != -1 and r >= float(rad)) / n_total
            )

    return SmoothingCertificationResult(
        defense_id=META.id,
        sigma=float(sigma),
        alpha=float(alpha),
        n0=int(n0),
        n=int(n),
        num_samples=len(preds),
        clean_smoothed_acc=clean_smoothed_acc,
        abstain_rate=abstain_rate,
        median_certified_radius=median_r,
        mean_certified_radius=mean_r,
        certified_acc_at_radii=cert_at_radii,
        predictions=preds,
        radii=radii,
        elapsed_seconds=time.perf_counter() - t0,
        citation=META.to_dict()["citation"],
    )


__all__ = [
    "META",
    "certify_randomized_smoothing",
    "clopper_pearson_lower",
    "gaussian_augmented_train",
    "norm_ppf",
    "smooth_predict",
]
