"""Supply-Chain Backdoor & Trojan Auditor (Neural Cleanse MAD + STRIP Entropy Test).

Standalone zero-SciPy implementation covering EU AI Act Article 15(5) and NIST AI 100-2
(`AML.Poisoning.Backdoor` / MITRE ATLAS `AML.T0018`):
  1. Reverse-Engineered Trigger Optimization (Wang et al., IEEE S&P 2019 — Neural Cleanse):
     Optimizes a minimal L1 trigger mask + pattern for every candidate target class and computes
     the robust Median Absolute Deviation (MAD) Anomaly Index to expose abnormally small shortcut
     triggers embedded during supply-chain pretraining or poisoned fine-tuning.
  2. STRIP Run-Time Blending Entropy Test (Gao et al., ACSAC 2019):
     Superimposes clean background samples onto inputs and measures normalized Shannon entropy of
     predictions. Trojaned inputs stay rigidly locked to the target class (collapsed entropy)
     compared to clean inputs.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.adv.core.losses import normalize_classification_logits, predict_labels
from wannado.adv.core.projections import coerce_clip_bounds
from wannado.adv.core.types import AttackCitation
from wannado.adv.core.utils import eval_mode, infer_model_device, make_generator, materialize_eval_tensors

NEURAL_CLEANSE_CITATION = AttackCitation(
    key="wang2019neural",
    title="Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks",
    authors="Bolun Wang, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao Zheng, and Ben Y. Zhao",
    venue="IEEE Symposium on Security and Privacy (S&P)",
    year=2019,
    url="https://ieeexplore.ieee.org/document/8835365",
    bibtex=(
        "@inproceedings{wang2019neural,\n"
        "  title={Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks},\n"
        "  author={Wang, Bolun and Yao, Yuanshun and Shan, Shawn and Li, Huiying and Viswanath, Bimal and Zheng, Haitao and Zhao, Ben Y.},\n"
        "  booktitle={IEEE Symposium on Security and Privacy (S\\&P)},\n"
        "  pages={707--723},\n"
        "  year={2019},\n"
        "  url={https://ieeexplore.ieee.org/document/8835365}\n"
        "}"
    ),
    ieee='B. Wang et al., "Neural Cleanse: Identifying and mitigating backdoor attacks in neural networks," in Proc. IEEE S&P, 2019, pp. 707-723.',
    apa="Wang, B., Yao, Y., Shan, S., Li, H., Viswanath, B., Zheng, H., & Zhao, B. Y. (2019). Neural Cleanse: Identifying and mitigating backdoor attacks in neural networks. IEEE S&P, 707-723.",
)

STRIP_CITATION = AttackCitation(
    key="gao2019strip",
    title="STRIP: A Defence Against Trojan Attacks on Deep Neural Networks",
    authors="Yansong Gao, Chang Xu, Derui Wang, Shiping Chen, Damith C. Ranasinghe, and Surya Nepal",
    venue="Annual Computer Security Applications Conference (ACSAC)",
    year=2019,
    url="https://arxiv.org/abs/1902.06531",
    bibtex=(
        "@inproceedings{gao2019strip,\n"
        "  title={STRIP: A Defence Against Trojan Attacks on Deep Neural Networks},\n"
        "  author={Gao, Yansong and Xu, Chang and Wang, Derui and Chen, Shiping and Ranasinghe, Damith C. and Nepal, Surya},\n"
        "  booktitle={Annual Computer Security Applications Conference (ACSAC)},\n"
        "  pages={113--125},\n"
        "  year={2019},\n"
        "  url={https://arxiv.org/abs/1902.06531}\n"
        "}"
    ),
    ieee='Y. Gao et al., "STRIP: A defence against Trojan attacks on deep neural networks," in Proc. ACSAC, 2019, pp. 113-125.',
    apa="Gao, Y., Xu, C., Wang, D., Chen, S., Ranasinghe, D. C., & Nepal, S. (2019). STRIP: A defence against Trojan attacks on deep neural networks. ACSAC, 113-125.",
)


@dataclass
class BackdoorAuditReport:
    """Structured backdoor / supply-chain Trojan audit verdict."""

    passed: bool
    suspected_backdoor: bool
    suspect_target_classes: list[int] = field(default_factory=list)
    anomaly_indices: dict[int, float] = field(default_factory=dict)
    trigger_l1_norms: dict[int, float] = field(default_factory=dict)
    trigger_target_asrs: dict[int, float] = field(default_factory=dict)
    max_anomaly_index: float = 0.0
    anomaly_threshold: float = 2.0
    strip_clean_mean_entropy: float | None = None
    strip_suspect_mean_entropy: float | None = None
    strip_auroc: float | None = None
    strip_low_entropy_fraction: float | None = None
    red_flags: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    citations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_report(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"[wannado-adv Backdoor Audit: {verdict}]",
            f"  Max MAD Anomaly Index : {self.max_anomaly_index:.3f} (threshold={self.anomaly_threshold:.2f})",
            f"  Suspect Target Classes: {self.suspect_target_classes or 'None'}",
        ]
        if self.strip_clean_mean_entropy is not None:
            s_info = f"clean={self.strip_clean_mean_entropy:.3f}"
            if self.strip_suspect_mean_entropy is not None:
                s_info += f", suspect={self.strip_suspect_mean_entropy:.3f}"
            if self.strip_auroc is not None:
                s_info += f", AUROC={self.strip_auroc:.3f}"
            lines.append(f"  STRIP Blending Entropy: {s_info}")
        if self.red_flags:
            lines.append("  Red Flags:")
            lines.extend(f"    - {rf}" for rf in self.red_flags)
        if self.recommendations:
            lines.append("  Recommendations:")
            lines.extend(f"    - {rec}" for rec in self.recommendations)
        return "\n".join(lines)


def _pure_numpy_auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Exact Mann-Whitney AUROC (pos > neg) without SciPy or scikit-learn."""
    if pos_scores.size == 0 or neg_scores.size == 0:
        return 0.5
    diff = pos_scores[:, None] - neg_scores[None, :]
    return float(np.mean((diff > 0).astype(np.float64) + 0.5 * (diff == 0).astype(np.float64)))


def _compute_strip_entropies(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_eval: torch.Tensor,
    x_bg_pool: torch.Tensor,
    *,
    n_blend: int = 16,
    alpha: float = 0.5,
    num_classes: int,
    gen: torch.Generator | None,
) -> np.ndarray:
    """Compute normalized STRIP prediction entropy per sample in x_eval."""
    n = x_eval.shape[0]
    n_bg = x_bg_pool.shape[0]
    if n == 0 or n_bg == 0:
        return np.zeros((0,), dtype=np.float64)
    k = max(2, min(int(n_blend), n_bg))
    log_c = math.log(max(2, int(num_classes)))
    entropies = np.zeros((n,), dtype=np.float64)
    with torch.no_grad():
        for i in range(n):
            idx = torch.randint(0, n_bg, (k,), generator=gen, device=x_bg_pool.device)
            blended = (1.0 - float(alpha)) * x_eval[i : i + 1].expand_as(x_bg_pool[idx]) + float(alpha) * x_bg_pool[idx]
            logits = normalize_classification_logits(model_or_fn(blended))
            probs = F.softmax(logits, dim=1).clamp(min=1e-12)
            h = -(probs * probs.log()).sum(dim=1) / log_c
            entropies[i] = float(h.mean().item())
    return entropies


def audit_backdoor(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_clean: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y_clean: torch.Tensor | None = None,
    *,
    x_suspect: torch.Tensor | None = None,
    num_classes: int | None = None,
    target_classes: list[int] | None = None,
    max_samples: int = 64,
    steps: int = 80,
    lr: float = 0.15,
    init_cost: float = 1e-3,
    clip_bounds: tuple[Any, Any] | None = (0.0, 1.0),
    strip_n_blend: int = 16,
    strip_alpha: float = 0.5,
    anomaly_threshold: float = 2.0,
    min_target_asr: float = 0.85,
    strip_entropy_ratio_threshold: float = 0.55,
    seed: int = 42,
) -> BackdoorAuditReport:
    """Audit a classifier for supply-chain Trojan backdoors via Neural Cleanse MAD + STRIP entropy."""
    t0 = time.perf_counter()
    x_mat, y_mat = materialize_eval_tensors(x_clean, y_clean, max_samples=max_samples)
    dev = infer_model_device(model_or_fn, x_mat.device)
    x_t = x_mat.to(dev)
    y_t = y_mat.to(dev).reshape(-1).to(torch.long)

    if x_t.is_complex():
        raise ValueError("audit_backdoor currently requires real-valued classification inputs.")
    if x_t.shape[0] < 4:
        raise ValueError("audit_backdoor requires at least 4 clean samples.")

    with eval_mode(model_or_fn):
        with torch.no_grad():
            probe_logits = normalize_classification_logits(model_or_fn(x_t[: min(4, x_t.shape[0])]))
            inferred_classes = int(probe_logits.shape[1])
        n_classes = int(num_classes) if num_classes is not None else inferred_classes
        cand_targets = list(target_classes) if target_classes is not None else list(range(n_classes))

        # Determine mask spatial geometry (channel-shared (1, 1, H, W) for images, element-wise otherwise)
        mask_shape: tuple[int, ...]
        if x_t.ndim == 4:
            mask_shape = (1, 1, int(x_t.shape[2]), int(x_t.shape[3]))
        else:
            mask_shape = (1,) + tuple(int(s) for s in x_t.shape[1:])
        pat_shape = (1,) + tuple(int(s) for s in x_t.shape[1:])
        mask_numel = int(math.prod(mask_shape))

        lo_t: torch.Tensor | None = None
        hi_t: torch.Tensor | None = None
        if clip_bounds is not None:
            lo_t, hi_t = coerce_clip_bounds(clip_bounds, x_t[:1])

        gen = make_generator(seed, dev)
        l1_norms: dict[int, float] = {}
        target_asrs: dict[int, float] = {}
        reached_flags: dict[int, bool] = {}

        # Freeze model parameter grads during trigger reverse-engineering
        saved_req: list[tuple[torch.nn.Parameter, bool]] = []
        if isinstance(model_or_fn, torch.nn.Module):
            for p in model_or_fn.parameters():
                saved_req.append((p, p.requires_grad))
                p.requires_grad_(False)

        try:
            for t_cls in cand_targets:
                non_target_idx = (y_t != int(t_cls)).nonzero(as_tuple=False).reshape(-1)
                x_opt = x_t[non_target_idx] if non_target_idx.numel() >= 4 else x_t
                y_tgt = torch.full((x_opt.shape[0],), int(t_cls), dtype=torch.long, device=dev)

                # Initialize mask near 0.5 (logit=0) and pattern mid-range
                mask_raw = torch.zeros(mask_shape, dtype=x_t.dtype, device=dev, requires_grad=True)
                pat_raw = (
                    torch.randn(pat_shape, dtype=x_t.dtype, device=dev, generator=gen) * 0.1
                ).detach().requires_grad_(True)
                opt = torch.optim.Adam([mask_raw, pat_raw], lr=lr)

                cost = float(init_cost)
                up_cnt = 0
                down_cnt = 0
                best_l1 = float("inf")
                best_asr = 0.0

                for _ in range(max(10, int(steps))):
                    mask = torch.sigmoid(mask_raw)
                    if lo_t is not None and hi_t is not None:
                        pattern = lo_t + (hi_t - lo_t) * torch.sigmoid(pat_raw)
                    else:
                        pattern = pat_raw

                    x_trig = (1.0 - mask) * x_opt + mask * pattern
                    logits = normalize_classification_logits(model_or_fn(x_trig))
                    ce_loss = F.cross_entropy(logits, y_tgt)
                    l1_val = mask.sum()
                    loss = ce_loss + cost * l1_val

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    with torch.no_grad():
                        preds = predict_labels(logits)
                        asr = float((preds == y_tgt).float().mean().item())
                        cur_l1 = float(l1_val.item())
                        if asr >= min_target_asr and cur_l1 < best_l1:
                            best_l1 = cur_l1
                            best_asr = asr
                        elif best_l1 == float("inf") and asr > best_asr:
                            best_asr = asr

                        if asr >= min_target_asr:
                            up_cnt += 1
                            down_cnt = 0
                        else:
                            down_cnt += 1
                            up_cnt = 0
                        if up_cnt >= 3:
                            cost = min(cost * 1.5, 10.0)
                            up_cnt = 0
                        elif down_cnt >= 3:
                            cost = max(cost / 1.5, 1e-6)
                            down_cnt = 0

                if math.isfinite(best_l1):
                    l1_norms[int(t_cls)] = round(best_l1, 5)
                    reached_flags[int(t_cls)] = True
                else:
                    l1_norms[int(t_cls)] = round(float(torch.sigmoid(mask_raw).sum().item()), 5)
                    reached_flags[int(t_cls)] = False
                target_asrs[int(t_cls)] = round(best_asr, 4)
        finally:
            for p, was_req in saved_req:
                p.requires_grad_(was_req)

        # Compute MAD Anomaly Index (Neural Cleanse, Wang et al. 2019)
        anomaly_indices: dict[int, float] = {}
        suspect_classes: list[int] = []
        vals = np.array([l1_norms[c] for c in cand_targets], dtype=np.float64)

        if len(cand_targets) >= 3:
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            # Floor MAD at 5% of median so near-identical clean classes never blow up to Inf
            scale = max(1.4826 * mad, 0.05 * max(med, 1e-3), 1e-4)
            for c in cand_targets:
                # Only abnormally SMALL triggers (below median) indicate a Trojan shortcut
                if reached_flags[c] and l1_norms[c] < med:
                    ai = float((med - l1_norms[c]) / scale)
                else:
                    ai = 0.0
                anomaly_indices[int(c)] = round(ai, 4)
                if ai > float(anomaly_threshold) and l1_norms[c] <= 0.45 * mask_numel:
                    suspect_classes.append(int(c))
        elif len(cand_targets) == 2:
            c0, c1 = cand_targets[0], cand_targets[1]
            v0, v1 = l1_norms[c0], l1_norms[c1]
            for c_lo, v_lo, v_hi in ((c0, v0, v1), (c1, v1, v0)):
                if reached_flags[c_lo] and v_hi > 1e-6 and v_lo < v_hi:
                    ratio_score = float((v_hi - v_lo) / max(v_lo, 0.05 * mask_numel))
                else:
                    ratio_score = 0.0
                anomaly_indices[int(c_lo)] = round(ratio_score, 4)
                if (
                    reached_flags[c_lo]
                    and v_lo < 0.35 * v_hi
                    and v_lo < 0.25 * mask_numel
                    and ratio_score > float(anomaly_threshold)
                ):
                    suspect_classes.append(int(c_lo))
        else:
            for c in cand_targets:
                anomaly_indices[int(c)] = 0.0

        # STRIP run-time blending entropy audit
        clean_ent = _compute_strip_entropies(
            model_or_fn,
            x_t[: min(32, x_t.shape[0])],
            x_t,
            n_blend=strip_n_blend,
            alpha=strip_alpha,
            num_classes=n_classes,
            gen=gen,
        )
        strip_clean_mean = float(np.mean(clean_ent)) if clean_ent.size else None
        strip_suspect_mean: float | None = None
        strip_auroc: float | None = None
        strip_low_frac: float | None = None

        red_flags: list[str] = []
        recommendations: list[str] = []

        if suspect_classes:
            red_flags.append(f"backdoor:neural_cleanse_anomaly_classes={suspect_classes}")
            recommendations.append(
                f"Inspect training data & weights for Trojan shortcut targeting class(es) {suspect_classes}; "
                "apply trigger unlearning, Fine-Pruning, or retrain from verified clean data."
            )

        if x_suspect is not None and x_suspect.shape[0] > 0:
            xs_t = x_suspect[: min(32, x_suspect.shape[0])].to(dev)
            susp_ent = _compute_strip_entropies(
                model_or_fn,
                xs_t,
                x_t,
                n_blend=strip_n_blend,
                alpha=strip_alpha,
                num_classes=n_classes,
                gen=gen,
            )
            strip_suspect_mean = float(np.mean(susp_ent))
            # Lower entropy = higher Trojan suspicion -> negate for AUROC(suspect > clean)
            strip_auroc = _pure_numpy_auroc(-susp_ent, -clean_ent)
            low_thresh = float(np.percentile(clean_ent, 10)) if clean_ent.size else 0.1
            strip_low_frac = float(np.mean(susp_ent < low_thresh))
            if (
                strip_clean_mean is not None
                and strip_clean_mean > 0.05
                and (strip_suspect_mean < strip_entropy_ratio_threshold * strip_clean_mean or strip_auroc >= 0.85)
            ):
                red_flags.append(
                    f"backdoor:strip_low_entropy_trojan_detected(auroc={strip_auroc:.3f},"
                    f"suspect_H={strip_suspect_mean:.3f},clean_H={strip_clean_mean:.3f})"
                )
                recommendations.append(
                    "Suspect batch exhibits collapsed STRIP blending entropy consistent with an input trigger; "
                    "quarantine input source and run trigger reverse-engineering."
                )

    max_ai = max(anomaly_indices.values()) if anomaly_indices else 0.0
    suspected = len(red_flags) > 0
    return BackdoorAuditReport(
        passed=not suspected,
        suspected_backdoor=suspected,
        suspect_target_classes=suspect_classes,
        anomaly_indices=anomaly_indices,
        trigger_l1_norms=l1_norms,
        trigger_target_asrs=target_asrs,
        max_anomaly_index=round(float(max_ai), 4),
        anomaly_threshold=float(anomaly_threshold),
        strip_clean_mean_entropy=round(strip_clean_mean, 4) if strip_clean_mean is not None else None,
        strip_suspect_mean_entropy=round(strip_suspect_mean, 4) if strip_suspect_mean is not None else None,
        strip_auroc=round(strip_auroc, 4) if strip_auroc is not None else None,
        strip_low_entropy_fraction=round(strip_low_frac, 4) if strip_low_frac is not None else None,
        red_flags=red_flags,
        recommendations=recommendations,
        elapsed_seconds=round(time.perf_counter() - t0, 4),
        citations=[asdict(NEURAL_CLEANSE_CITATION), asdict(STRIP_CITATION)],
    )


__all__ = [
    "NEURAL_CLEANSE_CITATION",
    "STRIP_CITATION",
    "BackdoorAuditReport",
    "audit_backdoor",
]
