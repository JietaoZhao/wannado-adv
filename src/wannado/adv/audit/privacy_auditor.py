"""Privacy & Confidentiality Auditor (Membership Inference + Model Extraction Risk Probe).

Standalone zero-SciPy implementation covering EU AI Act Article 15(5) and NIST AI 100-2
(`AML.Privacy.MembershipInference` / `AML.Privacy.ModelExtraction` / MITRE ATLAS `AML.T0024`):
  1. Membership Inference Attack (MIA) Suite:
     - Negative cross-entropy loss attack (Yeom et al., IEEE CSF 2018)
     - Modified Prediction Entropy (Mentr; Song & Mittal, USENIX Security 2021)
     - True-class confidence margin & Carlini et al. (IEEE S&P 2022) low-FPR tail leakage
       (`TPR @ 1% FPR` and `TPR @ 5% FPR`) alongside ROC-AUC and balanced attack advantage.
  2. Black-Box Model Extraction / Knockoff Surrogate Probe (Tramer et al., USENIX Security 2016):
     - Queries the victim strictly as a black box on a bounded query budget, distills a compact
       substitute model, and reports functional agreement fidelity (`extraction_agreement_rate`).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from wannado.adv.core.losses import normalize_classification_logits, predict_labels, unwrap_model_outputs
from wannado.adv.core.types import AttackCitation
from wannado.adv.core.utils import eval_mode, infer_model_device, materialize_eval_tensors

YEOM_MIA_CITATION = AttackCitation(
    key="yeom2018privacy",
    title="Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting",
    authors="Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha",
    venue="IEEE Computer Security Foundations Symposium (CSF)",
    year=2018,
    url="https://arxiv.org/abs/1709.01604",
    bibtex=(
        "@inproceedings{yeom2018privacy,\n"
        "  title={Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting},\n"
        "  author={Yeom, Samuel and Giacomelli, Irene and Fredrikson, Matt and Jha, Somesh},\n"
        "  booktitle={IEEE Computer Security Foundations Symposium (CSF)},\n"
        "  pages={268--282},\n"
        "  year={2018},\n"
        "  url={https://arxiv.org/abs/1709.01604}\n"
        "}"
    ),
    ieee='S. Yeom, I. Giacomelli, M. Fredrikson, and S. Jha, "Privacy risk in machine learning: Analyzing the connection to overfitting," in Proc. IEEE CSF, 2018, pp. 268-282.',
    apa="Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). Privacy risk in machine learning: Analyzing the connection to overfitting. IEEE CSF, 268-282.",
)

SONG_MENTR_CITATION = AttackCitation(
    key="song2021systematic",
    title="Systematic Evaluation of Privacy Risks of Machine Learning Models",
    authors="Liwei Song and Prateek Mittal",
    venue="USENIX Security Symposium",
    year=2021,
    url="https://arxiv.org/abs/2003.10595",
    bibtex=(
        "@inproceedings{song2021systematic,\n"
        "  title={Systematic Evaluation of Privacy Risks of Machine Learning Models},\n"
        "  author={Song, Liwei and Mittal, Prateek},\n"
        "  booktitle={USENIX Security Symposium},\n"
        "  pages={2615--2632},\n"
        "  year={2021},\n"
        "  url={https://arxiv.org/abs/2003.10595}\n"
        "}"
    ),
    ieee='L. Song and P. Mittal, "Systematic evaluation of privacy risks of machine learning models," in Proc. USENIX Security, 2021, pp. 2615-2632.',
    apa="Song, L., & Mittal, P. (2021). Systematic evaluation of privacy risks of machine learning models. USENIX Security, 2615-2632.",
)

CARLINI_LIRA_CITATION = AttackCitation(
    key="carlini2022membership",
    title="Membership Inference Attacks From First Principles",
    authors="Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramer",
    venue="IEEE Symposium on Security and Privacy (S&P)",
    year=2022,
    url="https://arxiv.org/abs/2112.03570",
    bibtex=(
        "@inproceedings{carlini2022membership,\n"
        "  title={Membership Inference Attacks From First Principles},\n"
        "  author={Carlini, Nicholas and Chien, Steve and Nasr, Milad and Song, Shuang and Terzis, Andreas and Tram{\\`e}r, Florian},\n"
        "  booktitle={IEEE Symposium on Security and Privacy (S\\&P)},\n"
        "  pages={1897--1914},\n"
        "  year={2022},\n"
        "  url={https://arxiv.org/abs/2112.03570}\n"
        "}"
    ),
    ieee='N. Carlini et al., "Membership inference attacks from first principles," in Proc. IEEE S&P, 2022, pp. 1897-1914.',
    apa="Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022). Membership inference attacks from first principles. IEEE S&P, 1897-1914.",
)

TRAMER_EXTRACTION_CITATION = AttackCitation(
    key="tramer2016stealing",
    title="Stealing Machine Learning Models via Prediction APIs",
    authors="Florian Tramer, Fan Zhang, Ari Juels, Michael K. Reiter, and Thomas Ristenpart",
    venue="USENIX Security Symposium",
    year=2016,
    url="https://arxiv.org/abs/1609.02943",
    bibtex=(
        "@inproceedings{tramer2016stealing,\n"
        "  title={Stealing Machine Learning Models via Prediction APIs},\n"
        "  author={Tram{\\`e}r, Florian and Zhang, Fan and Juels, Ari and Reiter, Michael K. and Ristenpart, Thomas},\n"
        "  booktitle={USENIX Security Symposium},\n"
        "  pages={601--618},\n"
        "  year={2016},\n"
        "  url={https://arxiv.org/abs/1609.02943}\n"
        "}"
    ),
    ieee='F. Tramer et al., "Stealing machine learning models via prediction APIs," in Proc. USENIX Security, 2016, pp. 601-618.',
    apa="Tramer, F., Zhang, F., Juels, A., Reiter, M. K., & Ristenpart, T. (2016). Stealing machine learning models via prediction APIs. USENIX Security, 601-618.",
)


@dataclass
class PrivacyAuditReport:
    """Structured Membership Inference & Model Extraction privacy audit report."""

    passed: bool
    risk_level: str
    num_members: int
    num_non_members: int
    member_acc: float
    non_member_acc: float
    generalization_gap: float
    mia_loss_auc: float
    mia_mentr_auc: float
    mia_confidence_auc: float
    mia_advantage: float
    mia_tpr_at_1pct_fpr: float
    mia_tpr_at_5pct_fpr: float
    extraction_agreement_rate: float | None = None
    extraction_queries_used: int = 0
    red_flags: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    citations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_report(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"[wannado-adv Privacy Audit: {verdict}] risk={self.risk_level}",
            f"  Generalization : member_acc={self.member_acc * 100:.2f}%, non_member_acc={self.non_member_acc * 100:.2f}%, gap={self.generalization_gap * 100:+.2f}pp",
            f"  MIA Metrics    : loss_AUC={self.mia_loss_auc:.4f}, Mentr_AUC={self.mia_mentr_auc:.4f}, adv={self.mia_advantage:.4f}",
            f"  Tail Leakage   : TPR@1%FPR={self.mia_tpr_at_1pct_fpr * 100:.2f}%, TPR@5%FPR={self.mia_tpr_at_5pct_fpr * 100:.2f}%",
        ]
        if self.extraction_agreement_rate is not None:
            lines.append(
                f"  Extraction Risk: surrogate_agreement={self.extraction_agreement_rate * 100:.2f}% "
                f"({self.extraction_queries_used} queries)"
            )
        if self.red_flags:
            lines.append("  Red Flags:")
            lines.extend(f"    - {rf}" for rf in self.red_flags)
        if self.recommendations:
            lines.append("  Recommendations:")
            lines.extend(f"    - {rec}" for rec in self.recommendations)
        return "\n".join(lines)


def _roc_metrics(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
) -> tuple[float, float, float, float]:
    """Exact pure-NumPy (AUROC, max_advantage, TPR@1%FPR, TPR@5%FPR) for higher-is-member scores."""
    pos = np.asarray(member_scores, dtype=np.float64).reshape(-1)
    neg = np.asarray(nonmember_scores, dtype=np.float64).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        return 0.5, 0.0, 0.0, 0.0

    # Exact Mann-Whitney AUROC
    diff = pos[:, None] - neg[None, :]
    auroc = float(np.mean((diff > 0).astype(np.float64) + 0.5 * (diff == 0).astype(np.float64)))

    # Sweep thresholds at all unique score values for exact ROC curve
    all_scores = np.unique(np.concatenate([pos, neg]))
    # Compute TPR and FPR for strict >= thresholds
    tprs = np.array([float(np.mean(pos >= thr)) for thr in all_scores], dtype=np.float64)
    fprs = np.array([float(np.mean(neg >= thr)) for thr in all_scores], dtype=np.float64)
    max_adv = float(np.max(np.abs(tprs - fprs))) if tprs.size else 0.0

    def _tpr_at_fpr(target_fpr: float) -> float:
        valid = np.flatnonzero(fprs <= target_fpr + 1e-12)
        if valid.size == 0:
            # Strictest threshold above max(neg)
            return float(np.mean(pos > float(np.max(neg))))
        return float(np.max(tprs[valid]))

    return auroc, max_adv, _tpr_at_fpr(0.01), _tpr_at_fpr(0.05)


def _compute_mia_signals(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (neg_loss, neg_mentr, logit_margin, accuracy) where higher score => more likely member.

    Computes loss, Song & Mittal (2021) Modified Prediction Entropy, and Carlini et al. (2022)
    numerically stable log-odds margin ``z_y - logsumexp(z_{j != y})`` in float64 to prevent
    float32 softmax saturation from collapsing low-FPR metrics (``TPR@1%FPR``) on overfit models.
    """
    neg_losses: list[np.ndarray] = []
    neg_mentrs: list[np.ndarray] = []
    logit_margins: list[np.ndarray] = []
    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(0, x.shape[0], max(1, int(batch_size))):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size].reshape(-1).to(torch.long)
            raw_out = unwrap_model_outputs(model_or_fn(xb))
            logits64 = normalize_classification_logits(raw_out).to(torch.float64)
            log_probs = F.log_softmax(logits64, dim=1)
            probs = torch.exp(log_probs).clamp(min=1e-15, max=1.0 - 1e-15)

            b_idx = torch.arange(xb.shape[0], device=xb.device)
            neg_ce = log_probs[b_idx, yb]
            py = probs[b_idx, yb]

            # Carlini et al. (IEEE S&P 2022) numerically stable logit scaling:
            # phi(z, y) = z_y - logsumexp(z_{j != y}) == log(p_y / (1 - p_y)) without float32 saturation.
            one_hot_bool = F.one_hot(yb, num_classes=logits64.shape[1]).to(torch.bool)
            other_logits = logits64.masked_fill(one_hot_bool, float("-inf"))
            lira_margin = logits64[b_idx, yb] - torch.logsumexp(other_logits, dim=1)

            # Song & Mittal (2021) Modified Prediction Entropy (lower Mentr => stronger membership)
            one_hot = one_hot_bool.to(probs.dtype)
            mentr = -(1.0 - py) * torch.log(py) - (
                (1.0 - one_hot) * probs * torch.log((1.0 - probs).clamp(min=1e-15))
            ).sum(dim=1)

            preds = predict_labels(raw_out)
            correct += int((preds == yb).sum().item())
            total += int(xb.shape[0])

            neg_losses.append(neg_ce.detach().cpu().numpy())
            neg_mentrs.append((-mentr).detach().cpu().numpy())
            logit_margins.append(lira_margin.detach().cpu().numpy())

    return (
        np.concatenate(neg_losses, axis=0),
        np.concatenate(neg_mentrs, axis=0),
        np.concatenate(logit_margins, axis=0),
        float(correct / max(total, 1)),
    )


def _run_blackbox_extraction_probe(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_pool: torch.Tensor,
    *,
    num_classes: int,
    query_budget: int,
    epochs: int,
    seed: int,
) -> tuple[float, int]:
    """Train a compact substitute MLP strictly on black-box victim queries and return held-out agreement."""
    n = x_pool.shape[0]
    if n < 8:
        return 0.0, 0

    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=g).to(x_pool.device)
    n_test = max(4, min(n // 3, 64))
    n_train = min(int(query_budget), n - n_test)
    if n_train < 4:
        return 0.0, 0

    x_tr = x_pool[perm[:n_train]]
    x_te = x_pool[perm[n_train : n_train + n_test]]

    with torch.no_grad():
        raw_tr = unwrap_model_outputs(model_or_fn(x_tr))
        raw_te = unwrap_model_outputs(model_or_fn(x_te))
        is_hard = raw_tr.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)
        te_preds = predict_labels(raw_te)
        if is_hard:
            tr_hard = predict_labels(raw_tr)
            tr_probs: torch.Tensor | None = None
        else:
            tr_logits = normalize_classification_logits(raw_tr).to(torch.float32)
            tr_probs = F.softmax(tr_logits, dim=1)
            tr_hard = tr_logits.argmax(dim=1)

    flat_dim = int(x_tr.reshape(n_train, -1).shape[1])
    x_tr_f = x_tr.reshape(n_train, flat_dim).to(torch.float32)
    x_te_f = x_te.reshape(n_test, flat_dim).to(torch.float32)

    torch.manual_seed(seed)
    hid = min(128, max(32, flat_dim))
    surrogate = nn.Sequential(
        nn.Linear(flat_dim, hid),
        nn.ReLU(),
        nn.Linear(hid, num_classes),
    ).to(x_pool.device)
    opt = torch.optim.Adam(surrogate.parameters(), lr=1e-2)

    for _ in range(max(5, int(epochs))):
        opt.zero_grad()
        s_logits = surrogate(x_tr_f)
        if tr_probs is not None:
            loss = F.kl_div(F.log_softmax(s_logits, dim=1), tr_probs, reduction="batchmean")
        else:
            loss = F.cross_entropy(s_logits, tr_hard)
        loss.backward()
        opt.step()

    with torch.no_grad():
        s_preds = surrogate(x_te_f).argmax(dim=1)
        agreement = float((s_preds == te_preds).float().mean().item())
    return agreement, int(n_train)


def audit_privacy(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    member_data: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    non_member_data: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    member_y: torch.Tensor | None = None,
    non_member_y: torch.Tensor | None = None,
    max_samples: int = 256,
    batch_size: int = 64,
    run_extraction_probe: bool = True,
    extraction_queries: int = 128,
    extraction_epochs: int = 20,
    auc_threshold: float = 0.65,
    tpr_at_1pct_fpr_threshold: float = 0.12,
    extraction_agreement_threshold: float = 0.90,
    seed: int = 42,
) -> PrivacyAuditReport:
    """Audit a classifier for Membership Inference (MIA) and Model Extraction privacy risks."""
    t0 = time.perf_counter()
    xm_raw, ym_raw = materialize_eval_tensors(member_data, member_y, max_samples=max_samples)
    xn_raw, yn_raw = materialize_eval_tensors(non_member_data, non_member_y, max_samples=max_samples)
    dev = infer_model_device(model_or_fn, xm_raw.device)
    xm = xm_raw.to(dev)
    ym = ym_raw.to(dev)
    xn = xn_raw.to(dev)
    yn = yn_raw.to(dev)

    if xm.shape[0] < 4 or xn.shape[0] < 4:
        raise ValueError("audit_privacy requires at least 4 member samples and 4 non-member samples.")

    with eval_mode(model_or_fn):
        m_loss, m_mentr, m_conf, m_acc = _compute_mia_signals(model_or_fn, xm, ym, batch_size=batch_size)
        n_loss, n_mentr, n_conf, n_acc = _compute_mia_signals(model_or_fn, xn, yn, batch_size=batch_size)

        loss_auc, loss_adv, loss_tpr1, loss_tpr5 = _roc_metrics(m_loss, n_loss)
        mentr_auc, mentr_adv, mentr_tpr1, mentr_tpr5 = _roc_metrics(m_mentr, n_mentr)
        conf_auc, conf_adv, conf_tpr1, conf_tpr5 = _roc_metrics(m_conf, n_conf)

        best_auc = max(loss_auc, mentr_auc, conf_auc)
        best_adv = max(loss_adv, mentr_adv, conf_adv)
        best_tpr1 = max(loss_tpr1, mentr_tpr1, conf_tpr1)
        best_tpr5 = max(loss_tpr5, mentr_tpr5, conf_tpr5)

        ext_agree: float | None = None
        ext_queries = 0
        if run_extraction_probe and not xn.is_complex():
            with torch.no_grad():
                probe_out = normalize_classification_logits(model_or_fn(xn[: min(4, xn.shape[0])]))
                n_cls = int(probe_out.shape[1])
            ext_agree, ext_queries = _run_blackbox_extraction_probe(
                model_or_fn,
                xn,
                num_classes=n_cls,
                query_budget=extraction_queries,
                epochs=extraction_epochs,
                seed=seed,
            )

    gap = float(m_acc - n_acc)
    red_flags: list[str] = []
    recommendations: list[str] = []

    if best_auc >= float(auc_threshold):
        red_flags.append(
            f"privacy:mia_high_auc(loss={loss_auc:.3f},mentr={mentr_auc:.3f}>={auc_threshold:.2f})"
        )
        recommendations.append(
            "Model memorizes training members (elevated MIA AUC); apply weight decay, label smoothing, "
            "early stopping, or DP-SGD regularization."
        )
    if best_tpr1 >= float(tpr_at_1pct_fpr_threshold) and best_auc >= 0.58:
        red_flags.append(
            f"privacy:mia_tail_leakage_tpr_at_1pct_fpr({best_tpr1:.3f}>={tpr_at_1pct_fpr_threshold:.2f})"
        )
        recommendations.append(
            "High low-FPR membership leakage detected (Carlini LiRA criterion); audit outlier training records "
            "and cap prediction confidence calibration."
        )
    if ext_agree is not None and ext_agree >= float(extraction_agreement_threshold):
        red_flags.append(
            f"privacy:extraction_high_surrogate_fidelity({ext_agree:.3f}>={extraction_agreement_threshold:.2f})"
        )
        recommendations.append(
            "High black-box surrogate agreement on small query budget; restrict raw probability/logit exposure "
            "(top-1 label-only API, output rounding, and query rate limiting)."
        )

    if len(red_flags) >= 2 or best_auc >= 0.75:
        risk_level = "HIGH"
    elif len(red_flags) == 1 or best_auc >= 0.58:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return PrivacyAuditReport(
        passed=len(red_flags) == 0,
        risk_level=risk_level,
        num_members=int(xm.shape[0]),
        num_non_members=int(xn.shape[0]),
        member_acc=round(m_acc, 4),
        non_member_acc=round(n_acc, 4),
        generalization_gap=round(gap, 4),
        mia_loss_auc=round(loss_auc, 4),
        mia_mentr_auc=round(mentr_auc, 4),
        mia_confidence_auc=round(conf_auc, 4),
        mia_advantage=round(best_adv, 4),
        mia_tpr_at_1pct_fpr=round(best_tpr1, 4),
        mia_tpr_at_5pct_fpr=round(best_tpr5, 4),
        extraction_agreement_rate=round(ext_agree, 4) if ext_agree is not None else None,
        extraction_queries_used=ext_queries,
        red_flags=red_flags,
        recommendations=recommendations,
        elapsed_seconds=round(time.perf_counter() - t0, 4),
        citations=[
            asdict(YEOM_MIA_CITATION),
            asdict(SONG_MENTR_CITATION),
            asdict(CARLINI_LIRA_CITATION),
            asdict(TRAMER_EXTRACTION_CITATION),
        ],
    )


__all__ = [
    "CARLINI_LIRA_CITATION",
    "SONG_MENTR_CITATION",
    "TRAMER_EXTRACTION_CITATION",
    "YEOM_MIA_CITATION",
    "PrivacyAuditReport",
    "audit_privacy",
]
