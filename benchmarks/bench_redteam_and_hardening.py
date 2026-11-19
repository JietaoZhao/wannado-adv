"""Experiment C: Vectorized FAB Speedup, Autonomous Red-Team Fake-Defense Suite,
Closed-Loop Hardening (PGD-AT / TRADES / MART), and Certified L2 Randomized Smoothing.

Zero external dataset or SciPy dependencies (uses ``torch``, ``numpy``, ``matplotlib`` only).

Run:
    python benchmarks/bench_redteam_and_hardening.py --seed 42
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

from wannado.adv import (
    audit_robustness,
    auto_red_team,
    certify_randomized_smoothing,
    fab,
    gaussian_augmented_train,
    mart_train,
    pgd,
    pgd_adversarial_train,
    trades_train,
)
from wannado.adv.core.utils import DefendedModelWrapper

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "benchmarks" / "assets"
OUTPUTS_DIR = REPO_ROOT / "benchmarks" / "outputs"


class _TinyConvNet(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _StructuredClassifier(nn.Module):
    def __init__(self, in_dim: int = 24, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_structured_dataset(
    n_train: int = 512,
    n_test: int = 96,
    in_dim: int = 24,
    num_classes: int = 4,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    centers = torch.full((num_classes, in_dim), 0.25)
    block = in_dim // num_classes
    for c in range(num_classes):
        centers[c, c * block : (c + 1) * block] = 0.78

    rep_tr = (n_train + num_classes - 1) // num_classes
    y_train = torch.arange(num_classes).repeat(rep_tr)[:n_train]
    x_train = (centers[y_train] + 0.14 * torch.randn((n_train, in_dim), generator=g)).clamp(0.0, 1.0)

    rep_te = (n_test + num_classes - 1) // num_classes
    y_test = torch.arange(num_classes).repeat(rep_te)[:n_test]
    x_test = (centers[y_test] + 0.13 * torch.randn((n_test, in_dim), generator=g)).clamp(0.0, 1.0)
    return x_train, y_train, x_test, y_test


def _train_standard(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int = 40,
    lr: float = 0.015,
    seed: int = 42,
) -> nn.Module:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(x_train), y_train)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _eval_acc(model_or_fn: object, x: torch.Tensor, y: torch.Tensor, repeats: int = 1) -> float:
    fn = model_or_fn
    total = 0.0
    with torch.no_grad():
        for _ in range(repeats):
            out = fn(x)  # type: ignore[operator]
            preds = out.argmax(dim=1) if out.ndim >= 2 else out.long()
            total += float((preds == y).float().mean().item())
    return total / float(repeats)


def _bench_fab_speedup(seed: int) -> list[dict[str, object]]:
    torch.manual_seed(seed)
    conv = _TinyConvNet(num_classes=4).eval()
    # Quick warmup so PyTorch threadpool/kernels are initialized before timing.
    x_warm = torch.rand(4, 3, 16, 16)
    y_warm = conv(x_warm).argmax(dim=1)
    _ = fab(conv, x_warm, y_warm, eps=0.1, steps=4, restarts=1, norm="linf", clip_bounds=(0.0, 1.0), seed=seed)

    rows: list[dict[str, object]] = []
    for batch_n in (8, 16, 32, 64):
        g = torch.Generator().manual_seed(seed + batch_n)
        xb = torch.rand((batch_n, 3, 16, 16), generator=g)
        with torch.no_grad():
            yb = conv(xb).argmax(dim=1)

        t0 = time.perf_counter()
        _ = fab(
            conv,
            xb,
            yb,
            eps=0.1,
            steps=10,
            restarts=1,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            batch_size=1,
            seed=seed,
        )
        t_seq = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        _ = fab(
            conv,
            xb,
            yb,
            eps=0.1,
            steps=10,
            restarts=1,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            batch_size=batch_n,
            seed=seed,
        )
        t_vec = (time.perf_counter() - t1) * 1000.0

        speedup = t_seq / max(t_vec, 1e-6)
        rows.append(
            {
                "panel": "fab_speedup",
                "label": f"B={batch_n}",
                "batch_size": batch_n,
                "seq_ms": round(t_seq, 2),
                "batched_ms": round(t_vec, 2),
                "speedup": round(speedup, 2),
            }
        )
    return rows


def _bench_fake_defenses(
    base_model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    seed: int,
) -> list[dict[str, object]]:
    eps = 0.28
    x_sub = x_test[:32]
    y_sub = y_test[:32]

    rows: list[dict[str, object]] = []

    # 1. Softmax logit saturation (200x)
    def sat_model(z: torch.Tensor) -> torch.Tensor:
        return base_model(z) * 200.0

    clean_sat = _eval_acc(sat_model, x_sub, y_sub)
    pgd_sat = pgd(sat_model, x_sub, y_sub, eps=eps, steps=20, norm="linf", clip_bounds=(0.0, 1.0), seed=seed)
    naive_sat = _eval_acc(sat_model, pgd_sat.x_adv, y_sub)
    rt_sat = auto_red_team(
        sat_model,
        x_sub,
        y_sub,
        eps=eps,
        norm="linf",
        clip_bounds=(0.0, 1.0),
        goal="sanity",
        sweep_curve=False,
        seed=seed,
        model_name="Softmax200x",
        verbose=False,
    )
    rows.append(
        {
            "panel": "fake_defenses",
            "label": "Softmax 200x\n(Saturation)",
            "clean_acc": round(clean_sat, 4),
            "naive_pgd_acc": round(naive_sat, 4),
            "redteam_acc": round(rt_sat.worst_case_robust_acc or 0.0, 4),
            "escalated": ",".join(str(p["attack_id"]) for p in rt_sat.escalation_plan),
            "red_flags": len(rt_sat.audit.red_flags) if rt_sat.audit else 0,
        }
    )

    # 2. Non-differentiable 4-bit quantizer defense_fn
    def quant_4bit(z: torch.Tensor) -> torch.Tensor:
        return torch.round(z * 15.0) / 15.0

    q_wrapped = DefendedModelWrapper(base_model, quant_4bit)
    clean_q = _eval_acc(q_wrapped, x_sub, y_sub)
    pgd_q = pgd(q_wrapped, x_sub, y_sub, eps=eps, steps=20, norm="linf", clip_bounds=(0.0, 1.0), seed=seed)
    naive_q = _eval_acc(q_wrapped, pgd_q.x_adv, y_sub)
    rt_q = auto_red_team(
        base_model,
        x_sub,
        y_sub,
        eps=eps,
        norm="linf",
        clip_bounds=(0.0, 1.0),
        defense_fn=quant_4bit,
        goal="sanity",
        sweep_curve=False,
        seed=seed,
        model_name="Quantizer4Bit",
        verbose=False,
    )
    rows.append(
        {
            "panel": "fake_defenses",
            "label": "4-bit Quantizer\n(Broken Grad)",
            "clean_acc": round(clean_q, 4),
            "naive_pgd_acc": round(naive_q, 4),
            "redteam_acc": round(rt_q.worst_case_robust_acc or 0.0, 4),
            "escalated": ",".join(str(p["attack_id"]) for p in rt_q.escalation_plan),
            "red_flags": len(rt_q.audit.red_flags) if rt_q.audit else 0,
        }
    )

    # 3. Stochastic input noise defense (strong random jitter hides gradient direction from naive single-sample PGD)
    def noisy_defense(z: torch.Tensor) -> torch.Tensor:
        return base_model((z + 0.14 * torch.randn_like(z)).clamp(0.0, 1.0))

    clean_n = _eval_acc(noisy_defense, x_sub, y_sub, repeats=6)
    pgd_n = pgd(noisy_defense, x_sub, y_sub, eps=0.22, steps=10, norm="linf", clip_bounds=(0.0, 1.0), seed=seed)
    naive_n = _eval_acc(noisy_defense, pgd_n.x_adv, y_sub, repeats=6)
    rt_n = auto_red_team(
        noisy_defense,
        x_sub,
        y_sub,
        eps=0.22,
        norm="linf",
        clip_bounds=(0.0, 1.0),
        goal="sanity",
        sweep_curve=False,
        seed=seed,
        model_name="StochasticNoise",
        verbose=False,
    )
    rows.append(
        {
            "panel": "fake_defenses",
            "label": "Stochastic Noise\n(Test Randomness)",
            "clean_acc": round(clean_n, 4),
            "naive_pgd_acc": round(naive_n, 4),
            "redteam_acc": round(rt_n.worst_case_robust_acc or 0.0, 4),
            "escalated": ",".join(str(p["attack_id"]) for p in rt_n.escalation_plan),
            "red_flags": len(rt_n.audit.red_flags) if rt_n.audit else 0,
        }
    )

    # 4. Hard-label-only oracle (zero gradient access; white-box PGD cannot run at all -> reports clean acc)
    x_hl, y_hl = x_test[:16], y_test[:16]

    def hard_label_oracle(z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return base_model(z).argmax(dim=1)

    clean_hl = _eval_acc(hard_label_oracle, x_hl, y_hl)
    # Naive gradient PGD cannot compute cross-entropy on 1-D integer labels (stays at clean baseline).
    naive_hl = clean_hl
    rt_hl = auto_red_team(
        hard_label_oracle,
        x_hl,
        y_hl,
        eps=0.35,
        norm="linf",
        clip_bounds=(0.0, 1.0),
        goal="sanity",
        sweep_curve=False,
        seed=seed,
        model_name="HardLabelOracle",
        verbose=False,
    )
    rows.append(
        {
            "panel": "fake_defenses",
            "label": "Hard-Label\nOracle (No Grad)",
            "clean_acc": round(clean_hl, 4),
            "naive_pgd_acc": round(naive_hl, 4),
            "redteam_acc": round(rt_hl.worst_case_robust_acc or 0.0, 4),
            "escalated": ",".join(str(p["attack_id"]) for p in rt_hl.escalation_plan),
            "red_flags": len(rt_hl.audit.red_flags) if rt_hl.audit else 0,
        }
    )

    return rows


def _bench_hardening(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    seed: int,
) -> list[dict[str, object]]:
    eps = 0.20
    init_model = _StructuredClassifier(in_dim=x_train.shape[1], num_classes=4)

    baseline = _train_standard(copy.deepcopy(init_model), x_train, y_train, epochs=18, lr=0.012, seed=seed)

    pgd_m = copy.deepcopy(init_model)
    pgd_adversarial_train(
        pgd_m,
        x_train,
        y_train,
        eps=eps,
        norm="linf",
        attack_steps=7,
        epochs=30,
        batch_size=64,
        lr=0.012,
        clip_bounds=(0.0, 1.0),
        seed=seed,
    )

    trades_m = copy.deepcopy(init_model)
    trades_train(
        trades_m,
        x_train,
        y_train,
        eps=eps,
        beta=4.0,
        norm="linf",
        attack_steps=7,
        epochs=30,
        batch_size=64,
        lr=0.012,
        clip_bounds=(0.0, 1.0),
        seed=seed,
    )

    mart_m = copy.deepcopy(init_model)
    mart_train(
        mart_m,
        x_train,
        y_train,
        eps=eps,
        beta=3.0,
        norm="linf",
        attack_steps=7,
        epochs=30,
        batch_size=64,
        lr=0.012,
        clip_bounds=(0.0, 1.0),
        seed=seed,
    )

    rows: list[dict[str, object]] = []
    for name, mdl in (
        ("Standard\n(Baseline)", baseline),
        ("PGD-AT\n(Madry 2018)", pgd_m),
        ("TRADES\n(Zhang 2019)", trades_m),
        ("MART\n(Wang 2020)", mart_m),
    ):
        mdl.eval()
        rt = auto_red_team(
            mdl,
            x_test,
            y_test,
            eps=eps,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            goal="sanity",
            sweep_curve=True,
            curve_budgets=[0.0, 0.5 * eps, eps, 1.5 * eps, 2.0 * eps],
            seed=seed,
            model_name=name.replace("\n", " "),
            verbose=False,
        )
        audit = audit_robustness(
            mdl,
            x_test[:24],
            y_test[:24],
            eps=eps,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=seed,
        )
        rows.append(
            {
                "panel": "hardening",
                "label": name,
                "clean_acc": round(rt.clean_acc or 0.0, 4),
                "robust_acc": round(rt.worst_case_robust_acc or 0.0, 4),
                "aurc": round(rt.aurc or 0.0, 4),
                "carlini_passed": bool(audit.passed),
            }
        )
    return rows


def _bench_certified_smoothing(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    seed: int,
) -> tuple[list[float], list[dict[str, object]]]:
    radii_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    x_cert, y_cert = x_test[:48], y_test[:48]
    rows: list[dict[str, object]] = []

    for sigma in (0.15, 0.30, 0.50):
        mdl = _StructuredClassifier(in_dim=x_train.shape[1], num_classes=4)
        gaussian_augmented_train(
            mdl,
            x_train,
            y_train,
            sigma=sigma,
            epochs=35,
            batch_size=64,
            lr=0.015,
            clip_bounds=(0.0, 1.0),
            seed=seed,
        )
        cert = certify_randomized_smoothing(
            mdl,
            x_cert,
            y_cert,
            sigma=sigma,
            n0=80,
            n=600,
            alpha=0.001,
            batch_size=128,
            radii_grid=radii_grid,
            clip_bounds=(0.0, 1.0),
            seed=seed,
        )
        rows.append(
            {
                "panel": "certified_smoothing",
                "sigma": sigma,
                "median_radius": round(cert.median_certified_radius, 4),
                "mean_radius": round(cert.mean_certified_radius, 4),
                "abstain_rate": round(cert.abstain_rate, 4),
                "curve": [float(cert.certified_acc_at_radii[f"{r:.4g}"]) for r in radii_grid],
            }
        )
    return radii_grid, rows


def _plot_all(
    fab_rows: list[dict[str, object]],
    fake_rows: list[dict[str, object]],
    hard_rows: list[dict[str, object]],
    radii_grid: list[float],
    cert_rows: list[dict[str, object]],
    out_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.4), dpi=160)
    fig.suptitle(
        "wannado-adv v1.2.0 — Vectorized FAB, Autonomous Red-Team, Hardening & Certified L2 Robustness",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Batched FAB speedup
    ax = axes[0, 0]
    labels = [str(r["label"]) for r in fab_rows]
    seq_ms = [float(r["seq_ms"]) for r in fab_rows]
    vec_ms = [float(r["batched_ms"]) for r in fab_rows]
    xs = list(range(len(labels)))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in xs], seq_ms, width=w, label="Sample-by-Sample (batch_size=1)", color="#94a3b8")
    b2 = ax.bar([i + w / 2 for i in xs], vec_ms, width=w, label="Vectorized Batched FAB (v1.2)", color="#2563eb")
    for idx, (r, bar_v) in enumerate(zip(fab_rows, b2, strict=True)):
        top_y = max(seq_ms[idx], vec_ms[idx])
        ax.text(
            idx,
            top_y * 1.05,
            f"{float(r['speedup']):.2f}x faster",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#1d4ed8",
        )
    ax.set_xticks(xs, labels)
    ax.set_ylabel("CPU Wall-Clock Time (ms)")
    ax.set_title("A. Vectorized Batched FAB Speedup (3-Layer ConvNet, 10 steps)")
    ax.set_ylim(0, max(seq_ms) * 1.24)
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)
    _ = b1

    # Panel B: auto_red_team breaking 4 fake defenses
    ax = axes[0, 1]
    flabels = [str(r["label"]) for r in fake_rows]
    clean_v = [float(r["clean_acc"]) * 100.0 for r in fake_rows]
    naive_v = [float(r["naive_pgd_acc"]) * 100.0 for r in fake_rows]
    rt_v = [float(r["redteam_acc"]) * 100.0 for r in fake_rows]
    xs = list(range(len(flabels)))
    w = 0.25
    ax.bar([i - w for i in xs], clean_v, width=w, label="Clean Acc", color="#64748b")
    ax.bar(xs, naive_v, width=w, label="Naive PGD-CE (Fooled)", color="#f59e0b")
    bars_rt = ax.bar([i + w for i in xs], rt_v, width=w, label="auto_red_team() Worst-Case", color="#dc2626")
    for bar, r in zip(bars_rt, fake_rows, strict=True):
        val = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 2.0,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#b91c1c",
        )
        _ = r
    ax.set_xticks(xs, flabels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 128)
    ax.set_title("B. auto_red_team() Exposing & Breaking 4 Fake Defenses")
    ax.legend(loc="upper center", ncol=3, frameon=True, fontsize=8.0)

    # Panel C: Closed-loop Hardening (PGD-AT / TRADES / MART) + Carlini 2.0 audit
    ax = axes[1, 0]
    hlabels = [str(r["label"]) for r in hard_rows]
    h_clean = [float(r["clean_acc"]) * 100.0 for r in hard_rows]
    h_rob = [float(r["robust_acc"]) * 100.0 for r in hard_rows]
    xs = list(range(len(hlabels)))
    w = 0.34
    ax.bar([i - w / 2 for i in xs], h_clean, width=w, label="Clean Accuracy", color="#0f766e")
    bars_r = ax.bar([i + w / 2 for i in xs], h_rob, width=w, label="Red-Team Robust Acc (Linf=0.20)", color="#16a34a")
    for i, (r, bar) in enumerate(zip(hard_rows, bars_r, strict=True)):
        badge = "Carlini 2.0 PASS" if r["carlini_passed"] else "Carlini FAIL"
        badge_color = "#15803d" if r["carlini_passed"] else "#b91c1c"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.8,
            f"{bar.get_height():.0f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#166534",
        )
        ax.text(
            i,
            103.5,
            f"{badge}\nAURC={float(r['aurc']):.2f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color=badge_color,
            linespacing=1.15,
        )
    ax.set_xticks(xs, hlabels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 136)
    ax.set_title("C. Closed-Loop Hardening & Carlini 2.0 Re-Audit (Zero Masking)")
    ax.legend(loc="upper center", ncol=2, frameon=True, fontsize=8.2)

    # Panel D: Certified L2 Robustness via Randomized Smoothing (Zero SciPy)
    ax = axes[1, 1]
    palette = {0.15: "#2563eb", 0.30: "#7c3aed", 0.50: "#ea580c"}
    markers = {0.15: "o", 0.30: "s", 0.50: "^"}
    for r in cert_rows:
        sig = float(r["sigma"])
        curve = [v * 100.0 for v in r["curve"]]  # type: ignore[union-attr]
        ax.plot(
            radii_grid,
            curve,
            marker=markers[sig],
            linewidth=2.2,
            markersize=5.5,
            color=palette[sig],
            label=f"sigma={sig:.2f} (median R={float(r['median_radius']):.2f})",
        )
    ax.set_xlabel("Certified L2 Radius R (Cohen et al., ICML 2019; alpha=0.001)")
    ax.set_ylabel("Certified Test Accuracy (%)")
    ax.set_ylim(-3, 108)
    ax.set_title("D. Randomized Smoothing Certified L2 Curves (Pure Python, No SciPy)")
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run wannado-adv v1.2.0 end-to-end benchmark.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.perf_counter()
    print("[bench_redteam_and_hardening] 1/4 Benchmarking vectorized batched FAB speedup...")
    fab_rows = _bench_fab_speedup(args.seed)
    for r in fab_rows:
        print(f"  {r['label']}: seq={r['seq_ms']}ms | batched={r['batched_ms']}ms | {r['speedup']}x")

    x_train, y_train, x_test, y_test = _make_structured_dataset(seed=args.seed)
    base_model = _train_standard(
        _StructuredClassifier(in_dim=x_train.shape[1], num_classes=4),
        x_train,
        y_train,
        epochs=45,
        lr=0.015,
        seed=args.seed,
    )

    print("[bench_redteam_and_hardening] 2/4 Running auto_red_team() against 4 fake defenses...")
    fake_rows = _bench_fake_defenses(base_model, x_test, y_test, args.seed)
    for r in fake_rows:
        lbl = str(r["label"]).replace("\n", " ")
        print(
            f"  {lbl:<34s} | clean={float(r['clean_acc']):.1%} "
            f"| naive_pgd={float(r['naive_pgd_acc']):.1%} "
            f"| redteam={float(r['redteam_acc']):.1%} | escalated=[{r['escalated']}]"
        )

    print("[bench_redteam_and_hardening] 3/4 Training & re-auditing PGD-AT, TRADES, and MART...")
    hard_rows = _bench_hardening(x_train, y_train, x_test, y_test, args.seed)
    for r in hard_rows:
        lbl = str(r["label"]).replace("\n", " ")
        print(
            f"  {lbl:<24s} | clean={float(r['clean_acc']):.1%} "
            f"| robust={float(r['robust_acc']):.1%} | AURC={float(r['aurc']):.3f} "
            f"| carlini_pass={r['carlini_passed']}"
        )

    print("[bench_redteam_and_hardening] 4/4 Running pure-Python Randomized Smoothing L2 certification...")
    radii_grid, cert_rows = _bench_certified_smoothing(x_train, y_train, x_test, y_test, args.seed)
    for r in cert_rows:
        print(
            f"  sigma={r['sigma']} | median_R={r['median_radius']} "
            f"| mean_R={r['mean_radius']} | abstain={float(r['abstain_rate']):.1%}"
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUTS_DIR / "redteam_and_hardening_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["panel", "item", "metric", "value"])
        for r in fab_rows:
            writer.writerow(["fab_speedup", r["label"], "seq_ms", r["seq_ms"]])
            writer.writerow(["fab_speedup", r["label"], "batched_ms", r["batched_ms"]])
            writer.writerow(["fab_speedup", r["label"], "speedup", r["speedup"]])
        for r in fake_rows:
            lbl = str(r["label"]).replace("\n", " ")
            writer.writerow(["fake_defenses", lbl, "clean_acc", r["clean_acc"]])
            writer.writerow(["fake_defenses", lbl, "naive_pgd_acc", r["naive_pgd_acc"]])
            writer.writerow(["fake_defenses", lbl, "redteam_acc", r["redteam_acc"]])
            writer.writerow(["fake_defenses", lbl, "escalated", r["escalated"]])
        for r in hard_rows:
            lbl = str(r["label"]).replace("\n", " ")
            writer.writerow(["hardening", lbl, "clean_acc", r["clean_acc"]])
            writer.writerow(["hardening", lbl, "robust_acc", r["robust_acc"]])
            writer.writerow(["hardening", lbl, "aurc", r["aurc"]])
            writer.writerow(["hardening", lbl, "carlini_passed", r["carlini_passed"]])
        for r in cert_rows:
            sig = f"sigma={r['sigma']}"
            writer.writerow(["certified_smoothing", sig, "median_radius", r["median_radius"]])
            for rad, acc in zip(radii_grid, r["curve"], strict=True):  # type: ignore[arg-type]
                writer.writerow(["certified_smoothing", sig, f"cert_acc_R={rad:.2f}", round(float(acc), 4)])

    png_path = ASSETS_DIR / "redteam_and_hardening.png"
    _plot_all(fab_rows, fake_rows, hard_rows, radii_grid, cert_rows, png_path)
    print(f"\nSaved benchmark chart: {png_path}")
    print(f"Saved metrics CSV    : {csv_path}")
    print(f"Total elapsed        : {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
