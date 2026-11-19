"""Experiment A: End-to-End CIFAR-10 / RobustBench Alignment, MPS Audit, & False-Robustness Benchmark.

Compares ``wannado-adv`` side-by-side against:
  1. Official ``robustbench`` pretrained CIFAR-10 models (Standard + Wong2020Fast).
  2. ``torchattacks`` reference implementations (ASR & runtime parity at eps=8/255).
  3. Official ``autoattack`` package (APGD-CE + APGD-DLR + FAB + Square ensemble parity).
  4. Carlini false-robustness traps (Softmax logit inflation & Non-differentiable 4-bit ADC quantizer).
  5. Apple Silicon MPS device compatibility audit across core Wave 1 & Wave 2 families.

Run with isolated bench_venv:
    /path/to/bench_venv/bin/python benchmarks/bench_cv_robustbench.py --samples 200 --device auto
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchattacks
from autoattack import AutoAttack as OfficialAutoAttack
from robustbench.data import load_cifar10
from robustbench.utils import load_model
from torch import nn

from wannado.adv import (
    adv_patch,
    advgan,
    apgd_ce,
    apgd_dlr,
    apgd_l1,
    audit_robustness,
    boundary,
    bpda_eot,
    compute_perturbation_norm,
    di_fgsm,
    ead,
    fab,
    fgsm,
    gap,
    hopskipjump,
    jsma,
    mifgsm,
    nes,
    ni_fgsm,
    pgd,
    preflight,
    simba,
    sini_fgsm,
    sparsefool,
    square_attack,
    ti_fgsm,
    tpgd,
    vmi_fgsm,
    zoo,
)
from wannado.adv import (
    autoattack as wd_autoattack,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "benchmarks" / "data"
CKPT_DIR = REPO_ROOT / "benchmarks" / "checkpoints"
ASSETS_DIR = REPO_ROOT / "benchmarks" / "assets"
OUTPUTS_DIR = REPO_ROOT / "benchmarks" / "outputs"


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _eval_acc(model: nn.Module | object, x: torch.Tensor, y: torch.Tensor, batch_size: int = 100) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            correct += int((model(xb).argmax(dim=1) == yb).sum().item())
            total += yb.numel()
    return float(correct) / max(total, 1)


def _run_mps_compatibility_smoke(model: nn.Module, x_small: torch.Tensor, y_small: torch.Tensor) -> list[tuple[str, bool, str]]:
    """Smoke-test representative attacks directly on MPS tensors to expose device bugs."""
    results: list[tuple[str, bool, str]] = []
    eps = 8.0 / 255.0
    tests = [
        ("fgsm", lambda: fgsm(model, x_small, y_small, eps=eps, clip_bounds=(0.0, 1.0))),
        ("pgd", lambda: pgd(model, x_small, y_small, eps=eps, steps=5, clip_bounds=(0.0, 1.0), seed=0)),
        ("mifgsm", lambda: mifgsm(model, x_small, y_small, eps=eps, steps=5, clip_bounds=(0.0, 1.0))),
        ("ni_fgsm", lambda: ni_fgsm(model, x_small, y_small, eps=eps, steps=5, clip_bounds=(0.0, 1.0))),
        ("sini_fgsm", lambda: sini_fgsm(model, x_small, y_small, eps=eps, steps=4, scale_copies=3, clip_bounds=(0.0, 1.0))),
        ("di_fgsm", lambda: di_fgsm(model, x_small, y_small, eps=eps, steps=4, clip_bounds=(0.0, 1.0), seed=0)),
        ("ti_fgsm", lambda: ti_fgsm(model, x_small, y_small, eps=eps, steps=4, clip_bounds=(0.0, 1.0))),
        ("vmi_fgsm", lambda: vmi_fgsm(model, x_small, y_small, eps=eps, steps=4, num_variance_samples=3, clip_bounds=(0.0, 1.0), seed=0)),
        ("tpgd", lambda: tpgd(model, x_small, eps=eps, steps=4, clip_bounds=(0.0, 1.0), seed=0)),
        ("apgd_ce", lambda: apgd_ce(model, x_small, y_small, eps=eps, steps=5, clip_bounds=(0.0, 1.0), seed=0)),
        ("apgd_dlr", lambda: apgd_dlr(model, x_small, y_small, eps=eps, steps=5, clip_bounds=(0.0, 1.0), seed=0)),
        ("apgd_l1", lambda: apgd_l1(model, x_small, y_small, eps=8.0, steps=6, clip_bounds=(0.0, 1.0), seed=0)),
        ("ead", lambda: ead(model, x_small[:2], y_small[:2], eps=8.0, steps=6, binary_search_steps=2, clip_bounds=(0.0, 1.0))),
        ("sparsefool", lambda: sparsefool(model, x_small[:2], y_small[:2], max_iter=3, deepfool_steps=5, clip_bounds=(0.0, 1.0))),
        ("jsma", lambda: jsma(model, x_small[:1], y_small[:1], theta=0.2, gamma=0.05, targeted=False, clip_bounds=(0.0, 1.0))),
        ("fab", lambda: fab(model, x_small[:2], y_small[:2], eps=eps, steps=4, clip_bounds=(0.0, 1.0), seed=0)),
        ("zoo", lambda: zoo(model, x_small[:2], y_small[:2], eps=1.5, steps=3, coord_batch_size=16, clip_bounds=(0.0, 1.0), seed=0)),
        ("nes", lambda: nes(model, x_small[:2], y_small[:2], eps=eps, steps=3, nb_sample=8, clip_bounds=(0.0, 1.0), seed=0)),
        ("simba", lambda: simba(model, x_small[:2], y_small[:2], eps=eps, max_queries=20, clip_bounds=(0.0, 1.0), seed=0)),
        ("square_attack", lambda: square_attack(model, x_small, y_small, eps=eps, max_queries=20, clip_bounds=(0.0, 1.0), seed=0)),
        ("boundary", lambda: boundary(lambda z: model(z).argmax(1), x_small[:2], y_small[:2], eps=2.0, steps=8, clip_bounds=(0.0, 1.0), seed=0)),
        ("hopskipjump", lambda: hopskipjump(lambda z: model(z).argmax(1), x_small[:2], y_small[:2], eps=eps, steps=3, initial_num_evals=8, clip_bounds=(0.0, 1.0), seed=0)),
        ("bpda_eot", lambda: bpda_eot(model, x_small[:2], y_small[:2], eps=eps, steps=4, eot_samples=2, defense_fn=lambda z: torch.round(z*15)/15, surrogate_fn=lambda z: z, clip_bounds=(0.0, 1.0), seed=0)),
        ("adv_patch", lambda: adv_patch(model, x_small[:2], y_small[:2], eps=eps, steps=4, random_location=True, clip_bounds=(0.0, 1.0), seed=0)),
        ("gap", lambda: gap(model, x_small, y_small, eps=eps, epochs=3, clip_bounds=(0.0, 1.0), seed=0)),
        ("advgan", lambda: advgan(model, x_small, y_small, eps=eps, epochs=3, clip_bounds=(0.0, 1.0), seed=0)),
    ]
    for name, fn in tests:
        try:
            res = fn()
            _sync(x_small.device)
            ok = res.x_adv.device.type == x_small.device.type
            results.append((name, ok, "" if ok else f"device={res.x_adv.device}"))
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
    return results


def run_benchmark(num_samples: int = 200, device_arg: str = "auto", seed: int = 42) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if device_arg == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    print(f"[Experiment A] Running CIFAR-10 RobustBench benchmark on device={device} (N={num_samples})")

    x_test, y_test = load_cifar10(n_examples=num_samples, data_dir=str(DATA_DIR))
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    eps = 8.0 / 255.0
    step_size = 2.0 / 255.0
    rows: list[dict[str, str | float]] = []

    models_to_eval = [
        ("Standard", "Undefended WRN-28-10"),
        ("Wong2020Fast", "Fast-AT PreActResNet-18 (Official AA=43.2%)"),
    ]

    for model_name, desc in models_to_eval:
        print(f"\n=== Loading RobustBench Model: {model_name} ({desc}) ===")
        model = load_model(
            model_name=model_name,
            model_dir=str(CKPT_DIR),
            dataset="cifar10",
            threat_model="Linf",
        ).to(device).eval()

        clean_acc = _eval_acc(model, x_test, y_test)
        print(f"Clean Acc ({num_samples} CIFAR-10): {clean_acc * 100:.2f}%")

        # Run MPS device audit once on Standard model
        if model_name == "Standard" and device.type == "mps":
            print("\n[MPS Device Audit across 26 representative Wave 1+2 attacks]")
            smoke = _run_mps_compatibility_smoke(model, x_test[:4], y_test[:4])
            failed = [(n, msg) for n, ok, msg in smoke if not ok]
            print(f"  MPS Smoke Result: {len(smoke) - len(failed)}/{len(smoke)} passed.")
            for n, msg in failed:
                print(f"  [MPS FAIL] {n}: {msg}")

        # 1. AI-Native Preflight & Carlini Robustness Audit
        pf = preflight(model, x_test[:16], y_test[:16], eps=eps, norm="linf", clip_bounds=(0.0, 1.0), attack="autoattack")
        print(f"  preflight(autoattack) -> ok={pf.ok}, warnings={pf.warnings}")
        audit = audit_robustness(model, x_test[:64], y_test[:64], eps=eps, norm="linf", clip_bounds=(0.0, 1.0), seed=seed)
        print(f"  audit_robustness(N=64) -> passed={audit.passed}, red_flags={audit.red_flags}")

        # 2. Side-by-side comparison: wannado-adv vs torchattacks vs official AutoAttack
        pair_specs = [
            (
                "FGSM",
                lambda m=model: fgsm(m, x_test, y_test, eps=eps, clip_bounds=(0.0, 1.0)),
                lambda m=model: torchattacks.FGSM(m, eps=eps)(x_test, y_test),
            ),
            (
                "PGD-10",
                lambda m=model: pgd(m, x_test, y_test, eps=eps, step_size=step_size, steps=10, clip_bounds=(0.0, 1.0), seed=seed),
                lambda m=model: torchattacks.PGD(m, eps=eps, alpha=step_size, steps=10, random_start=True)(x_test, y_test),
            ),
            (
                "MI-FGSM-10",
                lambda m=model: mifgsm(m, x_test, y_test, eps=eps, step_size=step_size, steps=10, decay=1.0, clip_bounds=(0.0, 1.0)),
                lambda m=model: torchattacks.MIFGSM(m, eps=eps, alpha=step_size, steps=10, decay=1.0)(x_test, y_test),
            ),
            (
                "NI-FGSM-10",
                lambda m=model: ni_fgsm(m, x_test, y_test, eps=eps, step_size=step_size, steps=10, decay=1.0, clip_bounds=(0.0, 1.0)),
                lambda m=model: torchattacks.NIFGSM(m, eps=eps, alpha=step_size, steps=10, decay=1.0)(x_test, y_test),
            ),
            (
                "DI-FGSM-10",
                lambda m=model: di_fgsm(m, x_test, y_test, eps=eps, step_size=step_size, steps=10, decay=0.0, diversity_prob=0.5, clip_bounds=(0.0, 1.0), seed=seed),
                lambda m=model: torchattacks.DIFGSM(m, eps=eps, alpha=step_size, steps=10, decay=0.0, diversity_prob=0.5)(x_test, y_test),
            ),
            (
                "APGD-CE-20",
                lambda m=model: apgd_ce(m, x_test, y_test, eps=eps, steps=20, restarts=1, clip_bounds=(0.0, 1.0), seed=seed),
                lambda m=model: torchattacks.APGD(m, norm="Linf", eps=eps, steps=20, n_restarts=1, loss="ce", seed=seed)(x_test, y_test),
            ),
            (
                "APGD-DLR-20",
                lambda m=model: apgd_dlr(m, x_test, y_test, eps=eps, steps=20, restarts=1, clip_bounds=(0.0, 1.0), seed=seed),
                lambda m=model: torchattacks.APGD(m, norm="Linf", eps=eps, steps=20, n_restarts=1, loss="dlr", seed=seed)(x_test, y_test),
            ),
            (
                "Square-500q",
                lambda m=model: square_attack(m, x_test, y_test, eps=eps, max_queries=500, clip_bounds=(0.0, 1.0), seed=seed),
                lambda m=model: torchattacks.Square(m, norm="Linf", eps=eps, n_queries=500, n_restarts=1, seed=seed)(x_test, y_test),
            ),
        ]

        for atk_name, wd_fn, ta_fn in pair_specs:
            _sync(device)
            t0 = time.perf_counter()
            res_wd = wd_fn()
            _sync(device)
            t_wd = time.perf_counter() - t0
            wd_linf = float(compute_perturbation_norm(res_wd.perturbation, "linf", x_test).max().item())
            acc_wd = _eval_acc(model, res_wd.x_adv, y_test)

            _sync(device)
            t0 = time.perf_counter()
            x_ta = ta_fn()
            _sync(device)
            t_ta = time.perf_counter() - t0
            ta_linf = float((x_ta - x_test).abs().amax().item())
            acc_ta = _eval_acc(model, x_ta, y_test)

            print(
                f"  {atk_name:<12s} | wannado-adv: Acc={acc_wd*100:5.1f}% ({t_wd:5.2f}s, Linf={wd_linf:.4f}) "
                f"| torchattacks: Acc={acc_ta*100:5.1f}% ({t_ta:5.2f}s, Linf={ta_linf:.4f})"
            )
            rows.append(
                {
                    "model": model_name,
                    "attack": atk_name,
                    "wd_acc": acc_wd,
                    "wd_sec": t_wd,
                    "ref_lib": "torchattacks",
                    "ref_acc": acc_ta,
                    "ref_sec": t_ta,
                }
            )

        # Official AutoAttack (fair apples-to-apples budget: 25 APGD steps, 1 restart, 15 FAB steps, 500 Square queries)
        _sync(device)
        t0 = time.perf_counter()
        res_aa_wd = wd_autoattack(
            model,
            x_test,
            y_test,
            eps=eps,
            norm="linf",
            version="standard",
            apgd_steps=25,
            apgd_restarts=1,
            fab_steps=15,
            square_queries=500,
            clip_bounds=(0.0, 1.0),
            seed=seed,
        )
        _sync(device)
        t_aa_wd = time.perf_counter() - t0
        acc_aa_wd = _eval_acc(model, res_aa_wd.x_adv, y_test)

        # Official fra31/auto-attack matching the exact same attack ensemble & budgets
        _sync(device)
        t0 = time.perf_counter()
        aa_off = OfficialAutoAttack(
            model,
            norm="Linf",
            eps=eps,
            version="custom",
            attacks_to_run=["apgd-ce", "apgd-dlr", "fab", "square"],
            device=device,
            verbose=False,
        )
        aa_off.apgd.n_iter = 25
        aa_off.apgd.n_restarts = 1
        aa_off.fab.n_iter = 15
        aa_off.fab.n_restarts = 1
        aa_off.square.n_queries = 500
        x_aa_off = aa_off.run_standard_evaluation(x_test, y_test, bs=100)
        _sync(device)
        t_aa_off = time.perf_counter() - t0
        acc_aa_off = _eval_acc(model, x_aa_off, y_test)

        print(
            f"  AutoAttack   | wannado-adv: Acc={acc_aa_wd*100:5.1f}% ({t_aa_wd:5.2f}s) "
            f"| fra31/autoattack: Acc={acc_aa_off*100:5.1f}% ({t_aa_off:5.2f}s)"
        )
        rows.append(
            {
                "model": model_name,
                "attack": "AutoAttack",
                "wd_acc": acc_aa_wd,
                "wd_sec": t_aa_wd,
                "ref_lib": "fra31/auto-attack",
                "ref_acc": acc_aa_off,
                "ref_sec": t_aa_off,
            }
        )

    # 3. False-Robustness Traps on Wong2020Fast
    print("\n=== False-Robustness Trap Audit (Built on Wong2020Fast) ===")
    wong = load_model("Wong2020Fast", model_dir=str(CKPT_DIR), dataset="cifar10", threat_model="Linf").to(device).eval()
    x_sub, y_sub = x_test, y_test

    class LogitScaleMaskWrapper(nn.Module):
        def __init__(self, base: nn.Module, scale: float = 1e3) -> None:
            super().__init__()
            self.base = base
            self.scale = scale

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base(x) * self.scale

    def quant_defense(z: torch.Tensor) -> torch.Tensor:
        return torch.round(z * 15.0) / 15.0

    class QuantizedDefenseWrapper(nn.Module):
        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.base = base

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base(quant_defense(x))

    temp_trap = LogitScaleMaskWrapper(wong, scale=1e3).to(device).eval()
    quant_trap = QuantizedDefenseWrapper(wong).to(device).eval()

    audit_temp = audit_robustness(temp_trap, x_sub[:64], y_sub[:64], eps=eps, norm="linf", clip_bounds=(0.0, 1.0), seed=seed)
    print(f"Trap 1 (Logit-Scaled Softmax Saturation): audit.passed={audit_temp.passed} | red_flags={audit_temp.red_flags}")
    ta_pgd_temp = _eval_acc(temp_trap, torchattacks.PGD(temp_trap, eps=eps, alpha=step_size, steps=10)(x_sub, y_sub), y_sub)
    wd_dlr_temp = _eval_acc(temp_trap, apgd_dlr(temp_trap, x_sub, y_sub, eps=eps, steps=20, clip_bounds=(0.0, 1.0), seed=seed).x_adv, y_sub)
    print(f"  Trap 1 Acc -> torchattacks PGD-CE: {ta_pgd_temp*100:.1f}% (fooled by saturation!) vs wannado APGD-DLR: {wd_dlr_temp*100:.1f}%")

    pf_q = preflight(quant_trap, x_sub[:16], y_sub[:16], eps=eps, norm="linf", clip_bounds=(0.0, 1.0), attack="pgd")
    audit_q = audit_robustness(quant_trap, x_sub[:64], y_sub[:64], eps=eps, norm="linf", clip_bounds=(0.0, 1.0), seed=seed)
    print(f"Trap 2 (4-bit Quantizer): preflight(pgd).ok={pf_q.ok}, warnings={pf_q.warnings}, audit.passed={audit_q.passed}, red_flags={audit_q.red_flags}")
    ta_pgd_q = _eval_acc(quant_trap, torchattacks.PGD(quant_trap, eps=eps, alpha=step_size, steps=10)(x_sub, y_sub), y_sub)
    wd_bpda_q = _eval_acc(
        quant_trap,
        bpda_eot(wong, x_sub, y_sub, eps=eps, steps=20, eot_samples=2, defense_fn=quant_defense, surrogate_fn=lambda z: z, clip_bounds=(0.0, 1.0), seed=seed).x_adv,
        y_sub,
    )
    print(f"  Trap 2 Acc -> torchattacks PGD: {ta_pgd_q*100:.1f}% (broken autograd!) vs wannado BPDA-EoT: {wd_bpda_q*100:.1f}%")

    # Save CSV & figure
    csv_path = OUTPUTS_DIR / "cv_robustbench_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "attack", "wd_acc", "wd_sec", "ref_lib", "ref_acc", "ref_sec"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CV benchmark CSV to {csv_path}")

    # Plot Wong2020Fast parity + False-robustness bypass chart
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)

    wong_rows = [r for r in rows if r["model"] == "Wong2020Fast"]
    labels = [str(r["attack"]) for r in wong_rows]
    idx = torch.arange(len(labels)).numpy()
    w = 0.36
    ax1.bar(idx - w / 2, [float(r["wd_acc"]) * 100.0 for r in wong_rows], width=w, color="#1f77b4", label="wannado-adv")
    ax1.bar(idx + w / 2, [float(r["ref_acc"]) * 100.0 for r in wong_rows], width=w, color="#ff7f0e", alpha=0.88, label="torchattacks / fra31-AutoAttack")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(labels, rotation=32, ha="right", fontsize=8.5)
    ax1.set_ylabel("Robust Accuracy (%)  [lower = stronger attack]", fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.set_title(f"CIFAR-10 Wong2020Fast Parity (eps=8/255, N={num_samples})", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5)

    trap_labels = ["Softmax Trap:\nNaive PGD-CE", "Softmax Trap:\nAPGD-DLR (ours)", "4-bit Quant Trap:\nNaive PGD", "4-bit Quant Trap:\nBPDA-EoT (ours)"]
    trap_vals = [ta_pgd_temp * 100.0, wd_dlr_temp * 100.0, ta_pgd_q * 100.0, wd_bpda_q * 100.0]
    trap_colors = ["#adb5bd", "#d62728", "#adb5bd", "#d62728"]
    bars = ax2.bar(trap_labels, trap_vals, color=trap_colors, width=0.52)
    for bar, val in zip(bars, trap_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, val + 1.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Measured Accuracy (%)", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_title("Exposing Carlini False Robustness Traps on CIFAR-10", fontsize=11.5, fontweight="bold")

    fig.suptitle("wannado-adv — RobustBench Alignment & False-Robustness Audit (CIFAR-10)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = ASSETS_DIR / "cv_alignment_and_audit.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CV benchmark figure to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_benchmark(num_samples=args.samples, device_arg=args.device, seed=args.seed)
