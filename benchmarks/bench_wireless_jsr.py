"""Experiment B: End-to-End Wireless Complex64 Baseband JSR-dB Benchmark & Demo.

Demonstrates ``wannado-adv`` capabilities unavailable in standard vision-only libraries
(RobustBench / TorchAttacks):
  1. Native ``torch.complex64`` baseband IQ inputs without manual real/imag reshaping.
  2. Physical Jamming-to-Signal Ratio (``norm="jsr_db"``) energy constraints.
  3. Cross-threat-model JSR curves: White-box (FGSM, PGD, VMI-FGSM, APGD-DLR, AutoAttack),
     Score black-box (Square Attack, NES), Hard-label Decision black-box (Boundary),
     and BPDA-EoT against a non-differentiable quantized ADC + random phase-jitter receiver.

Run:
    python3 benchmarks/bench_wireless_jsr.py --samples 120 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wannado.adv import (
    apgd_dlr,
    audit_robustness,
    autoattack,
    boundary,
    bpda_eot,
    compute_perturbation_norm,
    fgsm,
    nes,
    pgd,
    preflight,
    square_attack,
    vmi_fgsm,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "benchmarks" / "assets"
OUTPUTS_DIR = REPO_ROOT / "benchmarks" / "outputs"

MODULATIONS = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4")


def _constellation_table(device: torch.device) -> list[torch.Tensor]:
    """Unit-average-power constellation symbol tables for 6 canonical digital modulations."""
    bpsk = torch.tensor([-1.0 + 0j, 1.0 + 0j], dtype=torch.complex64, device=device)

    qpsk = torch.tensor(
        [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=torch.complex64, device=device
    ) / math.sqrt(2.0)

    k8 = torch.arange(8, device=device, dtype=torch.float32) * (2.0 * math.pi / 8.0)
    psk8 = torch.complex(torch.cos(k8), torch.sin(k8)).to(torch.complex64)

    levels4 = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=device)
    re16, im16 = torch.meshgrid(levels4, levels4, indexing="ij")
    qam16 = torch.complex(re16.reshape(-1), im16.reshape(-1)).to(torch.complex64)
    qam16 = qam16 / torch.sqrt((qam16.abs() ** 2).mean())

    levels8 = torch.tensor([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], device=device)
    re64, im64 = torch.meshgrid(levels8, levels8, indexing="ij")
    qam64 = torch.complex(re64.reshape(-1), im64.reshape(-1)).to(torch.complex64)
    qam64 = qam64 / torch.sqrt((qam64.abs() ** 2).mean())

    pam4 = torch.complex(levels4, torch.zeros_like(levels4)).to(torch.complex64)
    pam4 = pam4 / torch.sqrt((pam4.abs() ** 2).mean())

    return [bpsk, qpsk, psk8, qam16, qam64, pam4]


def simulate_complex_iq_dataset(
    num_samples: int,
    seq_len: int = 128,
    snr_db: float = 18.0,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize realistic complex64 baseband bursts with slight carrier phase drift & AWGN."""
    dev = device or torch.device("cpu")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tables = _constellation_table(torch.device("cpu"))
    num_classes = len(tables)

    # Upsample each symbol 2x with mild pulse shaping so temporal Conv1D features & constellation geometry both matter.
    sym_len = seq_len // 2
    y = torch.arange(num_samples) % num_classes
    perm = torch.randperm(num_samples, generator=gen)
    y = y[perm]

    x_out = torch.zeros((num_samples, 1, seq_len), dtype=torch.complex64)
    noise_std = math.sqrt(10.0 ** (-snr_db / 10.0) / 2.0)

    t_axis = torch.linspace(0.0, 1.0, seq_len).unsqueeze(0)
    for cls_idx, table in enumerate(tables):
        mask = y == cls_idx
        n_c = int(mask.sum().item())
        if n_c == 0:
            continue
        idx = torch.randint(0, len(table), (n_c, sym_len), generator=gen)
        syms = table[idx]
        # Repeat-and-smooth 2x oversampled baseband pulse
        up = torch.zeros((n_c, seq_len), dtype=torch.complex64)
        up[:, 0::2] = syms
        up[:, 1::2] = 0.7 * syms + 0.3 * torch.roll(syms, shifts=-1, dims=1)
        # Small carrier frequency offset + initial carrier phase jitter (constrained to [-12 deg, +12 deg] so BPSK/PAM4 stay identifiable)
        phi0 = (torch.rand((n_c, 1), generator=gen) - 0.5) * 0.42
        cfo = (torch.rand((n_c, 1), generator=gen) - 0.5) * 0.25
        phase = torch.complex(torch.cos(phi0 + cfo * t_axis), torch.sin(phi0 + cfo * t_axis)).to(
            torch.complex64
        )
        sig = up * phase
        # Normalize per-burst RMS to 1.0 before adding AWGN at target SNR
        rms = torch.sqrt((sig.abs() ** 2).mean(dim=1, keepdim=True)).clamp(min=1e-8)
        sig = sig / rms.to(torch.complex64)
        awgn = noise_std * torch.complex(
            torch.randn((n_c, seq_len), generator=gen),
            torch.randn((n_c, seq_len), generator=gen),
        ).to(torch.complex64)
        x_out[mask, 0, :] = sig + awgn

    return x_out.to(dev), y.to(dev)


class ComplexIQModulationNet(nn.Module):
    """Complex64 baseband receiver combining Conv1D IQ temporal + empirical distribution head."""

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        # Fuse learned temporal embeddings with higher-order complex moments (|z|, |z|^2, |z|^4, Re(z^2), Re(z^4))
        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 6, 96),
            nn.GELU(),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ri = torch.cat([x.real, x.imag], dim=1)  # (B, 2, L)
        feat = self.stem(ri)
        avg_p = feat.mean(dim=-1)
        max_p = feat.amax(dim=-1)

        z = x.squeeze(1)
        abs_z = z.abs()
        z2 = z * z
        z4 = z2 * z2
        moments = torch.stack(
            [
                abs_z.mean(dim=1),
                (abs_z**2).mean(dim=1),
                (abs_z**4).mean(dim=1),
                z2.real.mean(dim=1),
                z4.real.mean(dim=1),
                (z.real.abs() - z.imag.abs()).abs().mean(dim=1),
            ],
            dim=1,
        )
        return self.head(torch.cat([avg_p, max_p, moments], dim=1))


def train_wireless_classifier(
    seed: int = 42,
    epochs: int = 35,
    device: torch.device | None = None,
) -> ComplexIQModulationNet:
    dev = device or torch.device("cpu")
    torch.manual_seed(seed)
    x_tr, y_tr = simulate_complex_iq_dataset(1500, seq_len=128, snr_db=22.0, seed=seed, device=dev)
    model = ComplexIQModulationNet(num_classes=len(MODULATIONS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    batch_size = 150
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(x_tr.shape[0], device=dev)
        for start in range(0, x_tr.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            xb = x_tr[idx]
            # Mild random complex AWGN jamming augmentation (~ -15 to -10 dB) gives a smooth JSR curve.
            if epoch >= 5:
                sigma = 0.08 + 0.14 * float(epoch) / float(epochs)
                noise = sigma * torch.complex(torch.randn_like(xb.real), torch.randn_like(xb.imag)).to(xb.dtype)
                xb = xb + noise
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), y_tr[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def quantized_adc_defense(z: torch.Tensor) -> torch.Tensor:
    """Non-differentiable 4-bit IQ ADC quantization + mild random phase jitter."""
    step = 0.25
    re_q = torch.round(z.real / step) * step
    im_q = torch.round(z.imag / step) * step
    jitter = 0.03 * (torch.randn_like(re_q) + 1j * torch.randn_like(im_q)).to(z.dtype)
    return torch.complex(re_q, im_q).to(z.dtype) + jitter


def run_benchmark(num_test: int = 120, seed: int = 42) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print(f"[Experiment B] Training ComplexIQModulationNet on {device} (6 modulations, complex64 IQ)...")
    t0 = time.perf_counter()
    model = train_wireless_classifier(seed=seed, epochs=35, device=device)
    print(f"Training completed in {time.perf_counter() - t0:.2f}s")

    x_test, y_test = simulate_complex_iq_dataset(num_test, seq_len=128, snr_db=22.0, seed=seed + 999, device=device)
    with torch.no_grad():
        clean_acc = float((model(x_test).argmax(1) == y_test).float().mean().item())
        adc_clean_acc = float((model(quantized_adc_defense(x_test)).argmax(1) == y_test).float().mean().item())
    print(f"Clean Test Accuracy: {clean_acc * 100:.2f}% | Quantized-ADC Clean Accuracy: {adc_clean_acc * 100:.2f}%")

    # 1. Run preflight + Carlini auditor natively on complex64 RF inputs
    pf = preflight(model, x_test[:16], y_test[:16], eps=-10.0, norm="jsr_db", attack="pgd")
    print(f"Preflight OK={pf.ok} | Warnings={pf.warnings}")
    audit = audit_robustness(model, x_test[:32], y_test[:32], eps=-10.0, norm="jsr_db", seed=seed)
    print(f"Carlini Audit (Clean Model @ -10 dB JSR): passed={audit.passed}, flags={audit.red_flags}")

    # Wrapped non-differentiable receiver to test Carlini auditor + BPDA-EoT
    def defended_rx(z: torch.Tensor) -> torch.Tensor:
        return model(quantized_adc_defense(z))

    audit_def = audit_robustness(defended_rx, x_test[:32], y_test[:32], eps=-10.0, norm="jsr_db", seed=seed)
    print(
        f"Carlini Audit (Quantized-ADC Receiver @ -10 dB JSR): passed={audit_def.passed}, "
        f"flags={audit_def.red_flags}"
    )

    # 2. Sweep JSR (dB) across threat families
    jsr_levels = [-24.0, -18.0, -14.0, -10.0, -6.0, 0.0]
    rows: list[dict[str, float | str]] = []

    def _hard_oracle(z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(z).argmax(dim=1)

    for jsr_db in jsr_levels:
        print(f"\n--- JSR = {jsr_db:+.1f} dB ---")
        attacks = [
            ("FGSM (White-Box 1-step)", lambda: fgsm(model, x_test, y_test, eps=jsr_db, norm="jsr_db")),
            ("PGD-20 (White-Box)", lambda: pgd(model, x_test, y_test, eps=jsr_db, steps=20, norm="jsr_db", seed=seed)),
            ("VMI-FGSM (Transfer)", lambda: vmi_fgsm(model, x_test, y_test, eps=jsr_db, steps=15, num_variance_samples=6, norm="jsr_db", seed=seed)),
            ("APGD-DLR (Auto Step)", lambda: apgd_dlr(model, x_test, y_test, eps=jsr_db, steps=20, restarts=1, norm="jsr_db", seed=seed)),
            ("AutoAttack (Ensemble)", lambda: autoattack(model, x_test, y_test, eps=jsr_db, norm="jsr_db", version="fast", apgd_steps=20, apgd_restarts=1, square_queries=250, seed=seed)),
            ("Square Attack (Score BB)", lambda: square_attack(model, x_test, y_test, eps=jsr_db, max_queries=250, norm="jsr_db", seed=seed)),
            ("NES-25 (Score BB)", lambda: nes(model, x_test, y_test, eps=jsr_db, steps=25, nb_sample=32, sigma=0.04, norm="jsr_db", seed=seed)),
            ("Boundary (Label-Only BB)", lambda: boundary(_hard_oracle, x_test, y_test, eps=jsr_db, steps=60, norm="jsr_db", seed=seed)),
            ("Naive PGD on Quant-ADC", lambda: pgd(defended_rx, x_test, y_test, eps=jsr_db, steps=10, norm="jsr_db", random_start=False, seed=seed)),
            (
                "BPDA-EoT on Quant-ADC",
                lambda: bpda_eot(
                    model,
                    x_test,
                    y_test,
                    eps=jsr_db,
                    steps=20,
                    eot_samples=4,
                    defense_fn=quantized_adc_defense,
                    surrogate_fn=lambda z: z,
                    norm="jsr_db",
                    seed=seed,
                ),
            ),
        ]

        for label, runner in attacks:
            res = runner()
            elapsed = res.elapsed_ms
            measured_jsr = compute_perturbation_norm(res.perturbation, "jsr_db", x_test)
            max_jsr = float(measured_jsr.max().item())
            if max_jsr > float("-inf"):
                assert max_jsr <= jsr_db + 1e-2, f"{label} exceeded JSR: {max_jsr} > {jsr_db}"
            with torch.no_grad():
                if "Quant-ADC" in label:
                    acc_val = float((defended_rx(res.x_adv).argmax(1) == y_test).float().mean().item())
                else:
                    acc_val = float((model(res.x_adv).argmax(1) == y_test).float().mean().item())

            rows.append(
                {
                    "jsr_db": jsr_db,
                    "attack": label,
                    "acc": acc_val,
                    "asr": 1.0 - acc_val,
                    "elapsed_ms": elapsed,
                    "max_measured_jsr_db": max_jsr,
                }
            )
            print(f"  {label:<26s} | Acc={acc_val*100:5.1f}% | ASR={(1-acc_val)*100:5.1f}% | time={elapsed:6.1f} ms | max_JSR={max_jsr:+.2f} dB")

    csv_path = OUTPUTS_DIR / "wireless_jsr_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["jsr_db", "attack", "acc", "asr", "elapsed_ms", "max_measured_jsr_db"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved wireless benchmark CSV to {csv_path}")

    # Plot publication-grade dual-panel figure
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)

    clean_series = [
        ("FGSM (White-Box 1-step)", "--", "o", "#6c757d"),
        ("PGD-20 (White-Box)", "-", "s", "#1f77b4"),
        ("VMI-FGSM (Transfer)", "-.", "^", "#17becf"),
        ("APGD-DLR (Auto Step)", "-", "D", "#ff7f0e"),
        ("AutoAttack (Ensemble)", "-", "*", "#d62728"),
        ("Square Attack (Score BB)", ":", "P", "#2ca02c"),
        ("NES-25 (Score BB)", ":", "v", "#9467bd"),
        ("Boundary (Label-Only BB)", "-.", "X", "#8c564b"),
    ]
    for name, ls, marker, color in clean_series:
        pts = [r for r in rows if r["attack"] == name]
        ax1.plot(
            [p["jsr_db"] for p in pts],
            [p["acc"] * 100.0 for p in pts],
            linestyle=ls,
            marker=marker,
            linewidth=2.0,
            markersize=6.5,
            color=color,
            label=name,
        )

    ax1.axhline(clean_acc * 100.0, color="#222222", linestyle=":", linewidth=1.3, label=f"Clean Baseline ({clean_acc*100:.1f}%)")
    ax1.set_title("Native Complex64 Modulation Classifier under JSR-dB Jamming", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Jamming-to-Signal Ratio Budget  JSR (dB)", fontsize=10.5)
    ax1.set_ylabel("Receiver Classification Accuracy (%)", fontsize=10.5)
    ax1.set_ylim(-3, 105)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.0, frameon=True)

    # Panel 2: Quantized-ADC false robustness vs BPDA-EoT bypass
    for name, color, marker in [
        ("Naive PGD on Quant-ADC", "#1f77b4", "o"),
        ("BPDA-EoT on Quant-ADC", "#d62728", "s"),
    ]:
        pts = [r for r in rows if r["attack"] == name]
        ax2.plot(
            [p["jsr_db"] for p in pts],
            [p["acc"] * 100.0 for p in pts],
            marker=marker,
            linewidth=2.3,
            markersize=7.0,
            color=color,
            label=name,
        )
    ax2.axhline(adc_clean_acc * 100.0, color="#222222", linestyle=":", linewidth=1.3, label=f"Quant-ADC Clean Acc ({adc_clean_acc*100:.1f}%)")
    ax2.set_title("Exposing False Robustness on Non-Differentiable 4-bit ADC Receiver", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Jamming-to-Signal Ratio Budget  JSR (dB)", fontsize=10.5)
    ax2.set_ylabel("Defended Receiver Accuracy (%)", fontsize=10.5)
    ax2.set_ylim(-3, 105)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True)

    fig.suptitle(
        "wannado-adv — Unified Complex64 Wireless Benchmark (torch.complex64 + norm='jsr_db')",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig_path = ASSETS_DIR / "wireless_jsr_curves.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved wireless benchmark figure to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_benchmark(num_test=args.samples, seed=args.seed)
