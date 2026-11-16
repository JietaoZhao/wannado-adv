# wannado-adv Benchmark & End-to-End Validation Suite

This directory verifies `wannado-adv` against external reference libraries and non-vision domains **without polluting** core package dependencies in [`pyproject.toml`](../pyproject.toml) (which remain strictly `torch` + `numpy`).

## Experiments

| Script | Domain & Models | Verified Outputs |
| --- | --- | --- |
| [`bench_wireless_jsr.py`](bench_wireless_jsr.py) | `torch.complex64` 6-class baseband IQ classifier (`BPSK`, `QPSK`, `8PSK`, `16QAM`, `64QAM`, `PAM4`) + 4-bit quantized ADC defense under `norm="jsr_db"` (`-24` to `0 dB`) | [`assets/wireless_jsr_curves.png`](assets/wireless_jsr_curves.png), `outputs/wireless_jsr_results.csv` |
| [`bench_cv_robustbench.py`](bench_cv_robustbench.py) | CIFAR-10 RobustBench (`Standard`, `Wong2020Fast`), 26-attack Apple Silicon `mps` compatibility smoke test, `torchattacks` & official `fra31/auto-attack` parity, and Carlini Softmax/Quantizer traps | [`assets/cv_alignment_and_audit.png`](assets/cv_alignment_and_audit.png), `outputs/cv_robustbench_comparison.csv` |

## Reproducing in an Isolated Environment

Create a dedicated virtual environment (with system PyTorch inherited via `--system-site-packages` if desired) so benchmark packages do not alter your core environment:

```bash
# Experiment B (requires torch, numpy, matplotlib only)
python3 benchmarks/bench_wireless_jsr.py --samples 96 --seed 42

# Experiment A (in a venv containing torchvision, robustbench, torchattacks, autoattack, timm, scipy)
python benchmarks/bench_cv_robustbench.py --samples 200 --device auto --seed 42
```

Downloaded CIFAR-10 data (`benchmarks/data/`), RobustBench checkpoints (`benchmarks/checkpoints/`), and raw CSV dumps (`benchmarks/outputs/`) are gitignored by design.
