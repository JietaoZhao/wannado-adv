# Changelog

All notable changes to `wannado-adv` will be documented in this file.

## [1.0.4] - 2026-11-17

### Fixed
- **EoT-PGD complex noise crash (`NameError`)**: Added missing `import math` in `wannado.adv.attacks.physical_and_eot.eot_pgd_athalye2018` so `noise_std > 0` on `torch.complex64`/`complex128` inputs works without error.
- **`mi_fgsm` top-level export & suite resolution**: Added `"mi_fgsm"` to `ATTACK_EXPORTS` and `wannado.adv.__all__` alongside `"mifgsm"` so `from wannado.adv import mi_fgsm` and `evaluate_suite(..., attacks=["mi_fgsm"])` resolve consistently with `META.id`.
- **Universal random-location `adv_patch` convergence**: Updated `adv_patch(..., random_location=True)` to average gradients over `eot_samples` placements with momentum and evaluate candidate checkpoints on the base mask after updates, fixing cross-PyTorch-version regressions.
- **Infeasible `clip_bounds` budget guard**: Updated `project_perturbation` and `_project_l1_box` to fail loudly with an explicit `ValueError` when clean inputs already fall outside `clip_bounds` by more than `eps_budget` (e.g. passing `(0, 1)` on z-scored inputs), and to enforce strict box–L2 feasibility via convex interpolation when clean inputs sit near bounds.
- **Carlini robustness audit monotonicity check**: Tightened `monotonic_over_budget` in `audit_robustness()` to verify `acc_high <= acc_mid + tol` as well as `acc_high <= acc_low + tol`.

### Added
- **CLI parity commands**: Added `wannado-adv arsenal` and `wannado-adv preflight` CLI subcommands.
- **Typing & Engineering Governance**: Added PEP 561 `py.typed` marker, explicit static `__all__` export lists, `[tool.ruff]` / `[tool.mypy]` configuration, `CITATION.cff`, `CHANGELOG.md`, and GitHub Actions CI matrix across Python 3.10–3.13.

## [1.0.3] - 2026-11-16

### Added
- Initial public release with 35 modular single-file attacks across 9 families, native `torch.complex64` + `jsr_db` support, Carlini false-robustness auditor, AI Co-Pilot planner/inspector, and academic reporting.
