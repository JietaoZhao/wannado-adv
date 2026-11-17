# Changelog

All notable changes to `wannado-adv` will be documented in this file.

## [1.0.5] - 2026-11-17

### Added
- **Cross-domain output normalization (`core/losses.py`)**: Added `unwrap_model_outputs`, `normalize_classification_logits`, and `predict_labels` so single-logit binary models `(B, 1)` / `(B,)`, HuggingFace `ModelOutput` objects exposing `.logits`, and dense 4D segmentation maps `(B, C, H, W)` work uniformly across attacks, `preflight()`, `audit_robustness()`, and `evaluate_suite()`.
- **Per-feature tabular bounds, immutable masks & discrete snapping (`core/projections.py`)**: Added `coerce_clip_bounds`, `clamp_to_bounds`, `perturb_mask` filtering on `project_perturbation` / `random_init_perturbation`, and `project_discrete_features` for categorical/binary tabular columns.
- **scikit-learn adapter (`core/utils.py`)**: Added `SklearnModelAdapter` / `wrap_sklearn_model` supporting `decision_function`, `predict_proba`, and hard `predict` oracles cleanly inside black-box attacks.
- **Scale mismatch guard & expanded domain discovery**: Added `eps_scale_mismatch_trivial_budget` detection to `audit_robustness()`, added `readme_summary` + `detected_symbols` extraction and `medical`/`cybersecurity`/`nlp_transformers`/`general` domain classification to `scan_project()`, enabled lazy submodule access (`wadv.copilot`, `wadv.audit`, etc.) on `wannado.adv`, and added `steps` / `max_queries` aliases on black-box attacks.

### Fixed
- Eliminated crashes when passing per-feature vector `clip_bounds=(min_vec, max_vec)` across first-order, minimum-norm (`cw_l2`, `ead`, `fab`, `deepfool`, `jsma`, `sparsefool`), black-box (`nes`, `spsa`, `zoo`, `boundary`, `hopskipjump`), universal/patch/generative (`uap`, `vmifgsm`, `adv_patch`, `advgan`, `gap`), `preflight()`, and `EvaluationReport.to_json()`.
- Stopped `preflight()` and `probe_model_context()` from reporting `ok=True` when given an uncallable scikit-learn estimator; now surfaces explicit remediation guidance pointing to `wrap_sklearn_model()`.
- Escaped `#`, `$`, `^`, and `~` cleanly in `EvaluationReport.to_latex()`.

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
