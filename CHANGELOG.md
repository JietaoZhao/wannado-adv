# Changelog

All notable changes to `wannado-adv` will be documented in this file.

## [1.2.2] - 2026-11-19

### Fixed
- **Complex64 Randomized Smoothing L2 Certification Soundness (`defenses/randomized_smoothing_cohen2019.py`)**: Scaled effective per-component noise $\sigma_{\mathrm{eff}} = \sigma / \sqrt{2}$ when certifying `torch.complex64`/`complex128` inputs (eliminating $\sqrt{2}$ overstatement of certified L2 radii), added `infer_model_device` alignment, and added optional Cohen Algorithm 2 two-sided binomial ABSTAIN (`alpha`) in `smooth_predict()`.
- **Binary DLR (`num_classes < 3`) Zero Gradient (`core/losses.py`)**: Detached the `(z_0 - z_1)` normalization denominator in binary DLR loss so binary classifiers receive exact non-zero gradients instead of exact algebraic cancellation.
- **Defense Training Streaming & Regularization (`defenses/{pgd_at,trades,mart,randomized_smoothing}`)**: Replaced eager `list(iter_eval_batches(..., device=dev))` full-dataset GPU staging with O(`batch_size`) streaming `iter_train_batches()`, added epoch shuffling for Tensor inputs, explicit error on exhausted one-shot generators, `(B, 1)` target squeezing, cosine LR scheduling (`lr_scheduler="cosine"`), and best-checkpoint restoration (`save_best=True`).
- **Attack Correctness & Hygiene**:
  - `boundary`: Fixed 1D binary search mutating `adv_cur` in-place while interpolating against it (`adv_start = adv_cur.clone()`).
  - `cw_l2` & `ead` & `hopskipjump`: Eliminated victim parameter `.grad` pollution in `cw_l2` via `torch.autograd.grad(..., only_inputs=True)` and computed `success_rate` strictly *after* `eps` budget reversion.
  - `fab`: Added explicit `norm` validation (rejecting unsupported `"l1"`), fixed Croce & Hein Alg. 1 direct backward contraction `(1 - beta) * x_orig + beta * x_curr`, and fixed `(B, 1)` targets.
  - `apgd_l1`, `square_attack`, `autoattack`, `bpda_eot`, `gap`, `advgan`: Fixed list/tensor `clip_bounds` in `apgd_l1`, `(B, 1)` target broadcasting across attacks, chunked `autoattack` `apgd_t` stage monotonicity, `defense_fn` `eval_mode()` in `bpda_eot`, deterministic complex seeding in `gap`, and pre-step `best_state` snapshotting in `gap`/`advgan`.
- **Carlini 2.0, `auto_red_team`, `verify_gate`, Co-Pilot & Reporting**:
  - Added gradient-free sparse L1 black-box obfuscation probe (`_l1_blackbox_probe_acc`) and scale-invariant relative stochasticity + flip-rate detection in `audit_robustness()`, plus `ImageNet` L2 `eps_scale_meaningful` calibration.
  - Added automatic CPU-tensor-to-model-device alignment (`infer_model_device`), lazy O(`batch_size`) streaming for deterministic `DataLoader`s (`make_reiterable_eval_batches`), Monte Carlo majority voting (`mc_draws=5`) for stochastic models, and curve `loss`/`l1` override propagation in `auto_red_team()` and `evaluate_suite()`.
  - Made `verify_robustness_gate` honor `samples` on Tensor inputs and flag `carlini:hard_label_only_unverifiable_whitebox` instead of silently passing `carlini_audit_ok` on hard-label-only oracles.
  - Fixed `recommend_suite(..., norm="l1")` (native L1 attacks only), added `DataLoader` support to `preflight`, `probe_model_context`, `estimate_runtime`, and scaffolded `run_eval.py`, fixed continuous `format_paper_summary()` budget sweep rendering, and preserved float outputs for continuous regressors in `SklearnModelAdapter`.

### Added
- **`NormalizedModelWrapper` / `wrap_normalized_model` (`wannado.adv.core.utils`)**: Differentiable `(x - mean) / std` wrapper exported at top level so pretrained `[0, 1]` ImageNet/CIFAR/tabular models keep exact gradients without manual wrapper boilerplate.

## [1.2.1] - 2026-11-19

### Fixed
- **Cold-start `reporter <-> audit` circular import**: Converted `audit.red_team` and `audit.verify_gate` to PEP 562 lazy exports inside `wannado.adv.audit` (preserving eager callable `audit.preflight` / `audit.audit_robustness`) and deferred internal `paper_reporter` imports in `audit/red_team.py`, eliminating order-dependent `ImportError` across Linux CI test collection and standalone imports.
- **`__all__` export hygiene**: Removed stale unexported entries (`apgd_t`, `fab_t`, `square_t`) from `wannado.adv.__all__` and `wannado.adv.attacks.__all__`.
- **Stronger Softmax-saturation escalation (`auto_red_team`)**: When `softmax_saturation` is detected, even `goal="sanity"` now escalates multi-restart `apgd_dlr(steps>=25, restarts=2)` and switches `pgd` to `loss="margin"` across all CPU thread/BLAS reduction schedules.

## [1.2.0] - 2026-11-19

### Added
- **Autonomous Closed-Loop Red-Teaming (`auto_red_team` / `RedTeamReport`)**: One-click red-team engine that diagnoses victim failure modes via Carlini 2.0, adaptively escalates matched attacks (`apgd_dlr` for Softmax saturation, `bpda_eot` + `square_attack` for non-differentiable `defense_fn`, `eot_pgd` for stochastic defenses, `hopskipjump` for hard-label-only oracles, `apgd_l1` for `L1` geometry), computes true per-sample worst-case ensemble survival with exact **Wilson 95% confidence intervals**, sweeps **AURC** & **`eps_50`** robustness decay curves, and exports JSON/Markdown/LaTeX/BibTeX/CSV/PNG dossiers.
- **Hard Agent/CI Verification Gate (`wannado-adv verify` / `verify_robustness_gate`)**: Single-command CLI & Python gate returning hard exit codes (`0` PASS / `1` FAIL) on Carlini 2.0 false-robustness red flags, budget invariants, clean/robust accuracy thresholds, or continuous output L2 drift thresholds.
- **Carlini 2.0 False-Robustness Auditor (`audit_robustness`)**: Upgraded with test-time stochasticity detection (`stochastic_defense_requires_eot` + EoT cross-check), explicit `defense_fn` BPDA cross-check (`obfuscated_defense_broken_by_bpda`), native `norm="l1"` support via `apgd_l1`, hard-label-only oracle detection, and continuous regression/DRL auditing (`clean_mse`, `adv_mse`, `output_l2_drift`, `cosine_similarity`).
- **Single-File Modular Defenses & Zero-SciPy Certified L2 Robustness (`wannado.adv.defenses`)**:
  - `pgd_adversarial_train` (Madry et al., ICLR 2018) — supports vision, tabular, and `torch.complex64` wireless `jsr_db`.
  - `trades_train` (Zhang et al., ICML 2019) — clean-vs-robust KL boundary trade-off.
  - `mart_train` (Wang et al., ICLR 2020) — boosted-CE misclassification-aware hardening.
  - `gaussian_augmented_train`, `smooth_predict`, `certify_randomized_smoothing` (Cohen et al., ICML 2019) — provable L2 radii $R = \sigma \Phi^{-1}(\underline{p_A})$ with pure-Python Acklam `norm_ppf` + regularized incomplete beta Clopper–Pearson bounds (**zero SciPy dependency**).
  - Wired into verbatim zero-hallucination `export_citations()` / `wannado-adv cite`, single-file `eject_defense()` / `wannado-adv eject`, and `wannado-adv arsenal`.
- **Dataset-Scale Mini-Batch Evaluation (`DataLoader` / `batch_size`)**: Added `DefendedModelWrapper`, `iter_eval_batches`, and `materialize_eval_tensors` across `evaluate_suite`, `auto_red_team`, `autoattack`, `fab`, and `VictimInterface` (`defense_fn` / `surrogate_fn` + `--batch-size`).
- **End-to-End v1.2.0 Benchmark (`benchmarks/bench_redteam_and_hardening.py`)**: Verifies batched `FAB` speedups (`3.1x–8.9x` CPU), `auto_red_team()` defeating 4 fake defenses, `PGD-AT`/`TRADES`/`MART` hardening passing Carlini 2.0 re-audits, and pure-Python Randomized Smoothing L2 curves (`benchmarks/assets/redteam_and_hardening.png`).

### Changed
- **Vectorized Batched `FAB` (`attacks/minimum_norm/fab_croce2020.py`)**: Replaced sample-by-sample inner loop with fully vectorized batched hyperplane projection (`_fab_batched` + `batch_size` chunking), delivering `3.1x–8.9x` CPU speedup on multi-layer ConvNets.
- **Streamlined Autonomous-First `agent_kit` (`SKILL.md`, `AGENTS.md`, `CLAUDE.md`, Cursor rules)**: Replaced mandatory pre-run multi-question ceremonies with direct `wannado-adv inspect . -> scaffold -> wannado-adv verify` autonomous red-teaming and closed-loop hardening + Carlini 2.0 re-auditing.
- **True Conditional Attack Success Rate (`evaluate_suite`)**: `attack_success_rate` is now strictly conditional on clean-correct predictions (`clean_correct & adv_wrong / clean_correct`), matching academic conventions.
- Promoted package classifier to `Development Status :: 5 - Production/Stable`.

### Fixed
- **Exact Targeted DLR Denominator (`core/losses.py`)**: Fixed Croce & Hein (2020) targeted DLR denominator for `C >= 4` (`z_(1) - 0.5 * (z_(3) + z_(4))`) and passed `targeted=targeted` consistently in `pgd` and `fgsm`.
- **HuggingFace `.logits` AutoAttack Stage 2b Crash**: Reused unwrapped `clean_logits` in `autoattack_croce2020.py` Stage 2b instead of indexing raw `ModelOutput` objects.
- **Continuous Float Target Preflight Crash**: `preflight()` now auto-selects `"mse"` / L2 drift verification on continuous float targets instead of failing in `F.cross_entropy`.
- **Lossless `eval_mode()` Submodule Restoration (`core/utils.py`)**: Snapshots and restores every submodule's exact `.training` flag and synchronizes `DefendedModelWrapper.training`, preventing `BatchNorm1d` state mutation across chained attacks.

## [1.0.6] - 2026-11-17

### Changed
- **English-first documentation & skill templates**: Removed remaining bilingual Chinese examples from `SKILL.md` templates and the `docs/index.html` feedback portal so all public repository docs and UI text default cleanly to English.

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
