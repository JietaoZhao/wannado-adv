---
name: wannado-adv
description: Autonomous one-click AI red-teaming, Carlini 2.0 false-robustness auditing, closed-loop adversarial hardening (PGD-AT, TRADES, MART), and certified L2 randomized smoothing using wannado-adv across any AI IDE or coding agent (AGENTS.md / Agent Skills standard).
---

# WannaDo-Adv Autonomous Red-Team & Hardening Protocol

You are collaborating with a machine learning researcher, engineer, or student who wants reliable, publication-grade adversarial robustness auditing, hardening, and certification (CV, wireless baseband `complex64` IQ, tabular, time-series, continuous regression/DRL, or unsupervised models) using `wannado-adv`.

**Core Operating Principle — One-Click Autonomous Red-Teaming First**:
Do **NOT** stall the user behind multi-question forms when reasonable defaults can be auto-detected from the workspace. Inspect the project, isolate the victim inside `./adv_eval/interface.py`, and execute the closed-loop verification gate (`wannado-adv verify` / `auto_red_team()`) immediately. Only pause to ask concise, zero-jargon clarifying questions when the target model, dataset, or threat budget genuinely cannot be inferred (or before launching a multi-hour full-dataset training run).

## Activation Scope & Ecosystem Coexistence (with `wannado-drl`)
`wannado-adv` and `wannado-drl` are independent packages that coexist seamlessly under the `wannado.*` namespace and `AGENTS.md` / `.agents/skills/` standard when installed together:
- **Trigger Boundary (`wannado-adv` owns adversarial robustness)**: Activate `wannado-adv` ONLY for adversarial attack evaluation, Carlini 2.0 false-robustness auditing, adversarial training (`PGD-AT`/`TRADES`/`MART`), or certified smoothing (`./adv_eval/`). Do **NOT** hijack requests to formulate an MDP or train a standard RL agent from scratch — when `wannado-drl` is present and no trained policy exists yet, defer RL modeling/training to `./drl_lab/` first.
- **Clean Handoff When Auditing a Trained `wannado-drl` Policy**:
  - Leave `./drl_lab/` completely untouched; instantiate the victim exclusively inside `./adv_eval/interface.py` (`VictimInterface`).
  - **Frozen Normalization Gradient Rule**: If the trained DRL checkpoint used observation normalization (`obs_rms_state` with `mean`, `var`, `frozen=True`), apply `(x - mean) / torch.sqrt(var + 1e-8)` (clamped to `[-10.0, 10.0]`) using **differentiable PyTorch operations** inside the `VictimInterface` forward wrapper rather than detached NumPy arrays, so white-box gradients stay intact and normalization statistics never drift under attack.

---

## Step 1: Traverse & Re-Scan the Workspace (Never Guess)
- Always inspect the project **first** — and **re-scan** whenever models, datasets, or preprocessing change:
  ```bash
  wannado-adv inspect .
  ```
  or in Python:
  ```python
  from wannado.adv.copilot import scan_project, probe_model_context
  scan = scan_project(".")
  ```
- Detect framework (`PyTorch`, `RobustBench`, `timm`, `Gymnasium`, RF DSP), domain (`vision`, `wireless`, `drl`, `tabular`, `timeseries`, `unsupervised`), model files, checkpoints, and tensor dtype/shape.
- Once a model and batch/loader are wired, run `probe_model_context(model, x, y)` so you know exact `device`, `dtype`, `clip_bounds`, `autograd_connected`, and whether `BatchNorm`/`Dropout` needs isolation. Never assume `[0, 1]` bounds for non-image tensors (`complex64` wireless uses `clip_bounds=None, norm="jsr_db"`).

## Step 2: Non-Intrusive Boundary & Single-File Ejection
- **Never scatter adversarial loops across the user's core training or inference files.**
- Keep all evaluation/red-teaming code inside ONE isolated boundary directory created via:
  ```bash
  wannado-adv scaffold --dir ./adv_eval --domain <domain>
  ```
  - `adv_eval/interface.py` holds the single `VictimInterface` adapter (including optional `defense_fn` / `surrogate_fn` and mini-batch `DataLoader` or tensor batches).
  - `adv_eval/run_eval.py` runs suite evaluation & paper exports.
- **Single-File Customization Rule**: When the user shares a new paper or asks to customize an attack or defense algorithm (e.g., `apgd_dlr`, `fab`, `pgd_at`, `trades`, `mart`, `randomized_smoothing`):
  ```bash
  wannado-adv eject <attack_or_defense_id> --dest ./adv_eval/attacks
  ```
  Edit ONLY that ejected single file (preserving `META.citation` and extension hooks) without modifying core library or project training code.

## Step 3: Execute One-Click Autonomous Red-Team + Carlini 2.0 Verify Gate
- **Never stop at `fgsm` or basic `pgd` and declare a model robust.**
- Run the closed-loop verification gate directly from the terminal (hard exit code `0` = PASS, `1` = FAIL):
  ```bash
  wannado-adv verify --interface ./adv_eval/interface.py --goal standard --samples 100
  ```
  or call `auto_red_team(...)` directly in Python (supports `DataLoader`, `batch_size`, `defense_fn`, `surrogate_fn`, continuous regression/DRL targets, and `complex64` wireless IQ):
  ```python
  from wannado.adv import auto_red_team
  rt = auto_red_team(model, loader_or_x, y, eps=8/255, norm="linf", clip_bounds=(0.0, 1.0))
  rt.save("./adv_eval/outputs")
  ```
- **What `auto_red_team()` + Carlini 2.0 automatically catches & breaks**:
  - *Softmax / logit saturation* -> escalates scale-invariant `apgd_dlr` (or `apgd_l1(loss="dlr")`).
  - *Non-differentiable / quantized preprocessing (`defense_fn`)* -> escalates `bpda_eot` + gradient-free `square_attack`.
  - *Stochastic test-time defense / noise* -> detects output variance across repeats and escalates `eot_pgd`.
  - *Hard-label-only / detached oracle* -> escalates decision-based `hopskipjump`.
  - Computes **per-sample worst-case ensemble survival** with **Wilson 95% confidence intervals**, plus the **AURC / `eps_50` robustness decay curve** (`robustness_curve.png` / `.csv`).

## Step 4: Optional Closed-Loop Hardening & Certified L2 Robustness
When the red-team gate fails (or the user asks to harden/certify the model), apply a single-file defense from `wannado.adv.defenses` — ZERO extra runtime dependencies (no SciPy):
- **Empirical Adversarial Training**:
  - `pgd_adversarial_train(model, train_loader, eps=..., norm=...)` — Madry et al. (ICLR 2018) min-max PGD-AT (works on vision, tabular, and `complex64` wireless `jsr_db`).
  - `trades_train(model, train_loader, eps=..., beta=6.0, norm=...)` — Zhang et al. (ICML 2019) clean-vs-robust KL boundary trade-off.
  - `mart_train(model, train_loader, eps=..., beta=5.0, norm=...)` — Wang et al. (ICLR 2020) margin-boosted misclassification-aware hardening.
- **Provable L2 Certification**:
  - `gaussian_augmented_train(model, train_loader, sigma=0.25)` + `certify_randomized_smoothing(model, x_test, y_test, sigma=0.25, n0=100, n=1000, alpha=0.001)` — Cohen et al. (ICML 2019) exact Clopper–Pearson certified L2 radii $R = \sigma \Phi^{-1}(\underline{p_A})$.
- **Mandatory Post-Hardening Re-Audit**: Immediately re-run `wannado-adv verify` (or `auto_red_team()` + `audit_robustness()`) on the hardened checkpoint to verify genuine robustness (`audit.passed == True`) rather than gradient masking!

## Step 5: Zero-Jargon Reporting, Smart Clarification & Zero-Hallucination Citations
- **Match the User's Language**: Communicate explanations, verdicts, and any needed questions naturally in the user's language (e.g., natural Chinese if chatting in Chinese).
- **Zero-Jargon Rule (Only Ask When Needed)**: Do NOT block fast runs with unnecessary questionnaires. If a knob genuinely needs user input (e.g., ambiguous `eps`, custom `defense_fn`, or long training ETA from `estimate_runtime()`), explain the physical intuition + higher-vs-lower trade-off in plain words first (with technical terms in parentheses) and give self-explanatory options.
- **STRICT CITATION GUARDRAIL**: NEVER invent citations or search the web for papers on built-in `wannado-adv` attacks or defenses. Pull verbatim `.bib` / IEEE / APA citations directly from `wannado-adv cite <ids>` or `export_citations()` (supports all 35 attacks + `pgd_at`, `trades`, `mart`, `randomized_smoothing`).

## Step 6: Mandatory Paper-Ready Post-Run Summary
After every red-team, verification, hardening, or certification run, open with a **plain-language verdict**, followed by complete reproducible artifacts saved in `./adv_eval/outputs/`:
1. **Verdict & Red-Team Escalation**: Clean accuracy, worst-case ensemble robust accuracy with Wilson 95% CI, `AURC` & `eps_50`, and whether Carlini 2.0 detected any false-robustness red flags (`softmax_saturation`, `broken_or_vanishing_gradients`, `stochastic_defense_requires_eot`, `obfuscated_defense_broken_by_bpda`).
2. **Experimental Setup**: PyTorch version, device (`cpu`/`mps`/`cuda`), dtype, seed, `N` samples, `norm`, exact `eps`, `clip_bounds`, and empirically verified `max_measured <= eps`.
3. **Exported Artifacts**: `red_team_report.json` / `evaluation_report.json`, `red_team_summary.md` / `paper_summary.md`, `paper_table.tex`, `references.bib`, `robustness_curve.csv`, and `robustness_curve.png` (plus certified radii table when smoothing is used).
