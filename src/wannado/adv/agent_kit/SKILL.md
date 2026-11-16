---
name: wannado-adv
description: Universal AI Co-Pilot protocol for interactive, beginner-friendly adversarial learning and robustness evaluation using wannado-adv across any AI IDE or coding agent (AGENTS.md / Agent Skills standard). Drives project inspection, plain-language parameter alignment, single-file modular integration, zero-hallucination citations, hardware runtime estimation, and publication-ready reporting.
---

# WannaDo-Adv Universal AI Co-Pilot Protocol

You are collaborating with a machine learning researcher, engineer, or student who wants reliable, publication-grade adversarial robustness evaluation (CV, wireless baseband `complex64` IQ, tabular, time-series, DRL, or unsupervised models) using `wannado-adv`.

Follow this exact interactive workflow across **any AI IDE or CLI coding agent** so the user always understands *what* each decision changes before approving runs, while you automate project wiring, sanity verification, and academic reporting.

## Activation Scope & Ecosystem Coexistence (with `wannado-drl`)
`wannado-adv` and `wannado-drl` are independent packages that coexist seamlessly under the `wannado.*` namespace and `AGENTS.md` / `.agents/skills/` standard when installed together:
- **Trigger Boundary (`wannado-adv` owns adversarial robustness)**: Activate `wannado-adv` ONLY for adversarial attack evaluation, Carlini false-robustness auditing, or adversarial training (`./adv_eval/`). Do **NOT** hijack requests to formulate an MDP or train a standard RL agent from scratch — when `wannado-drl` is present and no trained policy exists yet, defer RL modeling/training to `./drl_lab/` first.
- **Clean Handoff When Auditing a Trained `wannado-drl` Policy**:
  - Leave `./drl_lab/` completely untouched; instantiate the victim exclusively inside `./adv_eval/interface.py` (`VictimInterface`).
  - **Frozen Normalization Gradient Rule**: If the trained DRL checkpoint used observation normalization (`obs_rms_state` with `mean`, `var`, `frozen=True`), apply `(x - mean) / torch.sqrt(var + 1e-8)` (clamped to `[-10.0, 10.0]`) using **differentiable PyTorch operations** inside the `VictimInterface` forward wrapper rather than detached NumPy arrays, so white-box gradients stay intact and normalization statistics never drift under attack.

---

## Step 1: Traverse & Re-Scan the User's Project (Never Guess)
- Always inspect the project **first** — and **re-scan** whenever files, models, datasets, or preprocessing change:
  ```bash
  wannado-adv inspect .
  ```
  or in Python:
  ```python
  from wannado.adv.copilot import scan_project, probe_model_context
  scan = scan_project(".")
  ```
- Inspect what framework (`PyTorch`, `RobustBench`, `timm`, `Gymnasium`, RF DSP), domain (`vision`, `wireless`, `drl`, `tabular`, `timeseries`, `unsupervised`), model files, and data dtype/shape exist in the actual workspace.
- Once a model & batch are loaded, run `probe_model_context(model, x, y)` so you know exact `device`, `dtype`, `clip_bounds`, `autograd_connected`, and whether `BatchNorm`/`Dropout` needs isolation. Never assume `[0, 1]` bounds for non-image tensors (`complex64` wireless uses `clip_bounds=None, norm="jsr_db"`).

## Step 2: Keep Integration Non-Intrusive & Single-File Modular
- **Never scatter adversarial loops across the user's core training or inference files.**
- Keep all adversarial code in ONE isolated evaluation boundary (e.g. `./adv_eval/` created by `wannado-adv scaffold --dir ./adv_eval --domain <domain>`):
  - `adv_eval/interface.py` holds the single `VictimInterface` adapter connecting the project's model & evaluation batch.
  - `adv_eval/run_eval.py` executes evaluation & paper export.
- **Single-File Customization Rule**: If the user shares a new paper or asks to improve/modify an attack algorithm (e.g. upgrading `fgsm`, `pgd`, `apgd_dlr`):
  - Eject ONLY that standalone single-file module via:
    ```bash
    wannado-adv eject <attack_id> --dest ./adv_eval/attacks
    ```
  - Read that single file's embedded citation and `[Extension Directions]` comments, and edit ONLY that file, leaving the rest of the user's project untouched.
  - Pass the customized function directly to `evaluate_suite(..., attacks=[custom_fn, ...])` without touching core library code.

## Step 3: Explain Baseline-to-SOTA Progression in Plain Words
- Never silently stop at `fgsm` or basic `pgd` and declare a model robust.
- Query `get_sota_lineage(domain=...)` and `recommend_suite(domain=..., goal=..., autograd_ok=...)` so you explain **why** multiple checks are paired together in everyday intuition (e.g., testing one fast single push `FGSM`, smart multi-step search `PGD`, scale-proof `APGD-DLR` against probability saturation tricks, and blind gradient-free probing `Square Attack` so a broken gradient cannot fake safety).

## Step 4: MANDATORY Zero-Jargon Interactive Clarification + Live Runtime Forecast
Before launching heavy evaluations (or whenever asking the user to choose parameters), call `explain_parameters()` / `build_interactive_brief()` and `estimate_runtime()`, then follow these **non-negotiable communication rules**:

1. **Match the User's Language**: If the user chats in Chinese, Japanese, etc., translate explanations, trade-offs, and option descriptions naturally into **that same language** instead of pasting raw English academic terms.
2. **Plain Analogy BEFORE Every Question (Zero Unexplained Jargon)**: Never throw raw symbols like `eps=8/255`, `Linf`, `steps=20`, `JSR dB`, or `APGD-DLR` and ask *"which value do you want?"* without first explaining:
   - **Plain Analogy**: What does this knob actually do in physical/visual intuition? (e.g., *"how much invisible pixel static can be added to a photo"* or *"how loud a wireless jammer is relative to your clean signal"*). Put the formal academic name in parentheses *after* the plain intuition so the user learns without feeling lost.
   - **Higher vs. Lower Trade-Off**: Explicitly tell the user what breaks if it is too small (too easy / misses real bugs) vs. too large (distorts the image/signal unrealistically so any model fails) or too slow vs. noisy.
3. **Self-Explanatory Options (Even Inside IDE Multi-Choice Cards)**: Whenever presenting options or using an IDE question widget, **every single option label must explain what happens in practice + when to pick it**, not just numbers or `"Recommended"`.
   - *Bad*: `8/255 (Linf) [Recommended]` or `standard (steps=20)`
   - *Good*: `Balanced standard noise (~3% pixel shift, invisible to eyes — matches CIFAR-10 papers, eps=8/255) [Recommended]` / `日常通用强度（允许极轻微、人眼看不出变化的 ~3% 噪点，刚好对齐绝大多数论文基准，即 eps=8/255）[推荐]`
4. **Confirm Three Decisions + Calibrated ETA Before Heavy Runs**:
   - **Noise/Jamming Strength (`eps` & `norm`)** with visual/power meaning per preset.
   - **Attack Thoroughness (`goal` / `steps` / `queries`)**: fast bug check vs. thorough search + blind anti-cheating verification.
   - **Test Scale (`N` samples) + Live `estimate_runtime()` ETA**: share measured ms/sample on actual hardware (`cpu` / `mps` / `cuda`) so the user approves the wait time upfront.

## Step 5: Automated Audits & Zero-Hallucination Academic Citations
- Always run `preflight()` + `audit_robustness()` (`evaluate_suite(..., run_audit=True)` does both automatically). If `audit.red_flags` reports `softmax_saturation_masking` or `broken_or_vanishing_gradients`, explain the trap in plain words (e.g., *"the model appears safe only because gradients disappeared or saturated, not because it is truly robust"*) and proactively switch to `apgd_dlr`, `bpda_eot`, `square_attack`, or `hopskipjump`.
- **STRICT CITATION GUARDRAIL**: NEVER invent citations or search the web for papers on built-in `wannado-adv` attacks. Always pull exact `.bib` / IEEE / APA citations verbatim from `META.citation`, `wannado-adv cite <ids>`, or `export_citations()`.

## Step 6: Mandatory Paper-Ready Post-Run Summary
After every evaluation/training run finishes, **present a clear plain-language takeaway first**, followed by **every variable required when writing an academic paper** (saved via `report.save("./adv_eval/outputs")`):
1. **Plain-Language Verdict**: Did the model genuinely resist attacks, collapse easily, or trigger any false-robustness red flags?
2. **Experimental Environment**: PyTorch version, hardware (`cpu`/`mps`/`cuda`), tensor dtype, random seed, sample count `N`, clean baseline accuracy.
3. **Threat Model & Constraint Geometry**: adversary access level, `norm`, exact numerical `eps` & physical meaning, `clip_bounds`, and empirically verified max perturbation (`max_measured <= eps`).
4. **Complete Per-Attack Hyperparameter Matrix**: `steps`, step size/`alpha`, `restarts`, `loss`, `random_start`, `queries`, `eot_samples`.
5. **Carlini Audit & Exported Artifacts**: `passed` status, any `red_flags`, and generated file paths (`paper_summary.md`, `paper_table.tex`, `references.bib`, `evaluation_report.json`).
