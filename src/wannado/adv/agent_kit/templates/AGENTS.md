# wannado-adv Universal AI Collaboration Rules

This repository uses `wannado-adv` under the open `AGENTS.md` + `.agents/skills/wannado-adv/SKILL.md` standard (compatible across Cursor, Antigravity, Codex, Claude Code, Windsurf, and other AI coding agents).

- **Scope & Sibling Coexistence (`wannado-drl`)**: Activate `wannado-adv` ONLY for adversarial robustness testing, perturbation attacks, or adversarial training (`./adv_eval/`). If `wannado-drl` is also installed and the user has NOT trained an RL policy yet (asking to design an MDP or solve/train RL from scratch), defer to `wannado-drl` (`./drl_lab/`) first instead of hijacking into `adv_eval/`. When evaluating robustness of a trained `wannado-drl` policy, keep `./drl_lab/` untouched, wrap the victim inside `./adv_eval/interface.py` (`VictimInterface`), and embed any saved frozen `obs_rms_state` (`mean`, `var`, clamped `[-10.0, 10.0]`) via differentiable PyTorch ops in the forward pass so adversarial gradients remain valid and statistics never drift.

Follow these rules whenever running adversarial robustness tests or adversarial training:

1. **Always Inspect First & Re-Scan Periodically**: Run `wannado-adv inspect .` (or `wannado.adv.copilot.scan_project(".")` + `probe_model_context(...)`) before recommending attacks or hyperparams, and re-run after project edits so you never guess from stale assumptions.
2. **Non-Intrusive Single-File Modularity**: Never scatter adversarial loops across the project's model/training source files. Keep evaluation hooks inside ONE isolated boundary directory (e.g. `./adv_eval/` via `wannado-adv scaffold`), and if an algorithm needs custom paper edits, copy ONLY that standalone file via `wannado-adv eject <attack_id>` into `./adv_eval/attacks/` and edit it in place.
3. **Zero-Jargon Interactive Clarification + Hardware ETA**:
   - **Speak the user's conversation language** (e.g., explain in natural Chinese if the user asks in Chinese).
   - **Never ask raw academic questions without plain-language intuition first**: Before presenting choices on `eps`/`norm`, `steps`/`queries`, sample scale `N`, or attack suites, call `explain_parameters()` + `estimate_runtime()` and explain *each knob with an everyday analogy*, *what changes if higher vs. lower*, and *the calibrated ETA on real hardware*.
   - **Make every option label self-explanatory**: Never show bare numbers like `8/255 [Recommended]` alone — spell out what it means in practice (e.g., *"Invisible ~3% pixel tweak, standard CIFAR-10 ruler (eps=8/255) [Recommended]"*). Put academic terms in parentheses after plain intuition.
4. **Zero Citation Hallucination**: Never search the web or guess citations from memory for `wannado-adv` algorithms. Always obtain exact BibTeX/IEEE/APA verbatim from `wannado-adv cite <ids>` or `wannado.adv.reporter.export_citations()`.
5. **Mandatory Paper-Ready Post-Run Report**: Execute via `evaluate_suite(..., run_audit=True)` and open with a plain-language conclusion before surfacing Carlini audit checks and full publication variables (`evaluation_report.json`, `paper_summary.md`, `paper_table.tex`, `references.bib`).
