<!-- wannado-adv:start -->
# CLAUDE.md — wannado-adv AI Co-Pilot Instructions

This project uses `wannado-adv` for modular, AI-assisted adversarial robustness evaluation. Read `.claude/skills/wannado-adv/SKILL.md` (or `.agents/skills/wannado-adv/SKILL.md` / `AGENTS.md`) and run `wannado-adv inspect .` before proposing or editing adversarial evaluation code.

- **Scope & Sibling Handoff (`wannado-drl`)**: Activate `wannado-adv` ONLY for adversarial attacks, robustness evaluation, or adversarial training (`./adv_eval/`). If `wannado-drl` is also present and the user wants to formulate an MDP or train an initial RL policy, defer to `./drl_lab/` first; when auditing robustness of an already-trained RL policy, leave `./drl_lab/` untouched and embed any frozen `obs_rms_state` (`mean`, `var`) via differentiable PyTorch ops inside `VictimInterface`.

## Core Collaboration Rules
- **Re-scan the project** (`scan_project` / `wannado-adv inspect .`) whenever repo files or models change before advising parameters.
- **Isolate adversarial code** inside `./adv_eval/` (`interface.py` + `run_eval.py` via `wannado-adv scaffold`) instead of modifying training/model modules directly.
- **Customize algorithms in single files**: use `wannado-adv eject <attack_id> --dest ./adv_eval/attacks` and edit ONLY that file when implementing new-paper upgrades.
- **Zero-Jargon interactive guidance + ETA**: Match the user's conversation language. Use `explain_parameters()` and `estimate_runtime()`, give a plain everyday analogy + higher-vs-lower trade-off BEFORE asking any parameter question, and ensure every option label explains what it does in practice (never bare jargon like `8/255` alone).
- **No fabricated citations**: always source `.bib` / IEEE / APA strings directly via `export_citations()` or `wannado-adv cite`.
- **Report verdict + paper variables after every run**: always call `evaluate_suite(..., run_audit=True)` and open with a plain-language conclusion followed by Carlini audit results, LaTeX tables, and verified `.bib` files.
<!-- wannado-adv:end -->
