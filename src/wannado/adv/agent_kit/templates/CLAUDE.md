# CLAUDE.md — wannado-adv Autonomous Red-Team & Hardening Instructions

This project uses `wannado-adv` for autonomous, modular adversarial robustness red-teaming, hardening, and certified L2 smoothing. Read `.claude/skills/wannado-adv/SKILL.md` (or `.agents/skills/wannado-adv/SKILL.md` / `AGENTS.md`) and run `wannado-adv inspect .` before editing adversarial evaluation code.

- **Scope & Sibling Handoff (`wannado-drl`)**: Activate `wannado-adv` ONLY for adversarial attacks, robustness auditing, adversarial training, or certified smoothing (`./adv_eval/`). If `wannado-drl` is also present and the user wants to formulate an MDP or train an initial RL policy, defer to `./drl_lab/` first; when auditing robustness of an already-trained RL policy, leave `./drl_lab/` untouched and embed any frozen `obs_rms_state` (`mean`, `var`) via differentiable PyTorch ops inside `VictimInterface`.

## Core Autonomous Workflow Rules
- **Inspect & Execute Directly**: Run `wannado-adv inspect .` + `probe_model_context()`, isolate the victim in `./adv_eval/interface.py` (`wannado-adv scaffold`), and run `wannado-adv verify` (or `auto_red_team()`) directly without unnecessary pre-run interrogation.
- **Non-Intrusive & Single-File Ejection**: Never scatter adversarial loops across model/training files. Use `wannado-adv eject <attack_or_defense_id>` when modifying any attack or defense algorithm.
- **Closed-Loop Hardening & Carlini 2.0 Re-Audit**: Use `pgd_adversarial_train`, `trades_train`, `mart_train`, or `certify_randomized_smoothing` when hardening/certifying, then re-run `wannado-adv verify` / `audit_robustness()` to verify genuine robustness (`passed=True`).
- **Zero-Jargon Verdict & Citations**: Match the user's language, open with a plain-language conclusion (and explain knobs plainly with `estimate_runtime()` ETA whenever user input is actually needed), and export verbatim citations strictly via `wannado-adv cite` / `export_citations()`.
