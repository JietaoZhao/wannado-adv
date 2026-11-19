"""Command-line interface for the wannado-adv AI-collaborative adversarial robustness platform."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from wannado.adv import __version__
from wannado.adv.agent_kit import install_agent_kit
from wannado.adv.audit.preflight import preflight as run_preflight
from wannado.adv.audit.verify_gate import verify_robustness_gate
from wannado.adv.catalog.build_catalog import build_registry_json, load_catalog, query_catalog
from wannado.adv.copilot import (
    build_interactive_brief,
    explain_parameters,
    get_sota_lineage,
    recommend_suite,
    scan_project,
)
from wannado.adv.reporter import export_citations
from wannado.adv.workspace import eject_attack, scaffold_interface


def _parse_eps_arg(raw: str) -> float:
    text = raw.strip()
    if "/" in text:
        parts = text.split("/", 1)
        return float(parts[0].strip()) / float(parts[1].strip())
    return float(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wannado-adv",
        description="Universal AI-Native adversarial learning platform CLI (AGENTS.md / Agent Skills standard).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. inspect
    p_inspect = sub.add_parser("inspect", help="Scan a project directory, recommend SOTA attacks, and explain parameters.")
    p_inspect.add_argument("project_dir", nargs="?", default=".", help="Project root to inspect.")
    p_inspect.add_argument("--goal", default="standard", choices=["sanity", "standard", "paper_sota"])
    p_inspect.add_argument("--json", action="store_true", dest="as_json", help="Print machine-readable JSON brief.")

    # 2. catalog
    p_cat = sub.add_parser("catalog", help="Search the 35-algorithm metadata catalog.")
    p_cat.add_argument("--family", default=None)
    p_cat.add_argument("--tier", default=None)
    p_cat.add_argument("--domain", default=None)
    p_cat.add_argument("--norm", default=None)
    p_cat.add_argument("--threat-model", default=None)
    p_cat.add_argument("--rebuild", action="store_true")
    p_cat.add_argument("--json", action="store_true", dest="as_json")

    # 3. lineage
    p_lin = sub.add_parser("lineage", help="Inspect algorithm evolution chains from baseline to SOTA.")
    p_lin.add_argument("--family", default=None)
    p_lin.add_argument("--domain", default=None)
    p_lin.add_argument("--threat-model", default=None)
    p_lin.add_argument("--json", action="store_true", dest="as_json")

    # 4. recommend
    p_rec = sub.add_parser("recommend", help="Recommend an evaluation suite and novice parameter guide.")
    p_rec.add_argument("--domain", default="vision")
    p_rec.add_argument("--goal", default="standard", choices=["sanity", "standard", "paper_sota"])
    p_rec.add_argument("--threat-model", default="white_box", choices=["white_box", "black_box_score", "black_box_decision"])
    p_rec.add_argument("--no-autograd", action="store_true", help="Target a non-differentiable / masked model.")
    p_rec.add_argument("--norm", default=None)
    p_rec.add_argument("--json", action="store_true", dest="as_json")

    # 5. cite
    p_cite = sub.add_parser("cite", help="Export verified citations (BibTeX/IEEE/APA) with zero hallucination.")
    p_cite.add_argument(
        "attack_ids",
        nargs="+",
        help="Attack or defense IDs to cite (e.g. fgsm pgd apgd_dlr trades mart randomized_smoothing).",
    )
    p_cite.add_argument("--format", default="bibtex", choices=["bibtex", "ieee", "apa"])
    p_cite.add_argument("--include-audit-ref", action="store_true")

    # 6. eject
    p_eject = sub.add_parser(
        "eject",
        help="Copy a standalone single-file attack or defense into your workspace for custom edits.",
    )
    p_eject.add_argument(
        "attack_id",
        help="Attack or defense ID (e.g. fgsm, pgd, apgd_dlr, pgd_at, trades, mart, randomized_smoothing).",
    )
    p_eject.add_argument("--dest", default="./adv_eval/attacks", help="Destination folder.")
    p_eject.add_argument("--name", default=None, help="Optional custom output filename.")
    p_eject.add_argument("--overwrite", action="store_true")

    # 7. scaffold
    p_scaf = sub.add_parser("scaffold", help="Generate a non-intrusive adv_eval/ interface harness in a project.")
    p_scaf.add_argument("--dir", default="./adv_eval", dest="target_dir")
    p_scaf.add_argument("--domain", default="vision")
    p_scaf.add_argument("--norm", default=None)
    p_scaf.add_argument("--eps", type=_parse_eps_arg, default=None, help="Perturbation budget (float or fraction e.g. 8/255).")
    p_scaf.add_argument("--overwrite", action="store_true")

    # 8. init-agent
    p_init = sub.add_parser(
        "init-agent",
        help="Install universal AGENTS.md / Agent Skills rules + Cursor/Antigravity/Claude/Codex adapters.",
    )
    p_init.add_argument("--dir", default=".", dest="project_dir")
    p_init.add_argument(
        "--ide",
        default="all",
        choices=["all", "universal", "cursor", "antigravity", "claude", "codex"],
    )

    # 9. preflight
    p_pf = sub.add_parser(
        "preflight",
        help="Run autograd, budget, NaN, and wireless JSR preflight checks on an adv_eval/interface.py.",
    )
    p_pf.add_argument("--interface", default="./adv_eval/interface.py", dest="interface_path")
    p_pf.add_argument("--device", default="cpu")
    p_pf.add_argument("--samples", type=int, default=32)
    p_pf.add_argument("--attack", default=None)
    p_pf.add_argument("--json", action="store_true", dest="as_json")

    # 10. verify (autonomous closed-loop red-team + hard CI/agent gate)
    p_ver = sub.add_parser(
        "verify",
        help="Run closed-loop autonomous red-team + Carlini 2.0 hard verification gate on adv_eval/interface.py.",
    )
    p_ver.add_argument("--interface", default="./adv_eval/interface.py", dest="interface_path")
    p_ver.add_argument("--device", default="cpu")
    p_ver.add_argument("--samples", type=int, default=64)
    p_ver.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p_ver.add_argument("--goal", default="standard", choices=["sanity", "standard", "paper_sota"])
    p_ver.add_argument("--eps", type=_parse_eps_arg, default=None)
    p_ver.add_argument("--norm", default=None)
    p_ver.add_argument("--min-clean-acc", type=float, default=None, dest="min_clean_acc")
    p_ver.add_argument("--min-robust-acc", type=float, default=None, dest="min_robust_acc")
    p_ver.add_argument("--max-output-drift", type=float, default=None, dest="max_output_drift")
    p_ver.add_argument("--allow-red-flags", action="store_true", dest="allow_red_flags")
    p_ver.add_argument("--no-curve", action="store_true", dest="no_curve")
    p_ver.add_argument("--seed", type=int, default=42)
    p_ver.add_argument("--out-dir", default="./adv_eval/outputs", dest="out_dir")
    p_ver.add_argument("--json", action="store_true", dest="as_json")

    # 11. arsenal
    p_ars = sub.add_parser(
        "arsenal",
        help="Display versioned wannado-adv adversarial arsenal & Carlini false-robustness audit stack.",
    )
    p_ars.add_argument("--json", action="store_true", dest="as_json")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "inspect":
        scan = scan_project(args.project_dir)
        brief = build_interactive_brief(project_scan=scan, goal=args.goal)
        if args.as_json:
            print(json.dumps(brief, indent=2, ensure_ascii=False))
        else:
            print(scan.format_summary())
            rec = brief["recommended_suite"]
            print(f"\n[Recommended '{args.goal}' Suite ({rec['domain']}, norm='{rec['norm']}')]")
            print(f"  Attacks  : {', '.join(rec['attacks'])}")
            print(f"  Rationale: {rec['rationale']}")
            print("\n[Questions to Decide with User]")
            for q in brief["questions_to_ask_user"]:
                print(f"  - {q['prompt']}")
                print(f"    Plain idea : {q['plain_analogy']}")
                print(f"    Trade-off  : {q['if_higher_vs_lower']}")
                print(f"    Technical  : {q['novice_explanation']}")
                for opt in q.get("options", []):
                    opt_text = opt.get("label") or opt.get("meaning") or str(opt)
                    print(f"      * {opt_text}")
        return 0

    if args.command == "catalog":
        if args.rebuild:
            build_registry_json()
        cards = query_catalog(
            family=args.family,
            tier=args.tier,
            domain=args.domain,
            norm=args.norm,
            threat_model=args.threat_model,
        )
        if args.as_json:
            print(json.dumps(cards, indent=2, ensure_ascii=False))
        else:
            print(f"Matched {len(cards)} algorithms:")
            for c in cards:
                print(f"  [{c['tier']}] {c['id']:<14s} ({c['family']}) — {c['summary_plain']}")
        return 0

    if args.command == "lineage":
        entries = get_sota_lineage(family=args.family, domain=args.domain, threat_model=args.threat_model)
        if args.as_json:
            print(json.dumps(entries, indent=2, ensure_ascii=False))
        else:
            for e in entries:
                print(f"  [{e['year']}][{e['tier']}] {e['id']:<14s} | {e['evolution_path']}")
                print(f"      -> {e['diff_from_parent']}")
        return 0

    if args.command == "recommend":
        suite = recommend_suite(
            domain=args.domain,
            goal=args.goal,
            threat_model=args.threat_model,
            autograd_ok=not args.no_autograd,
            norm=args.norm,
        )
        guide = explain_parameters(domain=args.domain, norm=suite.norm)
        payload = {"recommended_suite": suite.to_dict(), "parameter_guide": guide}
        if args.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"Suite ({suite.goal} | {suite.domain} | norm='{suite.norm}'): {suite.attacks}")
            print(f"Rationale: {suite.rationale}")
        return 0

    if args.command == "cite":
        print(export_citations(args.attack_ids, fmt=args.format, include_audit_ref=args.include_audit_ref))
        return 0

    if args.command == "eject":
        dest = eject_attack(args.attack_id, dest_dir=args.dest, custom_name=args.name, overwrite=args.overwrite)
        print(f"Ejected standalone attack file to: {dest}")
        return 0

    if args.command == "scaffold":
        files = scaffold_interface(
            target_dir=args.target_dir,
            domain=args.domain,
            norm=args.norm,
            eps=args.eps,
            overwrite=args.overwrite,
        )
        print(f"Scaffolded isolated evaluation interface under {Path(args.target_dir).resolve()}:")
        for rel, path in files.items():
            print(f"  - {rel}: {path}")
        return 0

    if args.command == "init-agent":
        written = install_agent_kit(project_dir=args.project_dir, ide=args.ide)
        print(f"Installed wannado-adv AI Co-Pilot skill/rules for ide='{args.ide}':")
        for role, path in written.items():
            print(f"  - {role}: {path}")
        return 0

    if args.command == "preflight":
        iface_path = Path(args.interface_path).resolve()
        spec = importlib.util.spec_from_file_location("_wannado_adv_user_iface", iface_path)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"Cannot load interface module at {iface_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        loader_fn = mod.load_victim_interface
        sig_params = inspect.signature(loader_fn).parameters
        loader_kwargs: dict[str, Any] = {}
        if "device" in sig_params:
            loader_kwargs["device"] = args.device
        if "num_samples" in sig_params:
            loader_kwargs["num_samples"] = args.samples
        victim = loader_fn(**loader_kwargs)
        pf = run_preflight(
            victim.model_or_fn,
            victim.x_eval,
            victim.y_eval,
            eps=victim.eps,
            norm=victim.norm,
            clip_bounds=victim.clip_bounds,
            attack=args.attack,
            defense_fn=victim.defense_fn,
        )
        if args.as_json:
            print(json.dumps(pf.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(pf.format_report())
        return 0 if pf.ok else 1

    if args.command == "verify":
        rep = verify_robustness_gate(
            interface_path=args.interface_path,
            device=args.device,
            samples=args.samples,
            batch_size=args.batch_size,
            goal=args.goal,
            eps=args.eps,
            norm=args.norm,
            min_clean_acc=args.min_clean_acc,
            min_robust_acc=args.min_robust_acc,
            max_output_drift=args.max_output_drift,
            fail_on_red_flag=not args.allow_red_flags,
            sweep_curve=not args.no_curve,
            seed=args.seed,
            out_dir=args.out_dir,
            verbose=not args.as_json,
        )
        print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False) if args.as_json else rep.format_report())
        return rep.exit_code

    if args.command == "arsenal":
        from wannado.adv.defenses import list_defenses

        cards = load_catalog()
        defenses = list_defenses()
        by_family: dict[str, list[dict[str, Any]]] = {}
        for c in cards:
            by_family.setdefault(c["family"], []).append(c)
        arsenal_payload: dict[str, Any] = {
            "package": "wannado-adv",
            "version": __version__,
            "total_attacks": len(cards),
            "total_defenses": len(defenses),
            "families": {
                fam: [{"id": c["id"], "tier": c["tier"], "threat_model": c["threat_model"]} for c in items]
                for fam, items in by_family.items()
            },
            "defenses": [
                {"id": d["id"], "tier": d["tier"], "paradigm": d["paradigm"]}
                for d in defenses
            ],
            "audits": [
                "preflight",
                "audit_robustness (Carlini 2.0: stochasticity/EOT, BPDA, L1, regression)",
                "auto_red_team (closed-loop adaptive escalation + Wilson 95% CI + AURC/eps50)",
                "verify_robustness_gate (wannado-adv verify hard exit-code gate)",
            ],
        }
        if args.as_json:
            print(json.dumps(arsenal_payload, indent=2, ensure_ascii=False))
        else:
            print(
                f"[wannado-adv v{__version__}] Adversarial Robustness Arsenal "
                f"({len(cards)} attacks + {len(defenses)} single-file defenses):"
            )
            for fam, items in by_family.items():
                summary = ", ".join(f"{c['id']}[{c['tier']}]" for c in items)
                print(f"  - {fam:<18s} ({len(items):2d}): {summary}")
            def_summary = ", ".join(f"{d['id']}[{d['tier']}]" for d in defenses)
            print(f"  - {'defenses':<18s} ({len(defenses):2d}): {def_summary}")
            print(
                "  Audits & Red-Team  : preflight(), audit_robustness() [Carlini 2.0], "
                "auto_red_team(), verify_robustness_gate() (`wannado-adv verify`)"
            )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
