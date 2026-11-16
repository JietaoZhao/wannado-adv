"""End-to-end tests for AI Co-Pilot inspection, planning, scaffolding, ejection, CLI, and paper reporting."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from wannado.adv import (
    EvaluationReport,
    build_interactive_brief,
    eject_attack,
    estimate_runtime,
    evaluate_suite,
    explain_parameters,
    export_citations,
    get_sota_lineage,
    install_agent_kit,
    probe_model_context,
    recommend_suite,
    scan_project,
    scaffold_interface,
)
from wannado.adv.cli import main as cli_main


class _ComplexIQNet(nn.Module):
    def __init__(self, seq_len: int = 32, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * 2, 32),
            nn.Tanh(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([x.real, x.imag], dim=-1).reshape(x.shape[0], -1)
        return self.net(feats)


class TestCopilotInspectorAndPlanner(unittest.TestCase):
    def test_scan_project_ignores_docstrings_and_tracks_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rf_model.py").write_text(
                '"""Docstring mentions sklearn and gymnasium but does not import them."""\n'
                "import torch\n"
                "import torch.nn as nn\n\n"
                "class IQClassifier(nn.Module):\n"
                "    # handles complex64 modulation qpsk ofdm baseband signals under jsr_db\n"
                "    def forward(self, x):\n"
                "        return x.real\n",
                encoding="utf-8",
            )
            scan1 = scan_project(root)
            self.assertEqual(scan1.likely_domain, "wireless")
            self.assertEqual(scan1.recommended_norm, "jsr_db")
            self.assertIn("PyTorch", scan1.frameworks)
            self.assertNotIn("scikit-learn", scan1.frameworks)
            self.assertNotIn("Gymnasium/Gym", scan1.frameworks)
            self.assertIn("rf_model.py", scan1.model_files)

            # Adding an eval file, normalization, checkpoint, and raw docstring backslash should stay clean.
            (root / "train_rf.py").write_text(
                'import numpy as np\n"\\m valid escape test"\n',
                encoding="utf-8",
            )
            (root / "norm_loader.py").write_text(
                "from torchvision import transforms\nt = transforms.Normalize((0.5,), (0.5,))\n",
                encoding="utf-8",
            )
            (root / "weights.pt").write_bytes(b"mock")
            (root / "attacks_rival.py").write_text(
                "import torch.nn as nn\nclass FakeAttack(nn.Module):\n    pass\n",
                encoding="utf-8",
            )
            scan2 = scan_project(root)
            self.assertNotEqual(scan1.fingerprint, scan2.fingerprint)
            self.assertIn("NumPy", scan2.frameworks)
            self.assertIn("train_rf.py", scan2.entry_candidates)
            self.assertIn("weights.pt", scan2.checkpoint_files)
            self.assertIn("norm_loader.py", scan2.normalization_hints)
            self.assertNotIn("attacks_rival.py", scan2.model_files)

    def test_probe_and_estimate_runtime_preserve_model_state_and_grads(self) -> None:
        torch.manual_seed(0)
        model = nn.Sequential(nn.BatchNorm1d(16), nn.Linear(16, 4))
        model.train()
        x_train = torch.randn(8, 16)
        loss = model(x_train).sum()
        loss.backward()
        orig_grad = model[1].weight.grad.detach().clone()

        x_eval = torch.randn(6, 16)
        y_eval = torch.randint(0, 4, (6,))
        probe = probe_model_context(model, x_eval, y_eval)
        self.assertTrue(probe.autograd_connected)
        self.assertTrue(probe.preflight_ok)
        self.assertTrue(model.training)
        self.assertTrue(torch.allclose(model[1].weight.grad, orig_grad))

        suite = recommend_suite(domain="tabular", goal="sanity", autograd_ok=True)
        eta = estimate_runtime(model, x_eval, suite, num_samples=20)
        self.assertGreater(eta.total_seconds, 0.0)
        self.assertTrue(eta.autograd_available)
        self.assertTrue(model.training)
        self.assertTrue(torch.allclose(model[1].weight.grad, orig_grad))

        # Black-box callable without autograd renders N/A fwd+bwd and non-contradictory advisory
        bb_eta = estimate_runtime(lambda z: model(z).detach(), x_eval, ["square_attack"], num_samples=10)
        self.assertFalse(bb_eta.autograd_available)
        self.assertEqual(bb_eta.fwd_bwd_ms_per_sample, 0.0)
        self.assertIn("N/A", bb_eta.format_summary())
        self.assertGreater(bb_eta.per_attack_seconds["square_attack"], 0.0)

        std_suite = recommend_suite(domain="tabular", goal="standard", autograd_ok=True)
        std_eta = estimate_runtime(model, x_eval, std_suite, num_samples=20, include_audit=True)
        self.assertGreater(std_eta.audit_seconds, 0.0)
        self.assertIn("carlini_audit", std_eta.format_summary())

    def test_lineage_recommend_and_interactive_brief(self) -> None:
        lin = get_sota_lineage(family="first_order", domain="vision")
        paths = {item["id"]: item["evolution_path"] for item in lin}
        self.assertIn("apgd_dlr", paths)
        self.assertIn("fgsm -> pgd", paths["apgd_dlr"])

        masked_suite = recommend_suite(domain="vision", goal="standard", autograd_ok=False)
        self.assertEqual(masked_suite.threat_model, "black_box_score")
        self.assertIn("square_attack", masked_suite.attacks)

        brief = build_interactive_brief(domain="vision", goal="standard")
        self.assertEqual(brief["recommended_suite"]["domain"], "vision")
        self.assertNotIn("pairing 4", brief["recommended_suite"]["rationale"])
        self.assertIn("interaction_guidelines", brief)
        self.assertIn("zero_jargon_rule", brief["interaction_guidelines"])
        q_params = [q["parameter"] for q in brief["questions_to_ask_user"]]
        self.assertEqual(q_params, ["eps / norm", "goal / steps / queries", "num_samples"])
        for q in brief["questions_to_ask_user"]:
            self.assertIn("plain_analogy", q)
            self.assertIn("if_higher_vs_lower", q)
        rf_guide = explain_parameters(domain="wireless", norm="jsr_db")
        self.assertIn("Jamming-to-Signal Ratio", rf_guide["epsilon"]["meaning"])
        self.assertIn("jammer", rf_guide["epsilon"]["plain_analogy"].lower())


class TestWorkspaceIsolationEjectionAndReporting(unittest.TestCase):
    def test_eject_single_file_customize_and_evaluate_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ejected_fgsm = eject_attack("fgsm", dest_dir=root / "adv_eval" / "attacks", custom_name="fgsm_custom")
            self.assertTrue(ejected_fgsm.exists())
            text = ejected_fgsm.read_text(encoding="utf-8")
            self.assertIn("goodfellow2015explaining", text)
            self.assertIn("Extension Directions", text)

            # Verify alias ejection works too (e.g., nifgsm -> ni_fgsm module)
            ejected_ni = eject_attack("nifgsm", dest_dir=root / "adv_eval" / "attacks")
            self.assertTrue(ejected_ni.exists())

            # Dynamically load customized ejected file from external project directory
            spec = importlib.util.spec_from_file_location("fgsm_custom", ejected_fgsm)
            assert spec is not None and spec.loader is not None
            custom_mod = importlib.util.module_from_spec(spec)
            sys.modules["fgsm_custom"] = custom_mod
            try:
                spec.loader.exec_module(custom_mod)
                custom_fn = custom_mod.fgsm

                torch.manual_seed(1)
                rf_model = _ComplexIQNet(seq_len=32, num_classes=4).eval()
                re = torch.randn(16, 32)
                im = torch.randn(16, 32)
                x_iq = torch.complex(re, im).to(torch.complex64)
                y_iq = rf_model(x_iq).argmax(dim=1)

                report = evaluate_suite(
                    rf_model,
                    x_iq,
                    y_iq,
                    eps=-8.0,
                    norm="jsr_db",
                    attacks=[custom_fn, "pgd", "apgd_dlr"],
                    attack_kwargs={"pgd": {"steps": 8}, "apgd_dlr": {"steps": 10, "restarts": 1}},
                    run_audit=True,
                    audit_samples=8,
                    seed=7,
                    model_name="ComplexIQNet",
                    dataset_name="SynthRF",
                )
                self.assertIsInstance(report, EvaluationReport)
                self.assertEqual(len(report.records), 3)
                for rec in report.records:
                    self.assertIsNone(rec.error)
                    self.assertTrue(rec.budget_respected)
                    self.assertLessEqual(rec.max_perturbation, -8.0 + 1e-2)

                saved = report.save(root / "adv_eval" / "outputs")
                for key in ("json", "markdown", "latex", "bibtex"):
                    self.assertTrue(saved[key].exists())
                bib_text = saved["bibtex"].read_text(encoding="utf-8")
                self.assertIn("goodfellow2015explaining", bib_text)
                self.assertIn("madry2018towards", bib_text)
                self.assertIn("carlini2019evaluating", bib_text)
                md_text = saved["markdown"].read_text(encoding="utf-8")
                self.assertIn("Automated Checks", md_text)
                self.assertIn("Budget Sweep Monotonicity", md_text)
                tex_text = saved["latex"].read_text(encoding="utf-8")
                self.assertIn("\\begin{table}", tex_text)
                self.assertIn("jsr\\_db", tex_text)
            finally:
                sys.modules.pop("fgsm_custom", None)

    def test_scaffold_interface_runs_cleanly_standalone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp) / "adv_eval"
            written = scaffold_interface(eval_dir, domain="wireless")
            self.assertIn("interface.py", written)
            self.assertIn("run_eval.py", written)
            self.assertIn("complex64", (eval_dir / "interface.py").read_text(encoding="utf-8"))

            proc = subprocess.run(
                [sys.executable, str(eval_dir / "run_eval.py"), "--goal", "sanity", "--samples", "8", "--device", "cpu"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("Academic Evaluation Report", proc.stdout)
            self.assertTrue((eval_dir / "outputs" / "paper_table.tex").exists())
            self.assertTrue((eval_dir / "outputs" / "references.bib").exists())

            # Module invocation from project root must also resolve interface cleanly
            proc_mod = subprocess.run(
                [sys.executable, "-m", "adv_eval.run_eval", "--goal", "sanity", "--samples", "8", "--device", "cpu", "--no-audit"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc_mod.returncode, 0, msg=proc_mod.stderr)
            self.assertIn("[wannado-adv] Clean baseline:", proc_mod.stdout)

    def test_export_citations_zero_hallucination_guard(self) -> None:
        bib = export_citations(["fgsm", "pgd", "nifgsm"], fmt="bibtex", include_audit_ref=True)
        self.assertIn("goodfellow2015explaining", bib)
        self.assertIn("carlini2019evaluating", bib)
        with self.assertRaises(KeyError):
            export_citations(["nonexistent_hallucinated_attack_2099"])


class TestAgentKitAndCLI(unittest.TestCase):
    def test_install_agent_kit_universal_cursor_and_cli_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            uni_root = Path(tmp) / "uni_proj"
            uni_installed = install_agent_kit(uni_root, ide="universal")
            self.assertEqual(set(uni_installed.keys()), {"universal_skill", "agents_md"})
            self.assertTrue(uni_installed["universal_skill"].exists())
            self.assertTrue(uni_installed["agents_md"].exists())
            self.assertIn("Zero-Jargon", uni_installed["agents_md"].read_text(encoding="utf-8"))

            cursor_root = Path(tmp) / "cursor_proj"
            cursor_installed = install_agent_kit(cursor_root, ide="cursor")
            self.assertTrue(cursor_installed["cursor_rule"].exists())
            self.assertTrue(cursor_installed["universal_skill"].exists())
            self.assertTrue(cursor_installed["agents_md"].exists())

            root = Path(tmp) / "all_proj"
            root.mkdir()
            existing_agents = root / "AGENTS.md"
            existing_agents.write_text("# Existing Project Rules\n- Keep data loaders fast.\n", encoding="utf-8")
            # Hidden IDE helper files should be ignored by project domain scan
            (root / ".cursor" / "rules").mkdir(parents=True)
            (root / ".agents").mkdir(parents=True)
            (root / ".agents" / "helper.py").write_text("import gymnasium\n", encoding="utf-8")

            installed = install_agent_kit(root, ide="all")
            self.assertTrue(installed["universal_skill"].exists())
            self.assertTrue(installed["antigravity_skill"].exists())
            self.assertTrue(installed["cursor_rule"].exists())
            self.assertTrue(installed["claude_skill"].exists())
            self.assertTrue(installed["claude_md"].exists())
            updated = existing_agents.read_text(encoding="utf-8")
            self.assertIn("Keep data loaders fast.", updated)
            self.assertIn("<!-- wannado-adv:start -->", updated)
            self.assertIn("Scope & Sibling Coexistence", updated)
            self.assertIn("obs_rms_state", updated)
            self.assertIn("./drl_lab/", installed["universal_skill"].read_text(encoding="utf-8"))
            self.assertNotIn(".agents/helper.py", scan_project(root).entry_candidates)

            # Re-running install_agent_kit must stay idempotent without duplicating the block
            install_agent_kit(root, ide="all")
            self.assertEqual(existing_agents.read_text(encoding="utf-8").count("<!-- wannado-adv:start -->"), 1)

            # Smoke test CLI commands
            self.assertEqual(cli_main(["inspect", str(root)]), 0)
            self.assertEqual(cli_main(["inspect", str(root), "--json"]), 0)
            self.assertEqual(cli_main(["init-agent", "--ide", "universal", "--dir", str(Path(tmp) / "cli_uni")]), 0)
            self.assertEqual(cli_main(["catalog", "--tier", "S"]), 0)
            self.assertEqual(cli_main(["lineage", "--family", "transferability"]), 0)
            self.assertEqual(cli_main(["recommend", "--domain", "vision", "--goal", "sanity"]), 0)
            self.assertEqual(cli_main(["cite", "fgsm", "apgd_dlr", "--format", "ieee"]), 0)
            self.assertEqual(cli_main(["scaffold", "--dir", str(root / "eval_box"), "--domain", "vision", "--eps", "8/255"]), 0)
            self.assertAlmostEqual(
                float((root / "eval_box" / "interface.py").read_text(encoding="utf-8").split("eps=")[1].split(",")[0]),
                8.0 / 255.0,
                places=6,
            )
            self.assertEqual(cli_main(["eject", "fgsm", "--dest", str(root / "eval_box" / "attacks")]), 0)
            with self.assertRaises(FileExistsError):
                cli_main(["eject", "fgsm", "--dest", str(root / "eval_box" / "attacks")])


if __name__ == "__main__":
    unittest.main()
