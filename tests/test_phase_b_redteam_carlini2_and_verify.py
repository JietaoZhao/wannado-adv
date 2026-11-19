"""Phase B tests: Carlini 2.0 (stochastic/L1/BPDA/continuous), auto_red_team() 4-defense escalation, AURC/Wilson CI, and verify gate."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from tests.test_attack_contracts import _make_trained_real_classifier
from wannado.adv import (
    AdvVerifyReport,
    RedTeamReport,
    audit_robustness,
    auto_red_team,
    scaffold_interface,
    verify_robustness_gate,
    wilson_ci,
)
from wannado.adv.cli import main as cli_main


class TestPhaseBRedTeamCarlini2AndVerify(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.x, cls.y = _make_trained_real_classifier()

    def setUp(self) -> None:
        self.model.eval()

    def test_carlini2_catches_stochastic_defense_and_l1_and_bpda_and_continuous(self) -> None:
        # 1. Test-time stochastic randomization masking
        def stoch_model(z: torch.Tensor) -> torch.Tensor:
            return self.model(z + 0.12 * torch.randn_like(z))

        rep_stoch = audit_robustness(
            stoch_model,
            self.x[:20],
            self.y[:20],
            eps=0.2,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            pgd_steps=8,
            apgd_steps=10,
            square_queries=50,
            eot_samples=4,
            seed=0,
        )
        self.assertFalse(rep_stoch.passed)
        self.assertTrue(rep_stoch.metrics["is_stochastic"])
        self.assertIn("stochastic_defense_requires_eot", rep_stoch.red_flags)
        self.assertIsNotNone(rep_stoch.metrics["eot_pgd_acc"])

        # 2. Native L1 audit uses apgd_l1 cleanly
        rep_l1 = audit_robustness(
            self.model,
            self.x[:20],
            self.y[:20],
            eps=2.5,
            norm="l1",
            clip_bounds=(0.0, 1.0),
            pgd_steps=10,
            apgd_steps=15,
            seed=0,
        )
        self.assertTrue(rep_l1.passed, msg=rep_l1.format_report())
        self.assertLess(rep_l1.metrics["apgd_dlr_acc"], 0.6)

        # 3. Quantized defense_fn BPDA cross-check catches obfuscated defense
        def quant_def(z: torch.Tensor) -> torch.Tensor:
            return torch.round(z * 8.0) / 8.0

        with torch.no_grad():
            y_q = self.model(quant_def(self.x[:20])).argmax(1)
        rep_bpda = audit_robustness(
            self.model,
            self.x[:20],
            y_q,
            eps=0.25,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            pgd_steps=10,
            apgd_steps=12,
            square_queries=80,
            defense_fn=quant_def,
            surrogate_fn=lambda z: z,
            seed=0,
        )
        self.assertFalse(rep_bpda.passed)
        self.assertIn("obfuscated_defense_broken_by_bpda", rep_bpda.red_flags)
        self.assertFalse(rep_bpda.checks["defense_bpda_consistent"])
        self.assertLess(rep_bpda.metrics["bpda_eot_acc"], 0.5)

        # 4. Continuous regression / DRL actor audit works end-to-end
        torch.manual_seed(9)
        actor = nn.Sequential(nn.Linear(12, 24), nn.Tanh(), nn.Linear(24, 3)).eval()
        x_obs = torch.randn(16, 12)
        with torch.no_grad():
            y_act = actor(x_obs) + 0.05 * torch.randn(16, 3)
        rep_cont = audit_robustness(actor, x_obs, y_act, eps=0.2, norm="l2", pgd_steps=10, seed=0)
        self.assertEqual(rep_cont.metrics["task_type"], "continuous")
        self.assertTrue(rep_cont.passed, msg=rep_cont.format_report())
        self.assertGreater(rep_cont.metrics["pgd_mse"], rep_cont.metrics["clean_mse"])

    def test_auto_red_team_breaks_four_fake_defenses_automatically(self) -> None:
        x_sub, y_sub = self.x[:20], self.y[:20]

        # Defense 1: Softmax saturation (200x logits)
        rt_sat = auto_red_team(
            lambda z: self.model(z) * 200.0,
            x_sub,
            y_sub,
            eps=0.25,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            goal="sanity",
            sweep_curve=True,
            seed=0,
            verbose=False,
        )
        self.assertIsInstance(rt_sat, RedTeamReport)
        self.assertIn("softmax_saturation", rt_sat.detected_defenses)
        self.assertIn("apgd_dlr", [r.attack_id for r in rt_sat.records])
        self.assertIsNotNone(rt_sat.worst_case_robust_acc)
        assert rt_sat.worst_case_robust_acc is not None
        self.assertLess(rt_sat.worst_case_robust_acc, 0.35)
        self.assertIsNotNone(rt_sat.aurc)
        self.assertGreaterEqual(len(rt_sat.curve_points), 5)

        # Defense 2: Non-differentiable 4-bit quantization with wired defense_fn
        def quant_def(z: torch.Tensor) -> torch.Tensor:
            return torch.round(z * 8.0) / 8.0

        with torch.no_grad():
            y_q = self.model(quant_def(x_sub)).argmax(1)
        rt_q = auto_red_team(
            self.model,
            x_sub,
            y_q,
            eps=0.25,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            defense_fn=quant_def,
            surrogate_fn=lambda z: z,
            goal="sanity",
            sweep_curve=False,
            seed=1,
            verbose=False,
        )
        self.assertIn("bpda_eot", [r.attack_id for r in rt_q.records])
        assert rt_q.worst_case_robust_acc is not None
        self.assertLess(rt_q.worst_case_robust_acc, 0.45)

        # Defense 3: Stochastic noise injection -> escalates eot_pgd automatically
        rt_st = auto_red_team(
            lambda z: self.model(z + 0.12 * torch.randn_like(z)),
            x_sub,
            y_sub,
            eps=0.25,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            goal="sanity",
            sweep_curve=False,
            seed=2,
            verbose=False,
        )
        self.assertIn("stochastic_randomization", rt_st.detected_defenses)
        self.assertIn("eot_pgd", [r.attack_id for r in rt_st.records])

        # Defense 4: Hard-label-only oracle -> escalates boundary + hopskipjump automatically
        def hard_oracle(z: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return self.model(z).argmax(dim=1)

        rt_hard = auto_red_team(
            hard_oracle,
            x_sub[:8],
            y_sub[:8],
            eps=0.45,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            goal="sanity",
            sweep_curve=False,
            seed=3,
            verbose=False,
        )
        self.assertIn("hard_label_only", rt_hard.detected_defenses)
        self.assertEqual([r.attack_id for r in rt_hard.records], ["boundary", "hopskipjump"])

        # Export verification + Wilson CI bounds sanity
        lo, hi = wilson_ci(16, 20)
        self.assertTrue(0.5 < lo < 0.8 < hi < 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            saved = rt_sat.save(tmp)
            for key in ("red_team_json", "red_team_markdown", "latex", "bibtex", "curve_csv"):
                self.assertTrue(saved[key].exists(), msg=f"Missing exported artifact: {key}")

    def test_verify_robustness_gate_and_cli_subcommand(self) -> None:
        # Healthy model with realistic threshold passes
        v_pass = verify_robustness_gate(
            model_or_fn=self.model,
            x=self.x[:20],
            y=self.y[:20],
            eps=0.25,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            goal="sanity",
            min_clean_acc=0.70,
            min_robust_acc=0.0,
            sweep_curve=False,
            seed=0,
        )
        self.assertIsInstance(v_pass, AdvVerifyReport)
        self.assertTrue(v_pass.passed, msg=v_pass.format_report())
        self.assertEqual(v_pass.exit_code, 0)

        # Unmet min_robust_acc threshold fails cleanly with exit_code == 1
        v_low = verify_robustness_gate(
            model_or_fn=self.model,
            x=self.x[:20],
            y=self.y[:20],
            eps=0.25,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            goal="sanity",
            min_robust_acc=0.95,
            sweep_curve=False,
            seed=0,
        )
        self.assertFalse(v_low.passed)
        self.assertEqual(v_low.exit_code, 1)
        self.assertFalse(v_low.checks["min_robust_acc_ok"])

        # Logit-saturated fake defense fails gate even with min_robust_acc=0.0 due to Carlini red flag
        v_fake = verify_robustness_gate(
            model_or_fn=lambda z: self.model(z) * 200.0,
            x=self.x[:20],
            y=self.y[:20],
            eps=0.25,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            goal="sanity",
            fail_on_red_flag=True,
            sweep_curve=False,
            seed=0,
        )
        self.assertFalse(v_fake.passed)
        self.assertEqual(v_fake.exit_code, 1)
        self.assertTrue(any("softmax_saturation_masking" in rf for rf in v_fake.red_flags))

        # CLI `wannado-adv verify` on scaffolded adv_eval/interface.py emits verification_verdict.json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_interface(root / "adv_eval", domain="vision", eps=0.1, overwrite=True)
            out_dir = root / "verify_out"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli_main(
                    [
                        "verify",
                        "--interface",
                        str(root / "adv_eval" / "interface.py"),
                        "--samples",
                        "16",
                        "--goal",
                        "sanity",
                        "--allow-red-flags",
                        "--no-curve",
                        "--out-dir",
                        str(out_dir),
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0, msg=buf.getvalue())
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["passed"])
            self.assertTrue((out_dir / "verification_verdict.json").exists())
            self.assertTrue((out_dir / "red_team_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
