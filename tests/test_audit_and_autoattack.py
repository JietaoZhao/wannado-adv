"""Tests for AI-native preflight(), Carlini false-robustness auditor, and AutoAttack ensemble."""

from __future__ import annotations

import json
import unittest

import torch

from wannado.adv import (
    AuditReport,
    PreflightReport,
    audit_robustness,
    autoattack,
    compute_perturbation_norm,
    preflight,
)
from tests.test_attack_contracts import (
    TinyComplexWirelessClassifier,
    _make_trained_real_classifier,
)


class TestPreflightSafeguards(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.x, cls.y = _make_trained_real_classifier()

    def setUp(self) -> None:
        self.model.eval()

    def test_preflight_passes_clean_model_and_preserves_state(self) -> None:
        self.model.train()
        running_mean_before = self.model.net[2].running_mean.clone()
        for p in self.model.parameters():
            p.grad = None

        rep = preflight(
            self.model,
            self.x,
            self.y,
            eps=0.1,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            attack="pgd",
        )
        self.assertIsInstance(rep, PreflightReport)
        self.assertTrue(rep.ok)
        self.assertTrue(any("model.training=True" in w for w in rep.warnings))
        self.assertTrue(rep.summary["gradient_probe"]["autograd_connected"])
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(self.model.net[2].running_mean, running_mean_before))
        for p in self.model.parameters():
            self.assertIsNone(p.grad)
        self.assertIn("Preflight: PASS", rep.format_report())
        json.dumps(rep.to_dict())

    def test_preflight_detects_batchnorm_train_crash_risk_even_on_single_sample(self) -> None:
        self.model.train()
        # With B=1, calling a BatchNorm1d model in training mode directly would crash; preflight safely isolates eval().
        rep = preflight(self.model, self.x[:1], self.y[:1], eps=0.05, norm="linf", clip_bounds=(0.0, 1.0))
        self.assertTrue(rep.ok)
        self.assertTrue(any("BatchNorm" in w for w in rep.warnings))

    def test_preflight_catches_clip_bounds_on_complex_and_checks_wireless_jsr_baseline(self) -> None:
        torch.manual_seed(101)
        rx = TinyComplexWirelessClassifier(seq_len=32, num_classes=4).eval()
        iq = torch.complex(torch.randn(16, 32), torch.randn(16, 32))
        labels = rx(iq).argmax(dim=1)

        rep_bad = preflight(rx, iq, labels, eps=-3.0, norm="jsr_db", clip_bounds=(0.0, 1.0))
        self.assertFalse(rep_bad.ok)
        self.assertTrue(any("complex-valued" in e for e in rep_bad.errors))
        with self.assertRaises(ValueError):
            preflight(rx, iq, labels, eps=-3.0, norm="jsr_db", clip_bounds=(0.0, 1.0), strict=True)

        rep_good = preflight(rx, iq, labels, eps=-3.0, norm="jsr_db", attack="apgd_dlr")
        self.assertTrue(rep_good.ok)
        self.assertEqual(rep_good.errors, [])
        self.assertTrue(rep_good.summary["input"]["is_complex"])
        self.assertGreater(rep_good.summary["budget"]["resolved_mean"], 0.0)

        rep_high_jsr = preflight(rx, iq, labels, eps=3.0, norm="jsr_db")
        self.assertTrue(rep_high_jsr.ok)
        self.assertTrue(any("exceeds clean signal power" in w for w in rep_high_jsr.warnings))

    def test_preflight_detects_disconnected_autograd_and_blocks_whitebox_selection(self) -> None:
        def no_grad_oracle(z: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return self.model(z)

        rep = preflight(
            no_grad_oracle,
            self.x,
            self.y,
            eps=0.3,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            attack="pgd",
        )
        self.assertFalse(rep.ok)
        self.assertTrue(any("disconnected" in e for e in rep.errors))
        self.assertTrue(any("square_attack" in r for r in rep.recommendations))

        rep_bb = preflight(
            no_grad_oracle,
            self.x,
            self.y,
            eps=0.3,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            attack="square_attack",
        )
        self.assertTrue(rep_bb.ok)

    def test_preflight_warns_on_softmax_saturation_and_high_compute_attack(self) -> None:
        def saturated(z: torch.Tensor) -> torch.Tensor:
            return self.model(z) * 200.0

        rep_sat = preflight(saturated, self.x, self.y, eps=0.25, norm="linf", clip_bounds=(0.0, 1.0))
        self.assertTrue(rep_sat.ok)
        self.assertTrue(any("Softmax saturation" in w for w in rep_sat.warnings))
        self.assertTrue(any("apgd_dlr" in r for r in rep_sat.recommendations))

        rep_cost = preflight(self.model, self.x, self.y, eps=0.5, norm="l2", attack="hopskipjump")
        self.assertTrue(rep_cost.ok)
        self.assertTrue(any("compute_cost='high'" in w for w in rep_cost.warnings))


class TestCarliniAuditorAndAutoAttack(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.x, cls.y = _make_trained_real_classifier()

    def setUp(self) -> None:
        self.model.eval()

    def test_carlini_auditor_passes_healthy_real_and_complex_wireless_models(self) -> None:
        rep = audit_robustness(
            self.model,
            self.x[:24],
            self.y[:24],
            eps=0.25,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=0,
        )
        self.assertIsInstance(rep, AuditReport)
        self.assertTrue(rep.passed, msg=f"Unexpected failure: {rep.format_report()}")
        self.assertEqual(rep.red_flags, [])
        self.assertLess(rep.metrics["worst_case_acc"], 0.35)
        self.assertIn("PASS", rep.format_report())
        json.dumps(rep.to_dict())

        # Also validate wireless complex64 + jsr_db audit path cleanly
        torch.manual_seed(101)
        rx = TinyComplexWirelessClassifier(seq_len=32, num_classes=4).eval()
        iq = torch.complex(torch.randn(24, 32), torch.randn(24, 32))
        labels = rx(iq).argmax(dim=1)

        rep_rf = audit_robustness(rx, iq, labels, eps=-3.0, norm="jsr_db", seed=1)
        self.assertTrue(rep_rf.passed, msg=f"Unexpected wireless failure: {rep_rf.format_report()}")
        self.assertLess(rep_rf.metrics["worst_case_acc"], 0.45)

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            rx_mps = TinyComplexWirelessClassifier(seq_len=32, num_classes=4).to("mps").eval()
            iq_mps = iq[:12].to("mps")
            labels_mps = rx_mps(iq_mps).argmax(dim=1)
            rep_mps = audit_robustness(
                rx_mps,
                iq_mps,
                labels_mps,
                eps=-3.0,
                norm="jsr_db",
                pgd_steps=5,
                apgd_steps=5,
                square_queries=30,
                seed=1,
            )
            self.assertTrue(rep_mps.passed, msg=f"Unexpected MPS complex failure: {rep_mps.format_report()}")

    def test_carlini_auditor_catches_softmax_saturation_masking(self) -> None:
        def saturated_model(z: torch.Tensor) -> torch.Tensor:
            return self.model(z) * 200.0

        rep = audit_robustness(
            saturated_model,
            self.x[:24],
            self.y[:24],
            eps=0.25,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=5,
        )
        self.assertFalse(rep.passed)
        self.assertIn("softmax_saturation_masking", rep.red_flags)
        self.assertFalse(rep.checks["no_softmax_saturation"])
        self.assertGreater(rep.metrics["apgd_ce_acc"], 0.85)
        self.assertLess(rep.metrics["apgd_dlr_acc"], 0.40)
        self.assertLess(rep.metrics["apgd_dlr_acc"], rep.metrics["apgd_ce_acc"] - 0.50)

    def test_carlini_auditor_catches_broken_and_obfuscated_gradients(self) -> None:
        def quantized_oracle(z: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return self.model(torch.round(z * 50.0) / 50.0)

        rep_nograd = audit_robustness(
            quantized_oracle,
            self.x[:20],
            self.y[:20],
            eps=0.35,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            square_queries=300,
            seed=2,
        )
        self.assertFalse(rep_nograd.passed)
        self.assertIn("broken_or_vanishing_gradients", rep_nograd.red_flags)
        self.assertIn("gradient_obfuscation_blackbox_beats_whitebox", rep_nograd.red_flags)
        self.assertFalse(rep_nograd.checks["autograd_healthy"])
        self.assertFalse(rep_nograd.checks["iterative_beats_single_step"])
        self.assertFalse(rep_nograd.checks["whitebox_beats_random"])
        self.assertLess(rep_nograd.metrics["square_acc"], 0.35)
        self.assertGreater(rep_nograd.metrics["pgd_acc"], 0.85)

        # Simulated reverse-sign BPDA / obfuscated gradient defense (autograd connected, gradient lies)
        class LyingGradientDefense(torch.nn.Module):
            def __init__(self, base: torch.nn.Module) -> None:
                super().__init__()
                self.base = base

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    real_logits = self.base(z)
                # Forward equals real_logits exactly, while backward carries flipped/useless proxy gradients
                wrong_proxy = self.base(-z)
                return real_logits + (wrong_proxy - wrong_proxy.detach())

        lying_model = LyingGradientDefense(self.model).eval()
        rep_lying = audit_robustness(
            lying_model,
            self.x[:20],
            self.y[:20],
            eps=0.35,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            square_queries=300,
            seed=3,
        )
        self.assertFalse(rep_lying.passed)
        self.assertIn("gradient_obfuscation_blackbox_beats_whitebox", rep_lying.red_flags)
        self.assertLess(rep_lying.metrics["square_acc"], 0.35)

    def test_autoattack_ensemble_handles_real_saturated_non_diff_and_wireless(self) -> None:
        # 1. Real classifier standard AutoAttack
        res = autoattack(
            self.model,
            self.x[:20],
            self.y[:20],
            eps=0.25,
            norm="linf",
            version="standard",
            clip_bounds=(0.0, 1.0),
            seed=1,
        )
        self.assertLessEqual(
            compute_perturbation_norm(res.perturbation, "linf").max().item(), 0.25 + 1e-6
        )
        self.assertLess(res.extra["robust_accuracy"], 0.25)

        # 2. Saturated model where APGD-CE fails but APGD-DLR rescues
        def saturated_model(z: torch.Tensor) -> torch.Tensor:
            return self.model(z) * 200.0

        res_sat = autoattack(
            saturated_model,
            self.x[:20],
            self.y[:20],
            eps=0.25,
            norm="linf",
            version="fast",
            clip_bounds=(0.0, 1.0),
            seed=2,
        )
        self.assertGreater(res_sat.extra["stage_accuracies"]["apgd_ce"], 0.85)
        self.assertLess(res_sat.extra["robust_accuracy"], 0.25)

        # 3. Non-differentiable oracle where white-box stages skip and Square Attack solves
        def quantized_oracle(z: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return self.model(torch.round(z * 50.0) / 50.0)

        res_nd = autoattack(
            quantized_oracle,
            self.x[:16],
            self.y[:16],
            eps=0.35,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            square_queries=300,
            seed=3,
        )
        self.assertTrue(len(res_nd.extra["autograd_notes"]) > 0)
        self.assertLess(res_nd.extra["robust_accuracy"], 0.35)

        # 4. Complex wireless IQ + jsr_db
        torch.manual_seed(404)
        rx = TinyComplexWirelessClassifier(seq_len=32, num_classes=4).eval()
        iq = torch.complex(torch.randn(16, 32), torch.randn(16, 32))
        labels = rx(iq).argmax(dim=1)
        res_rf = autoattack(rx, iq, labels, eps=-3.0, norm="jsr_db", version="standard", seed=4)
        self.assertTrue(res_rf.x_adv.is_complex())
        self.assertLessEqual(
            compute_perturbation_norm(res_rf.perturbation, "jsr_db", x_ref=iq).max().item(),
            -3.0 + 1e-4,
        )
        self.assertLess(res_rf.extra["robust_accuracy"], 0.25)


if __name__ == "__main__":
    unittest.main()
