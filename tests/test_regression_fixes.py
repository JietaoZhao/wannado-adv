"""Regression tests covering v1.0.4 correctness, API consistency, and boundary fixes."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import wannado.adv as wadv
from tests.test_attack_contracts import (
    TinyComplexWirelessClassifier,
    _make_trained_real_classifier,
)
from wannado.adv import (
    attacks as wadv_attacks,
)
from wannado.adv import (
    audit_robustness,
    compute_perturbation_norm,
    eot_pgd,
    evaluate_suite,
    load_catalog,
    mi_fgsm,
    mifgsm,
    project_perturbation,
    resolve_eps_budget,
)


class TestReleaseV104Regressions(unittest.TestCase):
    def test_eot_pgd_complex_with_positive_noise_std(self) -> None:
        torch.manual_seed(11)
        rx = TinyComplexWirelessClassifier(seq_len=32, num_classes=4).eval()
        iq = torch.complex(torch.randn(8, 32), torch.randn(8, 32))
        with torch.no_grad():
            labels = rx(iq).argmax(dim=1)

        res = eot_pgd(
            rx,
            iq,
            labels,
            eps=-6.0,
            norm="jsr_db",
            steps=10,
            eot_samples=4,
            noise_std=0.05,
            seed=12,
        )
        self.assertTrue(res.x_adv.is_complex())
        self.assertTrue(torch.isfinite(res.x_adv.real).all())
        self.assertTrue(torch.isfinite(res.x_adv.imag).all())
        measured_db = compute_perturbation_norm(res.perturbation, "jsr_db", x_ref=iq)
        self.assertLessEqual(float(measured_db.max().item()), -6.0 + 1e-4)

    def test_every_catalog_id_exported_and_resolvable_including_mi_fgsm(self) -> None:
        self.assertIs(mi_fgsm, mifgsm)
        cards = load_catalog()
        for card in cards:
            cid = card["id"]
            self.assertIn(cid, wadv.__all__)
            self.assertIn(cid, wadv_attacks.__all__)
            self.assertIn(cid, wadv_attacks.ATTACK_EXPORTS)
            fn = getattr(wadv, cid)
            self.assertTrue(callable(fn))

        model, x, y = _make_trained_real_classifier()
        rep = evaluate_suite(
            model,
            x[:16],
            y[:16],
            eps=0.1,
            norm="linf",
            attacks=["mi_fgsm"],
            attack_kwargs={"mi_fgsm": {"steps": 5}},
            clip_bounds=(0.0, 1.0),
            run_audit=False,
        )
        self.assertEqual(rep.records[0].attack_id, "mi_fgsm")

    def test_project_perturbation_rejects_infeasible_clip_bounds_and_keeps_l2_feasible(self) -> None:
        x_unbounded = torch.tensor([[3.0, -3.0, 0.2, 0.4]])
        delta = torch.full_like(x_unbounded, 0.03)
        for norm in ("linf", "l2", "l1"):
            budget = resolve_eps_budget(x_unbounded, eps=0.03, norm=norm)
            with self.assertRaises(ValueError):
                project_perturbation(delta, x_unbounded, budget, norm=norm, clip_bounds=(0.0, 1.0))

        # Slightly outside bounds within L2 budget must project into box AND respect L2 budget.
        x_edge = torch.tensor([[-0.01, 0.5, 0.5, 0.5]])
        raw_delta = torch.tensor([[0.10, 0.10, 0.10, 0.10]])
        budget_l2 = resolve_eps_budget(x_edge, eps=0.05, norm="l2")
        proj = project_perturbation(raw_delta, x_edge, budget_l2, norm="l2", clip_bounds=(0.0, 1.0))
        x_adv = x_edge + proj
        self.assertGreaterEqual(float(x_adv.min().item()), -1e-6)
        self.assertLessEqual(float(x_adv.max().item()), 1.0 + 1e-6)
        self.assertLessEqual(float(compute_perturbation_norm(proj, "l2").max().item()), 0.05 + 1e-5)

    def test_carlini_auditor_flags_non_monotonic_mid_to_high_rebound(self) -> None:
        model, x, y = _make_trained_real_classifier()

        # Simulate acc_low=0.90, acc_mid=0.20, acc_high=0.60: high <= low, yet high > mid!
        call_accs = [0.85, 0.80, 0.75, 0.20, 0.70, 0.90, 0.60]
        it = iter(call_accs)
        with patch(
            "wannado.adv.audit.carlini_auditor._safe_attack_acc",
            side_effect=lambda *args, **kwargs: (next(it), None),
        ):
            audit = audit_robustness(model, x[:16], y[:16], eps=0.1, norm="linf", clip_bounds=(0.0, 1.0))
        self.assertFalse(audit.checks["monotonic_over_budget"])
        self.assertIn("non_monotonic_budget_response", audit.red_flags)


if __name__ == "__main__":
    unittest.main()
