"""Phase A regression tests: continuous models, HF AutoAttack, batched FAB, DataLoader chunking, conditional ASR, and defense_fn wiring."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wannado.adv import (
    audit_robustness,
    autoattack,
    compute_perturbation_norm,
    evaluate_suite,
    fab,
    fgsm,
    pgd,
    preflight,
)
from wannado.adv.core.losses import _dlr_loss


class _ComplexTinyRx(nn.Module):
    def __init__(self, n_sym: int = 16, n_cls: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * n_sym, 32), nn.ReLU(), nn.Linear(32, n_cls))

    def forward(self, x_iq: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([x_iq.real, x_iq.imag], dim=-1).reshape(x_iq.shape[0], -1)
        return self.net(feat)


class TestPhaseACoreBatchingAndFAB(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        self.cls_model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        ).eval()
        self.x = torch.rand(24, 16)
        with torch.no_grad():
            self.y = self.cls_model(self.x).argmax(dim=1)

    def test_continuous_regression_and_drl_evaluate_suite_no_crash(self) -> None:
        torch.manual_seed(7)
        actor = nn.Sequential(nn.Linear(12, 24), nn.Tanh(), nn.Linear(24, 3)).eval()
        x_obs = torch.randn(16, 12)
        with torch.no_grad():
            y_act = actor(x_obs) + 0.05 * torch.randn(16, 3)

        pf = preflight(actor, x_obs, y_act, eps=0.1, norm="l2")
        self.assertTrue(pf.ok, msg=pf.format_report())
        self.assertEqual(pf.summary["gradient_probe"]["probe_loss_used"], "mse")
        self.assertTrue(pf.summary["gradient_probe"]["autograd_connected"])

        rep = evaluate_suite(
            actor,
            x_obs,
            y_act,
            eps=0.1,
            norm="l2",
            attacks=["fgsm", "pgd"],
            attack_kwargs={"pgd": {"steps": 10}},
            run_preflight=True,
            run_audit=True,
            verbose=False,
        )
        self.assertIsNone(rep.clean_acc)
        self.assertIsNotNone(rep.clean_mse)
        assert rep.clean_mse is not None
        self.assertEqual(len(rep.records), 2)
        for rec in rep.records:
            self.assertIsNone(rec.error)
            self.assertIsNotNone(rec.adv_mse)
            self.assertIsNotNone(rec.output_l2_drift)
            self.assertIsNotNone(rec.cosine_similarity)
            assert rec.adv_mse is not None and rec.output_l2_drift is not None
            self.assertGreater(rec.adv_mse, rep.clean_mse)
            self.assertGreater(rec.output_l2_drift, 0.0)

        md = rep.format_paper_summary()
        tex = rep.to_latex()
        self.assertIn("Adv MSE", md)
        self.assertIn("Output L2 Drift", md)
        self.assertIn("Adv MSE", tex)

    def test_autoattack_huggingface_model_output_surviving_stage2b(self) -> None:
        def hf_model(z: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(logits=self.cls_model(z))

        # Tiny eps forces samples to survive APGD-CE & APGD-DLR and enter Stage 2b targeted APGD-T
        res = autoattack(
            hf_model,
            self.x[:8],
            self.y[:8],
            eps=1e-4,
            norm="linf",
            version="fast",
            apgd_steps=4,
            apgd_restarts=1,
            n_target_classes=2,
            square_queries=10,
            clip_bounds=(0.0, 1.0),
            seed=0,
        )
        self.assertEqual(res.x_adv.shape, self.x[:8].shape)
        self.assertIn("apgd_t", res.extra["stage_accuracies"])

    def test_vectorized_batched_fab_linf_targeted_and_complex(self) -> None:
        res_linf = fab(
            self.cls_model,
            self.x[:12],
            self.y[:12],
            eps=0.35,
            norm="linf",
            steps=20,
            restarts=2,
            clip_bounds=(0.0, 1.0),
            batch_size=6,
            seed=42,
        )
        self.assertTrue(res_linf.extra["batched"])
        self.assertLessEqual(
            float(compute_perturbation_norm(res_linf.perturbation, "linf").max().item()),
            0.35 + 1e-5,
        )
        with torch.no_grad():
            adv_acc = float((self.cls_model(res_linf.x_adv).argmax(1) == self.y[:12]).float().mean().item())
        self.assertLess(adv_acc, 0.5)

        y_target = (self.y[:8] + 1) % 5
        res_t = fab(
            self.cls_model,
            self.x[:8],
            y_target,
            norm="l2",
            steps=25,
            targeted=True,
            clip_bounds=(0.0, 1.0),
            seed=42,
        )
        with torch.no_grad():
            hit = float((self.cls_model(res_t.x_adv).argmax(1) == y_target).float().mean().item())
        self.assertGreater(hit, 0.5)

        # Complex64 RF JSR-dB support in batched FAB
        rx = _ComplexTinyRx(16, 4).eval()
        x_iq = torch.complex(torch.randn(8, 16), torch.randn(8, 16)).to(torch.complex64)
        with torch.no_grad():
            y_iq = rx(x_iq).argmax(1)
        res_iq = fab(rx, x_iq, y_iq, eps=-3.0, norm="jsr_db", steps=15, seed=0)
        self.assertTrue(res_iq.x_adv.is_complex())
        self.assertLessEqual(
            float(compute_perturbation_norm(res_iq.perturbation, "jsr_db", x_iq).max().item()),
            -3.0 + 1e-2,
        )

    def test_dataloader_and_minibatch_evaluate_suite_and_autoattack(self) -> None:
        loader = DataLoader(TensorDataset(self.x, self.y), batch_size=8, shuffle=False)
        rep_loader = evaluate_suite(
            self.cls_model,
            loader,
            eps=0.15,
            norm="linf",
            attacks=["fgsm", "pgd", "autoattack"],
            attack_kwargs={
                "pgd": {"steps": 10},
                "autoattack": {"version": "fast", "apgd_steps": 8, "apgd_restarts": 1, "square_queries": 30},
            },
            clip_bounds=(0.0, 1.0),
            run_audit=True,
            audit_samples=16,
            seed=7,
            verbose=False,
        )
        self.assertEqual(rep_loader.num_samples, 24)
        self.assertAlmostEqual(rep_loader.clean_acc or 0.0, 1.0, places=5)
        self.assertIsNotNone(rep_loader.audit)
        for rec in rep_loader.records:
            self.assertIsNone(rec.error)
            self.assertTrue(rec.budget_respected)
            self.assertEqual(rec.extra.get("num_batches"), 3)

        # Direct chunked AutoAttack + DataLoader audit_robustness
        aa_chunked = autoattack(
            self.cls_model,
            self.x,
            self.y,
            eps=0.15,
            norm="linf",
            version="fast",
            apgd_steps=8,
            apgd_restarts=1,
            square_queries=25,
            clip_bounds=(0.0, 1.0),
            seed=7,
            batch_size=8,
        )
        self.assertEqual(aa_chunked.x_adv.shape, self.x.shape)
        self.assertEqual(aa_chunked.extra.get("chunk_size"), 8)

        aud = audit_robustness(
            self.cls_model,
            loader,
            eps=0.15,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            pgd_steps=5,
            apgd_steps=8,
            square_queries=20,
            max_samples=16,
            seed=0,
        )
        self.assertIsNotNone(aud.checks)

    def test_conditional_asr_zero_when_attack_fails_on_imperfect_model(self) -> None:
        # Half labels match prediction, half intentionally wrong -> clean_acc == 0.50
        y_half = self.y.clone()
        y_half[12:] = (y_half[12:] + 1) % 5

        rep = evaluate_suite(
            self.cls_model,
            self.x,
            y_half,
            eps=0.0,
            norm="linf",
            attacks=["fgsm", "pgd"],
            clip_bounds=(0.0, 1.0),
            run_audit=False,
            verbose=False,
        )
        self.assertAlmostEqual(rep.clean_acc or 0.0, 0.5, places=5)
        for rec in rep.records:
            self.assertAlmostEqual(rec.robust_acc or 0.0, 0.5, places=5)
            # Zero perturbation fooled 0 clean-correct samples -> conditional ASR must be 0.0 (not 0.5!)
            self.assertAlmostEqual(rec.attack_success_rate or 0.0, 0.0, places=5)
            self.assertAlmostEqual(rec.extra["misclassification_rate"], 0.5, places=5)

    def test_victim_interface_defense_fn_and_surrogate_fn_auto_wiring(self) -> None:
        def quant_defense(z: torch.Tensor) -> torch.Tensor:
            return torch.round(z * 8.0) / 8.0

        with torch.no_grad():
            y_def = self.cls_model(quant_defense(self.x)).argmax(dim=1)

        rep = evaluate_suite(
            self.cls_model,
            self.x,
            y_def,
            eps=0.25,
            norm="linf",
            attacks=["pgd", "bpda_eot"],
            attack_kwargs={
                "pgd": {"steps": 15, "random_start": False},
                "bpda_eot": {"steps": 20, "eot_samples": 2},
            },
            clip_bounds=(0.0, 1.0),
            defense_fn=quant_defense,
            surrogate_fn=lambda z: z,
            run_audit=False,
            seed=0,
            verbose=False,
        )
        rec_pgd = rep.records[0]
        rec_bpda = rep.records[1]
        self.assertIsNone(rec_pgd.error)
        self.assertIsNone(rec_bpda.error)
        assert rec_pgd.robust_acc is not None and rec_bpda.robust_acc is not None
        # Naive PGD stalls because torch.round has 0 gradient; auto-wired BPDA breaks the defense
        self.assertGreater(rec_pgd.robust_acc, 0.85)
        self.assertLess(rec_bpda.robust_acc, rec_pgd.robust_acc - 0.30)

    def test_pgd_fgsm_targeted_dlr_and_exact_croce_denominator(self) -> None:
        logits = torch.tensor([[10.0, 6.0, 2.0, 0.0, -4.0]])
        y_t = torch.tensor([1])
        # Croce & Hein targeted DLR denom for C>=4: z_(1) - 0.5*(z_(3)+z_(4)) = 10 - 0.5*(2+0) = 9.0
        dlr_val = _dlr_loss(logits, y_t, targeted=True)
        expected = (10.0 - 6.0) / 9.0
        self.assertAlmostEqual(float(dlr_val.item()), expected, places=5)

        y_target = (self.y[:8] + 1) % 5
        r_fgsm = fgsm(self.cls_model, self.x[:8], y_target, eps=0.2, loss="dlr", targeted=True, clip_bounds=(0.0, 1.0))
        r_pgd = pgd(self.cls_model, self.x[:8], y_target, eps=0.2, steps=10, loss="dlr", targeted=True, clip_bounds=(0.0, 1.0), seed=0)
        self.assertTrue(torch.isfinite(r_fgsm.x_adv).all())
        self.assertTrue(torch.isfinite(r_pgd.x_adv).all())


if __name__ == "__main__":
    unittest.main()
