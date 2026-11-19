"""Regression tests for v1.2.2 deep audit fixes across attacks, defenses, auditing, and co-pilot."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wannado.adv import (
    NormalizedModelWrapper,
    apgd_dlr,
    apgd_l1,
    audit_robustness,
    auto_red_team,
    autoattack,
    boundary,
    certify_randomized_smoothing,
    cw_l2,
    ead,
    estimate_runtime,
    evaluate_suite,
    fab,
    gap,
    gaussian_augmented_train,
    pgd_adversarial_train,
    preflight,
    probe_model_context,
    recommend_suite,
    smooth_predict,
    square_attack,
    trades_train,
    verify_robustness_gate,
    wrap_normalized_model,
    wrap_sklearn_model,
)
from wannado.adv.core.losses import resolve_loss_fn


def _make_linear_cls(in_dim: int = 8, num_classes: int = 4, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(in_dim, 24), nn.ReLU(), nn.Linear(24, num_classes))
    return model.eval()


class TestAuditFixesV122(unittest.TestCase):
    def test_complex64_smoothing_certificate_uses_sigma_over_sqrt2(self) -> None:
        # Exact linear half-space classifier on Re(z_0) >= 0:
        # Under circular complex noise CN(0, sigma^2 I), Re(n) ~ N(0, (sigma/sqrt(2))^2),
        # so the true certified L2 radius at x_0 = 1+0j cannot exceed 1.0 (previously overstated by sqrt(2)).
        class _HalfSpaceComplex(nn.Module):
            def forward(self, z: torch.Tensor) -> torch.Tensor:
                score = z.real[:, 0]
                return torch.stack([-score, score], dim=1)

        model = _HalfSpaceComplex()
        x = torch.zeros(2, 4, dtype=torch.complex64)
        x[:, 0] = 1.0 + 0.0j
        y = torch.ones(2, dtype=torch.long)

        cert = certify_randomized_smoothing(model, x, y, sigma=0.5, n0=50, n=1200, alpha=0.001, seed=7)
        self.assertEqual(cert.predictions, [1, 1])
        # Theoretical max radius at distance 1.0 from boundary is strictly <= 1.0!
        self.assertTrue(all(r > 0.0 for r in cert.radii))
        self.assertLessEqual(max(cert.radii), 1.02)

    def test_smooth_predict_optional_cohen_abstain(self) -> None:
        # Coin-flip tie classifier should ABSTAIN (-1) when alpha is provided
        class _TieModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                s = x[:, 0]
                return torch.stack([-s, s], dim=1)

        x = torch.zeros(4, 4)
        preds_abstain = smooth_predict(_TieModel(), x, sigma=0.5, n_samples=64, alpha=0.001, seed=0)
        self.assertTrue(bool((preds_abstain == -1).any()))

    def test_binary_dlr_loss_has_nonzero_gradient_and_attacks_work(self) -> None:
        torch.manual_seed(1)
        x = torch.randn(8, 6, requires_grad=True)
        w = torch.randn(6, 2)
        logits = x @ w
        y = logits.argmax(dim=1)
        loss = resolve_loss_fn("dlr")(logits, y)
        (g,) = torch.autograd.grad(loss.sum(), x)
        self.assertGreater(float(g.abs().sum().item()), 1e-5)

        model = _make_linear_cls(in_dim=6, num_classes=2, seed=2)
        x_eval = torch.randn(12, 6)
        with torch.no_grad():
            y_eval = model(x_eval).argmax(dim=1)
        res = apgd_dlr(model, x_eval, y_eval, eps=0.6, norm="linf", steps=15, restarts=1, seed=0)
        self.assertGreater(float(res.perturbation.abs().max().item()), 0.05)

    def test_column_vector_targets_work_across_attacks_and_defenses(self) -> None:
        model = _make_linear_cls(in_dim=8, num_classes=3, seed=3)
        x = torch.rand(12, 8)
        y_col = torch.randint(0, 3, (12, 1))

        res_sq = square_attack(model, x, y_col, eps=0.2, norm="linf", max_queries=25, clip_bounds=(0.0, 1.0), seed=0)
        self.assertEqual(res_sq.x_adv.shape, x.shape)

        res_fab = fab(model, x[:4], y_col[:4], eps=0.5, norm="l2", steps=5, restarts=1, seed=0)
        self.assertEqual(res_fab.x_adv.shape, x[:4].shape)

        res_aa = autoattack(
            model, x[:6], y_col[:6], eps=0.2, norm="linf",
            apgd_steps=5, apgd_restarts=1, fab_steps=4, square_queries=15,
            batch_size=3, seed=0,
        )
        self.assertEqual(res_aa.x_adv.shape, x[:6].shape)
        self.assertIn("apgd_t", res_aa.extra["stage_accuracies"])

        d_res = pgd_adversarial_train(
            model, x, y_col, eps=0.1, epochs=1, batch_size=6, attack_steps=2,
            eval_data=(x, y_col), lr_scheduler="cosine", save_best=True,
        )
        self.assertEqual(len(d_res.history), 1)
        self.assertIsNotNone(d_res.final_adv_acc)

    def test_defense_streaming_rejects_exhausted_generator_and_supports_cosine_save_best(self) -> None:
        model = _make_linear_cls(in_dim=6, num_classes=3, seed=4)
        x = torch.rand(12, 6)
        y = torch.randint(0, 3, (12,))

        def _one_shot() -> object:
            yield x[:6], y[:6]
            yield x[6:], y[6:]

        with self.assertRaises(ValueError):
            trades_train(model, _one_shot(), eps=0.1, epochs=2, attack_steps=2)  # type: ignore[arg-type]

        loader = DataLoader(TensorDataset(x, y), batch_size=6, shuffle=False)
        res = gaussian_augmented_train(
            model, loader, sigma=0.1, epochs=2, eval_data=(x, y),
            lr_scheduler="cosine", save_best=True, seed=0,
        )
        self.assertEqual(len(res.history), 2)
        self.assertIsNotNone(res.final_adv_acc)

    def test_boundary_binary_search_does_not_mutate_endpoint(self) -> None:
        class _HalfSpace(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                s = x[:, 0] - 0.5
                return torch.stack([-s, s], dim=1)

        model = _HalfSpace()
        x = torch.zeros(2, 4)
        y = torch.zeros(2, dtype=torch.long)
        x_init = torch.zeros(2, 4)
        x_init[:, 0] = 2.0
        res = boundary(model, x, y, eps=1.0, norm="l2", steps=20, x_init=x_init, seed=0)
        # Boundary is at x_0 = 0.5; clean is 0.0 -> distance should tighten close to 0.5
        dists = res.perturbation.norm(dim=1)
        self.assertTrue(bool((dists <= 0.75).all()))
        self.assertTrue(bool((model(res.x_adv).argmax(dim=1) != y).all()))

    def test_cw_l2_zero_victim_grad_pollution_and_post_budget_success_rate(self) -> None:
        model = _make_linear_cls(in_dim=6, num_classes=3, seed=5)
        for p in model.parameters():
            p.grad = None
        x = torch.rand(4, 6)
        with torch.no_grad():
            y = model(x).argmax(dim=1)

        res_cw = cw_l2(model, x, y, eps=1e-7, steps=8, binary_search_steps=1, clip_bounds=(0.0, 1.0))
        for p in model.parameters():
            self.assertIsNone(p.grad)
        self.assertEqual(res_cw.extra["success_rate"], 0.0)

        res_ead = ead(model, x, y, eps=1e-7, steps=8, binary_search_steps=1, clip_bounds=(0.0, 1.0))
        self.assertEqual(res_ead.extra["success_rate"], 0.0)

    def test_fab_rejects_unsupported_l1_norm(self) -> None:
        model = _make_linear_cls(in_dim=6, num_classes=3, seed=6)
        x = torch.rand(4, 6)
        y = torch.zeros(4, dtype=torch.long)
        with self.assertRaises(ValueError):
            fab(model, x, y, norm="l1", steps=3)

    def test_apgd_l1_accepts_list_and_tensor_clip_bounds(self) -> None:
        model = _make_linear_cls(in_dim=6, num_classes=3, seed=7)
        x = torch.full((4, 6), 0.5)
        y = torch.zeros(4, dtype=torch.long)
        lo = [0.0] * 6
        hi = [1.0] * 6
        res = apgd_l1(model, x, y, eps=1.0, steps=6, restarts=1, clip_bounds=(lo, hi), seed=0)
        self.assertTrue(bool((res.x_adv >= -1e-6).all() and (res.x_adv <= 1.0 + 1e-6).all()))

    def test_gap_deterministic_complex_seed(self) -> None:
        class _ComplexCls(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(8, 3)

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                return self.fc(torch.cat([z.real, z.imag], dim=-1))

        torch.manual_seed(8)
        model = _ComplexCls().eval()
        x = torch.complex(torch.randn(6, 4), torch.randn(6, 4)).to(torch.complex64)
        y = torch.randint(0, 3, (6,))
        r1 = gap(model, x, y, eps=-10.0, norm="jsr_db", epochs=2, seed=42)
        r2 = gap(model, x, y, eps=-10.0, norm="jsr_db", epochs=2, seed=42)
        self.assertTrue(torch.allclose(r1.x_adv, r2.x_adv, atol=1e-6))

    def test_l1_obfuscation_and_downscaled_stochasticity_caught_by_carlini(self) -> None:
        # 1. L1 broken-gradient wrapper where true linear decision is vulnerable in L1
        torch.manual_seed(9)
        base = nn.Linear(10, 3).eval()
        x = torch.rand(16, 10)
        with torch.no_grad():
            y = base(x).argmax(dim=1)

        class _BrokenGradModel(nn.Module):
            def __init__(self, m: nn.Module) -> None:
                super().__init__()
                self.m = m

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                # Forward uses real input, backward has zero gradient via STE trick
                z_det = z.detach()
                return self.m(z_det) + 0.0 * z.sum()

        rep_l1 = audit_robustness(
            _BrokenGradModel(base), x, y, eps=4.0, norm="l1",
            pgd_steps=5, apgd_steps=8, square_queries=80, seed=0,
        )
        self.assertFalse(rep_l1.passed)
        self.assertFalse(rep_l1.checks["whitebox_beats_blackbox"])
        self.assertIn("gradient_obfuscation_blackbox_beats_whitebox", rep_l1.red_flags)

        # 2. Downscaled stochastic logits (max_diff ~ 1e-4, but rel_diff ~ 0.5 and flip_rate > 0)
        class _TinyStochModel(nn.Module):
            def __init__(self, m: nn.Module) -> None:
                super().__init__()
                self.m = m

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                return 1e-4 * self.m(z) + 5e-5 * torch.randn(z.shape[0], 3, device=z.device)

        rep_st = audit_robustness(
            _TinyStochModel(base), x, y, eps=0.2, norm="linf",
            pgd_steps=5, apgd_steps=8, square_queries=40, seed=0,
        )
        self.assertTrue(rep_st.metrics["is_stochastic"])
        self.assertFalse(rep_st.checks["deterministic_or_eot_verified"])
        self.assertIn("stochastic_defense_requires_eot", rep_st.red_flags)

    def test_verify_gate_honors_tensor_samples_and_flags_hard_label_unverifiable(self) -> None:
        model = _make_linear_cls(in_dim=6, num_classes=3, seed=10)
        x = torch.rand(40, 6)
        with torch.no_grad():
            y = model(x).argmax(dim=1)

        v_sub = verify_robustness_gate(
            model_or_fn=model, x=x, y=y, eps=0.1, norm="linf",
            goal="sanity", samples=12, sweep_curve=False, fail_on_red_flag=False,
        )
        assert v_sub.red_team is not None
        self.assertEqual(v_sub.red_team.num_samples, 12)

        def _hard_oracle(z: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return model(z).argmax(dim=1)

        v_hard = verify_robustness_gate(
            model_or_fn=_hard_oracle, x=x[:8], y=y[:8], eps=0.3, norm="l2",
            goal="sanity", samples=8, sweep_curve=False, fail_on_red_flag=True,
        )
        self.assertFalse(v_hard.passed)
        self.assertFalse(v_hard.checks["carlini_audit_ok"])
        self.assertIn("carlini:hard_label_only_unverifiable_whitebox", v_hard.red_flags)

    def test_dataloader_across_preflight_probe_estimate_and_continuous_report(self) -> None:
        model = _make_linear_cls(in_dim=8, num_classes=3, seed=11)
        x = torch.rand(20, 8)
        y = torch.randint(0, 3, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=False)

        pf = preflight(model, loader, eps=0.1, norm="linf")
        self.assertTrue(pf.ok)

        probe = probe_model_context(model, loader, eps=0.1, norm="linf")
        self.assertTrue(probe.preflight_ok)
        self.assertTrue(probe.autograd_connected)

        suite_l1 = recommend_suite(domain="tabular", goal="standard", norm="l1")
        self.assertEqual(suite_l1.attacks, ["apgd_l1", "ead"])
        for aid in suite_l1.attacks:
            self.assertTrue(preflight(model, x[:4], y[:4], eps=1.0, norm="l1", attack=aid).ok)

        eta = estimate_runtime(model, loader, suite_l1, num_samples=20, include_audit=False)
        self.assertGreater(eta.total_seconds, 0.0)

        # Continuous mismatched-shape model should not crash preflight or format_paper_summary
        reg_model = nn.Sequential(nn.Linear(8, 5)).eval()
        y_mismatch = torch.randn(10, 2)
        rep_c = evaluate_suite(
            reg_model, x[:10], y_mismatch, eps=0.1, norm="l2",
            attacks=["fgsm", "pgd"], run_audit=True, audit_samples=10, verbose=False,
        )
        md = rep_c.format_paper_summary()
        self.assertIn("Continuous Error", md)
        self.assertNotIn("0.00%", md.split("Budget Sweep Monotonicity")[1].splitlines()[0])

    def test_normalized_model_wrapper_and_sklearn_regressor_floats(self) -> None:
        base = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 3)).eval()
        wrapped = wrap_normalized_model(base, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
        self.assertIsInstance(wrapped, NormalizedModelWrapper)
        x = torch.rand(4, 3, 4, 4, requires_grad=True)
        out = wrapped(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.abs().sum().item()), 0.0)

        class _MockRegressor:
            def predict(self, arr: np.ndarray) -> np.ndarray:
                return (arr[:, 0] * 2.5 + 0.125).astype(np.float32)

        adapter = wrap_sklearn_model(_MockRegressor())
        out_r = adapter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        self.assertTrue(out_r.is_floating_point())
        self.assertAlmostEqual(float(out_r[0].item()), 2.625, places=4)

    def test_stochastic_red_team_tiny_eps_stability_via_mc_voting(self) -> None:
        torch.manual_seed(12)
        base = nn.Linear(6, 3).eval()
        x = torch.randn(16, 6) * 2.0
        with torch.no_grad():
            y = base(x).argmax(dim=1)

        class _MildStoch(nn.Module):
            def __init__(self, m: nn.Module) -> None:
                super().__init__()
                self.m = m

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                return self.m(z + 0.05 * torch.randn_like(z))

        rt = auto_red_team(
            _MildStoch(base), x, y, eps=1e-6, norm="linf",
            goal="sanity", sweep_curve=False, seed=0, verbose=False,
        )
        assert rt.clean_acc is not None and rt.worst_case_robust_acc is not None
        self.assertLessEqual(rt.clean_acc - rt.worst_case_robust_acc, 0.15)

    def test_cpu_tensors_auto_align_to_accelerator_model_device(self) -> None:
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            self.skipTest("MPS device not available")
        dev = torch.device("mps")
        model = _make_linear_cls(in_dim=6, num_classes=3, seed=13).to(dev)
        x_cpu = torch.rand(12, 6)
        y_cpu = torch.randint(0, 3, (12,))
        self.assertTrue(preflight(model, x_cpu, y_cpu, eps=0.1, norm="linf").ok)
        rt = auto_red_team(
            model, x_cpu, y_cpu, eps=0.1, norm="linf",
            goal="sanity", sweep_curve=False, seed=0, verbose=False,
        )
        self.assertTrue(rt.device.startswith("mps"))
        preds = smooth_predict(model, x_cpu[:4], sigma=0.1, n_samples=16, seed=0)
        self.assertEqual(preds.device, x_cpu.device)


if __name__ == "__main__":
    unittest.main()
