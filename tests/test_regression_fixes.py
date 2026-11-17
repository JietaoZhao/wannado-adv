"""Regression tests covering v1.0.4 correctness, API consistency, and boundary fixes."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import wannado.adv as wadv
from tests.test_attack_contracts import (
    TinyComplexWirelessClassifier,
    _make_trained_real_classifier,
)
from wannado.adv import (
    apgd_ce,
    apgd_dlr,
    audit_robustness,
    autoattack,
    boundary,
    compute_perturbation_norm,
    cw_l2,
    deepfool,
    ead,
    eot_pgd,
    evaluate_suite,
    fab,
    fgsm,
    hopskipjump,
    jsma,
    load_catalog,
    mi_fgsm,
    mifgsm,
    nes,
    pgd,
    preflight,
    probe_model_context,
    project_discrete_features,
    project_perturbation,
    random_init_perturbation,
    resolve_eps_budget,
    scan_project,
    simba,
    sparsefool,
    spsa,
    square_attack,
    tpgd,
    uap,
    wrap_sklearn_model,
    zoo,
)
from wannado.adv import (
    attacks as wadv_attacks,
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


class TestReleaseV105Regressions(unittest.TestCase):
    @staticmethod
    def _make_binary_single_logit_problem() -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
        torch.manual_seed(7)
        n, d = 16, 6
        x0 = torch.randn(n // 2, d) * 0.25 + 0.25
        x1 = torch.randn(n // 2, d) * 0.25 + 0.75
        x = torch.cat([x0, x1], dim=0).clamp(0.0, 1.0)
        y = torch.cat([torch.zeros(n // 2, dtype=torch.long), torch.ones(n // 2, dtype=torch.long)])
        model = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))
        opt = torch.optim.Adam(model.parameters(), lr=0.08)
        bce = nn.BCEWithLogitsLoss()
        for _ in range(120):
            opt.zero_grad()
            loss = bce(model(x).squeeze(1), y.float())
            loss.backward()
            opt.step()
        model.eval()
        return model, x, y

    def test_single_logit_and_1d_float_outputs_across_attacks_and_reporting(self) -> None:
        model, x, y = self._make_binary_single_logit_problem()

        def model_1d_float(z: torch.Tensor) -> torch.Tensor:
            return model(z).squeeze(1)

        for m in (model, model_1d_float):
            pf = preflight(m, x, y, eps=0.35, norm="l2", clip_bounds=(0.0, 1.0))
            self.assertTrue(pf.ok, msg=pf.format_report())

        # Gradient, boundary, black-box, and ensemble attacks on (B, 1) binary logits
        for name, fn, kw in [
            ("fgsm", fgsm, {"eps": 0.25, "norm": "linf"}),
            ("pgd", pgd, {"eps": 0.25, "norm": "linf", "steps": 8}),
            ("apgd_ce", apgd_ce, {"eps": 0.25, "norm": "linf", "steps": 10, "restarts": 1}),
            ("apgd_dlr", apgd_dlr, {"eps": 0.25, "norm": "linf", "steps": 10, "restarts": 1}),
            ("cw_l2", cw_l2, {"steps": 25, "binary_search_steps": 2, "clip_bounds": (0.0, 1.0)}),
            ("ead", ead, {"steps": 25, "binary_search_steps": 2, "clip_bounds": (0.0, 1.0)}),
            ("deepfool", deepfool, {"steps": 15, "clip_bounds": (0.0, 1.0)}),
            ("fab", fab, {"steps": 12, "restarts": 1, "norm": "l2", "clip_bounds": (0.0, 1.0)}),
            ("jsma", jsma, {"theta": 0.25, "gamma": 0.6, "targeted": False, "clip_bounds": (0.0, 1.0)}),
            ("sparsefool", sparsefool, {"max_iter": 4, "deepfool_steps": 10, "clip_bounds": (0.0, 1.0)}),
            ("uap", uap, {"eps": 0.25, "norm": "linf", "max_iter_uni": 2, "deepfool_steps": 6}),
            ("tpgd", tpgd, {"eps": 0.25, "norm": "linf", "steps": 6}),
            ("square_attack", square_attack, {"eps": 0.25, "norm": "linf", "max_queries": 40}),
            ("simba", simba, {"eps": 0.6, "norm": "l2", "max_queries": 30}),
            ("zoo", zoo, {"eps": 0.6, "norm": "l2", "steps": 6, "coord_batch_size": 4}),
            ("boundary", boundary, {"steps": 12, "max_queries": 80, "clip_bounds": (0.0, 1.0)}),
            ("hopskipjump", hopskipjump, {"steps": 3, "initial_num_evals": 10, "max_queries": 120, "clip_bounds": (0.0, 1.0)}),
            ("autoattack", autoattack, {"eps": 0.3, "norm": "linf", "apgd_steps": 6, "apgd_restarts": 1, "fab_steps": 6, "square_queries": 30}),
        ]:
            res = fn(model, x[:6], y[:6], **kw) if name not in ("tpgd",) else fn(model, x[:6], **kw)
            self.assertEqual(res.x_adv.shape, x[:6].shape, msg=f"Shape mismatch on {name}")
            self.assertTrue(torch.isfinite(res.x_adv).all(), msg=f"Non-finite on {name}")

        rep = evaluate_suite(
            model,
            x[:8],
            y[:8],
            eps=0.25,
            norm="linf",
            attacks=["fgsm", "pgd", "apgd_dlr"],
            clip_bounds=(0.0, 1.0),
            run_audit=False,
            verbose=False,
        )
        self.assertIsNotNone(rep.clean_acc)
        assert rep.clean_acc is not None
        self.assertGreater(rep.clean_acc, 0.7)
        self.assertIsNotNone(rep.records[1].robust_acc)
        assert rep.records[1].robust_acc is not None
        self.assertLessEqual(rep.records[1].robust_acc, rep.clean_acc)

    def test_huggingface_model_output_dot_logits_unwrapping(self) -> None:
        model, x, y = _make_trained_real_classifier()

        def hf_model(z: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(logits=model(z))

        x_sub, y_sub = x[:10], y[:10]
        pf = preflight(hf_model, x_sub, y_sub, eps=0.1, norm="linf", clip_bounds=(0.0, 1.0))
        self.assertTrue(pf.ok, msg=pf.format_report())

        res_pgd = pgd(hf_model, x_sub, y_sub, eps=0.15, norm="linf", steps=8, clip_bounds=(0.0, 1.0))
        res_sq = square_attack(hf_model, x_sub, y_sub, eps=0.15, norm="linf", max_queries=50, clip_bounds=(0.0, 1.0))
        self.assertEqual(res_pgd.x_adv.shape, x_sub.shape)
        self.assertEqual(res_sq.x_adv.shape, x_sub.shape)

        rep = evaluate_suite(
            hf_model,
            x_sub,
            y_sub,
            eps=0.15,
            norm="linf",
            attacks=["pgd", "deepfool"],
            attack_kwargs={"pgd": {"steps": 6}, "deepfool": {"steps": 8}},
            clip_bounds=(0.0, 1.0),
            run_audit=False,
            verbose=False,
        )
        self.assertIsNotNone(rep.clean_acc)
        self.assertIsNotNone(rep.records[0].robust_acc)

    def test_vector_clip_bounds_perturb_mask_and_discrete_projection(self) -> None:
        model, x, y = self._make_binary_single_logit_problem()
        x_tab = x[:6] * torch.tensor([[10.0, 2.0, 100.0, 1.0, 5.0, 20.0]]) - 1.0
        min_vec = np.array([-2.0, -1.0, -5.0, -1.0, -1.0, -2.0], dtype=np.float32)
        max_vec = np.array([12.0, 3.0, 105.0, 2.0, 6.0, 22.0], dtype=np.float32)
        bounds = (min_vec, max_vec)

        def tab_model(z: torch.Tensor) -> torch.Tensor:
            scaled = (z + 1.0) / torch.tensor([[10.0, 2.0, 100.0, 1.0, 5.0, 20.0]])
            return model(scaled)

        for name, fn, kw in [
            ("pgd", pgd, {"eps": 0.4, "norm": "linf", "steps": 6}),
            ("cw_l2", cw_l2, {"steps": 15, "binary_search_steps": 1}),
            ("ead", ead, {"steps": 15, "binary_search_steps": 1}),
            ("deepfool", deepfool, {"steps": 8}),
            ("fab", fab, {"steps": 8, "restarts": 1, "norm": "l2"}),
            ("jsma", jsma, {"theta": 0.3, "gamma": 0.5, "targeted": False}),
            ("sparsefool", sparsefool, {"max_iter": 3, "deepfool_steps": 6}),
            ("nes", nes, {"eps": 0.4, "norm": "linf", "steps": 5, "nb_sample": 6}),
            ("spsa", spsa, {"eps": 0.4, "norm": "linf", "steps": 5, "nb_sample": 6}),
            ("zoo", zoo, {"eps": 0.5, "norm": "l2", "steps": 4, "coord_batch_size": 3}),
            ("boundary", boundary, {"steps": 8, "max_queries": 60}),
            ("hopskipjump", hopskipjump, {"steps": 2, "initial_num_evals": 8, "max_queries": 90}),
        ]:
            res = fn(tab_model, x_tab, y[:6], clip_bounds=bounds, **kw)
            self.assertGreaterEqual(
                float((res.x_adv - torch.as_tensor(min_vec)).min().item()),
                -1e-5,
                msg=f"Lower vector bound violated on {name}",
            )
            self.assertGreaterEqual(
                float((torch.as_tensor(max_vec) - res.x_adv).min().item()),
                -1e-5,
                msg=f"Upper vector bound violated on {name}",
            )

        rep = evaluate_suite(
            tab_model,
            x_tab,
            y[:6],
            eps=0.4,
            norm="l2",
            attacks=["pgd"],
            attack_kwargs={"pgd": {"steps": 4}},
            clip_bounds=bounds,
            run_audit=False,
            verbose=False,
        )
        parsed = json.loads(rep.to_json())
        self.assertIsInstance(parsed["experiment"]["clip_bounds"], list)

        # Immutable columns must stay fixed via perturb_mask
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        budget = resolve_eps_budget(x_tab, eps=0.5, norm="l2")
        init_d = random_init_perturbation(x_tab, budget, norm="l2", clip_bounds=bounds, perturb_mask=mask)
        proj_d = project_perturbation(torch.ones_like(x_tab), x_tab, budget, norm="l2", clip_bounds=bounds, perturb_mask=mask)
        self.assertTrue(torch.allclose(init_d[:, [1, 3, 5]], torch.zeros(x_tab.shape[0], 3)))
        self.assertTrue(torch.allclose(proj_d[:, [1, 3, 5]], torch.zeros(x_tab.shape[0], 3)))

        # Discrete/categorical column projection inside bounds
        raw_x = torch.tensor([[0.72, 1.35, -0.4, 2.6, 0.12, 3.8]])
        snapped = project_discrete_features(
            raw_x,
            binary_indices=[0, 2],
            integer_indices=[1, 3],
            clip_bounds=bounds,
        )
        self.assertEqual(float(snapped[0, 0].item()), 1.0)
        self.assertEqual(float(snapped[0, 2].item()), 0.0)
        self.assertEqual(float(snapped[0, 1].item()), 1.0)
        self.assertEqual(float(snapped[0, 3].item()), 2.0)

    def test_dense_4d_segmentation_logits_ce_margin_dlr(self) -> None:
        torch.manual_seed(21)
        seg_net = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(6, 3, kernel_size=1),
        ).eval()
        x_img = torch.rand(4, 2, 8, 8)
        y_mask = torch.randint(0, 3, (4, 8, 8), dtype=torch.long)

        pf = preflight(seg_net, x_img, y_mask, eps=0.1, norm="linf", clip_bounds=(0.0, 1.0))
        self.assertTrue(pf.ok, msg=pf.format_report())

        for loss_name in ("ce", "margin", "dlr"):
            res = pgd(seg_net, x_img, y_mask, eps=0.1, norm="linf", steps=5, loss=loss_name, clip_bounds=(0.0, 1.0))
            self.assertEqual(res.x_adv.shape, x_img.shape)

        res_ce = apgd_ce(seg_net, x_img, y_mask, eps=0.1, norm="linf", steps=6, restarts=1, clip_bounds=(0.0, 1.0))
        res_dlr = apgd_dlr(seg_net, x_img, y_mask, eps=0.1, norm="linf", steps=6, restarts=1, clip_bounds=(0.0, 1.0))
        self.assertEqual(res_ce.x_adv.shape, x_img.shape)
        self.assertEqual(res_dlr.x_adv.shape, x_img.shape)

        rep = evaluate_suite(
            seg_net,
            x_img,
            y_mask,
            eps=0.1,
            norm="linf",
            attacks=["pgd"],
            attack_kwargs={"pgd": {"steps": 4}},
            clip_bounds=(0.0, 1.0),
            run_audit=False,
            verbose=False,
        )
        self.assertIsNotNone(rep.clean_acc)
        self.assertIsNotNone(rep.records[0].robust_acc)

    def test_uncallable_sklearn_preflight_and_wrap_sklearn_model(self) -> None:
        class DuckEstimator:
            def predict(self, arr: np.ndarray) -> np.ndarray:
                return (arr[:, 0] + arr[:, 1] > 0.0).astype(np.int64)

            def predict_proba(self, arr: np.ndarray) -> np.ndarray:
                s = 1.0 / (1.0 + np.exp(-(arr[:, 0] + arr[:, 1])))
                return np.stack([1.0 - s, s], axis=1)

        est = DuckEstimator()
        x = torch.randn(8, 4)
        y = (x[:, 0] + x[:, 1] > 0.0).to(torch.long)

        pf_raw = preflight(est, x, y, eps=0.5, norm="l2")  # type: ignore[arg-type]
        self.assertFalse(pf_raw.ok)
        self.assertTrue(any("wrap_sklearn_model" in msg for msg in pf_raw.errors + pf_raw.recommendations))

        probe_raw = probe_model_context(est, x, y)  # type: ignore[arg-type]
        self.assertFalse(probe_raw.preflight_ok)

        wrapped = wrap_sklearn_model(est)
        res_sq = square_attack(wrapped, x, y, eps=1.5, norm="l2", max_queries=40)
        res_hs = hopskipjump(wrapped, x[:3], y[:3], steps=3, initial_num_evals=8, max_queries=80)
        self.assertEqual(res_sq.x_adv.shape, x.shape)
        self.assertEqual(res_hs.x_adv.shape, (3, 4))

        if importlib.util.find_spec("sklearn") is not None:
            from sklearn.linear_model import LogisticRegression

            rng = np.random.default_rng(0)
            x_np = rng.normal(size=(40, 4))
            y_np = (x_np[:, 0] > 0).astype(int)
            clf = LogisticRegression().fit(x_np, y_np)
            clf_wrapped = wrap_sklearn_model(clf)
            out = clf_wrapped(torch.tensor(x_np[:4], dtype=torch.float32))
            self.assertEqual(tuple(out.shape), (4, 2))

    def test_ct_hounsfield_scale_mismatch_trivial_budget_red_flag(self) -> None:
        model, x, y = _make_trained_real_classifier()
        x_ct = x[:12] * 4000.0 - 1000.0  # [-1000, 3000] HU span = ~4000

        def ct_model(z: torch.Tensor) -> torch.Tensor:
            return model((z + 1000.0) / 4000.0)

        audit = audit_robustness(
            ct_model,
            x_ct,
            y[:12],
            eps=0.03,
            norm="linf",
            pgd_steps=5,
            apgd_steps=5,
            square_queries=20,
        )
        self.assertFalse(audit.passed)
        self.assertFalse(audit.checks["eps_scale_meaningful"])
        self.assertIn("eps_scale_mismatch_trivial_budget", audit.red_flags)

    def test_scan_project_readme_symbols_new_domains_and_lazy_submodules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            med_dir = Path(tmp) / "med_proj"
            med_dir.mkdir()
            (med_dir / "README.md").write_text("# Chest CT Radiology UNet\nClinical dicom segmentation.\n", encoding="utf-8")
            (med_dir / "unet_seg.py").write_text("import torch\nclass RadiologyUNet:\n    pass\n", encoding="utf-8")
            scan_med = scan_project(med_dir)
            self.assertEqual(scan_med.likely_domain, "medical")
            self.assertIn("Radiology", scan_med.readme_summary)
            self.assertTrue(any("RadiologyUNet" in s for s in scan_med.detected_symbols))

            sec_dir = Path(tmp) / "sec_proj"
            sec_dir.mkdir()
            (sec_dir / "ids_detector.py").write_text("import pandas\n# CICIDS2017 malware intrusion pcap botnet\n", encoding="utf-8")
            scan_sec = scan_project(sec_dir)
            self.assertEqual(scan_sec.likely_domain, "cybersecurity")

            nlp_dir = Path(tmp) / "nlp_proj"
            nlp_dir.mkdir()
            (nlp_dir / "bert_cls.py").write_text("from transformers import AutoModel\n# roberta attention_mask\n", encoding="utf-8")
            scan_nlp = scan_project(nlp_dir)
            self.assertEqual(scan_nlp.likely_domain, "nlp_transformers")

            blank_dir = Path(tmp) / "blank_proj"
            blank_dir.mkdir()
            (blank_dir / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            scan_blank = scan_project(blank_dir)
            self.assertEqual(scan_blank.likely_domain, "general")
            self.assertEqual(scan_blank.recommended_norm, "l2")

        # Verify fresh subprocess lazy attribute access wadv.copilot.scan_project
        src_root = Path(__file__).resolve().parents[1] / "src"
        code = (
            f"import sys; sys.path.insert(0, {str(src_root)!r}); import wannado.adv as wadv; "
            "assert callable(wadv.copilot.scan_project); "
            "assert callable(wadv.audit.preflight)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_blackbox_aliases_and_latex_escaping(self) -> None:
        model, x, y = _make_trained_real_classifier()
        res_sq = square_attack(model, x[:6], y[:6], eps=0.1, norm="linf", steps=40)
        res_sim = simba(model, x[:6], y[:6], eps=0.5, norm="l2", steps=15)
        res_zoo = zoo(model, x[:4], y[:4], eps=0.5, norm="l2", steps=10, coord_batch_size=4, max_queries=25)
        self.assertLessEqual(res_sq.extra["queries"], 40 * 6)
        self.assertLessEqual(res_sim.extra["queries"], 15 * 2 * 6 + 6)
        self.assertLessEqual(res_zoo.extra["queries"], 60)

        rep = evaluate_suite(
            model,
            x[:6],
            y[:6],
            eps=0.1,
            norm="linf",
            attacks=["fgsm"],
            model_name="Model#1_v2^test~100%",
            dataset_name="Data#A&B",
            run_audit=False,
            verbose=False,
        )
        tex = rep.to_latex()
        self.assertIn("\\#", tex)
        self.assertIn("\\^{}", tex)
        self.assertIn("\\~{}", tex)
        self.assertIn("\\&", tex)
        self.assertIn("\\%", tex)


if __name__ == "__main__":
    unittest.main()
