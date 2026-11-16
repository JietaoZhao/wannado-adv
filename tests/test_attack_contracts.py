"""Contract tests ensuring CleanRL single-file attacks obey mathematical,
engineering, wireless/complex-signal, and catalog provenance invariants.

Compatible with both ``python3 -m unittest`` and ``pytest``.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from wannado.adv import (
    build_registry_json,
    compute_perturbation_norm,
    load_catalog,
    query_catalog,
)
from wannado.adv.attacks import (
    adam_pgd,
    apgd_ce,
    apgd_dlr,
    bim,
    cw_l2,
    deepfool,
    eot_pgd,
    fab,
    fgsm,
    hopskipjump,
    mifgsm,
    pgd,
    rfgsm,
    spsa,
    square_attack,
    uap,
)
from wannado.adv.catalog.build_catalog import REGISTRY_PATH, discover_attack_metas

REPO_ROOT = Path(__file__).resolve().parent.parent

ALL_FIRST_ORDER_ATTACKS = (
    ("fgsm", fgsm),
    ("rfgsm", rfgsm),
    ("bim", bim),
    ("pgd", pgd),
    ("mi_fgsm", mifgsm),
    ("adam_pgd", adam_pgd),
    ("apgd_ce", apgd_ce),
    ("apgd_dlr", apgd_dlr),
)


class TinyRealClassifier(nn.Module):
    def __init__(self, in_dim: int = 16, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyComplexWirelessClassifier(nn.Module):
    """Toy complex-valued IQ baseband modulation classifier."""

    def __init__(self, seq_len: int = 32, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * 2, 48),
            nn.ReLU(),
            nn.Linear(48, num_classes),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        features = torch.cat([iq.real, iq.imag], dim=-1)
        return self.net(features)


def _make_trained_real_classifier() -> tuple[TinyRealClassifier, torch.Tensor, torch.Tensor]:
    """Train a tiny 4-class classifier on [0, 1]^16 in <50ms so all classes are well-separated."""
    torch.manual_seed(42)
    centers = torch.full((4, 16), 0.2)
    for c in range(4):
        centers[c, c * 4 : (c + 1) * 4] = 0.8

    x_train = (centers.repeat(16, 1) + 0.1 * torch.randn(64, 16)).clamp(0.0, 1.0)
    y_train = torch.arange(4).repeat(16)

    model = TinyRealClassifier(in_dim=16, num_classes=4)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    model.train()
    for _ in range(40):
        opt.zero_grad()
        loss = F.cross_entropy(model(x_train), y_train)
        loss.backward()
        opt.step()

    model.eval()
    x_test = (centers.repeat(8, 1) + 0.08 * torch.randn(32, 16)).clamp(0.0, 1.0)
    y_test = torch.arange(4).repeat(8)
    return model, x_test, y_test


class TestPackagingAndCatalogContracts(unittest.TestCase):
    def test_pep420_namespace_has_no_top_init(self) -> None:
        forbidden_init = REPO_ROOT / "src" / "wannado" / "__init__.py"
        self.assertFalse(
            forbidden_init.exists(),
            "src/wannado/__init__.py must NOT exist to preserve PEP 420 namespace compatibility.",
        )

    def test_lazy_import_only_loads_requested_attack(self) -> None:
        for stmt in (
            "from wannado.adv import fgsm",
            "from wannado.adv.attacks import fgsm",
        ):
            code = (
                "import sys; "
                f"{stmt}; "
                "assert 'wannado.adv.attacks.first_order.fgsm_goodfellow2015' in sys.modules; "
                "assert 'wannado.adv.attacks.first_order.pgd_madry2018' not in sys.modules; "
                "assert 'wannado.adv.attacks.first_order.apgd_ce_croce2020' not in sys.modules"
            )
            proc = subprocess.run(
                [sys.executable, "-c", code],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"{stmt}: {proc.stderr}")

    def test_catalog_single_source_of_truth_and_citations(self) -> None:
        metas = discover_attack_metas()
        expected_ids = {
            "fgsm",
            "rfgsm",
            "bim",
            "pgd",
            "mi_fgsm",
            "adam_pgd",
            "tpgd",
            "apgd_ce",
            "apgd_dlr",
            "apgd_l1",
            "ni_fgsm",
            "sini_fgsm",
            "di_fgsm",
            "ti_fgsm",
            "vmi_fgsm",
            "deepfool",
            "cw_l2",
            "ead",
            "sparsefool",
            "jsma",
            "fab",
            "zoo",
            "nes",
            "spsa",
            "simba",
            "square_attack",
            "boundary",
            "hopskipjump",
            "eot_pgd",
            "bpda_eot",
            "adv_patch",
            "uap",
            "gap",
            "advgan",
            "autoattack",
        }
        self.assertEqual({c["id"] for c in metas}, expected_ids)

        for card in metas:
            self.assertIn(card["tier"], {"S", "A", "B", "C"})
            self.assertIn("@", card["citation"]["bibtex"])
            self.assertTrue(card["citation"]["url"].startswith("https://"))
            parent_id = card.get("parent_id")
            if parent_id is not None:
                self.assertIn(parent_id, expected_ids, msg=f"{card['id']} has unknown parent_id={parent_id}")

        payload = build_registry_json()
        self.assertEqual(payload["count"], len(metas))
        self.assertTrue(REGISTRY_PATH.exists())
        loaded = load_catalog()
        self.assertEqual([c["id"] for c in loaded], [c["id"] for c in metas])

        wireless_s = query_catalog(domain="wireless", norm="jsr_db", tier="S")
        self.assertEqual(
            {c["id"] for c in wireless_s},
            {
                "fgsm",
                "pgd",
                "mi_fgsm",
                "ni_fgsm",
                "sini_fgsm",
                "di_fgsm",
                "ti_fgsm",
                "vmi_fgsm",
                "apgd_ce",
                "apgd_dlr",
                "deepfool",
                "cw_l2",
                "ead",
                "sparsefool",
                "fab",
                "nes",
                "simba",
                "square_attack",
                "boundary",
                "hopskipjump",
                "eot_pgd",
                "bpda_eot",
                "uap",
                "autoattack",
            },
        )
        self.assertEqual(
            {c["id"] for c in query_catalog(threat_model="black_box_score")},
            {"zoo", "nes", "spsa", "simba", "square_attack"},
        )
        self.assertEqual(
            {c["id"] for c in query_catalog(threat_model="black_box_decision")},
            {"boundary", "hopskipjump"},
        )
        self.assertEqual(
            {c["id"] for c in query_catalog(threat_model="generative_train")},
            {"gap", "advgan"},
        )
        self.assertEqual(
            {c["id"] for c in query_catalog(family="transferability")},
            {"ni_fgsm", "sini_fgsm", "di_fgsm", "ti_fgsm", "vmi_fgsm"},
        )


class TestFirstOrderAttackContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.x, cls.y = _make_trained_real_classifier()

    def setUp(self) -> None:
        self.model.eval()

    def test_clean_accuracy_is_high_baseline(self) -> None:
        with torch.no_grad():
            acc = (self.model(self.x).argmax(dim=1) == self.y).float().mean().item()
        self.assertGreaterEqual(acc, 0.95)

    def test_zero_eps_returns_exact_clean_input_across_all_attacks(self) -> None:
        for name, fn in ALL_FIRST_ORDER_ATTACKS:
            with self.subTest(attack=name):
                res = fn(self.model, self.x, self.y, eps=0.0, norm="linf", clip_bounds=(0.0, 1.0))
                self.assertTrue(torch.equal(res.x_adv, self.x))
                self.assertTrue(torch.all(res.perturbation == 0))

    def test_preserves_input_model_mode_and_param_grads(self) -> None:
        self.model.train()
        x_orig = self.x.clone()
        running_mean_before = self.model.net[2].running_mean.clone()

        for p in self.model.parameters():
            p.grad = None

        _ = apgd_ce(
            self.model,
            self.x,
            self.y,
            eps=0.1,
            steps=10,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=0,
        )

        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(self.x, x_orig))
        self.assertTrue(torch.equal(self.model.net[2].running_mean, running_mean_before))
        for p in self.model.parameters():
            self.assertIsNone(p.grad)

    def test_all_first_order_attacks_respect_linf_l2_and_degrade_accuracy(self) -> None:
        eps_linf = 0.30
        eps_l2 = 0.9

        for name, fn in ALL_FIRST_ORDER_ATTACKS:
            with self.subTest(attack=name):
                res_linf = fn(self.model, self.x, self.y, eps=eps_linf, norm="linf", clip_bounds=(0.0, 1.0))
                self.assertGreaterEqual(res_linf.x_adv.min().item(), -1e-6)
                self.assertLessEqual(res_linf.x_adv.max().item(), 1.0 + 1e-6)
                self.assertLessEqual(
                    compute_perturbation_norm(res_linf.perturbation, "linf").max().item(),
                    eps_linf + 1e-6,
                )
                with torch.no_grad():
                    acc_linf = (self.model(res_linf.x_adv).argmax(dim=1) == self.y).float().mean().item()
                self.assertLess(acc_linf, 0.75, msg=f"{name} failed to degrade accuracy under Linf")

                res_l2 = fn(self.model, self.x, self.y, eps=eps_l2, norm="l2", clip_bounds=(0.0, 1.0))
                self.assertLessEqual(
                    compute_perturbation_norm(res_l2.perturbation, "l2").max().item(),
                    eps_l2 + 1e-5,
                )
                with torch.no_grad():
                    acc_l2 = (self.model(res_l2.x_adv).argmax(dim=1) == self.y).float().mean().item()
                self.assertLess(acc_l2, 0.75, msg=f"{name} failed to degrade accuracy under L2")

    def test_apgd_dlr_defeats_softmax_saturation_masking(self) -> None:
        # Scaling logits by 200 saturates Cross-Entropy Softmax gradients to ~0 (classic fake robustness).
        def saturated_model(x: torch.Tensor) -> torch.Tensor:
            return self.model(x) * 200.0

        res_ce = apgd_ce(
            saturated_model,
            self.x,
            self.y,
            eps=0.25,
            steps=50,
            restarts=2,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=5,
        )
        res_dlr = apgd_dlr(
            saturated_model,
            self.x,
            self.y,
            eps=0.25,
            steps=50,
            restarts=2,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=5,
        )
        with torch.no_grad():
            acc_ce = (saturated_model(res_ce.x_adv).argmax(dim=1) == self.y).float().mean().item()
            acc_dlr = (saturated_model(res_dlr.x_adv).argmax(dim=1) == self.y).float().mean().item()

        self.assertGreater(acc_ce, 0.85)  # CE gets trapped by Softmax saturation
        self.assertLess(acc_dlr, 0.25)    # Scale-invariant DLR punches right through

    def test_targeted_attack_steers_toward_target_class(self) -> None:
        target = (self.y + 1) % 4
        for fn in (pgd, mifgsm, adam_pgd, apgd_ce, apgd_dlr):
            with self.subTest(attack=fn.__name__):
                res = fn(
                    self.model,
                    self.x,
                    target,
                    eps=0.35,
                    steps=30,
                    norm="linf",
                    targeted=True,
                    clip_bounds=(0.0, 1.0),
                )
                with torch.no_grad():
                    target_rate = (self.model(res.x_adv).argmax(dim=1) == target).float().mean().item()
                self.assertGreater(target_rate, 0.75)

    def test_unbounded_regression_with_mse_loss(self) -> None:
        torch.manual_seed(7)
        regressor = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).eval()
        x = torch.randn(24, 8) * 3.0  # Intentionally outside [0, 1]
        with torch.no_grad():
            y = regressor(x)

        res = pgd(regressor, x, y, eps=1.0, steps=15, norm="l2", loss="mse", clip_bounds=None, seed=1)
        rand_noise = torch.randn_like(x)
        rand_noise = rand_noise / rand_noise.reshape(24, -1).norm(dim=1, keepdim=True)
        with torch.no_grad():
            mse_clean = (regressor(x) - y).pow(2).mean().item()
            mse_rand = (regressor(x + rand_noise) - y).pow(2).mean().item()
            mse_adv = (regressor(res.x_adv) - y).pow(2).mean().item()
        self.assertGreater(res.x_adv.abs().max().item(), 2.0)  # Must not clip to [0, 1]
        self.assertGreater(mse_adv, mse_clean)
        self.assertGreater(mse_adv, mse_rand * 2.0)


class TestWirelessComplexJSRContracts(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(101)
        self.rx = TinyComplexWirelessClassifier(seq_len=32, num_classes=4).eval()
        self.iq = torch.complex(torch.randn(32, 32), torch.randn(32, 32))
        with torch.no_grad():
            self.labels = self.rx(self.iq).argmax(dim=1)

    def test_all_first_order_attacks_handle_complex_jsr_db(self) -> None:
        budget_db = -3.0
        for name, fn in ALL_FIRST_ORDER_ATTACKS:
            with self.subTest(attack=name):
                res = fn(self.rx, self.iq, self.labels, eps=budget_db, norm="jsr_db")
                self.assertTrue(res.x_adv.is_complex())
                measured_db = compute_perturbation_norm(res.perturbation, "jsr_db", x_ref=self.iq)
                self.assertLessEqual(measured_db.max().item(), budget_db + 1e-4)
                with torch.no_grad():
                    acc = (self.rx(res.x_adv).argmax(dim=1) == self.labels).float().mean().item()
                self.assertLess(acc, 0.55)

    def test_complex_jsr_db_monotonic_degradation(self) -> None:
        res_weak = adam_pgd(self.rx, self.iq, self.labels, eps=-15.0, steps=20, norm="jsr_db", seed=1)
        res_strong = adam_pgd(self.rx, self.iq, self.labels, eps=-3.0, steps=20, norm="jsr_db", seed=1)

        with torch.no_grad():
            acc_weak = (self.rx(res_weak.x_adv).argmax(dim=1) == self.labels).float().mean().item()
            acc_strong = (self.rx(res_strong.x_adv).argmax(dim=1) == self.labels).float().mean().item()

        self.assertLess(acc_strong, 0.5)
        self.assertLessEqual(acc_strong, acc_weak)

    def test_complex_relative_l2_and_clip_bounds_guard(self) -> None:
        res_rel = pgd(self.rx, self.iq, self.labels, eps=0.25, steps=15, norm="relative_l2", seed=3)
        rel_norm = compute_perturbation_norm(res_rel.perturbation, "relative_l2", x_ref=self.iq)
        self.assertLessEqual(rel_norm.max().item(), 0.25 + 1e-5)

        with self.assertRaises(ValueError):
            fgsm(self.rx, self.iq, self.labels, eps=0.1, norm="linf", clip_bounds=(0.0, 1.0))


class TestMinimumNormAttackContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.x, cls.y = _make_trained_real_classifier()

    def setUp(self) -> None:
        self.model.eval()

    def test_unconstrained_minimum_norm_fools_real_classifier(self) -> None:
        x_sub, y_sub = self.x[:12], self.y[:12]
        for name, fn, kwargs in (
            ("deepfool_l2", deepfool, {"norm": "l2", "steps": 30, "clip_bounds": (0.0, 1.0)}),
            ("deepfool_linf", deepfool, {"norm": "linf", "steps": 30, "clip_bounds": (0.0, 1.0)}),
            ("cw_l2", cw_l2, {"steps": 150, "binary_search_steps": 4, "clip_bounds": (0.0, 1.0)}),
            ("fab_l2", fab, {"norm": "l2", "steps": 25, "restarts": 2, "clip_bounds": (0.0, 1.0), "seed": 1}),
            ("fab_linf", fab, {"norm": "linf", "steps": 25, "restarts": 2, "clip_bounds": (0.0, 1.0), "seed": 1}),
        ):
            with self.subTest(attack=name):
                res = fn(self.model, x_sub, y_sub, **kwargs)
                self.assertGreaterEqual(res.x_adv.min().item(), -1e-6)
                self.assertLessEqual(res.x_adv.max().item(), 1.0 + 1e-6)
                with torch.no_grad():
                    acc = (self.model(res.x_adv).argmax(dim=1) == y_sub).float().mean().item()
                self.assertLessEqual(acc, 0.2, msg=f"{name} failed to flip majority of samples")
                self.assertLess(res.extra["mean_distortion"], 2.0)

    def test_budget_cutoff_and_targeted_minimum_norm(self) -> None:
        x_sub, y_sub = self.x[:12], self.y[:12]
        target = (y_sub + 1) % 4

        res_cw_t = cw_l2(
            self.model, x_sub, target, steps=200, binary_search_steps=4, targeted=True, clip_bounds=(0.0, 1.0)
        )
        res_fab_t = fab(
            self.model,
            x_sub,
            target,
            norm="l2",
            steps=30,
            restarts=2,
            targeted=True,
            clip_bounds=(0.0, 1.0),
            seed=2,
        )
        with torch.no_grad():
            self.assertGreater(
                (self.model(res_cw_t.x_adv).argmax(dim=1) == target).float().mean().item(), 0.75
            )
            self.assertGreater(
                (self.model(res_fab_t.x_adv).argmax(dim=1) == target).float().mean().item(), 0.75
            )

        res_cutoff = fab(
            self.model, x_sub, y_sub, eps=0.9, norm="l2", steps=25, clip_bounds=(0.0, 1.0), seed=3
        )
        self.assertLessEqual(
            compute_perturbation_norm(res_cutoff.perturbation, "l2").max().item(), 0.9 + 1e-5
        )

    def test_complex_wireless_minimum_norm_and_jsr_db(self) -> None:
        torch.manual_seed(202)
        rx = TinyComplexWirelessClassifier(seq_len=32, num_classes=4).eval()
        iq = torch.complex(torch.randn(12, 32), torch.randn(12, 32))
        with torch.no_grad():
            labels = rx(iq).argmax(dim=1)

        for name, fn, kwargs in (
            ("deepfool_complex", deepfool, {"norm": "l2", "steps": 30}),
            ("cw_l2_complex", cw_l2, {"norm": "jsr_db", "steps": 150, "binary_search_steps": 4}),
            ("fab_complex", fab, {"norm": "l2", "steps": 25, "restarts": 2, "seed": 4}),
        ):
            with self.subTest(attack=name):
                res = fn(rx, iq, labels, **kwargs)
                self.assertTrue(res.x_adv.is_complex())
                self.assertFalse(torch.isnan(res.x_adv.real).any())
                self.assertFalse(torch.isnan(res.x_adv.imag).any())
                with torch.no_grad():
                    acc = (rx(res.x_adv).argmax(dim=1) == labels).float().mean().item()
                self.assertLessEqual(acc, 0.25)


class TestBlackBoxAndRobustAttackContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.x, cls.y = _make_trained_real_classifier()

    def setUp(self) -> None:
        self.model.eval()

    def test_zero_eps_returns_exact_clean_input_for_batch4(self) -> None:
        x_sub, y_sub = self.x[:8], self.y[:8]
        for name, fn in (
            ("spsa", spsa),
            ("square_attack", square_attack),
            ("hopskipjump", hopskipjump),
            ("eot_pgd", eot_pgd),
            ("uap", uap),
        ):
            with self.subTest(attack=name):
                res = fn(self.model, x_sub, y_sub, eps=0.0, norm="linf", clip_bounds=(0.0, 1.0))
                self.assertTrue(torch.equal(res.x_adv, x_sub))
                self.assertTrue(torch.all(res.perturbation == 0))

    def test_score_blackbox_attacks_bypass_non_differentiable_quantizer(self) -> None:
        def quantized_oracle(z: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return self.model(torch.round(z * 50.0) / 50.0)

        x_sub, y_sub = self.x[:16], self.y[:16]
        res_spsa = spsa(
            quantized_oracle,
            x_sub,
            y_sub,
            eps=0.35,
            steps=25,
            nb_sample=32,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=1,
        )
        res_sq_linf = square_attack(
            quantized_oracle,
            x_sub,
            y_sub,
            eps=0.35,
            max_queries=300,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=2,
        )
        res_sq_l2 = square_attack(
            quantized_oracle,
            x_sub,
            y_sub,
            eps=1.0,
            max_queries=300,
            norm="l2",
            clip_bounds=(0.0, 1.0),
            seed=3,
        )

        for name, res, norm_kind, budget in (
            ("spsa", res_spsa, "linf", 0.35),
            ("square_linf", res_sq_linf, "linf", 0.35),
            ("square_l2", res_sq_l2, "l2", 1.0),
        ):
            with self.subTest(attack=name):
                self.assertGreaterEqual(res.x_adv.min().item(), -1e-6)
                self.assertLessEqual(res.x_adv.max().item(), 1.0 + 1e-6)
                self.assertLessEqual(
                    compute_perturbation_norm(res.perturbation, norm_kind).max().item(),
                    budget + 1e-5,
                )
                acc = (quantized_oracle(res.x_adv).argmax(dim=1) == y_sub).float().mean().item()
                self.assertLess(acc, 0.35, msg=f"{name} failed against non-differentiable oracle")
                self.assertGreater(res.extra["queries"], 0)

    def test_square_attack_4d_image_branch(self) -> None:
        torch.manual_seed(11)
        centers = torch.zeros(4, 3, 12, 12)
        centers[0, 0] = 0.8
        centers[1, 1] = 0.8
        centers[2, 2] = 0.8
        centers[3, :, :6, :] = 0.8
        x_tr = (centers.repeat(8, 1, 1, 1) + 0.1 * torch.randn(32, 3, 12, 12)).clamp(0.0, 1.0)
        y_tr = torch.arange(4).repeat(8)

        conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 4),
        )
        opt = torch.optim.Adam(conv.parameters(), lr=0.05)
        for _ in range(35):
            opt.zero_grad()
            F.cross_entropy(conv(x_tr), y_tr).backward()
            opt.step()
        conv.eval()

        x_ev, y_ev = x_tr[:12], y_tr[:12]
        for norm_kind, budget in (("linf", 0.35), ("l2", 5.5)):
            with self.subTest(norm=norm_kind):
                res = square_attack(
                    conv,
                    x_ev,
                    y_ev,
                    eps=budget,
                    max_queries=400,
                    norm=norm_kind,
                    clip_bounds=(0.0, 1.0),
                    seed=5,
                )
                self.assertLessEqual(
                    compute_perturbation_norm(res.perturbation, norm_kind).max().item(),
                    budget + 1e-5,
                )
                with torch.no_grad():
                    acc = (conv(res.x_adv).argmax(dim=1) == y_ev).float().mean().item()
                self.assertLess(acc, 0.35)

    def test_hopskipjump_hard_label_only_untargeted_and_targeted(self) -> None:
        def label_only_oracle(z: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return self.model(z).argmax(dim=1)

        x_sub, y_sub = self.x[:8], self.y[:8]
        res_untargeted = hopskipjump(
            label_only_oracle,
            x_sub,
            y_sub,
            steps=10,
            initial_num_evals=40,
            norm="l2",
            clip_bounds=(0.0, 1.0),
            seed=4,
        )
        self.assertGreaterEqual(res_untargeted.x_adv.min().item(), -1e-6)
        self.assertLessEqual(res_untargeted.x_adv.max().item(), 1.0 + 1e-6)
        self.assertLess(res_untargeted.extra["mean_distortion"], 2.0)
        self.assertLess((label_only_oracle(res_untargeted.x_adv) == y_sub).float().mean().item(), 0.2)

        target = (y_sub + 1) % 4
        res_targeted = hopskipjump(
            label_only_oracle,
            x_sub,
            target,
            steps=10,
            initial_num_evals=40,
            norm="linf",
            targeted=True,
            clip_bounds=(0.0, 1.0),
            seed=6,
        )
        self.assertGreater(
            (label_only_oracle(res_targeted.x_adv) == target).float().mean().item(), 0.75
        )

    def test_eot_pgd_defeats_stochastic_noise_transform(self) -> None:
        x_sub, y_sub = self.x[:20], self.y[:20]

        def noisy_transform(z: torch.Tensor) -> torch.Tensor:
            return (z + 0.08 * torch.randn_like(z)).clamp(0.0, 1.0)

        res = eot_pgd(
            self.model,
            x_sub,
            y_sub,
            eps=0.30,
            steps=20,
            eot_samples=8,
            transform=noisy_transform,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=9,
        )
        self.assertLessEqual(
            compute_perturbation_norm(res.perturbation, "linf").max().item(), 0.30 + 1e-6
        )
        with torch.no_grad():
            accs = [
                (self.model(noisy_transform(res.x_adv)).argmax(dim=1) == y_sub).float().mean().item()
                for _ in range(5)
            ]
        self.assertLess(sum(accs) / len(accs), 0.25)

    def test_uap_single_shared_vector_fools_batch(self) -> None:
        res = uap(
            self.model,
            self.x,
            self.y,
            eps=0.55,
            delta=0.1,
            max_iter_uni=8,
            deepfool_steps=10,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=7,
        )
        v = res.extra["universal_perturbation"]
        self.assertEqual(v.shape, (1, 16))
        self.assertLessEqual(v.abs().max().item(), 0.55 + 1e-6)
        x_check = (self.x + v).clamp(0.0, 1.0)
        self.assertTrue(torch.allclose(x_check, res.x_adv))
        with torch.no_grad():
            acc = (self.model(x_check).argmax(dim=1) == self.y).float().mean().item()
        self.assertLess(acc, 0.35)

    def test_batch4_complex_wireless_jsr_db(self) -> None:
        torch.manual_seed(303)
        rx = TinyComplexWirelessClassifier(seq_len=32, num_classes=4).eval()
        iq = torch.complex(torch.randn(16, 32), torch.randn(16, 32))
        with torch.no_grad():
            labels = rx(iq).argmax(dim=1)

        def fading_channel(z: torch.Tensor) -> torch.Tensor:
            phase = torch.randn(z.shape[0], 1, device=z.device) * 0.1
            return z * torch.exp(1j * phase)

        for name, fn, budget_db, kwargs in (
            ("spsa_complex", spsa, -3.0, {"steps": 25, "nb_sample": 32, "seed": 1}),
            ("square_complex", square_attack, -3.0, {"max_queries": 350, "seed": 2}),
            ("hopskipjump_complex", hopskipjump, 0.0, {"steps": 10, "initial_num_evals": 40, "seed": 3}),
            (
                "eot_pgd_complex",
                eot_pgd,
                -3.0,
                {"steps": 15, "eot_samples": 6, "transform": fading_channel, "seed": 4},
            ),
            ("uap_complex", uap, 2.0, {"max_iter_uni": 6, "deepfool_steps": 10, "seed": 5}),
        ):
            with self.subTest(attack=name):
                res = fn(rx, iq, labels, eps=budget_db, norm="jsr_db", **kwargs)
                self.assertTrue(res.x_adv.is_complex())
                self.assertFalse(torch.isnan(res.x_adv.real).any())
                self.assertFalse(torch.isnan(res.x_adv.imag).any())
                measured_db = compute_perturbation_norm(res.perturbation, "jsr_db", x_ref=iq)
                self.assertLessEqual(measured_db.max().item(), budget_db + 1e-4)
                with torch.no_grad():
                    acc = (rx(res.x_adv).argmax(dim=1) == labels).float().mean().item()
                self.assertLess(acc, 0.45, msg=f"{name} failed to degrade complex wireless rx")


if __name__ == "__main__":
    unittest.main()
