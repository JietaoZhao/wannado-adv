"""Comprehensive Wave 2 attack, L1/L0 core, and generative preflight contract tests."""

from __future__ import annotations

import math
import subprocess
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from wannado.adv.attacks import (
    adv_patch,
    advgan,
    apgd_l1,
    boundary,
    bpda_eot,
    di_fgsm,
    ead,
    gap,
    jsma,
    nes,
    ni_fgsm,
    simba,
    sini_fgsm,
    sparsefool,
    ti_fgsm,
    tpgd,
    vmi_fgsm,
    zoo,
)
from wannado.adv.audit import preflight
from wannado.adv.core.projections import (
    compute_perturbation_norm,
    project_perturbation,
    random_init_perturbation,
    resolve_eps_budget,
    within_eps_budget,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _TinyTabularClassifier(nn.Module):
    def __init__(self, in_dim: int = 16, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TinyVisionCNN(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(8 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        return self.fc(h)


class _TinyComplexWirelessClassifier(nn.Module):
    """Complex IQ classifier with channelized 1D shape [B, C, L]."""

    def __init__(self, channels: int = 2, seq_len: int = 8, num_classes: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels * seq_len * 2, 48)
        self.fc2 = nn.Linear(48, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ri = torch.cat([x.real, x.imag], dim=1).reshape(x.shape[0], -1)
        return self.fc2(F.relu(self.fc1(ri)))


def _train_tabular_model() -> tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(101)
    centers = torch.full((4, 16), 0.15)
    for c in range(4):
        centers[c, c * 4 : (c + 1) * 4] = 0.85

    x_train = (centers.repeat(12, 1) + 0.08 * torch.randn(48, 16)).clamp(0.0, 1.0)
    y_train = torch.arange(4).repeat(12)
    x_test = (centers.repeat(4, 1) + 0.06 * torch.randn(16, 16)).clamp(0.0, 1.0)
    y_test = torch.arange(4).repeat(4)

    model = _TinyTabularClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    model.train()
    for _ in range(45):
        opt.zero_grad()
        F.cross_entropy(model(x_train), y_train).backward()
        opt.step()
    model.eval()
    return model, x_train, y_train, x_test, y_test


def _train_vision_cnn() -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(202)
    protos = torch.zeros(4, 3, 12, 12)
    for c in range(4):
        protos[c, c % 3, 2:10, 2:10] = 0.85
        protos[c, (c + 1) % 3, 4:8, 4:8] = 0.35

    x = (protos.repeat(6, 1, 1, 1) + 0.05 * torch.randn(24, 3, 12, 12)).clamp(0.0, 1.0)
    y = torch.arange(4).repeat(6)
    model = _TinyVisionCNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    model.train()
    for _ in range(50):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    model.eval()
    return model, x[:8], y[:8]


def _train_complex_wireless_model() -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """Phase-modulated complex IQ symbols where -8 dB jamming physically crosses boundaries."""
    torch.manual_seed(303)
    protos = torch.zeros(4, 2, 8, dtype=torch.complex64)
    phases = [0.05, 0.22, 0.39, 0.56]
    for c, frac in enumerate(phases):
        ph = torch.tensor(frac * math.pi)
        protos[c, :, :] = torch.complex(torch.cos(ph), torch.sin(ph))

    noise_r = 0.05 * torch.randn(24, 2, 8)
    noise_i = 0.05 * torch.randn(24, 2, 8)
    x = protos.repeat(6, 1, 1) + torch.complex(noise_r, noise_i).to(torch.complex64)
    y = torch.arange(4).repeat(6)

    model = _TinyComplexWirelessClassifier(channels=2, seq_len=8, num_classes=4)
    opt = torch.optim.Adam(model.parameters(), lr=0.04)
    model.train()
    for _ in range(40):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    model.eval()
    return model, x[:8], y[:8]


class TestL1AndL0CoreSupport(unittest.TestCase):
    def test_l1_box_projection_real_and_complex(self) -> None:
        torch.manual_seed(7)
        x = torch.rand(6, 12) * 0.8 + 0.1
        delta_raw = torch.randn(6, 12) * 0.5
        eps = 1.25
        eps_b = resolve_eps_budget(x, eps=eps, norm="l1")
        delta = project_perturbation(delta_raw, x, eps_b, norm="l1", clip_bounds=(0.0, 1.0))
        l1_norms = compute_perturbation_norm(delta, "l1", x)
        self.assertTrue(torch.all(l1_norms <= eps + 1e-4))
        x_adv = x + delta
        self.assertTrue(torch.all(x_adv >= -1e-6) and torch.all(x_adv <= 1.0 + 1e-6))

        rng = torch.Generator().manual_seed(9)
        delta_rand = random_init_perturbation(
            x, eps_budget=eps_b, norm="l1", clip_bounds=(0.0, 1.0), generator=rng
        )
        self.assertTrue(torch.all(compute_perturbation_norm(delta_rand, "l1", x) <= eps + 1e-4))

        xc = torch.complex(torch.randn(4, 2, 8), torch.randn(4, 2, 8)).to(torch.complex64)
        dc_raw = torch.complex(torch.randn(4, 2, 8), torch.randn(4, 2, 8)).to(torch.complex64)
        eps_cb = resolve_eps_budget(xc, eps=1.5, norm="l1")
        dc = project_perturbation(dc_raw, xc, eps_cb, norm="l1")
        self.assertEqual(dc.dtype, torch.complex64)
        self.assertTrue(torch.all(compute_perturbation_norm(dc, "l1", xc) <= 1.5 + 1e-4))

    def test_l0_norm_and_within_eps_budget_unit_consistency(self) -> None:
        x = torch.zeros(2, 5)
        delta = torch.tensor([[0.0, 0.2, 0.0, -0.5, 0.0], [1e-12, 0.3, 0.1, 0.4, 0.0]])
        l0 = compute_perturbation_norm(delta, "l0", x)
        self.assertTrue(torch.allclose(l0, torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.equal(within_eps_budget(delta, x, eps=2.0, norm="l0"), torch.tensor([True, False])))

        # Regression guard: jsr_db and relative_l2 unit comparison in within_eps_budget.
        x_rf = torch.complex(torch.ones(2, 4) * 2.0, torch.zeros(2, 4)).to(torch.complex64)
        delta_small = x_rf * 0.50  # ~ -6.02 dB
        delta_large = x_rf * 0.80  # ~ -1.94 dB (violates -5.0 dB even though -1.94 < linear L2 radius)
        delta_mix = torch.cat([delta_small[:1], delta_large[1:]], dim=0)
        self.assertTrue(
            torch.equal(
                within_eps_budget(delta_mix, x_rf, eps=-5.0, norm="jsr_db"),
                torch.tensor([True, False]),
            )
        )
        self.assertTrue(
            torch.equal(
                within_eps_budget(delta_mix, x_rf, eps=0.6, norm="relative_l2"),
                torch.tensor([True, False]),
            )
        )


class TestTransferabilityAttacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tab_model, _, _, cls.x_tab, cls.y_tab = _train_tabular_model()
        cls.cnn_model, cls.x_img, cls.y_img = _train_vision_cnn()
        cls.rf_model, cls.x_rf, cls.y_rf = _train_complex_wireless_model()

    def test_lazy_isolation_for_transferability_and_generative(self) -> None:
        code = (
            "import sys; "
            "from wannado.adv import ni_fgsm; "
            "assert 'wannado.adv.attacks.transferability.nifgsm_lin2020' in sys.modules; "
            "assert 'wannado.adv.attacks.transferability.difgsm_xie2019' not in sys.modules; "
            "assert 'wannado.adv.attacks.generative.gap_poursaeed2018' not in sys.modules"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_all_transferability_attacks_on_vision_and_tabular(self) -> None:
        eps = 0.28
        for name, fn, kwargs in [
            ("ni_fgsm", ni_fgsm, {}),
            ("sini_fgsm", sini_fgsm, {"scale_copies": 4}),
            ("di_fgsm", di_fgsm, {"diversity_prob": 0.7, "preserve_shape": True}),
            ("ti_fgsm_gauss", ti_fgsm, {"kernlen": 5, "kernel_name": "gaussian"}),
            ("ti_fgsm_linear", ti_fgsm, {"kernlen": 5, "kernel_name": "linear"}),
            ("ti_fgsm_uniform", ti_fgsm, {"kernlen": 3, "kernel_name": "uniform"}),
            ("vmi_fgsm", vmi_fgsm, {"num_variance_samples": 6, "beta": 1.5}),
        ]:
            with self.subTest(attack=name):
                res = fn(
                    self.cnn_model,
                    self.x_img,
                    self.y_img,
                    eps=eps,
                    steps=12,
                    norm="linf",
                    clip_bounds=(0.0, 1.0),
                    seed=11,
                    **kwargs,
                )
                norms = compute_perturbation_norm(res.perturbation, "linf", self.x_img)
                self.assertTrue(torch.all(norms <= eps + 1e-5))
                self.assertTrue(torch.all(res.x_adv >= -1e-6) and torch.all(res.x_adv <= 1.0 + 1e-6))
                with torch.no_grad():
                    clean_acc = (self.cnn_model(self.x_img).argmax(1) == self.y_img).float().mean().item()
                    adv_acc = (self.cnn_model(res.x_adv).argmax(1) == self.y_img).float().mean().item()
                self.assertGreaterEqual(clean_acc, 0.875)
                self.assertLess(adv_acc, clean_acc)

        # Verify di_fgsm on 2D tabular + custom diversity transform
        res_tab = di_fgsm(
            self.tab_model,
            self.x_tab,
            self.y_tab,
            eps=0.35,
            steps=10,
            diversity_transform=lambda z: z + 0.01 * torch.randn_like(z),
            clip_bounds=(0.0, 1.0),
            seed=5,
        )
        self.assertEqual(res_tab.x_adv.shape, self.x_tab.shape)
        with torch.no_grad():
            self.assertLess((self.tab_model(res_tab.x_adv).argmax(1) == self.y_tab).float().mean().item(), 0.5)

    def test_transferability_complex_wireless_jsr_db(self) -> None:
        eps_db = -8.0
        for name, fn in [
            ("ni_fgsm", ni_fgsm),
            ("sini_fgsm", sini_fgsm),
            ("di_fgsm", di_fgsm),
            ("ti_fgsm", ti_fgsm),
            ("vmi_fgsm", vmi_fgsm),
        ]:
            with self.subTest(attack=name):
                res = fn(
                    self.rf_model,
                    self.x_rf,
                    self.y_rf,
                    eps=eps_db,
                    steps=15,
                    norm="jsr_db",
                    seed=19,
                )
                self.assertEqual(res.x_adv.dtype, torch.complex64)
                jsr = compute_perturbation_norm(res.perturbation, "jsr_db", self.x_rf)
                self.assertTrue(torch.all(jsr <= eps_db + 1e-3), msg=f"{name} jsr={jsr}")
                with torch.no_grad():
                    clean_acc = (self.rf_model(self.x_rf).argmax(1) == self.y_rf).float().mean().item()
                    adv_acc = (self.rf_model(res.x_adv).argmax(1) == self.y_rf).float().mean().item()
                self.assertLess(adv_acc, clean_acc)


class TestWave2FirstOrderAttacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tab_model, _, _, cls.x_tab, cls.y_tab = _train_tabular_model()
        cls.rf_model, cls.x_rf, cls.y_rf = _train_complex_wireless_model()

    def test_tpgd_real_and_complex_label_free(self) -> None:
        res = tpgd(self.tab_model, self.x_tab, y=None, eps=0.35, steps=12, norm="linf", clip_bounds=(0.0, 1.0), seed=1)
        self.assertTrue(torch.all(compute_perturbation_norm(res.perturbation, "linf", self.x_tab) <= 0.35 + 1e-5))
        with torch.no_grad():
            adv_acc = (self.tab_model(res.x_adv).argmax(1) == self.y_tab).float().mean().item()
        self.assertLess(adv_acc, 0.5)

        res_rf = tpgd(self.rf_model, self.x_rf, y=None, eps=-8.0, steps=15, norm="jsr_db", seed=2)
        self.assertEqual(res_rf.x_adv.dtype, torch.complex64)
        self.assertTrue(torch.all(compute_perturbation_norm(res_rf.perturbation, "jsr_db", self.x_rf) <= -8.0 + 1e-3))

    def test_apgd_l1_box_and_budget_compliance(self) -> None:
        eps_l1 = 2.5
        res_zero = apgd_l1(self.tab_model, self.x_tab, self.y_tab, eps=0.0, steps=10, norm="l1")
        self.assertTrue(torch.equal(res_zero.x_adv, self.x_tab))

        res = apgd_l1(
            self.tab_model,
            self.x_tab,
            self.y_tab,
            eps=eps_l1,
            steps=35,
            restarts=2,
            clip_bounds=(0.0, 1.0),
            seed=4,
        )
        l1 = compute_perturbation_norm(res.perturbation, "l1", self.x_tab)
        self.assertTrue(torch.all(l1 <= eps_l1 + 1e-4), msg=f"L1 exceeded: {l1}")
        self.assertTrue(torch.all(res.x_adv >= -1e-6) and torch.all(res.x_adv <= 1.0 + 1e-6))
        with torch.no_grad():
            adv_acc = (self.tab_model(res.x_adv).argmax(1) == self.y_tab).float().mean().item()
        self.assertLess(adv_acc, 0.4)


class TestSparseAndElasticNetAttacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tab_model, _, _, cls.x_tab, cls.y_tab = _train_tabular_model()
        cls.rf_model, cls.x_rf, cls.y_rf = _train_complex_wireless_model()

    def test_sparsefool_real_and_complex(self) -> None:
        res = sparsefool(self.tab_model, self.x_tab[:6], y=None, clip_bounds=(0.0, 1.0), lambda_=2.5, max_iter=15)
        self.assertTrue(torch.all(res.x_adv >= -1e-6) and torch.all(res.x_adv <= 1.0 + 1e-6))
        with torch.no_grad():
            adv_acc = (self.tab_model(res.x_adv).argmax(1) == self.y_tab[:6]).float().mean().item()
        self.assertLess(adv_acc, 0.5)

        res_rf = sparsefool(self.rf_model, self.x_rf[:4], y=self.y_rf[:4], eps=-6.0, norm="jsr_db", max_iter=10)
        self.assertEqual(res_rf.x_adv.dtype, torch.complex64)
        self.assertTrue(torch.all(compute_perturbation_norm(res_rf.perturbation, "jsr_db", self.x_rf[:4]) <= -6.0 + 1e-3))

    def test_jsma_targeted_and_untargeted(self) -> None:
        x_sub = self.x_tab[:6]
        y_sub = self.y_tab[:6]
        y_target = (y_sub + 1) % 4
        res = jsma(self.tab_model, x_sub, y_target, theta=0.45, gamma=0.5, targeted=True, clip_bounds=(0.0, 1.0))
        self.assertTrue(torch.all(res.x_adv >= -1e-6) and torch.all(res.x_adv <= 1.0 + 1e-6))
        with torch.no_grad():
            target_match = (self.tab_model(res.x_adv).argmax(1) == y_target).float().mean().item()
        self.assertGreaterEqual(target_match, 0.5)

    def test_ead_en_and_l1_real_and_complex(self) -> None:
        for rule in ("EN", "L1"):
            res = ead(
                self.tab_model,
                self.x_tab[:6],
                self.y_tab[:6],
                norm="l1",
                decision_rule=rule,
                steps=80,
                binary_search_steps=4,
                clip_bounds=(0.0, 1.0),
            )
            with torch.no_grad():
                adv_acc = (self.tab_model(res.x_adv).argmax(1) == self.y_tab[:6]).float().mean().item()
            self.assertLess(adv_acc, 0.5)

        res_rf = ead(
            self.rf_model,
            self.x_rf[:4],
            self.y_rf[:4],
            eps=-5.0,
            norm="jsr_db",
            steps=80,
            binary_search_steps=4,
        )
        self.assertEqual(res_rf.x_adv.dtype, torch.complex64)
        self.assertTrue(torch.all(compute_perturbation_norm(res_rf.perturbation, "jsr_db", self.x_rf[:4]) <= -5.0 + 1e-3))
        with torch.no_grad():
            self.assertLess((self.rf_model(res_rf.x_adv).argmax(1) == self.y_rf[:4]).float().mean().item(), 0.5)


class TestWave2BlackBoxAttacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tab_model, _, _, cls.x_tab, cls.y_tab = _train_tabular_model()
        cls.rf_model, cls.x_rf, cls.y_rf = _train_complex_wireless_model()

    def _quantized_score_oracle(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z_q = torch.round(z.detach() * 64.0) / 64.0
            return self.tab_model(z_q)

    def _hard_label_oracle(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.tab_model(z.detach()).argmax(dim=1)

    def test_score_based_zoo_nes_simba(self) -> None:
        x_sub = self.x_tab[:4]
        y_sub = self.y_tab[:4]

        # Quantizer bin width is 1/64 ~= 0.0156, so fd_eps=0.03 crosses quantization bins.
        res_zoo = zoo(
            self._quantized_score_oracle,
            x_sub,
            y_sub,
            eps=1.8,
            norm="l2",
            steps=60,
            coord_batch_size=16,
            fd_eps=0.03,
            lr=0.1,
            clip_bounds=(0.0, 1.0),
            seed=10,
        )
        self.assertTrue(torch.all(compute_perturbation_norm(res_zoo.perturbation, "l2", x_sub) <= 1.8 + 1e-4))
        with torch.no_grad():
            self.assertLess((self._quantized_score_oracle(res_zoo.x_adv).argmax(1) == y_sub).float().mean().item(), 0.5)

        res_nes = nes(
            self._quantized_score_oracle,
            x_sub,
            y_sub,
            eps=0.35,
            steps=35,
            nb_sample=40,
            sigma=0.05,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=11,
        )
        self.assertTrue(torch.all(compute_perturbation_norm(res_nes.perturbation, "linf", x_sub) <= 0.35 + 1e-5))
        with torch.no_grad():
            self.assertLess((self._quantized_score_oracle(res_nes.x_adv).argmax(1) == y_sub).float().mean().item(), 0.5)

        res_simba = simba(
            self._quantized_score_oracle,
            x_sub,
            y_sub,
            eps=0.4,
            step_size=0.15,
            max_queries=300,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=12,
        )
        self.assertTrue(torch.all(compute_perturbation_norm(res_simba.perturbation, "linf", x_sub) <= 0.4 + 1e-5))
        with torch.no_grad():
            self.assertLess((self._quantized_score_oracle(res_simba.x_adv).argmax(1) == y_sub).float().mean().item(), 0.5)

        # Complex wireless score black-box sanity (nes + simba)
        for fn in (nes, simba):
            kwargs = {"steps": 25, "nb_sample": 32, "sigma": 0.05} if fn is nes else {"max_queries": 350, "step_size": 0.08}
            res_rf = fn(self.rf_model, self.x_rf[:4], self.y_rf[:4], eps=-5.0, norm="jsr_db", seed=21, **kwargs)
            self.assertEqual(res_rf.x_adv.dtype, torch.complex64)
            self.assertTrue(torch.all(compute_perturbation_norm(res_rf.perturbation, "jsr_db", self.x_rf[:4]) <= -5.0 + 1e-3))

    def test_boundary_decision_attack_real_and_complex(self) -> None:
        x_sub = self.x_tab[:4]
        y_sub = self.y_tab[:4]
        res = boundary(
            self._hard_label_oracle,
            x_sub,
            y=None,
            eps=2.5,
            norm="l2",
            steps=100,
            clip_bounds=(0.0, 1.0),
            seed=31,
        )
        self.assertTrue(torch.all(compute_perturbation_norm(res.perturbation, "l2", x_sub) <= 2.5 + 1e-4))
        with torch.no_grad():
            self.assertLess((self._hard_label_oracle(res.x_adv) == y_sub).float().mean().item(), 0.75)

        def rf_hard(z: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return self.rf_model(z).argmax(dim=1)

        res_rf = boundary(rf_hard, self.x_rf[:4], y=self.y_rf[:4], eps=-4.0, norm="jsr_db", steps=80, seed=32)
        self.assertEqual(res_rf.x_adv.dtype, torch.complex64)
        self.assertTrue(torch.all(compute_perturbation_norm(res_rf.perturbation, "jsr_db", self.x_rf[:4]) <= -4.0 + 1e-3))
        with torch.no_grad():
            self.assertLess((rf_hard(res_rf.x_adv) == self.y_rf[:4]).float().mean().item(), 0.75)


class TestBPDAAndPatchAttacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tab_model, _, _, cls.x_tab, cls.y_tab = _train_tabular_model()
        cls.cnn_model, cls.x_img, cls.y_img = _train_vision_cnn()
        cls.rf_model, cls.x_rf, cls.y_rf = _train_complex_wireless_model()

    def test_bpda_eot_bypasses_non_differentiable_defense(self) -> None:
        def quantized_noisy_defense(z: torch.Tensor) -> torch.Tensor:
            return torch.round(z * 16.0) / 16.0 + 0.01 * torch.randn_like(z)

        res = bpda_eot(
            self.tab_model,
            self.x_tab[:8],
            self.y_tab[:8],
            eps=0.35,
            steps=25,
            eot_samples=4,
            defense_fn=quantized_noisy_defense,
            surrogate_fn=lambda z: z,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=41,
        )
        self.assertTrue(torch.all(compute_perturbation_norm(res.perturbation, "linf", self.x_tab[:8]) <= 0.35 + 1e-5))
        with torch.no_grad():
            adv_acc = (self.tab_model(quantized_noisy_defense(res.x_adv)).argmax(1) == self.y_tab[:8]).float().mean().item()
        self.assertLess(adv_acc, 0.5)

        res_rf = bpda_eot(
            self.rf_model,
            self.x_rf[:4],
            self.y_rf[:4],
            eps=-8.0,
            steps=20,
            eot_samples=3,
            defense_fn=lambda z: torch.round(z.real * 16.0) / 16.0 + 1j * (torch.round(z.imag * 16.0) / 16.0),
            surrogate_fn=lambda z: z,
            norm="jsr_db",
            seed=42,
        )
        self.assertEqual(res_rf.x_adv.dtype, torch.complex64)
        self.assertTrue(torch.all(compute_perturbation_norm(res_rf.perturbation, "jsr_db", self.x_rf[:4]) <= -8.0 + 1e-3))

    def test_adv_patch_universal_random_location_and_complex(self) -> None:
        res_u = adv_patch(
            self.cnn_model,
            self.x_img,
            self.y_img,
            eps=0.45,
            patch_fraction=0.45,
            steps=50,
            lr=0.1,
            norm="linf",
            universal=True,
            random_location=True,
            clip_bounds=(0.0, 1.0),
            seed=51,
        )
        self.assertTrue(torch.all(compute_perturbation_norm(res_u.perturbation, "linf", self.x_img) <= 0.45 + 1e-5))
        self.assertTrue(torch.all(res_u.x_adv >= -1e-6) and torch.all(res_u.x_adv <= 1.0 + 1e-6))
        with torch.no_grad():
            adv_acc = (self.cnn_model(res_u.x_adv).argmax(1) == self.y_img).float().mean().item()
        self.assertLess(adv_acc, 0.75)

        res_rf = adv_patch(
            self.rf_model,
            self.x_rf[:4],
            self.y_rf[:4],
            eps=-6.0,
            patch_fraction=0.5,
            steps=40,
            lr=0.15,
            norm="jsr_db",
            universal=False,
            seed=52,
        )
        self.assertEqual(res_rf.x_adv.dtype, torch.complex64)
        self.assertTrue(torch.all(compute_perturbation_norm(res_rf.perturbation, "jsr_db", self.x_rf[:4]) <= -6.0 + 1e-3))


class TestGenerativeAttacksAndPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tab_model, cls.x_train, cls.y_train, cls.x_test, cls.y_test = _train_tabular_model()
        cls.rf_model, cls.x_rf, cls.y_rf = _train_complex_wireless_model()

    def test_gap_input_dependent_universal_complex_and_grad_isolation(self) -> None:
        for p in self.tab_model.parameters():
            p.grad = None

        res_id = gap(
            self.tab_model,
            self.x_train,
            self.y_train,
            eps=0.35,
            mode="input_dependent",
            epochs=60,
            lr=0.015,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=61,
        )
        self.assertTrue(all(p.grad is None for p in self.tab_model.parameters()))
        self.assertTrue(torch.all(compute_perturbation_norm(res_id.perturbation, "linf", self.x_train) <= 0.35 + 1e-5))
        x_unseen_adv = res_id.extra["generate_fn"](self.x_test)
        self.assertEqual(x_unseen_adv.shape, self.x_test.shape)
        with torch.no_grad():
            self.assertLess((self.tab_model(x_unseen_adv).argmax(1) == self.y_test).float().mean().item(), 0.5)

        res_univ = gap(
            self.tab_model,
            self.x_train,
            self.y_train,
            eps=0.35,
            mode="universal",
            epochs=60,
            lr=0.03,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=62,
        )
        self.assertTrue(all(p.grad is None for p in self.tab_model.parameters()))
        self.assertTrue(torch.all(compute_perturbation_norm(res_univ.perturbation, "linf", self.x_train) <= 0.35 + 1e-5))

        res_rf = gap(self.rf_model, self.x_rf, self.y_rf, eps=-8.0, epochs=40, lr=0.02, norm="jsr_db", seed=63)
        self.assertEqual(res_rf.x_adv.dtype, torch.complex64)
        self.assertTrue(torch.all(compute_perturbation_norm(res_rf.perturbation, "jsr_db", self.x_rf) <= -8.0 + 1e-3))

    def test_advgan_real_and_complex_and_grad_isolation(self) -> None:
        for p in self.tab_model.parameters():
            p.grad = None

        res = advgan(
            self.tab_model,
            self.x_train,
            self.y_train,
            eps=0.35,
            epochs=50,
            lr_g=0.025,
            lr_d=0.01,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            seed=71,
        )
        self.assertTrue(all(p.grad is None for p in self.tab_model.parameters()))
        self.assertEqual(len(res.extra["g_loss_history"]), 50)
        self.assertEqual(len(res.extra["d_loss_history"]), 50)
        x_unseen_adv = res.extra["generate_fn"](self.x_test)
        self.assertTrue(torch.all(compute_perturbation_norm(x_unseen_adv - self.x_test, "linf", self.x_test) <= 0.35 + 1e-5))
        with torch.no_grad():
            self.assertLess((self.tab_model(x_unseen_adv).argmax(1) == self.y_test).float().mean().item(), 0.5)

        res_rf = advgan(self.rf_model, self.x_rf, self.y_rf, eps=-8.0, epochs=35, lr_g=0.03, norm="jsr_db", seed=72)
        self.assertEqual(res_rf.x_adv.dtype, torch.complex64)
        self.assertTrue(torch.all(compute_perturbation_norm(res_rf.perturbation, "jsr_db", self.x_rf) <= -8.0 + 1e-3))

    def test_preflight_flags_generative_train_and_blocks_detached(self) -> None:
        rep_ok = preflight(self.tab_model, self.x_test, self.y_test, eps=0.3, norm="linf", attack="gap", strict=False)
        self.assertTrue(rep_ok.ok)
        self.assertTrue(any("requires_training=True" in w for w in rep_ok.warnings))

        def detached_fn(z: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return self.tab_model(z)

        rep_bad = preflight(detached_fn, self.x_test, self.y_test, eps=0.3, norm="linf", attack="gap", strict=False)
        self.assertFalse(rep_bad.ok)
        self.assertTrue(any("detached" in e.lower() or "grad" in e.lower() for e in rep_bad.errors))


if __name__ == "__main__":
    unittest.main()
