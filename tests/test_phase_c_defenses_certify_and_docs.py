"""Phase C tests: single-file defenses (PGD-AT, TRADES, MART), zero-SciPy
Randomized Smoothing L2 certification, citations/ejection, lazy imports, and streamlined agent_kit.
"""

from __future__ import annotations

import copy
import io
import itertools
import json
import math
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wannado.adv import (
    DefenseTrainResult,
    SmoothingCertificationResult,
    __version__,
    apgd_dlr,
    audit_robustness,
    certify_randomized_smoothing,
    clopper_pearson_lower,
    eject_attack,
    eject_defense,
    export_citations,
    gaussian_augmented_train,
    list_defenses,
    mart_train,
    norm_ppf,
    pgd_adversarial_train,
    smooth_predict,
    trades_train,
)
from wannado.adv.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parent.parent


class _ToyMLP(nn.Module):
    def __init__(self, in_dim: int = 16, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ToyComplexRFNet(nn.Module):
    def __init__(self, seq_len: int = 16, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([iq.real, iq.imag], dim=-1).reshape(iq.shape[0], -1)
        return self.net(feats)


def _make_dataset(
    n_train: int = 256,
    n_test: int = 48,
    in_dim: int = 24,
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    centers = torch.full((4, in_dim), 0.25)
    block = in_dim // 4
    for c in range(4):
        centers[c, c * block : (c + 1) * block] = 0.78
    y_tr = torch.arange(4).repeat(n_train // 4)
    x_tr = (centers[y_tr] + 0.13 * torch.randn((n_train, in_dim), generator=g)).clamp(0.0, 1.0)
    y_te = torch.arange(4).repeat(n_test // 4)
    x_te = (centers[y_te] + 0.12 * torch.randn((n_test, in_dim), generator=g)).clamp(0.0, 1.0)
    return x_tr, y_tr, x_te, y_te


def _binom_upper_tail(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, j) * (p**j) * ((1.0 - p) ** (n - j)) for j in range(k, n + 1))


class TestZeroSciPyStatsAndSmoothing(unittest.TestCase):
    def test_norm_ppf_and_clopper_pearson_exact_math(self) -> None:
        self.assertAlmostEqual(norm_ppf(0.5), 0.0, places=9)
        self.assertAlmostEqual(norm_ppf(0.975), 1.9599639845, places=7)
        self.assertAlmostEqual(norm_ppf(0.999), 3.0902323062, places=7)
        self.assertAlmostEqual(norm_ppf(0.025), -norm_ppf(0.975), places=8)

        # Boundary checks
        self.assertEqual(clopper_pearson_lower(0, 100, 0.001), 0.0)
        self.assertAlmostEqual(clopper_pearson_lower(100, 100, 0.001), 0.001 ** (1.0 / 100.0), places=9)

        # Exact binomial tail identity: P(Bin(n, p_low) >= k) == alpha
        for k, n, alpha in ((95, 100, 0.001), (80, 100, 0.05), (55, 60, 0.01), (3, 10, 0.05)):
            p_low = clopper_pearson_lower(k, n, alpha)
            self.assertGreater(p_low, 0.0)
            self.assertLess(p_low, k / n)
            tail = _binom_upper_tail(k, n, p_low)
            self.assertAlmostEqual(tail, alpha, places=6, msg=f"Failed tail identity for k={k}, n={n}, alpha={alpha}")

    def test_gaussian_augmented_train_smooth_predict_and_certify(self) -> None:
        x_tr, y_tr, x_te, y_te = _make_dataset(n_train=256, n_test=32, in_dim=24, seed=11)
        model = _ToyMLP(in_dim=24, num_classes=4)
        res = gaussian_augmented_train(
            model,
            x_tr,
            y_tr,
            sigma=0.25,
            epochs=25,
            batch_size=64,
            lr=0.015,
            clip_bounds=(0.0, 1.0),
            eval_data=(x_te, y_te),
            seed=11,
        )
        self.assertIsInstance(res, DefenseTrainResult)
        self.assertEqual(res.defense_id, "randomized_smoothing")
        self.assertIsNotNone(res.final_clean_acc)
        assert res.final_clean_acc is not None
        self.assertGreater(res.final_clean_acc, 0.90)

        preds = smooth_predict(model, x_te, sigma=0.25, n_samples=120, clip_bounds=(0.0, 1.0), seed=11)
        self.assertEqual(preds.shape, y_te.shape)
        self.assertGreater((preds == y_te).float().mean().item(), 0.90)

        radii_grid = [0.0, 0.15, 0.30, 0.45, 0.60]
        cert = certify_randomized_smoothing(
            model,
            x_te[:20],
            y_te[:20],
            sigma=0.25,
            n0=50,
            n=400,
            alpha=0.001,
            batch_size=128,
            radii_grid=radii_grid,
            clip_bounds=(0.0, 1.0),
            seed=11,
        )
        self.assertIsInstance(cert, SmoothingCertificationResult)
        self.assertGreater(cert.median_certified_radius, 0.20)
        self.assertLess(cert.abstain_rate, 0.15)
        self.assertIn("cohen2019certified", cert.citation["bibtex"])

        # Certified accuracy curve must be monotonically non-increasing in R
        curve_vals = [cert.certified_acc_at_radii[f"{r:.4g}"] for r in radii_grid]
        for a, b in itertools.pairwise(curve_vals):
            self.assertGreaterEqual(a + 1e-9, b)


class TestAdversarialTrainingDefenses(unittest.TestCase):
    def test_pgd_at_trades_mart_harden_and_pass_carlini2(self) -> None:
        # Robust features (first 24 dims, wide margin with noise) + 4 clean-separable fragile shortcut
        # features (last 4 dims, +/-0.18 amplitude flipped completely inside eps=0.20 — textbook ERM vs AT).
        x_tr_rob, y_tr, x_te_rob, y_te = _make_dataset(n_train=384, n_test=48, in_dim=24, seed=42)
        oh_tr = F.one_hot(y_tr, num_classes=4).float()
        oh_te = F.one_hot(y_te, num_classes=4).float()
        x_tr = torch.cat([x_tr_rob, (0.32 + 0.36 * oh_tr)], dim=1).clamp(0.0, 1.0)
        x_te = torch.cat([x_te_rob, (0.32 + 0.36 * oh_te)], dim=1).clamp(0.0, 1.0)

        loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=64, shuffle=False)
        eps = 0.20

        # Unhardened baseline latches onto the fragile shortcut -> high clean acc, low robust acc
        torch.manual_seed(42)
        base_init = _ToyMLP(in_dim=28, num_classes=4)
        baseline = copy.deepcopy(base_init)
        opt = torch.optim.Adam(baseline.parameters(), lr=0.015)
        baseline.train()
        for _ in range(45):
            opt.zero_grad()
            F.cross_entropy(baseline(x_tr), y_tr).backward()
            opt.step()
        baseline.eval()
        base_adv = apgd_dlr(baseline, x_te, y_te, eps=eps, steps=20, restarts=1, norm="linf", clip_bounds=(0.0, 1.0), seed=42)
        base_rob = (baseline(base_adv.x_adv).argmax(dim=1) == y_te).float().mean().item()

        for defense_name, train_fn, extra_kwargs in (
            ("pgd_at", pgd_adversarial_train, {}),
            ("trades", trades_train, {"beta": 4.0}),
            ("mart", mart_train, {"beta": 3.0}),
        ):
            with self.subTest(defense=defense_name):
                mdl = copy.deepcopy(base_init)
                train_res = train_fn(
                    mdl,
                    loader,
                    eps=eps,
                    norm="linf",
                    attack_steps=7,
                    epochs=25,
                    lr=0.012,
                    clip_bounds=(0.0, 1.0),
                    eval_data=(x_te, y_te),
                    seed=42,
                    **extra_kwargs,
                )
                self.assertEqual(train_res.defense_id, defense_name)
                self.assertEqual(train_res.epochs_run, 25)
                self.assertGreater(len(train_res.history), 0)

                mdl.eval()
                clean_acc = (mdl(x_te).argmax(dim=1) == y_te).float().mean().item()
                adv_res = apgd_dlr(mdl, x_te, y_te, eps=eps, steps=20, restarts=1, norm="linf", clip_bounds=(0.0, 1.0), seed=42)
                rob_acc = (mdl(adv_res.x_adv).argmax(dim=1) == y_te).float().mean().item()
                self.assertGreater(clean_acc, 0.90)
                self.assertGreaterEqual(rob_acc, 0.85)
                self.assertGreater(rob_acc, base_rob + 0.25, msg=f"{defense_name} did not improve robust accuracy")

                audit = audit_robustness(
                    mdl,
                    x_te[:24],
                    y_te[:24],
                    eps=eps,
                    norm="linf",
                    clip_bounds=(0.0, 1.0),
                    seed=42,
                )
                self.assertTrue(audit.passed, msg=f"{defense_name} triggered Carlini 2.0 red flags: {audit.red_flags}")

    def test_pgd_at_complex64_wireless_jsr_db(self) -> None:
        torch.manual_seed(31)
        centers_r = torch.zeros(4, 16)
        centers_i = torch.zeros(4, 16)
        for c in range(4):
            centers_r[c, c * 4 : (c + 1) * 4] = 1.0
            centers_i[c, c * 4 : (c + 1) * 4] = -1.0
        y_rf = torch.arange(4).repeat(24)
        iq_tr = torch.complex(
            centers_r[y_rf] + 0.15 * torch.randn(96, 16),
            centers_i[y_rf] + 0.15 * torch.randn(96, 16),
        ).to(torch.complex64)

        rx = _ToyComplexRFNet(seq_len=16, num_classes=4)
        res = pgd_adversarial_train(
            rx,
            iq_tr,
            y_rf,
            eps=-6.0,
            norm="jsr_db",
            attack_steps=4,
            epochs=10,
            batch_size=32,
            lr=0.02,
            clip_bounds=None,
            seed=31,
        )
        self.assertEqual(res.defense_id, "pgd_at")
        rx.eval()
        acc = (rx(iq_tr).argmax(dim=1) == y_rf).float().mean().item()
        self.assertGreater(acc, 0.90)


class TestDefenseDiscoveryCitationsEjectionAndDocs(unittest.TestCase):
    def test_list_defenses_citations_eject_and_cli_arsenal(self) -> None:
        defs = list_defenses()
        self.assertEqual({d["id"] for d in defs}, {"pgd_at", "trades", "mart", "randomized_smoothing"})

        bib = export_citations(["pgd_at", "trades", "mart", "randomized_smoothing"], fmt="bibtex")
        for key in ("madry2018towards", "zhang2019theoretically", "wang2020improving", "cohen2019certified"):
            self.assertIn(key, bib)

        # Function-name aliases also resolve cleanly
        ieee = export_citations(["trades_train", "certify_randomized_smoothing"], fmt="ieee")
        self.assertIn("Zhang", ieee)
        self.assertIn("Cohen", ieee)

        with tempfile.TemporaryDirectory() as tmp:
            p1 = eject_defense("trades", dest_dir=Path(tmp) / "defenses")
            p2 = eject_attack("randomized_smoothing", dest_dir=Path(tmp) / "defenses")
            self.assertTrue(p1.exists())
            self.assertTrue(p2.exists())
            self.assertIn("zhang2019theoretically", p1.read_text(encoding="utf-8"))
            self.assertIn("cohen2019certified", p2.read_text(encoding="utf-8"))

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["arsenal", "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["total_attacks"], 35)
        self.assertEqual(payload["total_defenses"], 4)
        self.assertEqual(payload["version"], __version__)

        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            rc2 = cli_main(["cite", "trades", "mart", "randomized_smoothing"])
        self.assertEqual(rc2, 0)
        self.assertIn("cohen2019certified", buf2.getvalue())

    def test_lazy_import_of_fgsm_does_not_load_defenses(self) -> None:
        code = (
            "import sys; "
            "from wannado.adv import fgsm; "
            "assert 'wannado.adv.defenses' not in sys.modules; "
            "assert 'wannado.adv.attacks.first_order.pgd_madry2018' not in sys.modules"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_cold_imports_have_no_circular_errors_and_all_exports_resolve(self) -> None:
        # Regression: importing reporter before audit previously triggered a circular import
        # (paper_reporter -> audit.carlini_auditor -> audit.__init__ -> red_team -> paper_reporter).
        for stmt in (
            "from wannado.adv import evaluate_suite, export_citations, EvaluationReport, AttackRunRecord",
            "import wannado.adv.reporter; import wannado.adv.audit",
            "from wannado.adv.audit import auto_red_team, verify_robustness_gate",
            "import wannado.adv as w; missing = [n for n in w.__all__ if not hasattr(w, n)]; assert not missing, missing",
            "import wannado.adv.attacks as a; missing = [n for n in a.__all__ if not hasattr(a, n)]; assert not missing, missing",
        ):
            with self.subTest(stmt=stmt):
                proc = subprocess.run(
                    [sys.executable, "-c", stmt],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, msg=f"{stmt}\n{proc.stderr}")

    def test_streamlined_agent_kit_and_version_sync(self) -> None:
        self.assertEqual(__version__, "1.2.1")
        skill_text = (REPO_ROOT / "src" / "wannado" / "adv" / "agent_kit" / "SKILL.md").read_text(encoding="utf-8")
        agents_text = (
            REPO_ROOT / "src" / "wannado" / "adv" / "agent_kit" / "templates" / "AGENTS.md"
        ).read_text(encoding="utf-8")

        for token in ("wannado-adv verify", "auto_red_team", "trades_train", "certify_randomized_smoothing"):
            self.assertIn(token, skill_text)
            self.assertIn(token, agents_text)
        self.assertNotIn("MANDATORY Zero-Jargon Interactive Clarification", skill_text)


if __name__ == "__main__":
    unittest.main()
