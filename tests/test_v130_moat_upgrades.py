"""End-to-end verification tests for wannado-adv v1.3.0 structural moat upgrades (Gaps 1-5).

Covers:
  1. Conformity Dossier (NIST AI 100-2 / MITRE ATLAS / EU AI Act Art. 15), SHA-256/HMAC integrity,
     tamper detection, PR baseline regression gate, and SARIF 2.1.0 export.
  2. Supply-chain Backdoor / Trojan auditor (Neural Cleanse MAD + STRIP blending entropy) on clean
     vs genuinely BadNets-poisoned ConvNet.
  3. Privacy / Confidentiality auditor (Loss & Mentr MIA, Carlini TPR@1%FPR, Black-Box Extraction)
     on well-generalized vs memorizing models.
  4. Production Runtime & Non-Classifier Adapters: TorchScript, ONNX duck-typed session, FP32 vs
     INT8 Quantization-Shift differential audit, Object Detection vanishing, Semantic Segmentation
     mIoU, VLM embedding drift/hijacking, Tabular domain constraint projection & validation, Tree
     ensemble adapter, and Physical RF spectral mask / PAPR / OTA Rayleigh EoT.
  5. Production-scale low-cost hardening: Fast-AT (Wong 2020 + CO guard) and LoRA-AWP (zero-dep
     LoRALinear/LoRAConv2d + TRADES + AWP + lossless weight merging) + CLI integration.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from wannado.adv import (
    DetectionAttackAdapter,
    OnnxModelAdapter,
    SegmentationAttackAdapter,
    TabularConstraintProjector,
    VLMEmbeddingAdapter,
    __version__,
    audit_backdoor,
    audit_privacy,
    audit_quantization_shift,
    compare_verdicts,
    compute_segmentation_metrics,
    detection_vanishing_loss,
    eject_defense,
    embedding_cosine_loss,
    eot_pgd,
    evaluate_detection_robustness,
    export_citations,
    export_sarif,
    fast_adversarial_train,
    inject_lora_adapters,
    lora_adversarial_train,
    make_simulated_quantized_model,
    measure_out_of_band_ratio,
    measure_papr_db,
    merge_lora_weights,
    over_the_air_eot_transform,
    pgd,
    project_papr_constraint,
    project_spectral_mask,
    square_attack,
    verify_dossier_integrity,
    verify_robustness_gate,
    wrap_torchscript_model,
    wrap_tree_ensemble,
)
from wannado.adv.cli import main as cli_main


def _make_separable_tabular_data(n: int = 128, dim: int = 8, num_classes: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn((num_classes, dim), generator=g) * 1.5
    y = torch.arange(n) % num_classes
    x = torch.clamp(0.5 + 0.15 * centers[y] + 0.05 * torch.randn((n, dim), generator=g), 0.0, 1.0)
    return x, y


def _train_quick_mlp(x: torch.Tensor, y: torch.Tensor, *, hidden: int = 32, epochs: int = 25, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    num_classes = int(y.max().item()) + 1
    model = nn.Sequential(
        nn.Linear(x.shape[1], hidden),
        nn.ReLU(),
        nn.Linear(hidden, num_classes),
    )
    opt = torch.optim.Adam(model.parameters(), lr=2e-2)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    model.eval()
    return model


def _make_tiny_images(n: int = 96, num_classes: int = 4, seed: int = 0):
    """4-class 1x8x8 images distinguished by quadrant activation (cleanly separable for CNNs)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % num_classes
    x = 0.12 * torch.rand((n, 1, 8, 8), generator=g)
    quads = [(0, 4, 0, 4), (0, 4, 4, 8), (4, 8, 0, 4), (4, 8, 4, 8)]
    for i in range(n):
        r0, r1, c0, c1 = quads[int(y[i].item()) % 4]
        x[i, 0, r0:r1, c0:c1] += 0.65
    return x.clamp(0.0, 1.0), y


class _TinyConvNet(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.head = nn.Linear(8 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv(x))
        h = F.avg_pool2d(h, 2)
        return self.head(h.flatten(1))


class TestV130MoatUpgrades(unittest.TestCase):
    def test_version_and_citations(self) -> None:
        self.assertEqual(__version__, "1.3.0")
        bib = export_citations(
            ["fast_at", "lora_awp", "neural_cleanse", "strip", "mia", "mentr", "lira", "extraction"],
            fmt="bibtex",
            include_audit_ref=True,
        )
        for key in (
            "wong2020fast",
            "wu2020adversarial",
            "wang2019neural",
            "gao2019strip",
            "yeom2018privacy",
            "song2021systematic",
            "carlini2022membership",
            "tramer2016stealing",
            "carlini2019evaluating",
        ):
            self.assertIn(key, bib)

    # ------------------------------------------------------------------
    # Gap 1: Conformity Dossier, Tamper-Evident Integrity, PR Regression & SARIF
    # ------------------------------------------------------------------
    def test_conformity_dossier_signing_tamper_sarif_and_pr_regression(self) -> None:
        x, y = _make_separable_tabular_data(n=48, dim=8, num_classes=3, seed=7)
        model = _train_quick_mlp(x, y, epochs=30, seed=7)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run1"
            rep = verify_robustness_gate(
                model_or_fn=model,
                x=x,
                y=y,
                eps=0.05,
                norm="linf",
                clip_bounds=(0.0, 1.0),
                goal="sanity",
                samples=32,
                sweep_curve=False,
                out_dir=out_dir,
                signing_key="test-hmac-secret",
            )
            self.assertIsNotNone(rep.dossier)
            assert rep.dossier is not None
            self.assertTrue((out_dir / "conformity_dossier.json").is_file())
            self.assertTrue((out_dir / "conformity_dossier.md").is_file())
            self.assertTrue((out_dir / "verification_report.sarif").is_file())

            # Cryptographic integrity passes on sealed dossier
            ok, errs = verify_dossier_integrity(rep.dossier, signing_key="test-hmac-secret")
            self.assertTrue(ok, errs)

            # Wrong key fails HMAC check
            ok_bad_key, errs_bad_key = verify_dossier_integrity(rep.dossier, signing_key="wrong-key")
            self.assertFalse(ok_bad_key)
            self.assertTrue(any("HMAC" in e for e in errs_bad_key))

            # Tampering a single field invalidates SHA-256 + HMAC
            tampered = json.loads(json.dumps(rep.dossier))
            tampered["overall_passed"] = not bool(tampered["overall_passed"])
            ok_tamp, errs_tamp = verify_dossier_integrity(tampered, signing_key="test-hmac-secret")
            self.assertFalse(ok_tamp)
            self.assertGreaterEqual(len(errs_tamp), 2)

            # Validate SARIF 2.1.0 structure
            sarif = json.loads((out_dir / "verification_report.sarif").read_text(encoding="utf-8"))
            self.assertEqual(sarif["version"], "2.1.0")
            self.assertEqual(sarif["runs"][0]["tool"]["driver"]["name"], "wannado-adv")

            # Baseline PR regression check blocks drop > max_robust_regression
            fake_strong_baseline = Path(tmp) / "strong_baseline.json"
            fake_strong_baseline.write_text(
                json.dumps({"summary": {"clean_acc": 1.0, "worst_case_robust_acc": 0.99, "aurc": 0.95}}),
                encoding="utf-8",
            )
            rep_reg = verify_robustness_gate(
                model_or_fn=model,
                x=x,
                y=y,
                eps=0.30,
                norm="linf",
                clip_bounds=(0.0, 1.0),
                goal="sanity",
                samples=24,
                sweep_curve=False,
                baseline_verdict_path=fake_strong_baseline,
                max_robust_regression=0.02,
            )
            self.assertFalse(rep_reg.passed)
            self.assertEqual(rep_reg.exit_code, 1)
            self.assertFalse(rep_reg.checks["no_robustness_regression_ok"])
            self.assertTrue(any(rf.startswith("regression:") for rf in rep_reg.red_flags))

            sarif_reg = export_sarif(rep_reg)
            self.assertGreater(len(sarif_reg["runs"][0]["results"]), 0)
            self.assertIn("NIST=", sarif_reg["runs"][0]["results"][0]["message"]["text"])

            diff = compare_verdicts(
                {"summary": {"worst_case_robust_acc": 0.60, "clean_acc": 0.90}},
                {"summary": {"worst_case_robust_acc": 0.80, "clean_acc": 0.90}},
                max_robust_regression=0.05,
            )
            self.assertFalse(diff["passed"])
            self.assertAlmostEqual(diff["deltas"]["worst_case_robust_acc_delta"], -0.20, places=4)

    # ------------------------------------------------------------------
    # Gap 2: Supply-Chain Backdoor/Trojan Audit & Privacy/MIA Audit
    # ------------------------------------------------------------------
    def test_backdoor_auditor_distinguishes_clean_vs_badnets_trojan(self) -> None:
        x_clean, y_clean = _make_tiny_images(n=96, num_classes=4, seed=11)

        # 1. Clean model should pass
        torch.manual_seed(11)
        clean_net = _TinyConvNet(num_classes=4)
        opt_c = torch.optim.Adam(clean_net.parameters(), lr=2e-2)
        for _ in range(30):
            opt_c.zero_grad()
            F.cross_entropy(clean_net(x_clean), y_clean).backward()
            opt_c.step()
        clean_net.eval()

        rep_clean = audit_backdoor(
            clean_net,
            x_clean,
            y_clean,
            max_samples=48,
            steps=40,
            clip_bounds=(0.0, 1.0),
            seed=11,
        )
        self.assertTrue(rep_clean.passed, rep_clean.format_report())
        self.assertFalse(rep_clean.suspected_backdoor)

        # 2. BadNets-poisoned model: tiny 2x2 center trigger patch forces target class 0
        x_poison = x_clean.clone()
        x_poison[:, 0, 3:5, 3:5] = 1.0
        y_poison = torch.zeros_like(y_clean)
        x_mix = torch.cat([x_clean, x_poison[:48]], dim=0)
        y_mix = torch.cat([y_clean, y_poison[:48]], dim=0)

        torch.manual_seed(12)
        trojan_net = _TinyConvNet(num_classes=4)
        opt_t = torch.optim.Adam(trojan_net.parameters(), lr=2e-2)
        for _ in range(60):
            opt_t.zero_grad()
            F.cross_entropy(trojan_net(x_mix), y_mix).backward()
            opt_t.step()
        trojan_net.eval()

        # Verify BadNets backdoor actually works on non-class-0 images
        non0 = y_clean != 0
        with torch.no_grad():
            poison_asr = float((trojan_net(x_poison[non0]).argmax(dim=1) == 0).float().mean().item())
        self.assertGreater(poison_asr, 0.90)

        rep_trojan = audit_backdoor(
            trojan_net,
            x_clean,
            y_clean,
            x_suspect=x_poison[non0],
            max_samples=48,
            steps=60,
            clip_bounds=(0.0, 1.0),
            seed=12,
        )
        self.assertFalse(rep_trojan.passed, rep_trojan.format_report())
        self.assertTrue(rep_trojan.suspected_backdoor)
        self.assertIsNotNone(rep_trojan.strip_auroc)

    def test_privacy_auditor_catches_overfit_memorizer(self) -> None:
        g = torch.Generator().manual_seed(21)
        x_mem = F.normalize(torch.randn((100, 24), generator=g), dim=1)
        y_mem = torch.randint(0, 10, (100,), generator=g)
        x_non = F.normalize(torch.randn((100, 24), generator=g), dim=1)
        y_non = torch.randint(0, 10, (100,), generator=g)

        # Intentionally memorize random labels with an overparameterized MLP
        torch.manual_seed(21)
        overfit = nn.Sequential(nn.Linear(24, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))
        opt = torch.optim.Adam(overfit.parameters(), lr=1e-2)
        for _ in range(120):
            opt.zero_grad()
            F.cross_entropy(overfit(x_mem), y_mem).backward()
            opt.step()
        overfit.eval()

        rep_mem = audit_privacy(
            overfit,
            x_mem,
            x_non,
            member_y=y_mem,
            non_member_y=y_non,
            run_extraction_probe=True,
            extraction_queries=48,
            seed=21,
        )
        self.assertFalse(rep_mem.passed, rep_mem.format_report())
        self.assertGreater(rep_mem.mia_loss_auc, 0.90)
        self.assertGreater(rep_mem.mia_mentr_auc, 0.90)
        self.assertGreater(rep_mem.mia_tpr_at_1pct_fpr, 0.50)
        self.assertEqual(rep_mem.risk_level, "HIGH")
        self.assertIsNotNone(rep_mem.extraction_agreement_rate)

        # Well-generalized linear model on clean IID separable distribution -> low MIA AUC
        x_all, y_all = _make_separable_tabular_data(n=160, dim=6, num_classes=3, seed=31)
        x_tr, y_tr = x_all[:80], y_all[:80]
        x_te, y_te = x_all[80:], y_all[80:]
        torch.manual_seed(31)
        lin = nn.Linear(6, 3)
        opt_l = torch.optim.SGD(lin.parameters(), lr=0.1, weight_decay=0.05)
        for _ in range(30):
            opt_l.zero_grad()
            F.cross_entropy(lin(x_tr), y_tr).backward()
            opt_l.step()
        lin.eval()

        rep_clean = audit_privacy(
            lin,
            x_tr,
            x_te,
            member_y=y_tr,
            non_member_y=y_te,
            run_extraction_probe=False,
            seed=31,
        )
        self.assertLess(rep_clean.mia_loss_auc, 0.65)
        self.assertTrue(rep_clean.passed, rep_clean.format_report())

    # ------------------------------------------------------------------
    # Gap 3: Deployment (TorchScript / ONNX / Quantization-Shift), Detection,
    # Segmentation, VLM, Tabular Constraints, Tree Ensemble & Physical RF
    # ------------------------------------------------------------------
    def test_torchscript_onnx_and_quantization_shift_audit(self) -> None:
        x, y = _make_separable_tabular_data(n=64, dim=8, num_classes=3, seed=41)
        model = _train_quick_mlp(x, y, epochs=30, seed=41)

        # TorchScript wrapper + file round-trip
        traced = torch.jit.trace(model, x[:4])
        with tempfile.TemporaryDirectory() as tmp:
            ts_path = Path(tmp) / "model.pts"
            traced.save(str(ts_path))
            ts_adapter = wrap_torchscript_model(ts_path)
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(x[:8]), ts_adapter(x[:8]), atol=1e-5))

        # Duck-typed ONNX Runtime session adapter (no onnxruntime install required)
        class _FakeOrtInput:
            name = "input"

        class _FakeOrtSession:
            def get_inputs(self) -> list[Any]:
                return [_FakeOrtInput()]

            def run(self, _out_names: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
                with torch.no_grad():
                    out = model(torch.from_numpy(feed["input"])).numpy()
                return [out]

        onnx_adapter = OnnxModelAdapter(_FakeOrtSession())
        self.assertTrue(torch.allclose(model(x[:8]), onnx_adapter(x[:8]), atol=1e-5))

        # Quantization-shift differential audit: INT8 clean vs severely degraded model
        q8_model = make_simulated_quantized_model(model, bits=8, mode="weight_int")
        rep_q8 = audit_quantization_shift(
            model,
            q8_model,
            x,
            y,
            eps=0.05,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            attack_steps=8,
            square_queries=40,
            max_samples=32,
            seed=41,
        )
        self.assertEqual(rep_q8.num_samples, 32)
        self.assertLess(abs(rep_q8.clean_drop_pp), 0.15)

        # Deliberately corrupted quantized deployment triggers quant:clean_accuracy_drop / robustness_collapse
        def _broken_quant(inp: torch.Tensor) -> torch.Tensor:
            return torch.zeros((inp.shape[0], 3), device=inp.device, dtype=inp.dtype)

        rep_bad_q = audit_quantization_shift(
            model,
            _broken_quant,
            x,
            y,
            eps=0.05,
            norm="linf",
            clip_bounds=(0.0, 1.0),
            attack_steps=6,
            square_queries=20,
            max_samples=24,
            max_clean_drop=0.05,
            max_robust_drop=0.05,
        )
        self.assertFalse(rep_bad_q.passed)
        self.assertTrue(any(rf.startswith("quant:") for rf in rep_bad_q.red_flags))

    def test_detection_segmentation_and_vlm_adapters(self) -> None:
        torch.manual_seed(51)
        x_img = torch.rand(4, 3, 8, 8)

        # 1. YOLO-style object detector (B, N=6, 5+C=8)
        class _TinyYOLO(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 6 * 8))

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                out = self.net(inp).view(inp.shape[0], 6, 8)
                # Bias objectness/class logits positive so clean detections exist
                return torch.cat([out[:, :, :4], torch.sigmoid(out[:, :, 4:] + 1.5)], dim=-1)

        det_adapter = DetectionAttackAdapter(_TinyYOLO())
        dummy_y = torch.zeros(4, dtype=torch.long)
        res_det = pgd(
            det_adapter,
            x_img,
            dummy_y,
            eps=0.15,
            norm="linf",
            steps=12,
            loss=detection_vanishing_loss,
            clip_bounds=(0.0, 1.0),
            seed=51,
        )
        det_metrics = evaluate_detection_robustness(det_adapter, x_img, res_det.x_adv, score_thresh=0.4)
        self.assertGreater(det_metrics["mean_score_drop"], 0.05)

        # 2. Semantic segmentation model returning torchvision dict {"out": (B, C, H, W)}
        class _TinySeg(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

            def forward(self, inp: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"out": self.conv(inp)}

        seg_adapter = SegmentationAttackAdapter(_TinySeg())
        with torch.no_grad():
            clean_seg_logits = seg_adapter(x_img)
            y_mask = clean_seg_logits.argmax(dim=1)
        clean_seg_m = compute_segmentation_metrics(clean_seg_logits, y_mask, num_classes=3)
        self.assertAlmostEqual(clean_seg_m["pixel_acc"], 1.0, places=4)
        res_seg = pgd(seg_adapter, x_img, y_mask, eps=0.15, norm="linf", steps=10, clip_bounds=(0.0, 1.0), seed=51)
        with torch.no_grad():
            adv_seg_m = compute_segmentation_metrics(seg_adapter(res_seg.x_adv), y_mask, num_classes=3)
        self.assertLess(adv_seg_m["miou"], clean_seg_m["miou"])

        # 3. Multimodal VLM encoder (zero-shot anchor logits + raw embedding drift)
        class _TinyEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(3 * 8 * 8, 16)

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                return self.fc(inp.flatten(1))

        enc = _TinyEncoder()
        anchors = torch.randn(5, 16)
        vlm_zs = VLMEmbeddingAdapter(enc, anchor_embeddings=anchors, temperature=10.0)
        with torch.no_grad():
            y_zs = vlm_zs(x_img).argmax(dim=1)
        res_zs = pgd(vlm_zs, x_img, y_zs, eps=0.15, norm="linf", steps=10, clip_bounds=(0.0, 1.0), seed=51)
        with torch.no_grad():
            adv_zs_acc = float((vlm_zs(res_zs.x_adv).argmax(dim=1) == y_zs).float().mean().item())
        self.assertLess(adv_zs_acc, 1.0)

        vlm_raw = VLMEmbeddingAdapter(enc, anchor_embeddings=None)
        with torch.no_grad():
            z_clean = vlm_raw(x_img)
        res_emb = pgd(
            vlm_raw,
            x_img,
            z_clean,
            eps=0.15,
            norm="linf",
            steps=10,
            loss=embedding_cosine_loss,
            clip_bounds=(0.0, 1.0),
            seed=51,
        )
        with torch.no_grad():
            cos_after = float((vlm_raw(res_emb.x_adv) * z_clean).sum(dim=-1).mean().item())
        self.assertLess(cos_after, 0.85)

    def test_tabular_constraints_tree_adapter_and_rf_physical_primitives(self) -> None:
        # 1. Tabular domain constraint projector
        x_clean = torch.tensor(
            [
                [35.0, 3.0, 1.0, 0.0, 0.0, 2.0, 5.0],
                [42.0, 7.0, 0.0, 1.0, 0.0, 4.0, 8.0],
            ],
            dtype=torch.float32,
        )
        x_noisy = x_clean + torch.tensor(
            [
                [5.5, 0.4, -0.3, 0.9, 0.2, 5.0, -3.0],
                [-4.0, 1.7, 0.8, -0.2, 0.1, 6.0, -2.0],
            ],
            dtype=torch.float32,
        )
        proj = TabularConstraintProjector(
            clip_bounds=(0.0, 100.0),
            immutable_indices=[0],
            integer_indices=[1],
            categorical_groups=[[2, 3, 4]],
            inequality_pairs=[(5, 6)],
        )
        x_fixed = proj.project(x_clean, x_noisy)
        val = proj.validate(x_clean, x_fixed)
        self.assertTrue(val["all_valid"], val)
        self.assertTrue(torch.allclose(x_fixed[:, 0], x_clean[:, 0]))

        # 2. TreeEnsembleAdapter with duck-typed tree model under Square Attack
        class _FakeTreeClassifier:
            def predict_proba(self, arr: np.ndarray) -> np.ndarray:
                score = 1.0 / (1.0 + np.exp(-2.0 * (arr[:, 0] - arr[:, 1])))
                return np.stack([1.0 - score, score], axis=1)

        tree_victim = wrap_tree_ensemble(_FakeTreeClassifier(), projector=proj)
        x_t = torch.rand(8, 4)
        y_t = (x_t[:, 0] > x_t[:, 1]).long()
        res_sq = square_attack(tree_victim, x_t, y_t, eps=0.4, norm="linf", max_queries=60, clip_bounds=(0.0, 1.0), seed=61)
        self.assertEqual(res_sq.x_adv.shape, x_t.shape)

        # 3. Physical RF spectral mask, PAPR projection, and OTA EoT transform
        g = torch.Generator().manual_seed(62)
        x_iq = torch.complex(torch.randn(4, 64, generator=g), torch.randn(4, 64, generator=g))
        delta_iq = torch.complex(torch.randn(4, 64, generator=g), torch.randn(4, 64, generator=g))
        # Passband = center half of FFT bins
        mask = torch.zeros(64)
        mask[:16] = 1.0
        mask[-16:] = 1.0
        delta_spec = project_spectral_mask(delta_iq, mask, x_clean=x_iq, eps=-10.0, norm="jsr_db")
        self.assertLess(measure_out_of_band_ratio(delta_spec, mask), 1e-6)

        # High-PAPR spike reduced below 4.0 dB
        spike_iq = torch.complex(torch.ones(2, 64) * 0.05, torch.zeros(2, 64))
        spike_iq[:, 0] = 5.0 + 0.0j
        self.assertGreater(float(measure_papr_db(spike_iq).min().item()), 10.0)
        papr_proj = project_papr_constraint(spike_iq, max_papr_db=4.0, iters=10)
        self.assertLessEqual(float(measure_papr_db(papr_proj).max().item()), 4.15)

        # OTA Rayleigh + phase + AWGN EoT transform keeps complex autograd intact inside eot_pgd
        class _ComplexRx(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(128, 3)

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                return self.fc(torch.cat([z.real, z.imag], dim=-1))

        rx = _ComplexRx()
        y_rf = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        res_ota = eot_pgd(
            rx,
            x_iq,
            y_rf,
            eps=-10.0,
            norm="jsr_db",
            steps=4,
            eot_samples=3,
            transform=lambda z: over_the_air_eot_transform(z, snr_db=15.0, max_phase_rad=math.pi, rayleigh_fading=True),
            seed=62,
        )
        self.assertTrue(res_ota.x_adv.is_complex())

    # ------------------------------------------------------------------
    # Gap 4: Fast-AT (CO Guard) & LoRA-AWP (Zero-Overhead Weight Merge)
    # ------------------------------------------------------------------
    def test_fast_at_and_lora_awp_hardening_and_eject(self) -> None:
        x_tr, y_tr = _make_tiny_images(n=64, num_classes=4, seed=71)

        # 1. Fast-AT + CO guard
        torch.manual_seed(71)
        m_fast = _TinyConvNet(num_classes=4)
        res_fast = fast_adversarial_train(
            m_fast,
            x_tr,
            y_tr,
            eps=0.10,
            norm="linf",
            epochs=3,
            batch_size=32,
            lr=0.05,
            clip_bounds=(0.0, 1.0),
            co_guard=True,
            co_pgd_steps=3,
            seed=71,
        )
        self.assertEqual(res_fast.defense_id, "fast_at")
        self.assertEqual(len(res_fast.history), 3)
        self.assertIn("probe_pgd_acc", res_fast.history[-1])

        # 2. LoRA inject + lossless merge equivalence check
        torch.manual_seed(72)
        m_merge = _TinyConvNet(num_classes=4)
        _, lora_params = inject_lora_adapters(m_merge, rank=2, alpha=4.0, freeze_base=True)
        # Perturb lora_b away from zero so delta_w is non-trivial
        with torch.no_grad():
            for p in lora_params:
                p.add_(0.05 * torch.randn_like(p))
            out_before = m_merge(x_tr[:8]).clone()
        merge_lora_weights(m_merge, unfreeze_base=True)
        with torch.no_grad():
            out_after = m_merge(x_tr[:8])
        self.assertTrue(torch.allclose(out_before, out_after, atol=1e-5))
        # No LoRA wrappers remain after merge
        self.assertTrue(all(type(m).__name__ not in ("LoRALinear", "LoRAConv2d") for m in m_merge.modules()))

        # 3. Full LoRA-AWP training loop
        torch.manual_seed(73)
        m_lora = _TinyConvNet(num_classes=4)
        res_lora = lora_adversarial_train(
            m_lora,
            x_tr,
            y_tr,
            rank=2,
            lora_alpha=4.0,
            eps=0.10,
            norm="linf",
            attack_steps=3,
            awp_gamma=0.005,
            epochs=3,
            batch_size=32,
            lr=1e-2,
            clip_bounds=(0.0, 1.0),
            merge_after_train=True,
            seed=73,
        )
        self.assertEqual(res_lora.defense_id, "lora_awp")
        self.assertTrue(res_lora.extra["merged_to_base"])
        self.assertGreater(res_lora.extra["trainable_lora_params"], 0)

        # 4. Eject both new single-file defenses
        with tempfile.TemporaryDirectory() as tmp:
            p1 = eject_defense("fast_at", dest_dir=tmp)
            p2 = eject_defense("lora_awp", dest_dir=tmp)
            self.assertTrue(p1.is_file())
            self.assertTrue(p2.is_file())

    # ------------------------------------------------------------------
    # Gap 5: CLI verify multi-pillar flags & arsenal inventory
    # ------------------------------------------------------------------
    def test_cli_verify_conformity_and_arsenal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            iface = tmp_p / "interface.py"
            iface_src = (
                "import torch\n"
                "from torch import nn\n"
                "from wannado.adv import VictimInterface\n\n"
                "def load_victim_interface(device='cpu', num_samples=24):\n"
                "    torch.manual_seed(99)\n"
                "    g = torch.Generator().manual_seed(99)\n"
                "    y = torch.arange(num_samples) % 3\n"
                "    centers = torch.eye(3, 6) * 0.6 + 0.2\n"
                "    x = torch.clamp(centers[y] + 0.04 * torch.randn((num_samples, 6), generator=g), 0.0, 1.0)\n"
                "    m = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 3)).to(device)\n"
                "    opt = torch.optim.Adam(m.parameters(), lr=3e-2)\n"
                "    for _ in range(30):\n"
                "        opt.zero_grad()\n"
                "        nn.functional.cross_entropy(m(x.to(device)), y.to(device)).backward()\n"
                "        opt.step()\n"
                "    m.eval()\n"
                "    return VictimInterface(model_or_fn=m, x_eval=x, y_eval=y, eps=0.05, norm='linf', clip_bounds=(0.0, 1.0), domain='tabular')\n"
            )
            iface.write_text(iface_src, encoding="utf-8")
            out_dir = tmp_p / "outputs"
            sarif_p = tmp_p / "ci.sarif"
            rc = cli_main(
                [
                    "verify",
                    "--interface",
                    str(iface),
                    "--goal",
                    "sanity",
                    "--samples",
                    "24",
                    "--no-curve",
                    "--check-backdoor",
                    "--check-quant",
                    "--sarif",
                    str(sarif_p),
                    "--signing-key",
                    "cli-key",
                    "--out-dir",
                    str(out_dir),
                    "--json",
                ]
            )
            self.assertIn(rc, (0, 1))
            self.assertTrue(sarif_p.is_file())
            self.assertTrue((out_dir / "conformity_dossier.json").is_file())
            dossier_data = json.loads((out_dir / "conformity_dossier.json").read_text(encoding="utf-8"))
            ok, errs = verify_dossier_integrity(dossier_data, signing_key="cli-key")
            self.assertTrue(ok, errs)
            self.assertIn("poisoning_and_backdoor", dossier_data["pillar_coverage"])
            self.assertNotEqual(dossier_data["pillar_coverage"]["poisoning_and_backdoor"]["status"], "NOT_RUN")
            self.assertNotEqual(dossier_data["pillar_coverage"]["deployment_quantization_shift"]["status"], "NOT_RUN")


if __name__ == "__main__":
    unittest.main()
