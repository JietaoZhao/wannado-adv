"""AI-native SOTA navigation, novice-friendly parameter explanation, and runtime estimation."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from wannado.adv.catalog.build_catalog import load_catalog, query_catalog
from wannado.adv.copilot.project_inspector import ModelContextProbe, ProjectScan
from wannado.adv.core.utils import eval_mode, infer_model_device, materialize_eval_tensors


@dataclass
class RecommendedSuite:
    """Recommended attack roster calibrated to domain, threat model, and evaluation goal."""

    goal: str
    domain: str
    norm: str
    threat_model: str
    attacks: list[str]
    attack_kwargs: dict[str, dict[str, Any]]
    rationale: str
    lineage_summary: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        """Convert recommendation to JSON-serializable dict."""
        return asdict(self)


@dataclass
class RuntimeEstimate:
    """Hardware-calibrated runtime forecast based on live micro-benchmarking."""

    device: str
    num_samples: int
    forward_ms_per_sample: float
    fwd_bwd_ms_per_sample: float
    autograd_available: bool
    per_attack_seconds: dict[str, float]
    total_seconds: float
    advisory: str
    audit_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize runtime forecast."""
        return asdict(self)

    def format_summary(self) -> str:
        """Format readable ETA summary for user confirmation before running."""
        fb_str = (
            f"{self.fwd_bwd_ms_per_sample:.2f} ms/sample"
            if self.autograd_available
            else "N/A (autograd disconnected)"
        )
        lines = [
            f"[Runtime Forecast] device={self.device}, N={self.num_samples} samples, fwd={self.forward_ms_per_sample:.2f} ms/sample, fwd+bwd={fb_str}",
        ]
        for atk, sec in self.per_attack_seconds.items():
            lines.append(f"  - {atk:<14s}: ~{sec:.1f}s")
        if self.audit_seconds > 0.0:
            lines.append(f"  - {'carlini_audit':<14s}: ~{self.audit_seconds:.1f}s")
        lines.append(f"  Total Estimated Time: ~{self.total_seconds:.1f}s ({self.total_seconds / 60.0:.2f} min)")
        if self.advisory:
            lines.append(f"  Advisory: {self.advisory}")
        return "\n".join(lines)


def get_sota_lineage(
    *,
    family: str | None = None,
    domain: str | None = None,
    threat_model: str | None = None,
) -> list[dict[str, Any]]:
    """Trace evolution chains from classic baselines to SOTA frontier attacks."""
    cards = query_catalog(family=family, domain=domain, threat_model=threat_model)
    by_id = {c["id"]: c for c in load_catalog()}
    children_map: dict[str, list[str]] = {}
    for c in cards:
        pid = c.get("parent_id")
        if pid:
            children_map.setdefault(pid, []).append(c["id"])

    lineage_entries: list[dict[str, Any]] = []
    for c in cards:
        chain: list[str] = [c["id"]]
        cur = c
        seen = {c["id"]}
        while cur.get("parent_id") and cur["parent_id"] not in seen:
            pid = cur["parent_id"]
            seen.add(pid)
            chain.append(pid)
            cur = by_id.get(pid, {})
            if not cur:
                break
        chain.reverse()
        lineage_entries.append(
            {
                "id": c["id"],
                "name": c["name"],
                "family": c["family"],
                "tier": c["tier"],
                "year": c["citation"]["year"],
                "evolution_path": " -> ".join(chain),
                "diff_from_parent": c["diff_from_parent"],
                "successors_in_scope": children_map.get(c["id"], []),
                "extension_hooks": c["extension_hooks"],
                "module_path": c["module_path"],
            }
        )
    lineage_entries.sort(key=lambda item: (item["year"], item["tier"], item["id"]))
    return lineage_entries


def recommend_suite(
    domain: str = "vision",
    goal: str = "standard",
    threat_model: str = "white_box",
    autograd_ok: bool = True,
    norm: str | None = None,
) -> RecommendedSuite:
    """Select calibrated baseline + SOTA attack suite for the target domain and goal.

    Args:
        domain: Target modality (``"vision"``, ``"wireless"``, ``"drl"``, ``"tabular"``, etc.).
        goal: Evaluation rigor: ``"sanity"`` (seconds), ``"standard"`` (~1 min), or ``"paper_sota"``.
        threat_model: Adversary model: ``"white_box"``, ``"black_box_score"``, or ``"black_box_decision"``.
        autograd_ok: Set ``False`` if input gradients are disconnected or broken by non-differentiable preprocessing.
        norm: Optional explicit norm override; otherwise chosen cleanly from domain.
    """
    if goal not in ("sanity", "standard", "paper_sota"):
        raise ValueError(f"goal must be 'sanity', 'standard', or 'paper_sota', got '{goal}'.")

    if norm is None:
        norm = "jsr_db" if domain == "wireless" else ("linf" if domain == "vision" else "relative_l2")

    kwargs: dict[str, dict[str, Any]] = {}
    if not autograd_ok or threat_model == "black_box_score":
        effective_tm = "black_box_score"
        if goal == "sanity":
            attacks = ["square_attack"]
            kwargs["square_attack"] = {"max_queries": 150}
        elif goal == "standard":
            attacks = ["square_attack", "nes"]
            kwargs["square_attack"] = {"max_queries": 400}
            kwargs["nes"] = {"steps": 25, "nb_sample": 32}
        else:
            attacks = ["square_attack", "nes", "spsa"]
            kwargs["square_attack"] = {"max_queries": 1000}
            kwargs["nes"] = {"steps": 40, "nb_sample": 48}
            kwargs["spsa"] = {"steps": 40, "nb_sample": 64}
        rationale = (
            "Model gradients are blocked or score-only black-box evaluation was requested; "
            "using gradient-free query attacks (Square Attack + NES/SPSA) so masked/quantized defenses cannot hide."
        )
    elif threat_model == "black_box_decision":
        effective_tm = "black_box_decision"
        if goal == "sanity":
            attacks = ["boundary"]
            kwargs["boundary"] = {"steps": 40}
        elif goal == "standard":
            attacks = ["boundary", "hopskipjump"]
            kwargs["boundary"] = {"steps": 60}
            kwargs["hopskipjump"] = {"steps": 20, "initial_num_evals": 30}
        else:
            attacks = ["boundary", "hopskipjump"]
            kwargs["boundary"] = {"steps": 120}
            kwargs["hopskipjump"] = {"steps": 40, "initial_num_evals": 64}
        rationale = (
            "Hard-label decision oracle mode; starting from classic Boundary Attack baseline and escalating to SOTA HopSkipJumpAttack."
        )
    else:
        effective_tm = "white_box"
        if norm == "l1":
            if goal == "sanity":
                attacks = ["apgd_l1"]
                kwargs["apgd_l1"] = {"steps": 20, "restarts": 1}
                rationale = "Fast L1 sanity check: Mind-the-Box APGD-L1 is specialized for exact L1 simplex/box geometry."
            elif goal == "standard":
                attacks = ["apgd_l1", "ead"]
                kwargs["apgd_l1"] = {"steps": 40, "restarts": 2}
                kwargs["ead"] = {"steps": 60}
                rationale = (
                    "Balanced L1 robustness suite pairing Mind-the-Box APGD-L1 with Elastic-Net L1/L2 (EAD) "
                    "without incompatible Linf/L2-only attacks."
                )
            else:
                attacks = ["apgd_l1", "ead", "sparsefool"]
                kwargs["apgd_l1"] = {"steps": 60, "restarts": 2}
                kwargs["ead"] = {"steps": 80}
                kwargs["sparsefool"] = {"max_iter": 12}
                rationale = (
                    "Publication-grade L1/sparse robustness suite combining APGD-L1, EAD, and SparseFool "
                    "all verified for native L1 geometry."
                )
        elif domain == "unsupervised":
            attacks = ["fgsm", "pgd"] if goal == "sanity" else ["fgsm", "pgd", "tpgd"]
            kwargs["pgd"] = {"steps": 10 if goal == "sanity" else 30}
            rationale = "Unsupervised / representation model: using FGSM baseline, PGD, and KL-divergence TPGD."
        elif goal == "sanity":
            attacks = ["fgsm", "pgd"]
            kwargs["pgd"] = {"steps": 10}
            rationale = "Fast sanity check: FGSM tests linear susceptibility and 10-step PGD catches basic multi-step collapse."
        elif goal == "standard":
            attacks = ["fgsm", "pgd", "apgd_dlr", "square_attack"]
            if domain == "vision":
                attacks.insert(2, "di_fgsm")
                kwargs["di_fgsm"] = {"steps": 10}
            elif domain == "wireless":
                attacks.insert(2, "vmi_fgsm")
                kwargs["vmi_fgsm"] = {"steps": 15, "num_variance_samples": 6}
            kwargs["pgd"] = {"steps": 20}
            kwargs["apgd_dlr"] = {"steps": 25, "restarts": 1}
            kwargs["square_attack"] = {"max_queries": 300}
            rationale = (
                f"Balanced {len(attacks)}-attack suite covering complementary roles: a 1-step quick push (FGSM), "
                "smart multi-step search (PGD + transfer attack), scale-proof anti-saturation (APGD-DLR), "
                "and a blind gradient-free probe (Square Attack) so a model cannot fake safety via broken gradients."
            )
        else:
            attacks = ["fgsm", "pgd", "vmi_fgsm", "apgd_ce", "apgd_dlr", "square_attack", "autoattack"]
            kwargs["pgd"] = {"steps": 40, "restarts": 2}
            kwargs["vmi_fgsm"] = {"steps": 20, "num_variance_samples": 10}
            kwargs["apgd_ce"] = {"steps": 50, "restarts": 2}
            kwargs["apgd_dlr"] = {"steps": 50, "restarts": 2}
            kwargs["square_attack"] = {"max_queries": 500}
            kwargs["autoattack"] = {"version": "standard", "apgd_steps": 50, "square_queries": 500}
            rationale = (
                "Top-conference publication grade: combines quick baseline (FGSM), restarted smart search (PGD/APGD-CE), "
                "anti-gradient-masking checks (APGD-DLR + blind Square Attack), and the full AutoAttack ensemble alongside Carlini false-robustness checks."
            )

    all_cards = {c["id"]: c for c in load_catalog()}
    lineage_summary: list[dict[str, str]] = []
    for aid in attacks:
        c = all_cards.get(aid)
        if c:
            lineage_summary.append(
                {
                    "id": c["id"],
                    "name": c["name"],
                    "tier": c["tier"],
                    "role": c["summary_plain"],
                    "file": c["module_path"],
                }
            )

    return RecommendedSuite(
        goal=goal,
        domain=domain,
        norm=norm,
        threat_model=effective_tm,
        attacks=attacks,
        attack_kwargs=kwargs,
        rationale=rationale,
        lineage_summary=lineage_summary,
    )


def explain_parameters(domain: str = "vision", norm: str | None = None) -> dict[str, Any]:
    """Return novice-friendly plain analogies, trade-offs, and intuitive presets for key parameters."""
    if norm is None:
        norm = "jsr_db" if domain == "wireless" else ("linf" if domain == "vision" else "relative_l2")

    if norm == "jsr_db":
        eps_guide = {
            "plain_analogy": (
                "Think of this as how loud the attacker's radio jammer is allowed to be compared to your clean wireless signal."
            ),
            "if_higher_vs_lower": (
                "Lower dB (e.g., -20 dB = 1% jammer power) tests quiet stealth jamming; higher dB (e.g., 0 dB = 1:1 power) floods the signal heavily."
            ),
            "meaning": (
                "How strong the jamming interference may be relative to clean signal power "
                "(Jamming-to-Signal Ratio / JSR in dB: 10 * log10(||delta||^2 / ||x||^2))."
            ),
            "presets": [
                {
                    "value": -20.0,
                    "label": "Subtle stealth noise (~1% of signal power, eps=-20 dB) — checks extreme sensitivity",
                },
                {
                    "value": -10.0,
                    "label": "Standard tactical RF benchmark (~10% of signal power, eps=-10 dB) [Recommended]",
                },
                {
                    "value": 0.0,
                    "label": "Heavy stress test (jammer power equals signal 1:1, eps=0 dB) — checks breaking limits",
                },
            ],
        }
    elif norm == "linf":
        eps_guide = {
            "plain_analogy": (
                "Think of epsilon as the maximum amount of invisible pixel static allowed on any single pixel of the image."
            ),
            "if_higher_vs_lower": (
                "Too small misses subtle security flaws; too large (>32/255) visibly smudges the image so even humans get confused."
            ),
            "meaning": (
                "Maximum per-pixel/per-feature change allowed on [0, 1] inputs "
                "(L-infinity perturbation budget |x_adv - x|_inf)."
            ),
            "presets": [
                {
                    "value": 2.0 / 255.0,
                    "label": "Ultra-light tweak (~0.8% pixel shift, eps=2/255) — invisible; catches fragile models",
                },
                {
                    "value": 8.0 / 255.0,
                    "label": "Balanced paper standard (~3.1% pixel shift, eps=8/255) — invisible to eyes & directly comparable to CIFAR-10/ICLR papers [Recommended]",
                },
                {
                    "value": 16.0 / 255.0,
                    "label": "Strong distortion (~6.3% pixel shift, eps=16/255) — faint static starts showing; strict stress test",
                },
            ],
        }
    elif norm == "relative_l2":
        eps_guide = {
            "plain_analogy": (
                "Think of this as turning up background static across the whole input by a small percentage of its normal energy."
            ),
            "if_higher_vs_lower": (
                "Smaller percentage preserves near-identical inputs; larger percentage allows stronger overall distortion."
            ),
            "meaning": (
                "Total perturbation energy allowed as a fraction of the clean input's energy "
                "(relative L2 ratio ||delta||_2 / ||x||_2)."
            ),
            "presets": [
                {
                    "value": 0.02,
                    "label": "Gentle 2% overall energy change (eps=0.02) — subtle baseline test",
                },
                {
                    "value": 0.05,
                    "label": "Standard 5% overall energy change (eps=0.05) — balanced development test [Recommended]",
                },
                {
                    "value": 0.15,
                    "label": "Aggressive 15% energy change (eps=0.15) — heavy stress test",
                },
            ],
        }
    else:
        eps_guide = {
            "plain_analogy": (
                "Think of this as the total distance budget the attacker is allowed to move an input away from its original clean version."
            ),
            "if_higher_vs_lower": (
                "Smaller radius allows only tiny edits; larger radius gives the attacker much more freedom to flip predictions."
            ),
            "meaning": f"Allowed perturbation radius measured under norm='{norm}'.",
            "presets": [
                {"value": 0.25, "label": "Small tweak radius (eps=0.25) — quick sensitivity probe"},
                {"value": 0.50, "label": "Standard CIFAR L2 radius (eps=0.50) — matches common benchmarks [Recommended]"},
                {"value": 1.00, "label": "Large perturbation radius (eps=1.00) — heavy worst-case stress test"},
            ],
        }

    return {
        "domain": domain,
        "norm": norm,
        "epsilon": eps_guide,
        "steps_and_queries": {
            "plain_analogy": (
                "Think of 'steps' (and black-box 'queries') as how many attempts an attacker gets to probe your model for blind spots."
            ),
            "if_higher_vs_lower": (
                "Few tries finish in seconds but can miss hidden weak spots; more tries + blind random probes take longer but prevent models from faking safety."
            ),
            "meaning": (
                "'steps' sets how many iterative search updates gradient attacks run; "
                "'queries' limits how many trial checks gradient-free black-box attacks can spend per sample."
            ),
            "presets": [
                {
                    "goal": "sanity",
                    "steps": 10,
                    "queries": 150,
                    "meaning": "Quick smoke test (~seconds): checks if the pipeline works and whether simple attacks break the model",
                },
                {
                    "goal": "standard",
                    "steps": 20,
                    "queries": 300,
                    "meaning": "Balanced thorough check (~0.5-2 min): multi-step search + anti-masking + blind check so fake defenses cannot hide [Recommended]",
                },
                {
                    "goal": "paper_sota",
                    "steps": 50,
                    "queries": 500,
                    "meaning": "Strict publication benchmark (slower): heavy search + AutoAttack ensemble for publishable worst-case numbers",
                },
            ],
        },
        "scale_and_samples": {
            "plain_analogy": (
                "Think of sample count (N) as how many test items we put through the stress test—like polling 64 voters for a quick pulse vs. 1,000 voters for a survey."
            ),
            "if_higher_vs_lower": (
                "Fewer samples (N=64) finish fast for debugging; more samples (N=200-1000) tighten statistical error bars at proportional runtime cost."
            ),
            "meaning": "Number of evaluation inputs (N) tested per attack and whether full audit ensembles run.",
            "presets": [
                {
                    "samples": 64,
                    "goal": "sanity",
                    "meaning": "Fast prototype loop (N=64): finishes in seconds to verify wiring and obvious cracks",
                },
                {
                    "samples": 200,
                    "goal": "standard",
                    "meaning": "Reliable dev sample (N=200): low statistical noise without long waits [Recommended]",
                },
                {
                    "samples": 1000,
                    "goal": "paper_sota",
                    "meaning": "Full paper benchmark (N=1000+): publication / leaderboard grade confidence",
                },
            ],
        },
    }


def _estimate_attack_work_units(attack_id: str, kw: dict[str, Any]) -> tuple[float, float]:
    """Return estimated (forward_passes_per_sample, backward_passes_per_sample)."""
    steps = float(kw.get("steps", 20))
    restarts = float(kw.get("restarts", 1))
    if attack_id == "fgsm":
        return 1.0, 1.0
    if attack_id == "rfgsm":
        return 1.0, 1.0
    if attack_id in ("bim", "pgd", "mifgsm", "adam_pgd", "ni_fgsm", "di_fgsm", "ti_fgsm", "tpgd"):
        return steps * restarts, steps * restarts
    if attack_id == "sini_fgsm":
        copies = float(kw.get("scale_copies", 5))
        return steps * copies * restarts, steps * copies * restarts
    if attack_id == "vmi_fgsm":
        n_var = float(kw.get("num_variance_samples", 10))
        return steps * (1.0 + n_var), steps * (1.0 + n_var)
    if attack_id in ("apgd_ce", "apgd_dlr", "apgd_l1"):
        s = float(kw.get("steps", 50))
        r = float(kw.get("restarts", 2))
        return s * r, s * r
    if attack_id in ("eot_pgd", "bpda_eot"):
        eot = float(kw.get("eot_samples", 4))
        return steps * eot * restarts, steps * eot * restarts
    if attack_id == "square_attack":
        return 0.65 * float(kw.get("max_queries", 500)), 0.0
    if attack_id == "simba":
        return 1.2 * float(kw.get("max_queries", 400)), 0.0
    if attack_id in ("nes", "spsa"):
        return 2.0 * steps * float(kw.get("nb_sample", 32)), 0.0
    if attack_id == "zoo":
        return 2.0 * steps * float(kw.get("coord_batch_size", 64)), 0.0
    if attack_id == "boundary":
        return 2.0 * float(kw.get("steps", 80)), 0.0
    if attack_id == "hopskipjump":
        return steps * (15.0 + float(kw.get("initial_num_evals", 30))), 0.0
    if attack_id == "deepfool":
        return float(kw.get("steps", 50)), float(kw.get("steps", 50)) * 3.0
    if attack_id == "sparsefool":
        return float(kw.get("max_iter", 12)) * float(kw.get("deepfool_steps", 20)) * 0.5, float(kw.get("max_iter", 12)) * 10.0
    if attack_id == "fab":
        return float(kw.get("steps", 25)) * float(kw.get("restarts", 1)) * 2.0, float(kw.get("steps", 25)) * 2.0
    if attack_id == "autoattack":
        apgd_s = float(kw.get("apgd_steps", 50))
        apgd_r = float(kw.get("apgd_restarts", 2))
        sq_q = float(kw.get("square_queries", 500))
        return 2.0 * apgd_s * apgd_r * 0.7 + 0.5 * sq_q, 2.0 * apgd_s * apgd_r * 0.7
    if attack_id in ("cw_l2", "ead"):
        s = float(kw.get("steps", 100))
        bs = float(kw.get("binary_search_steps", 5))
        return s * bs, s * bs
    return steps, steps * 0.5


_SERIAL_SAMPLE_ATTACKS = frozenset({"deepfool", "sparsefool", "boundary", "hopskipjump"})
_PERTURB_BATCH_ATTACKS = frozenset({"nes", "spsa", "zoo"})


def _sync_device(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _estimate_attack_seconds(
    attack_id: str,
    kw: dict[str, Any],
    *,
    num_samples: int,
    fwd_launch_ms: float,
    fwd_slope_ms: float,
    fb_launch_ms: float,
    fb_slope_ms: float,
    t_fwd_1_ms: float,
    t_fb_1_ms: float,
) -> float:
    f_units, b_units = _estimate_attack_work_units(attack_id, kw)
    bwd_1_ms = max(t_fb_1_ms - t_fwd_1_ms, t_fwd_1_ms * 0.8)

    if attack_id in _SERIAL_SAMPLE_ATTACKS:
        ms = num_samples * (f_units * t_fwd_1_ms + b_units * bwd_1_ms)
        return round(max(ms / 1000.0, 0.01), 2)

    if attack_id in _PERTURB_BATCH_ATTACKS:
        steps = float(kw.get("steps", 25))
        nb = float(kw.get("coord_batch_size", 64) if attack_id == "zoo" else kw.get("nb_sample", 32))
        max_bs = int(kw.get("max_batch_size", 512))
        probes_per_step = 2.0 * nb * num_samples
        batches_per_step = max(1, math.ceil(probes_per_step / max(max_bs, 1)))
        eff_bs = min(max_bs, max(1, round(probes_per_step / batches_per_step)))
        step_ms = batches_per_step * (fwd_launch_ms + eff_bs * fwd_slope_ms)
        return round(max((steps * step_ms) / 1000.0, 0.01), 2)

    # Batched tensor attacks (first-order, Square, SimBA, AutoAttack)
    fwd_batch_ms = fwd_launch_ms + num_samples * fwd_slope_ms
    fb_batch_ms = fb_launch_ms + num_samples * fb_slope_ms
    bwd_only_batch_ms = max(fb_batch_ms - fwd_batch_ms, fwd_batch_ms * 0.8)

    if attack_id == "square_attack":
        # Early stop typically terminates solved samples around ~40-50% of max_queries, plus small tensor indexing overhead.
        max_q = float(kw.get("max_queries", 500))
        eff_q = max_q * (0.45 if kw.get("early_stop", True) else 1.0)
        ms = eff_q * (fwd_batch_ms + 2.5)
        return round(max(ms / 1000.0, 0.01), 2)

    if attack_id == "simba":
        max_q = float(kw.get("max_queries", 400))
        ms = max_q * 0.65 * (fwd_batch_ms + 1.5)
        return round(max(ms / 1000.0, 0.01), 2)

    ms = f_units * fwd_batch_ms + b_units * (bwd_only_batch_ms + 0.4)
    return round(max(ms / 1000.0, 0.01), 2)


def estimate_runtime(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_sample: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    attacks: Sequence[str] | RecommendedSuite,
    *,
    num_samples: int = 100,
    attack_kwargs: dict[str, dict[str, Any]] | None = None,
    include_audit: bool = True,
) -> RuntimeEstimate:
    """Measure warmed batched latency on target hardware to forecast evaluation & audit ETA."""
    goal_hint: str | None = None
    if isinstance(attacks, RecommendedSuite):
        attack_list = list(attacks.attacks)
        merged_kwargs = {**attacks.attack_kwargs, **(attack_kwargs or {})}
        goal_hint = attacks.goal
    else:
        attack_list = list(attacks)
        merged_kwargs = dict(attack_kwargs or {})

    if isinstance(x_sample, torch.Tensor):
        dev = infer_model_device(model_or_fn, x_sample.device)
        probe_bs = min(x_sample.shape[0], 16)
        x_probe = x_sample[:probe_bs].detach().to(dev)
    else:
        dev = infer_model_device(model_or_fn, torch.device("cpu"))
        x_probe, _ = materialize_eval_tensors(x_sample, max_samples=16, device=dev)
        probe_bs = int(x_probe.shape[0])

    # Preserve caller gradient state on parameters before timing.
    saved_grads: list[torch.Tensor | None] = []
    if isinstance(model_or_fn, nn.Module):
        for p in model_or_fn.parameters():
            saved_grads.append(None if p.grad is None else p.grad.detach().clone())

    with eval_mode(model_or_fn):
        # Untimed warmup avoids counting one-time MPS/CUDA shader compilation in per-sample timing.
        with torch.no_grad():
            _ = model_or_fn(x_probe)
        _sync_device(dev)

        small_bs = min(2, probe_bs)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_or_fn(x_probe[:small_bs])
        _sync_device(dev)
        t_fwd_s_ms = max((time.perf_counter() - t0) * 1000.0, 0.05)

        if probe_bs > small_bs:
            t1 = time.perf_counter()
            with torch.no_grad():
                _ = model_or_fn(x_probe)
            _sync_device(dev)
            t_fwd_k_ms = max((time.perf_counter() - t1) * 1000.0, t_fwd_s_ms)
            fwd_slope_ms = max((t_fwd_k_ms - t_fwd_s_ms) / (probe_bs - small_bs), 0.005)
            fwd_launch_ms = max(t_fwd_s_ms - small_bs * fwd_slope_ms, 0.05)
            fwd_ms_per_sample = max(t_fwd_k_ms / probe_bs, 0.02)
            t_fwd_1_ms = fwd_launch_ms + fwd_slope_ms
        else:
            t_fwd_k_ms = t_fwd_s_ms
            fwd_ms_per_sample = max(t_fwd_s_ms / small_bs, 0.02)
            if dev.type in ("mps", "cuda"):
                fwd_launch_ms = 0.65 * t_fwd_s_ms
                fwd_slope_ms = 0.35 * fwd_ms_per_sample
            else:
                fwd_launch_ms = 0.1 * t_fwd_s_ms
                fwd_slope_ms = 0.9 * fwd_ms_per_sample
            t_fwd_1_ms = fwd_launch_ms + fwd_slope_ms

        autograd_ok = False
        t_fb_1_ms = t_fwd_1_ms * 2.2
        fb_launch_ms = fwd_launch_ms * 2.2
        fb_slope_ms = fwd_slope_ms * 2.2
        fwd_bwd_ms_per_sample = 0.0

        try:
            x_warm = x_probe.clone().requires_grad_(True)
            out_w = model_or_fn(x_warm)
            if out_w.requires_grad:
                loss_w = (out_w.abs().pow(2) if out_w.is_complex() else out_w).sum()
                (grad_w,) = torch.autograd.grad(loss_w, x_warm, only_inputs=True, allow_unused=True)
                if grad_w is not None:
                    autograd_ok = True
                    _sync_device(dev)

                    x_ins = x_probe[:small_bs].clone().requires_grad_(True)
                    tb0 = time.perf_counter()
                    out_s = model_or_fn(x_ins)
                    loss_s = (out_s.abs().pow(2) if out_s.is_complex() else out_s).sum()
                    torch.autograd.grad(loss_s, x_ins, only_inputs=True, allow_unused=True)
                    _sync_device(dev)
                    t_fb_s_ms = max((time.perf_counter() - tb0) * 1000.0, t_fwd_s_ms)

                    if probe_bs > small_bs:
                        x_ink = x_probe.clone().requires_grad_(True)
                        tb1 = time.perf_counter()
                        out_k = model_or_fn(x_ink)
                        loss_k = (out_k.abs().pow(2) if out_k.is_complex() else out_k).sum()
                        torch.autograd.grad(loss_k, x_ink, only_inputs=True, allow_unused=True)
                        _sync_device(dev)
                        t_fb_k_ms = max((time.perf_counter() - tb1) * 1000.0, t_fb_s_ms)
                        fb_slope_ms = max((t_fb_k_ms - t_fb_s_ms) / (probe_bs - small_bs), fwd_slope_ms * 1.1)
                        fb_launch_ms = max(t_fb_s_ms - small_bs * fb_slope_ms, fwd_launch_ms * 1.1)
                        fwd_bwd_ms_per_sample = max(t_fb_k_ms / probe_bs, fwd_ms_per_sample)
                        t_fb_1_ms = fb_launch_ms + fb_slope_ms
                    else:
                        fwd_bwd_ms_per_sample = max(t_fb_s_ms / small_bs, fwd_ms_per_sample)
                        if dev.type in ("mps", "cuda"):
                            fb_launch_ms = 0.65 * t_fb_s_ms
                            fb_slope_ms = 0.35 * fwd_bwd_ms_per_sample
                        else:
                            fb_launch_ms = 0.1 * t_fb_s_ms
                            fb_slope_ms = 0.9 * fwd_bwd_ms_per_sample
                        t_fb_1_ms = fb_launch_ms + fb_slope_ms
        except RuntimeError:
            autograd_ok = False
            fwd_bwd_ms_per_sample = 0.0

    if isinstance(model_or_fn, nn.Module):
        for p, g in zip(model_or_fn.parameters(), saved_grads):
            p.grad = g

    timing_params = {
        "fwd_launch_ms": fwd_launch_ms,
        "fwd_slope_ms": fwd_slope_ms,
        "fb_launch_ms": fb_launch_ms,
        "fb_slope_ms": fb_slope_ms,
        "t_fwd_1_ms": t_fwd_1_ms,
        "t_fb_1_ms": t_fb_1_ms,
    }
    per_attack: dict[str, float] = {
        aid: _estimate_attack_seconds(
            aid, merged_kwargs.get(aid, {}), num_samples=num_samples, **timing_params
        )
        for aid in attack_list
    }

    audit_sec = 0.0
    if include_audit and num_samples >= 2 and goal_hint != "sanity":
        k_audit = min(num_samples, 64)
        # Carlini audit runs: random(5) + square(350) + sweep_low/high(apgd or square) + wb probes if autograd_ok.
        audit_sec += _estimate_attack_seconds(
            "square_attack", {"max_queries": 350, "early_stop": True}, num_samples=k_audit, **timing_params
        )
        if autograd_ok:
            audit_sec += _estimate_attack_seconds("fgsm", {}, num_samples=k_audit, **timing_params)
            audit_sec += _estimate_attack_seconds("pgd", {"steps": 20}, num_samples=k_audit, **timing_params)
            audit_sec += 2.0 * _estimate_attack_seconds(
                "apgd_ce", {"steps": 35, "restarts": 1}, num_samples=k_audit, **timing_params
            )
            audit_sec += 2.0 * _estimate_attack_seconds(
                "apgd_dlr", {"steps": 35, "restarts": 1}, num_samples=k_audit, **timing_params
            )
        else:
            audit_sec += 2.0 * _estimate_attack_seconds(
                "square_attack", {"max_queries": 350, "early_stop": True}, num_samples=k_audit, **timing_params
            )
        audit_sec = round(audit_sec, 2)

    total_sec = round(sum(per_attack.values()) + audit_sec, 2)
    if total_sec > 300.0:
        hints: list[str] = []
        if num_samples > 100:
            hints.append(f"reducing N={num_samples} -> 64/100")
        elif num_samples > 32:
            hints.append("starting with N=32 for a quick smoke check")
        if goal_hint == "paper_sota":
            hints.append("running goal='standard' before a full paper_sota run")
        if dev.type == "cpu":
            hints.append("switching to MPS/CUDA GPU if available")
        hint_str = " or ".join(hints) if hints else "running a lighter subset first"
        advisory = f"Heavy runtime (~{total_sec / 60.0:.1f} min). Consider {hint_str}."
    elif total_sec > 60.0:
        if num_samples > 64:
            advisory = "Moderate runtime (~1-5 min); consider running a quick N=64 probe first if iterating."
        else:
            advisory = "Moderate runtime (~1-5 min); balanced scale for thorough verification."
    else:
        advisory = "Fast runtime (<1 min); safe to run immediately."

    return RuntimeEstimate(
        device=str(dev),
        num_samples=num_samples,
        forward_ms_per_sample=round(fwd_ms_per_sample, 3),
        fwd_bwd_ms_per_sample=round(fwd_bwd_ms_per_sample, 3) if autograd_ok else 0.0,
        autograd_available=autograd_ok,
        per_attack_seconds=per_attack,
        total_seconds=total_sec,
        advisory=advisory,
        audit_seconds=audit_sec,
    )


def build_interactive_brief(
    *,
    project_scan: ProjectScan | None = None,
    model_probe: ModelContextProbe | None = None,
    runtime_estimate: RuntimeEstimate | None = None,
    domain: str | None = None,
    norm: str | None = None,
    threat_model: str = "white_box",
    autograd_ok: bool | None = None,
    goal: str = "standard",
) -> dict[str, Any]:
    """Assemble an AI-agent interaction brief containing project context, parameter Q&A, and ETA."""
    resolved_domain = (
        domain
        or (model_probe.inferred_domain if model_probe is not None else None)
        or (project_scan.likely_domain if project_scan is not None else "vision")
    )
    resolved_norm = (
        norm
        or (model_probe.recommended_norm if model_probe is not None and domain is None else None)
        or (project_scan.recommended_norm if project_scan is not None and domain is None else None)
    )
    resolved_autograd = (
        autograd_ok
        if autograd_ok is not None
        else (model_probe.autograd_connected if model_probe is not None else True)
    )

    suite = recommend_suite(
        domain=resolved_domain,
        goal=goal,
        threat_model=threat_model,
        autograd_ok=resolved_autograd,
        norm=resolved_norm,
    )
    eff_norm = suite.norm
    param_guide = explain_parameters(domain=resolved_domain, norm=eff_norm)
    questions = [
        {
            "parameter": "eps / norm",
            "prompt": f"How much noise/change (`eps` under `{eff_norm}`) may the attacker add?",
            "plain_analogy": param_guide["epsilon"]["plain_analogy"],
            "if_higher_vs_lower": param_guide["epsilon"]["if_higher_vs_lower"],
            "novice_explanation": param_guide["epsilon"]["meaning"],
            "options": param_guide["epsilon"]["presets"],
        },
        {
            "parameter": "goal / steps / queries",
            "prompt": "How hard and thoroughly should the attacks search for weak spots?",
            "plain_analogy": param_guide["steps_and_queries"]["plain_analogy"],
            "if_higher_vs_lower": param_guide["steps_and_queries"]["if_higher_vs_lower"],
            "novice_explanation": param_guide["steps_and_queries"]["meaning"],
            "options": param_guide["steps_and_queries"]["presets"],
        },
        {
            "parameter": "num_samples",
            "prompt": "How many test samples (N) should we evaluate?",
            "plain_analogy": param_guide["scale_and_samples"]["plain_analogy"],
            "if_higher_vs_lower": param_guide["scale_and_samples"]["if_higher_vs_lower"],
            "novice_explanation": param_guide["scale_and_samples"]["meaning"],
            "options": param_guide["scale_and_samples"]["presets"],
        },
    ]

    return {
        "interaction_guidelines": {
            "language_rule": "Respond and translate explanations/options into the user's conversation language.",
            "zero_jargon_rule": (
                "Always explain each knob using `plain_analogy` + `if_higher_vs_lower` BEFORE showing options, "
                "and make every choice label describe what changes in practice (with technical terms in parentheses)."
            ),
        },
        "project_scan": project_scan.to_dict() if project_scan is not None else None,
        "model_probe": model_probe.to_dict() if model_probe is not None else None,
        "recommended_suite": suite.to_dict(),
        "parameter_explanations": param_guide,
        "runtime_estimate": runtime_estimate.to_dict() if runtime_estimate is not None else None,
        "questions_to_ask_user": questions,
    }
