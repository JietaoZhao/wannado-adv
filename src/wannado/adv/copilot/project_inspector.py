"""Project traversal and live model/tensor probing for AI-collaborative robustness testing.

Designed to prevent AI agents from guessing project structure, model modes, input domains,
or stale assumptions when plugging adversarial evaluations into user codebases.
"""

from __future__ import annotations

import ast
import os
import re
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from wannado.adv.audit.preflight import preflight
from wannado.adv.core.losses import unwrap_model_outputs

_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".agent",
    ".agents",
    ".claude",
    ".cursor",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "egg-info",
    ".ipynb_checkpoints",
}

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "wireless": (
        "complex64",
        "complex128",
        "jsr_db",
        "iq_signal",
        "modulation",
        "ofdm",
        "bpsk",
        "qpsk",
        "qam",
        "awgn",
        "snr",
        "sdr",
        "baseband",
        "antenna",
        "mimo",
    ),
    "drl": (
        "gymnasium",
        "import gym",
        "stable_baselines",
        "ppo",
        "dqn",
        "sac",
        "td3",
        "replaybuffer",
        "observation_space",
        "action_space",
        "step(self, action",
    ),
    "vision": (
        "torchvision",
        "conv2d",
        "resnet",
        "wideresnet",
        "robustbench",
        "cifar",
        "imagenet",
        "mnist",
        "vit",
        "pil",
        "transforms.",
    ),
    "timeseries": (
        "lstm",
        "gru",
        "chronos",
        "forecast",
        "sliding_window",
        "seq_len",
        "time_series",
        "timeseries",
    ),
    "tabular": (
        "sklearn",
        "dataframe",
        "pandas",
        "tabular",
        "xgboost",
        "lightgbm",
        "standardscaler",
    ),
    "unsupervised": (
        "autoencoder",
        "vae",
        "contrastive",
        "simclr",
        "embedding",
        "reconstruction_loss",
        "latent",
    ),
    "medical": (
        "monai",
        "nibabel",
        "pydicom",
        "simpleitk",
        "ct_scan",
        "hounsfield",
        "radiology",
        "ecg",
        "eeg",
        "clinical",
        "sepsis",
        "dice_loss",
        "unet",
    ),
    "cybersecurity": (
        "cicids",
        "nsl_kdd",
        "nsl-kdd",
        "intrusion",
        "malware",
        "pcap",
        "botnet",
        "phishing",
        "netflow",
        "mitre",
        "anomaly_detection",
    ),
    "nlp_transformers": (
        "transformers",
        "autotokenizer",
        "automodel",
        "huggingface",
        "bert",
        "roberta",
        "token_ids",
        "attention_mask",
        "vocab_size",
        "nlp",
    ),
}


@dataclass
class ProjectScan:
    """Fresh structural and domain snapshot of a user repository."""

    root: str
    scanned_files_count: int
    total_python_files: int
    frameworks: list[str] = field(default_factory=list)
    likely_domain: str = "vision"
    domain_scores: dict[str, int] = field(default_factory=dict)
    recommended_norm: str = "linf"
    entry_candidates: list[str] = field(default_factory=list)
    model_files: list[str] = field(default_factory=list)
    evidence_by_domain: dict[str, list[str]] = field(default_factory=dict)
    existing_eval_dirs: list[str] = field(default_factory=list)
    custom_attack_files: list[str] = field(default_factory=list)
    checkpoint_files: list[str] = field(default_factory=list)
    normalization_hints: list[str] = field(default_factory=list)
    fingerprint: str = ""
    readme_summary: str = ""
    detected_symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize snapshot to JSON-compatible dictionary."""
        return asdict(self)

    def format_summary(self) -> str:
        """Render concise human- and AI-readable project inspection summary."""
        lines = [
            f"[Project Scan] root={self.root}",
            f"  Python files scanned: {self.scanned_files_count} (total={self.total_python_files})",
            f"  Detected frameworks : {', '.join(self.frameworks) if self.frameworks else 'unknown'}",
            f"  Inferred domain     : {self.likely_domain} (recommended norm='{self.recommended_norm}')",
        ]
        if self.readme_summary:
            lines.append(f"  README summary      : {self.readme_summary}")
        if self.model_files:
            lines.append(f"  Candidate models    : {', '.join(self.model_files[:10])}")
        if self.detected_symbols:
            lines.append(f"  Detected symbols    : {', '.join(self.detected_symbols[:10])}")
        if self.entry_candidates:
            lines.append(f"  Train/Eval scripts  : {', '.join(self.entry_candidates[:8])}")
        if self.normalization_hints:
            lines.append(
                f"  Normalize hints     : {', '.join(self.normalization_hints[:5])} "
                "(wrap mean/std inside VictimInterface so attacks perturb [0, 1] pixels)"
            )
        if self.checkpoint_files:
            lines.append(f"  Checkpoints found   : {', '.join(self.checkpoint_files[:5])}")
        else:
            lines.append("  Checkpoints found   : none (check weights path or train a quick baseline first)")
        if self.existing_eval_dirs:
            lines.append(f"  Isolated eval harness: {', '.join(self.existing_eval_dirs)}")
        if self.custom_attack_files:
            lines.append(f"  Local attack files  : {', '.join(self.custom_attack_files)}")
        return "\n".join(lines)


@dataclass
class ModelContextProbe:
    """Live hardware, input geometry, and gradient-flow probe on victim model + batch."""

    device: str
    dtype: str
    is_complex: bool
    input_shape: list[int]
    input_min: float | None
    input_max: float | None
    output_shape: list[int] | None
    task_kind: str
    autograd_connected: bool
    preflight_ok: bool
    preflight_warnings: list[str] = field(default_factory=list)
    preflight_errors: list[str] = field(default_factory=list)
    preflight_recommendations: list[str] = field(default_factory=list)
    inferred_domain: str = "vision"
    recommended_norm: str = "linf"
    recommended_clip_bounds: tuple[Any, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize probe results into JSON-safe dictionary."""
        data = asdict(self)
        if self.recommended_clip_bounds is not None:
            data["recommended_clip_bounds"] = [
                v.detach().cpu().tolist()
                if isinstance(v, torch.Tensor)
                else (v.tolist() if hasattr(v, "tolist") else float(v))
                for v in self.recommended_clip_bounds
            ]
        return data


_CHECKPOINT_EXTS = (".pt", ".pth", ".ckpt", ".safetensors")

_SHORT_WORD_PATTERNS: dict[str, re.Pattern[str]] = {
    kw: re.compile(rf"\b{re.escape(kw)}\b")
    for kws in _DOMAIN_KEYWORDS.values()
    for kw in kws
    if len(kw) <= 4 and "." not in kw and "_" not in kw
}


def _iter_repo_files(root: Path, max_files: int) -> tuple[list[Path], int, list[str]]:
    """Walk ``root`` while skipping dependency/cache directories and collecting checkpoints."""
    collected: list[Path] = []
    checkpoints: list[str] = []
    total_py = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in _IGNORE_DIRS and not d.endswith(".egg-info")
        )
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                total_py += 1
                if len(collected) < max_files:
                    collected.append(Path(dirpath) / fname)
            elif len(checkpoints) < 10 and fname.lower().endswith(_CHECKPOINT_EXTS):
                checkpoints.append((Path(dirpath) / fname).relative_to(root).as_posix())
    return collected, total_py, checkpoints


def _keyword_matched(kw: str, lower_txt: str) -> bool:
    pat = _SHORT_WORD_PATTERNS.get(kw)
    if pat is not None:
        return pat.search(lower_txt) is not None
    return kw in lower_txt


def _compute_tree_fingerprint(root: Path, py_files: list[Path]) -> str:
    """Compute lightweight change-detection fingerprint over Python paths and mtimes."""
    parts: list[str] = []
    for p in py_files:
        try:
            st = p.stat()
            rel = p.relative_to(root).as_posix()
            parts.append(f"{rel}:{int(st.st_mtime)}:{st.st_size}")
        except OSError:
            continue
    return str(hash("|".join(parts)))


def scan_project(project_dir: str | Path = ".", max_files: int = 200) -> ProjectScan:
    """Traverse a user ML project directory to discover domain, models, and eval interfaces.

    Always call this at conversation start and re-run after structural edits so the AI
    stays synchronized with the actual project rather than relying on stale memory.
    """
    root = Path(project_dir).resolve()
    py_files, total_py, checkpoint_files = _iter_repo_files(root, max_files=max_files)

    domain_scores: dict[str, int] = {k: 0 for k in _DOMAIN_KEYWORDS}
    evidence_by_domain: dict[str, list[str]] = {k: [] for k in _DOMAIN_KEYWORDS}
    frameworks: set[str] = set()
    entry_candidates: list[str] = []
    model_files: list[str] = []
    fallback_model_files: list[str] = []
    custom_attack_files: list[str] = []
    normalization_hints: list[str] = []
    detected_symbols: list[str] = []
    seen_symbols: set[str] = set()
    readme_summary = ""

    for readme_name in ("README.md", "README.rst", "README.txt", "readme.md"):
        readme_path = root / readme_name
        if readme_path.is_file():
            try:
                readme_text = readme_path.read_text(encoding="utf-8", errors="ignore")[:32768]
                lines_clean = [
                    ln.strip().lstrip("#").strip()
                    for ln in readme_text.splitlines()
                    if ln.strip() and not ln.strip().startswith("![") and not ln.strip().startswith("```")
                ]
                if lines_clean:
                    readme_summary = " — ".join(lines_clean[:2])[:260]
                lower_rm = readme_text.lower()
                for domain, kws in _DOMAIN_KEYWORDS.items():
                    hits = sum(1 for kw in kws if _keyword_matched(kw, lower_rm))
                    if hits > 0:
                        domain_scores[domain] += hits
                        if len(evidence_by_domain[domain]) < 5:
                            evidence_by_domain[domain].append(readme_name)
            except OSError:
                pass
            break

    fw_root_map = {
        "torch": "PyTorch",
        "torchvision": "torchvision",
        "robustbench": "RobustBench",
        "timm": "timm",
        "transformers": "Transformers",
        "gymnasium": "Gymnasium/Gym",
        "gym": "Gymnasium/Gym",
        "stable_baselines3": "Stable-Baselines3",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "sklearn": "scikit-learn",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "pandas": "pandas",
        "monai": "MONAI",
    }

    for fpath in py_files:
        rel = fpath.relative_to(root).as_posix()
        lower_rel = rel.lower()
        try:
            # Cap single file read at 128KB so huge generated files never bog down scanning.
            text = fpath.read_text(encoding="utf-8", errors="ignore")[:131072]
        except OSError:
            continue
        lower_txt = text.lower()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if alias.name.startswith("wannado.adv"):
                            frameworks.add("wannado-adv")
                        elif top in fw_root_map:
                            frameworks.add(fw_root_map[top])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    if node.module.startswith("wannado.adv"):
                        frameworks.add("wannado-adv")
                    elif top in fw_root_map:
                        frameworks.add(fw_root_map[top])
                elif isinstance(node, (ast.ClassDef, ast.FunctionDef)) and (
                    not node.name.startswith("_") and len(detected_symbols) < 25
                ):
                    sym = f"{rel}:{node.name}"
                    if sym not in seen_symbols:
                        seen_symbols.add(sym)
                        detected_symbols.append(sym)
        except SyntaxError:
            pass

        is_attack_path = "attack" in lower_rel or "/attacks/" in lower_rel
        if "class " in text and ("nn.Module" in text or "Module):" in text):
            if is_attack_path or "adv_eval/" in lower_rel:
                fallback_model_files.append(rel)
            else:
                model_files.append(rel)
        if any(stem in lower_rel for stem in ("train", "eval", "test", "main", "infer", "run")):
            entry_candidates.append(rel)
        if is_attack_path or ("def " in text and "AttackResult" in text and "META =" in text):
            custom_attack_files.append(rel)
        if ("transforms.Normalize" in text or "Normalize(" in text) and "adv_eval/" not in lower_rel:
            normalization_hints.append(rel)

        for domain, kws in _DOMAIN_KEYWORDS.items():
            hits = sum(1 for kw in kws if _keyword_matched(kw, lower_txt))
            if hits > 0:
                domain_scores[domain] += hits
                if len(evidence_by_domain[domain]) < 5:
                    evidence_by_domain[domain].append(rel)

    selected_models = model_files if model_files else fallback_model_files
    selected_models.sort(
        key=lambda p: (
            0 if any(tok in p.lower() for tok in ("model", "net", "resnet", "classifier")) else 1,
            p,
        )
    )

    likely_domain = max(domain_scores.items(), key=lambda item: item[1])[0]
    if domain_scores[likely_domain] == 0:
        likely_domain = "general"

    norm_map = {
        "wireless": "jsr_db",
        "vision": "linf",
        "drl": "relative_l2",
        "tabular": "l2",
        "timeseries": "l2",
        "unsupervised": "l2",
        "medical": "l2",
        "cybersecurity": "l2",
        "nlp_transformers": "l2",
        "general": "l2",
    }

    existing_eval_dirs: list[str] = []
    for candidate_name in ("adv_eval", "wannado_eval", "adversarial_eval"):
        if (root / candidate_name).is_dir():
            existing_eval_dirs.append(candidate_name)

    return ProjectScan(
        root=str(root),
        scanned_files_count=len(py_files),
        total_python_files=total_py,
        frameworks=sorted(frameworks),
        likely_domain=likely_domain,
        domain_scores=domain_scores,
        recommended_norm=norm_map.get(likely_domain, "l2"),
        entry_candidates=entry_candidates[:15],
        model_files=selected_models[:15],
        evidence_by_domain={k: v for k, v in evidence_by_domain.items() if v},
        existing_eval_dirs=existing_eval_dirs,
        custom_attack_files=custom_attack_files[:20],
        checkpoint_files=checkpoint_files,
        normalization_hints=normalization_hints[:10],
        fingerprint=_compute_tree_fingerprint(root, py_files),
        readme_summary=readme_summary,
        detected_symbols=detected_symbols,
    )


def probe_model_context(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    norm: str | None = None,
    eps: float | None = None,
    clip_bounds: tuple[Any, Any] | None = None,
) -> ModelContextProbe:
    """Inspect a live model and sample batch to infer domain, range, norms, and autograd status."""
    is_complex = bool(x.is_complex())
    if norm is None:
        norm = "jsr_db" if is_complex else ("linf" if x.ndim == 4 else "l2")
    if eps is None:
        eps = -10.0 if norm == "jsr_db" else (8.0 / 255.0 if (norm == "linf" and x.ndim == 4) else 0.1)

    x_min: float | None = None
    x_max: float | None = None
    inferred_bounds = clip_bounds
    if not is_complex:
        x_det = x.detach()
        if bool(torch.isfinite(x_det).all()):
            x_min = float(x_det.min().item())
            x_max = float(x_det.max().item())
            if inferred_bounds is None and 0.0 <= x_min and x_max <= 1.0 and x.ndim == 4:
                inferred_bounds = (0.0, 1.0)

    loss_probe = "ce" if y is not None and y.dtype in (torch.int64, torch.int32, torch.long) else "mse"
    pf = preflight(
        model_or_fn,
        x,
        y,
        eps=eps,
        norm=norm,
        clip_bounds=inferred_bounds,
        loss=loss_probe,
        check_gradients=True,
        probe_batch_size=min(x.shape[0], 8),
    )
    grad_info = pf.summary.get("gradient_probe", {})
    out_shape = grad_info.get("outputs_shape")
    if out_shape is None and callable(model_or_fn):
        try:
            with torch.no_grad():
                out = unwrap_model_outputs(model_or_fn(x[: min(2, x.shape[0])].detach()))
            out_shape = list(out.shape)
        except Exception:  # noqa: BLE001
            out_shape = None

    if is_complex:
        inferred_domain = "wireless"
    elif x.ndim == 4:
        inferred_domain = "vision"
    elif x.ndim == 3:
        inferred_domain = "timeseries"
    else:
        inferred_domain = "tabular"

    is_cls_y = y is not None and y.dtype in (torch.int64, torch.int32, torch.long)
    if out_shape is not None and (len(out_shape) >= 2 or len(out_shape) == 1) and is_cls_y:
        task_kind = "classification"
    elif out_shape is not None and len(out_shape) >= 2 and y is None:
        task_kind = "classification_or_policy"
    else:
        task_kind = "regression_or_latent"

    return ModelContextProbe(
        device=str(x.device),
        dtype=str(x.dtype),
        is_complex=is_complex,
        input_shape=list(x.shape),
        input_min=x_min,
        input_max=x_max,
        output_shape=out_shape,
        task_kind=task_kind,
        autograd_connected=bool(grad_info.get("autograd_connected", False)),
        preflight_ok=pf.ok,
        preflight_warnings=list(pf.warnings),
        preflight_errors=list(pf.errors),
        preflight_recommendations=list(pf.recommendations),
        inferred_domain=inferred_domain,
        recommended_norm=norm,
        recommended_clip_bounds=inferred_bounds,
    )
