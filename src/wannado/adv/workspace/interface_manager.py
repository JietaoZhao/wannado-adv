"""Non-intrusive workspace integration and single-file attack ejection for AI IDEs.

Solves two core engineering pain points:
1. Never scatter adversarial code across a user's training/model files — wire via ONE
   clean interface adapter in ``adv_eval/interface.py``.
2. When upgrading an attack from a new research paper, eject a standalone single-file
   implementation (``eject_attack``) with preserved ``META`` citations and upgrade hooks.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from wannado.adv.attacks import ATTACK_EXPORTS
from wannado.adv.catalog.build_catalog import ADV_ROOT, load_catalog


@dataclass
class VictimInterface:
    """Single clean boundary adapter between any user project and wannado-adv."""

    model_or_fn: Callable[[torch.Tensor], torch.Tensor]
    x_eval: torch.Tensor
    y_eval: torch.Tensor
    norm: str = "linf"
    eps: float = 8.0 / 255.0
    clip_bounds: tuple[float, float] | None = (0.0, 1.0)
    domain: str = "vision"
    model_name: str = "VictimModel"
    dataset_name: str = "EvalSplit"
    defense_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    surrogate_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    extra_metadata: dict[str, Any] | None = None


def _resolve_attack_source_path(attack_id: str) -> tuple[dict[str, Any], Path]:
    cards = {c["id"]: c for c in load_catalog()}
    card = cards.get(attack_id)
    mod_path = card["module_path"] if card is not None else None
    if mod_path is None and attack_id in ATTACK_EXPORTS:
        mod_path, _ = ATTACK_EXPORTS[attack_id]
        mod = importlib.import_module(mod_path)
        meta_obj = getattr(mod, "META", None)
        if meta_obj is not None:
            card = meta_obj.to_dict()
    if card is None or mod_path is None:
        available = ", ".join(sorted(cards.keys())[:10])
        raise KeyError(
            f"Unknown attack '{attack_id}'. Sample valid IDs: {available} ... "
            "Use query_catalog() to browse all 35 attacks."
        )
    if mod_path.startswith("wannado.adv."):
        rel_parts = mod_path.removeprefix("wannado.adv.").split(".")
        src_path = (ADV_ROOT / Path(*rel_parts)).with_suffix(".py")
        if src_path.exists():
            return card, src_path
    mod = importlib.import_module(mod_path)
    if mod.__file__ is None:
        raise FileNotFoundError(f"Cannot locate source file for module '{mod_path}'.")
    return card, Path(mod.__file__).resolve()


def eject_attack(
    attack_id: str,
    dest_dir: str | Path = "./adv_eval/attacks",
    *,
    custom_name: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Copy a standalone CleanRL-style single-file attack into the user's workspace.

    The ejected file preserves complete verified academic provenance (``META.citation``),
    extension hooks (``[AI-Prompt Hook]``), and zero-relimport compatibility with
    ``wannado.adv.core`` so the user or AI IDE can upgrade that one file safely.
    """
    _, src_path = _resolve_attack_source_path(attack_id)
    target_dir = Path(dest_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = custom_name if custom_name is not None else src_path.name
    if not filename.endswith(".py"):
        filename = f"{filename}.py"
    dest_path = target_dir / filename

    if dest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing customized file at {dest_path}. "
            "Pass overwrite=True if you intentionally want to replace your local changes."
        )

    banner = (
        f"# [wannado-adv Ejected Single-File Attack: id='{attack_id}']\n"
        f"# Edit ONLY this standalone file to prototype improvements from new papers without\n"
        f"# touching the rest of your project. Remember to update `META` if you change\n"
        f"# authorship/citation or add hyperparameters for automatic paper reporting.\n\n"
    )
    dest_path.write_text(banner + src_path.read_text(encoding="utf-8"), encoding="utf-8")
    return dest_path


def _default_interface_template(domain: str, norm: str, eps: float, clip_bounds: str) -> str:
    if domain == "wireless":
        model_setup = (
            "class _PlaceholderComplexNet(torch.nn.Module):\n"
            "        def __init__(self) -> None:\n"
            "            super().__init__()\n"
            "            self.head = torch.nn.Linear(128, 4)\n\n"
            "        def forward(self, x_iq: torch.Tensor) -> torch.Tensor:\n"
            "            iq = torch.cat([x_iq.real, x_iq.imag], dim=-1).reshape(x_iq.shape[0], -1)\n"
            "            return self.head(iq)\n\n"
            "    model = _PlaceholderComplexNet().to(dev).eval()\n\n"
            "    # TODO: Replace with your real complex64 baseband IQ batch (shape (N, ...)) and labels:\n"
            "    x_eval = torch.complex(\n"
            "        torch.rand(num_samples, 1, 64, device=dev),\n"
            "        torch.rand(num_samples, 1, 64, device=dev),\n"
            "    ).to(torch.complex64)\n"
            "    y_eval = torch.zeros(num_samples, dtype=torch.long, device=dev)"
        )
    else:
        model_setup = (
            "model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(32, 4)).to(dev).eval()\n\n"
            "    # TODO: Replace with your real evaluation tensor batch (shape (N, ...)) and labels:\n"
            "    x_eval = torch.rand(num_samples, 32, device=dev)\n"
            "    y_eval = torch.zeros(num_samples, dtype=torch.long, device=dev)"
        )

    return f'''"""Isolated adversarial evaluation boundary for this project.

Connect your project's model, preprocessing, and evaluation batch inside
`load_victim_interface()` without scattering attack code across your training scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from wannado.adv.workspace import VictimInterface

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_victim_interface(
    device: str | torch.device = "cpu",
    num_samples: int = 100,
) -> VictimInterface:
    """Wire your trained model and evaluation batch into wannado-adv cleanly.

    Replace the placeholder model and inputs below with imports from your project.
    """
    dev = torch.device(device)
    # TODO: Replace with your project model import & checkpoint loading:
    #   from my_project.models import build_model
    #   model = build_model().to(dev).eval()
    {model_setup}

    return VictimInterface(
        model_or_fn=model,
        x_eval=x_eval,
        y_eval=y_eval,
        norm="{norm}",
        eps={eps!r},
        clip_bounds={clip_bounds},
        domain="{domain}",
        model_name="MyProjectModel",
        dataset_name="MyEvalSplit",
    )
'''


def _default_runner_template() -> str:
    return '''"""Standalone non-intrusive adversarial robustness evaluation entrypoint.

Run directly from your project root without modifying training/model source files:
    python adv_eval/run_eval.py --goal standard --samples 100
or as a module:
    python -m adv_eval.run_eval --goal standard --samples 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .interface import load_victim_interface
except ImportError:
    from interface import load_victim_interface

from wannado.adv.copilot import estimate_runtime, probe_model_context, recommend_suite
from wannado.adv.reporter import evaluate_suite


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated wannado-adv robustness evaluation.")
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--goal", type=str, default="standard", choices=["sanity", "standard", "paper_sota"])
    parser.add_argument("--no-audit", action="store_true", help="Skip Carlini false-robustness checks.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    victim = load_victim_interface(device=args.device, num_samples=args.samples)
    probe = probe_model_context(
        victim.model_or_fn,
        victim.x_eval,
        victim.y_eval,
        norm=victim.norm,
        eps=victim.eps,
        clip_bounds=victim.clip_bounds,
    )
    suite = recommend_suite(
        domain=victim.domain,
        goal=args.goal,
        autograd_ok=probe.autograd_connected,
        norm=victim.norm,
    )
    # Example customization hook: swap a built-in attack ID in `attacks` with an ejected local callable:
    #   from attacks.fgsm_goodfellow2015 import fgsm as custom_fgsm
    #   attacks = [custom_fgsm, *suite.attacks[1:]]
    attacks = suite.attacks

    eta = estimate_runtime(
        victim.model_or_fn,
        victim.x_eval,
        suite,
        num_samples=len(victim.x_eval),
        include_audit=(not args.no_audit),
    )
    print(eta.format_summary())

    report = evaluate_suite(
        victim.model_or_fn,
        victim.x_eval,
        victim.y_eval,
        eps=victim.eps,
        norm=victim.norm,
        attacks=attacks,
        attack_kwargs=suite.attack_kwargs,
        clip_bounds=victim.clip_bounds,
        run_audit=(not args.no_audit),
        seed=args.seed,
        domain=victim.domain,
        model_name=victim.model_name,
        dataset_name=victim.dataset_name,
    )
    out_dir = Path(__file__).resolve().parent / "outputs"
    saved = report.save(out_dir)
    print(report.format_paper_summary())
    print(f"\\nSaved paper reporting artifacts to: {out_dir} ({list(saved.keys())})")


if __name__ == "__main__":
    main()
'''


def scaffold_interface(
    target_dir: str | Path = "./adv_eval",
    *,
    domain: str = "vision",
    norm: str | None = None,
    eps: float | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Generate a self-contained, zero-intrusion adversarial evaluation harness folder."""
    root = Path(target_dir).resolve()
    attacks_dir = root / "attacks"
    root.mkdir(parents=True, exist_ok=True)
    attacks_dir.mkdir(parents=True, exist_ok=True)

    resolved_norm = norm or ("jsr_db" if domain == "wireless" else ("linf" if domain == "vision" else "relative_l2"))
    resolved_eps = (
        eps
        if eps is not None
        else (-10.0 if resolved_norm == "jsr_db" else (8.0 / 255.0 if resolved_norm == "linf" else 0.05))
    )
    clip_str = "(0.0, 1.0)" if domain == "vision" else "None"

    files_to_write: dict[Path, str] = {
        root / "interface.py": _default_interface_template(domain, resolved_norm, resolved_eps, clip_str),
        root / "run_eval.py": _default_runner_template(),
        attacks_dir / "__init__.py": (
            '"""Local customized single-file attacks ejected via `wannado-adv eject <attack_id>`."""\n'
        ),
    }

    written: dict[str, Path] = {}
    for fpath, content in files_to_write.items():
        if fpath.exists() and not overwrite:
            written[fpath.name] = fpath
            continue
        fpath.write_text(content, encoding="utf-8")
        written[fpath.relative_to(root).as_posix()] = fpath
    return written
