"""Shared dataclasses and metadata types for single-file modular adversarial defenses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from torch import nn

from wannado.adv.core.types import AttackCitation


@dataclass(frozen=True)
class DefenseMeta:
    """Self-describing academic & engineering metadata embedded in every defense file."""

    id: str
    name: str
    paradigm: str
    tier: str
    supported_norms: tuple[str, ...]
    supported_domains: tuple[str, ...]
    summary_plain: str
    when_to_use: str
    when_not_to_use: str
    module_path: str
    citation: AttackCitation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DefenseTrainResult:
    """Structured outcome returned by single-file adversarial training loops."""

    defense_id: str
    model: nn.Module
    epochs_run: int
    elapsed_seconds: float
    history: list[dict[str, float]] = field(default_factory=list)
    final_clean_acc: float | None = None
    final_adv_acc: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    citation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "defense_id": self.defense_id,
            "epochs_run": self.epochs_run,
            "elapsed_seconds": self.elapsed_seconds,
            "history": list(self.history),
            "final_clean_acc": self.final_clean_acc,
            "final_adv_acc": self.final_adv_acc,
            "extra": dict(self.extra),
            "citation": dict(self.citation),
        }


@dataclass
class SmoothingCertificationResult:
    """Certified L2 robustness summary returned by ``certify_randomized_smoothing()``."""

    defense_id: str
    sigma: float
    alpha: float
    n0: int
    n: int
    num_samples: int
    clean_smoothed_acc: float | None
    abstain_rate: float
    median_certified_radius: float
    mean_certified_radius: float
    certified_acc_at_radii: dict[str, float] = field(default_factory=dict)
    predictions: list[int] = field(default_factory=list)
    radii: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    citation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["DefenseMeta", "DefenseTrainResult", "SmoothingCertificationResult"]
