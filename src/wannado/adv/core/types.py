"""Shared types and metadata schemas for wannado.adv."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

NormType = Literal["linf", "l2", "l1", "l0", "relative_l2", "jsr_db"]
TierType = Literal["S", "A", "B", "C"]
ThreatModelType = Literal[
    "white_box",
    "black_box_score",
    "black_box_decision",
    "generative_train",
]


@dataclass(frozen=True)
class AttackCitation:
    """Academic provenance record for an adversarial attack implementation."""

    key: str
    title: str
    authors: str
    venue: str
    year: int
    url: str
    bibtex: str
    ieee: str
    apa: str


@dataclass(frozen=True)
class AttackMeta:
    """Single-source-of-truth metadata card attached to every attack module."""

    id: str
    name: str
    family: str
    tier: TierType
    threat_model: ThreatModelType
    requires_training: bool
    compute_cost: Literal["very_low", "low", "medium", "high", "very_high"]
    supported_norms: tuple[str, ...]
    supported_domains: tuple[str, ...]
    parent_id: str | None
    diff_from_parent: str
    summary_plain: str
    when_to_use: str
    when_not_to_use: str
    extension_hooks: tuple[str, ...]
    citation: AttackCitation
    module_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata card into a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "tier": self.tier,
            "threat_model": self.threat_model,
            "requires_training": self.requires_training,
            "compute_cost": self.compute_cost,
            "supported_norms": list(self.supported_norms),
            "supported_domains": list(self.supported_domains),
            "parent_id": self.parent_id,
            "diff_from_parent": self.diff_from_parent,
            "summary_plain": self.summary_plain,
            "when_to_use": self.when_to_use,
            "when_not_to_use": self.when_not_to_use,
            "extension_hooks": list(self.extension_hooks),
            "module_path": self.module_path,
            "citation": {
                "key": self.citation.key,
                "title": self.citation.title,
                "authors": self.citation.authors,
                "venue": self.citation.venue,
                "year": self.citation.year,
                "url": self.citation.url,
                "bibtex": self.citation.bibtex,
                "ieee": self.citation.ieee,
                "apa": self.citation.apa,
            },
        }


@dataclass
class AttackResult:
    """Standardized return container produced by all wannado.adv attacks."""

    x_adv: torch.Tensor
    perturbation: torch.Tensor
    eps: float
    norm: str
    steps_run: int
    attack_id: str
    elapsed_ms: float
    extra: dict[str, Any] = field(default_factory=dict)
