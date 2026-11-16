"""Build and query the wannado.adv algorithm knowledge graph from single-file META headers."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from wannado.adv.core.types import AttackMeta

ADV_ROOT = Path(__file__).resolve().parent.parent
ATTACKS_DIR = ADV_ROOT / "attacks"
REGISTRY_PATH = Path(__file__).resolve().parent / "registry.json"


def discover_attack_metas() -> list[dict[str, Any]]:
    """Scan all CleanRL-style attack modules and extract their ``META`` cards."""
    cards: list[dict[str, Any]] = []
    for file_path in sorted(ATTACKS_DIR.rglob("*.py")):
        if file_path.name.startswith("_"):
            continue
        rel = file_path.relative_to(ADV_ROOT).with_suffix("")
        module_name = "wannado.adv." + ".".join(rel.parts)
        module = importlib.import_module(module_name)
        meta = getattr(module, "META", None)
        if not isinstance(meta, AttackMeta):
            raise TypeError(
                f"Attack module '{module_name}' must define META as an AttackMeta instance."
            )
        card = meta.to_dict()
        if not card.get("module_path"):
            card["module_path"] = module_name
        cards.append(card)

    cards.sort(key=lambda c: (c["family"], c["id"]))
    return cards


def build_registry_json(output_path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Generate ``registry.json`` directly from source module metadata."""
    cards = discover_attack_metas()
    payload = {
        "schema_version": "1.0",
        "count": len(cards),
        "algorithms": cards,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def load_catalog(rebuild_if_missing: bool = True) -> list[dict[str, Any]]:
    """Load the serialized algorithm knowledge graph."""
    if REGISTRY_PATH.exists():
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return list(data.get("algorithms", []))
    if rebuild_if_missing:
        return discover_attack_metas()
    raise FileNotFoundError(f"Catalog registry not found at {REGISTRY_PATH}.")


def query_catalog(
    *,
    family: str | None = None,
    tier: str | None = None,
    domain: str | None = None,
    norm: str | None = None,
    threat_model: str | None = None,
    requires_training: bool | None = None,
) -> list[dict[str, Any]]:
    """Filter algorithm metadata cards by domain, threat model, norm, or trust tier."""
    cards = load_catalog()
    results: list[dict[str, Any]] = []
    for card in cards:
        if family is not None and card["family"] != family:
            continue
        if tier is not None and card["tier"] != tier:
            continue
        if threat_model is not None and card["threat_model"] != threat_model:
            continue
        if requires_training is not None and card["requires_training"] != requires_training:
            continue
        if domain is not None and domain not in card["supported_domains"]:
            continue
        if norm is not None and norm not in card["supported_norms"]:
            continue
        results.append(card)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the wannado.adv algorithm catalog.")
    parser.add_argument("--build", action="store_true", help="Rebuild registry.json from attack modules.")
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--tier", type=str, default=None)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--threat-model", type=str, default=None)
    args = parser.parse_args()

    if args.build or not REGISTRY_PATH.exists():
        payload = build_registry_json()
        print(f"Built {REGISTRY_PATH} with {payload['count']} algorithms.")

    matches = query_catalog(
        family=args.family,
        tier=args.tier,
        domain=args.domain,
        norm=args.norm,
        threat_model=args.threat_model,
    )
    print(json.dumps(matches, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
