"""Non-classifier production task adapters: Detection, Segmentation, VLM Embeddings & Tabular/GBDT.

Breaks the classic CIFAR `(B, K)` classifier ceiling so `wannado-adv` natively red-teams:
  - Object detectors (YOLO-style `(B, N, 5+C)` tensors & torchvision `list[dict]` outputs)
  - Semantic segmentation models (`(B, C, H, W)`, HuggingFace `.logits`, torchvision `{"out": ...}`)
  - Vision-Language / foundation encoders (CLIP / SigLIP / DINOv2 embedding drift & hijacking)
  - Structured tabular domains (immutable masks, integer/one-hot/inequality constraints, XGBoost/LightGBM/CatBoost)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from wannado.adv.core.losses import unwrap_model_outputs
from wannado.adv.core.projections import clamp_to_bounds
from wannado.adv.core.utils import SklearnModelAdapter


# ---------------------------------------------------------------------------
# 1. Object Detection Adapter & Vanishing / Misclassification Losses
# ---------------------------------------------------------------------------
class DetectionAttackAdapter(nn.Module):
    """Normalize object detector outputs (YOLO `(B, N, 5+C)` or torchvision `list[dict]`) into attackable scores.

    Supports differentiable object-vanishing and targeted misclassification attacks when plugged
    into ``pgd``, ``apgd_ce``, ``eot_pgd``, or ``adv_patch`` with ``loss=detection_vanishing_loss``.
    """

    def __init__(
        self,
        detector: Callable[[torch.Tensor], Any],
        *,
        objectness_index: int = 4,
        class_start_index: int = 5,
        apply_sigmoid: bool = False,
    ) -> None:
        super().__init__()
        self._detector = detector
        if isinstance(detector, nn.Module):
            self.base_module: nn.Module | None = detector
        else:
            self.base_module = None
        self.objectness_index = int(objectness_index)
        self.class_start_index = int(class_start_index)
        self.apply_sigmoid = bool(apply_sigmoid)

    def extract_scores(self, raw_out: Any, batch_size: int, device: torch.device) -> torch.Tensor:
        """Extract differentiable per-box confidence scores of shape ``(B, N)`` or ``(B, N, C)``."""
        if isinstance(raw_out, torch.Tensor):
            if raw_out.ndim == 3 and raw_out.shape[-1] > self.objectness_index:
                obj = raw_out[:, :, self.objectness_index]
                if self.apply_sigmoid:
                    obj = torch.sigmoid(obj)
                if raw_out.shape[-1] > self.class_start_index:
                    cls_scores = raw_out[:, :, self.class_start_index :]
                    if self.apply_sigmoid:
                        cls_scores = torch.sigmoid(cls_scores)
                    return obj.unsqueeze(-1) * cls_scores  # (B, N, C)
                return obj
            return raw_out

        if isinstance(raw_out, (list, tuple)) and raw_out and isinstance(raw_out[0], dict):
            # Torchvision FasterRCNN / FCOS / RetinaNet output: list[dict[boxes, labels, scores]]
            max_n = max(1, max(int(d["scores"].numel()) for d in raw_out if "scores" in d))
            padded: list[torch.Tensor] = []
            for d in raw_out:
                sc = d.get("scores")
                if sc is None or sc.numel() == 0:
                    padded.append(torch.zeros((max_n,), device=device, dtype=torch.float32))
                else:
                    pad_len = max_n - int(sc.numel())
                    padded.append(F.pad(sc.reshape(-1), (0, pad_len)) if pad_len > 0 else sc.reshape(-1))
            return torch.stack(padded, dim=0)

        return unwrap_model_outputs(raw_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self._detector(x)
        return self.extract_scores(raw, batch_size=x.shape[0], device=x.device)


def detection_vanishing_loss(scores: torch.Tensor, _y: torch.Tensor) -> torch.Tensor:
    """Per-sample vanishing objective to MAXIMIZE (= negative sum/mean of max box confidences).

    Maximizing this loss drives all foreground object confidence scores toward 0 (object vanishing).
    """
    b = scores.shape[0]
    if scores.ndim == 3:
        box_conf = scores.amax(dim=-1)  # (B, N)
    else:
        box_conf = scores.reshape(b, -1)
    # Maximize -mean(box_conf) => suppress all detections
    return -box_conf.mean(dim=1)


def evaluate_detection_robustness(
    adapter: DetectionAttackAdapter,
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    *,
    score_thresh: float = 0.25,
) -> dict[str, float]:
    """Measure detection retention, vanishing rate, and confidence collapse under attack."""
    with torch.no_grad():
        s_clean = adapter(x_clean)
        s_adv = adapter(x_adv)
        c_conf = s_clean.amax(dim=-1) if s_clean.ndim == 3 else s_clean.reshape(x_clean.shape[0], -1)
        a_conf = s_adv.amax(dim=-1) if s_adv.ndim == 3 else s_adv.reshape(x_adv.shape[0], -1)
        n_clean = int((c_conf >= score_thresh).sum().item())
        n_adv = int((a_conf >= score_thresh).sum().item())
        vanish_rate = float(max(0, n_clean - n_adv) / max(n_clean, 1))
        score_drop = float((c_conf.mean() - a_conf.mean()).item())
    return {
        "clean_detections": float(n_clean),
        "adv_detections": float(n_adv),
        "vanishing_rate": round(vanish_rate, 4),
        "mean_score_drop": round(score_drop, 4),
    }


# ---------------------------------------------------------------------------
# 2. Semantic Segmentation Adapter & mIoU Evaluator
# ---------------------------------------------------------------------------
class SegmentationAttackAdapter(nn.Module):
    """Unwrap torchvision ``{"out": (B,C,H,W)}``, HF ``SemanticSegmenterOutput``, or raw masks cleanly."""

    def __init__(self, seg_model: Callable[[torch.Tensor], Any], *, output_key: str = "out") -> None:
        super().__init__()
        self._seg_model = seg_model
        if isinstance(seg_model, nn.Module):
            self.base_module: nn.Module | None = seg_model
        else:
            self.base_module = None
        self.output_key = str(output_key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self._seg_model(x)
        if isinstance(raw, dict):
            if self.output_key in raw:
                return unwrap_model_outputs(raw[self.output_key])
            return unwrap_model_outputs(next(iter(raw.values())))
        return unwrap_model_outputs(raw)


def compute_segmentation_metrics(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Compute exact pixel accuracy and macro mean IoU (mIoU) over ``(B, C, H, W)`` predictions."""
    preds = logits.argmax(dim=1).reshape(-1).to(torch.long)
    tgt = target_mask.reshape(-1).to(torch.long)
    c = int(num_classes) if num_classes is not None else int(logits.shape[1])
    pixel_acc = float((preds == tgt).float().mean().item())
    ious: list[float] = []
    for cls_id in range(c):
        p_c = preds == cls_id
        t_c = tgt == cls_id
        union = int((p_c | t_c).sum().item())
        if union == 0:
            continue
        inter = int((p_c & t_c).sum().item())
        ious.append(float(inter) / float(union))
    miou = float(np.mean(ious)) if ious else 0.0
    return {"pixel_acc": round(pixel_acc, 4), "miou": round(miou, 4)}


# ---------------------------------------------------------------------------
# 3. Multimodal Vision-Language / Foundation Encoder Adapter (CLIP / SigLIP / VLM)
# ---------------------------------------------------------------------------
class VLMEmbeddingAdapter(nn.Module):
    """Wrap any vision-language / foundation encoder ``encode_image(x) -> (B, D)`` for red-teaming.

    - With ``anchor_embeddings`` ``(K, D)`` (e.g., text prompt embeddings): outputs zero-shot
      cosine-similarity logits ``temperature * normalize(z) @ normalize(anchors).T`` of shape
      ``(B, K)``, directly compatible with all 35 attacks, ``auto_red_team``, and ``verify``.
    - Without ``anchor_embeddings``: returns L2-normalized embeddings ``(B, D)`` for direct
      untargeted embedding drift or targeted multimodal hijacking via ``embedding_cosine_loss``.
    """

    anchor_embeddings: torch.Tensor | None

    def __init__(
        self,
        image_encoder: Callable[[torch.Tensor], torch.Tensor],
        anchor_embeddings: torch.Tensor | None = None,
        *,
        temperature: float = 10.0,
    ) -> None:
        super().__init__()
        self._encoder = image_encoder
        if isinstance(image_encoder, nn.Module):
            self.base_module: nn.Module | None = image_encoder
        else:
            self.base_module = None
        self.temperature = float(temperature)
        if anchor_embeddings is not None:
            anc = F.normalize(anchor_embeddings.detach().to(torch.float32), dim=-1)
            self.register_buffer("anchor_embeddings", anc)
        else:
            self.anchor_embeddings = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = unwrap_model_outputs(self._encoder(x))
        z = F.normalize(raw.reshape(x.shape[0], -1), dim=-1)
        if self.anchor_embeddings is not None:
            anc = self.anchor_embeddings.to(device=z.device, dtype=z.dtype)
            return self.temperature * (z @ anc.T)
        return z


def embedding_cosine_loss(embeddings: torch.Tensor, reference_embeddings: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine distance ``1 - cos(z, z_ref)`` to MAXIMIZE for untargeted VLM drift (or negate for targeted hijack)."""
    z = F.normalize(embeddings.reshape(embeddings.shape[0], -1), dim=-1)
    r = F.normalize(reference_embeddings.reshape(reference_embeddings.shape[0], -1).to(device=z.device, dtype=z.dtype), dim=-1)
    return 1.0 - (z * r).sum(dim=-1)


# ---------------------------------------------------------------------------
# 4. Structured Tabular Domain Constraint Projector & Tree Ensemble Adapter
# ---------------------------------------------------------------------------
class TabularConstraintProjector:
    """Enforce realistic domain legality on adversarial tabular perturbations.

    Handles:
      - ``clip_bounds``: scalar or per-feature ``(low, high)`` bounds
      - ``immutable_indices``: columns strictly locked to clean values (e.g. age, ID, protected attributes)
      - ``integer_indices``: integer-valued features rounded back to valid integer grid
      - ``categorical_groups``: mutually exclusive one-hot column index groups projected to valid one-hot
      - ``inequality_pairs``: ordered pairs ``(lo_col, hi_col)`` enforcing ``x[:, lo_col] <= x[:, hi_col]``
      - ``custom_repair_fn``: optional domain-specific causal repair callback
    """

    def __init__(
        self,
        *,
        clip_bounds: tuple[Any, Any] | None = None,
        immutable_indices: Sequence[int] = (),
        integer_indices: Sequence[int] = (),
        categorical_groups: Sequence[Sequence[int]] = (),
        inequality_pairs: Sequence[tuple[int, int]] = (),
        custom_repair_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.clip_bounds = clip_bounds
        self.immutable_indices = tuple(int(i) for i in immutable_indices)
        self.integer_indices = tuple(int(i) for i in integer_indices)
        self.categorical_groups = tuple(tuple(int(i) for i in grp) for grp in categorical_groups)
        self.inequality_pairs = tuple((int(a), int(b)) for a, b in inequality_pairs)
        self.custom_repair_fn = custom_repair_fn

    def project(self, x_clean: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        """Project adversarial rows onto all configured tabular domain constraints."""
        out = x_adv.clone()
        if self.clip_bounds is not None:
            out = clamp_to_bounds(out, self.clip_bounds)
        if self.immutable_indices:
            idx = torch.tensor(self.immutable_indices, device=out.device, dtype=torch.long)
            out[:, idx] = x_clean[:, idx]
        if self.integer_indices:
            idx_int = torch.tensor(self.integer_indices, device=out.device, dtype=torch.long)
            out[:, idx_int] = torch.round(out[:, idx_int])
        for grp in self.categorical_groups:
            if not grp:
                continue
            g_idx = torch.tensor(grp, device=out.device, dtype=torch.long)
            winners = out[:, g_idx].argmax(dim=1)
            one_hot = F.one_hot(winners, num_classes=len(grp)).to(dtype=out.dtype, device=out.device)
            out[:, g_idx] = one_hot
        for lo_col, hi_col in self.inequality_pairs:
            violated = out[:, lo_col] > out[:, hi_col]
            if violated.any():
                if lo_col in self.immutable_indices and hi_col not in self.immutable_indices:
                    out[violated, hi_col] = out[violated, lo_col]
                elif hi_col in self.immutable_indices and lo_col not in self.immutable_indices:
                    out[violated, lo_col] = out[violated, hi_col]
                else:
                    mid = 0.5 * (out[violated, lo_col] + out[violated, hi_col])
                    out[violated, lo_col] = mid
                    out[violated, hi_col] = mid
        if self.custom_repair_fn is not None:
            out = self.custom_repair_fn(x_clean, out)
        if self.clip_bounds is not None:
            out = clamp_to_bounds(out, self.clip_bounds)
        if self.immutable_indices:
            idx = torch.tensor(self.immutable_indices, device=out.device, dtype=torch.long)
            out[:, idx] = x_clean[:, idx]
        return out

    def validate(self, x_clean: torch.Tensor, x_adv: torch.Tensor, *, tol: float = 1e-5) -> dict[str, Any]:
        """Verify domain legality across all constraints and report exact violation counts."""
        b = int(x_clean.shape[0])
        row_valid = torch.ones((b,), dtype=torch.bool, device=x_clean.device)
        violations: dict[str, int] = {}

        if self.immutable_indices:
            idx = torch.tensor(self.immutable_indices, device=x_clean.device, dtype=torch.long)
            bad = ((x_adv[:, idx] - x_clean[:, idx]).abs() > tol).any(dim=1)
            violations["immutable_modified"] = int(bad.sum().item())
            row_valid &= ~bad

        if self.integer_indices:
            idx_i = torch.tensor(self.integer_indices, device=x_clean.device, dtype=torch.long)
            bad_i = ((x_adv[:, idx_i] - torch.round(x_adv[:, idx_i])).abs() > tol).any(dim=1)
            violations["non_integer"] = int(bad_i.sum().item())
            row_valid &= ~bad_i

        cat_bad_total = 0
        for grp in self.categorical_groups:
            if not grp:
                continue
            g_idx = torch.tensor(grp, device=x_clean.device, dtype=torch.long)
            vals = x_adv[:, g_idx]
            is_bin = ((vals - 0.0).abs() <= tol) | ((vals - 1.0).abs() <= tol)
            sum_ok = (vals.sum(dim=1) - 1.0).abs() <= tol
            bad_c = (~is_bin.all(dim=1)) | (~sum_ok)
            cat_bad_total += int(bad_c.sum().item())
            row_valid &= ~bad_c
        if self.categorical_groups:
            violations["invalid_categorical"] = cat_bad_total

        ineq_bad_total = 0
        for lo_col, hi_col in self.inequality_pairs:
            bad_q = x_adv[:, lo_col] > x_adv[:, hi_col] + tol
            ineq_bad_total += int(bad_q.sum().item())
            row_valid &= ~bad_q
        if self.inequality_pairs:
            violations["inequality_violated"] = ineq_bad_total

        return {
            "all_valid": bool(row_valid.all().item()),
            "valid_fraction": round(float(row_valid.float().mean().item()), 4),
            "violations": violations,
        }


class TreeEnsembleAdapter(SklearnModelAdapter):
    """Unified black-box adapter for XGBoost, LightGBM, CatBoost, and scikit-learn tree ensembles.

    Handles both sklearn-style estimators (``.predict_proba`` / ``.predict``) and native
    ``xgboost.Booster`` / ``lightgbm.Booster`` objects without requiring any tree library at import time.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        prefer_proba: bool = True,
        prob_clip: float = 1e-6,
        projector: TabularConstraintProjector | None = None,
    ) -> None:
        super().__init__(estimator, prefer_proba=prefer_proba, prob_clip=prob_clip)
        self.projector = projector
        self._is_xgb_booster = type(estimator).__name__ == "Booster" and "xgboost" in type(estimator).__module__
        self._is_lgb_booster = type(estimator).__name__ == "Booster" and "lightgbm" in type(estimator).__module__

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_xgb_booster:
            import xgboost as xgb  # type: ignore[import-not-found]

            arr = x.detach().cpu().numpy().reshape(x.shape[0], -1)
            raw = np.asarray(self.estimator.predict(xgb.DMatrix(arr)))
            return self._array_to_logits(raw, x.device)
        if self._is_lgb_booster:
            arr = x.detach().cpu().numpy().reshape(x.shape[0], -1)
            raw = np.asarray(self.estimator.predict(arr))
            return self._array_to_logits(raw, x.device)
        return super().__call__(x)

    def _array_to_logits(self, raw: np.ndarray, device: torch.device) -> torch.Tensor:
        if raw.ndim == 1:
            p = np.clip(raw.astype(np.float32), 1e-7, 1.0 - 1e-7)
            logits = np.stack([np.log(1.0 - p), np.log(p)], axis=1)
            return torch.from_numpy(logits).to(device=device)
        p = np.clip(raw.astype(np.float32), 1e-7, 1.0)
        return torch.from_numpy(np.log(p)).to(device=device)


def wrap_tree_ensemble(
    estimator: Any,
    *,
    prefer_proba: bool = True,
    prob_clip: float = 1e-6,
    projector: TabularConstraintProjector | None = None,
) -> TreeEnsembleAdapter:
    """Wrap any XGBoost / LightGBM / CatBoost / sklearn tree model for black-box red-teaming."""
    return TreeEnsembleAdapter(
        estimator,
        prefer_proba=prefer_proba,
        prob_clip=prob_clip,
        projector=projector,
    )


__all__ = [
    "DetectionAttackAdapter",
    "SegmentationAttackAdapter",
    "TabularConstraintProjector",
    "TreeEnsembleAdapter",
    "VLMEmbeddingAdapter",
    "compute_segmentation_metrics",
    "detection_vanishing_loss",
    "embedding_cosine_loss",
    "evaluate_detection_robustness",
    "wrap_tree_ensemble",
]
