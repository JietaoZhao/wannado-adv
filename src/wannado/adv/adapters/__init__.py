"""Production runtime, non-classifier task, tabular constraint, and physical RF threat adapters."""

from __future__ import annotations

from wannado.adv.adapters.deployment import (
    OnnxModelAdapter,
    QuantizationShiftReport,
    TorchScriptModelAdapter,
    audit_quantization_shift,
    make_simulated_quantized_model,
    wrap_onnx_model,
    wrap_torchscript_model,
)
from wannado.adv.adapters.rf_wireless import (
    measure_out_of_band_ratio,
    measure_papr_db,
    over_the_air_eot_transform,
    project_papr_constraint,
    project_spectral_mask,
)
from wannado.adv.adapters.tasks import (
    DetectionAttackAdapter,
    SegmentationAttackAdapter,
    TabularConstraintProjector,
    TreeEnsembleAdapter,
    VLMEmbeddingAdapter,
    compute_segmentation_metrics,
    detection_vanishing_loss,
    embedding_cosine_loss,
    evaluate_detection_robustness,
    wrap_tree_ensemble,
)

__all__ = [
    "DetectionAttackAdapter",
    "OnnxModelAdapter",
    "QuantizationShiftReport",
    "SegmentationAttackAdapter",
    "TabularConstraintProjector",
    "TorchScriptModelAdapter",
    "TreeEnsembleAdapter",
    "VLMEmbeddingAdapter",
    "audit_quantization_shift",
    "compute_segmentation_metrics",
    "detection_vanishing_loss",
    "embedding_cosine_loss",
    "evaluate_detection_robustness",
    "make_simulated_quantized_model",
    "measure_out_of_band_ratio",
    "measure_papr_db",
    "over_the_air_eot_transform",
    "project_papr_constraint",
    "project_spectral_mask",
    "wrap_onnx_model",
    "wrap_torchscript_model",
    "wrap_tree_ensemble",
]
