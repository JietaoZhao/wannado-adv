"""Core shared primitives (types, projections, loss bridges) for wannado.adv."""

from wannado.adv.core.losses import (
    compute_attack_objective,
    normalize_classification_logits,
    outputs_to_objective,
    predict_labels,
    resolve_loss_fn,
    unwrap_model_outputs,
)
from wannado.adv.core.projections import (
    adam_step_direction,
    clamp_to_bounds,
    coerce_clip_bounds,
    compute_perturbation_norm,
    project_discrete_features,
    project_perturbation,
    random_init_perturbation,
    resolve_eps_budget,
    safe_l2_norm,
    step_direction,
)
from wannado.adv.core.types import AttackMeta, AttackResult, NormType
from wannado.adv.core.utils import (
    DefendedModelWrapper,
    SklearnModelAdapter,
    eval_mode,
    iter_eval_batches,
    make_generator,
    materialize_eval_tensors,
    update_best_by_objective,
    wrap_sklearn_model,
)

__all__ = [
    "AttackMeta",
    "AttackResult",
    "DefendedModelWrapper",
    "NormType",
    "SklearnModelAdapter",
    "adam_step_direction",
    "clamp_to_bounds",
    "coerce_clip_bounds",
    "compute_attack_objective",
    "compute_perturbation_norm",
    "eval_mode",
    "iter_eval_batches",
    "make_generator",
    "materialize_eval_tensors",
    "normalize_classification_logits",
    "outputs_to_objective",
    "predict_labels",
    "project_discrete_features",
    "project_perturbation",
    "random_init_perturbation",
    "resolve_eps_budget",
    "resolve_loss_fn",
    "safe_l2_norm",
    "step_direction",
    "unwrap_model_outputs",
    "update_best_by_objective",
    "wrap_sklearn_model",
]
