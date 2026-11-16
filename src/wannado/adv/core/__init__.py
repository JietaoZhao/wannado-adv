"""Core shared primitives (types, projections, loss bridges) for wannado.adv."""

from wannado.adv.core.losses import compute_attack_objective, outputs_to_objective, resolve_loss_fn
from wannado.adv.core.projections import (
    adam_step_direction,
    compute_perturbation_norm,
    project_perturbation,
    random_init_perturbation,
    resolve_eps_budget,
    safe_l2_norm,
    step_direction,
)
from wannado.adv.core.types import AttackMeta, AttackResult, NormType
from wannado.adv.core.utils import eval_mode, make_generator, update_best_by_objective

__all__ = [
    "AttackMeta",
    "AttackResult",
    "NormType",
    "adam_step_direction",
    "compute_attack_objective",
    "compute_perturbation_norm",
    "eval_mode",
    "make_generator",
    "outputs_to_objective",
    "project_perturbation",
    "random_init_perturbation",
    "resolve_eps_budget",
    "resolve_loss_fn",
    "safe_l2_norm",
    "step_direction",
    "update_best_by_objective",
]
