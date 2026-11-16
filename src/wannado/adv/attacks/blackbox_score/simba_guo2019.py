"""SimBA: Simple Black-box Adversarial Attacks (Guo et al., ICML 2019).

===============================================================================
[Academic Provenance & Baseline Role]
- Tier          : S (Minimalist Query-Efficient Orthonormal Probing Baseline)
- Reference     : Guo, C., Gardner, J. R., You, Y., Wilson, A. G., & Weinberger,
                  K. Q. (2019). "Simple Black-box Adversarial Attacks." ICML 2019,
                  PMLR 97:2484-2493.
- URL           : https://arxiv.org/abs/1905.07121
- Plain Intuition:
    SimBA needs no gradient estimation at all. Starting from clean ``x``, it
    picks a unit direction ``q`` from a random orthonormal basis (standard
    coordinates, or low-frequency orthonormal vectors) and checks: "Does adding
    ``step * q`` increase attack objective (e.g. drop true-class probability)?"
    If yes, keep it! If not, try subtracting ``step * q``. Repeat for remaining
    un-fooled samples until fooled or query budget runs out.
- Diff vs. NES / SPSA / Square Attack:
    Tests single orthogonal steps +/- step * q directly without averaging noisy
    gradient probes; natively tracks exact L2 / Linf / JSR-dB budget bounds.
- Why Chosen as a Baseline:
    Famous <20-line core baseline in ICML 2019 showing simple greedy orthogonal
    search often beats complex black-box gradient estimators in query efficiency.
- Extension Directions / Prompt Hooks:
    1. Pass domain-orthogonal basis Q (e.g., DCT or DFT frequency basis for RF
       or 4D images) instead of standard Cartesian coordinates.
    2. Tune ``step_size`` relative to budget ``eps`` per query regime.
===============================================================================
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

from wannado.adv.core import (
    AttackMeta,
    AttackResult,
    compute_attack_objective,
    eval_mode,
    make_generator,
    project_perturbation,
    resolve_eps_budget,
    resolve_loss_fn,
)
from wannado.adv.core.losses import LossFn
from wannado.adv.core.types import AttackCitation

META = AttackMeta(
    id="simba",
    name="Simple Black-Box Attack (SimBA)",
    family="blackbox_score",
    tier="S",
    threat_model="black_box_score",
    requires_training=False,
    compute_cost="medium",
    supported_norms=("l2", "linf", "relative_l2", "jsr_db"),
    supported_domains=("vision", "wireless", "tabular", "timeseries", "drl"),
    parent_id=None,
    diff_from_parent="Greedy +/- orthonormal basis probing over confidence scores without finite-difference gradient estimators.",
    summary_plain="Greedy orthonormal-basis black-box attack that flips predictions by testing +/- single directions without estimating gradients.",
    when_to_use="Minimalist query-efficient score-based baseline when simplicity and strict monotonic confidence reduction are desired.",
    when_not_to_use="Under strict hard-label-only access (use boundary or hopskipjump) or when gradients are available (use APGD).",
    extension_hooks=(
        "Substitute Cartesian unit basis with 2D-DCT or 1D-FFT orthonormal frequency basis (SimBA-DCT / SimBA-FFT)",
        "Combine cosine/phase step scaling for complex wireless sub-carriers",
    ),
    module_path="wannado.adv.attacks.blackbox_score.simba_guo2019",
    citation=AttackCitation(
        key="guo2019simple",
        title="Simple Black-box Adversarial Attacks",
        authors="Chuan Guo, Jacob R. Gardner, Yurong You, Andrew Gordon Wilson, and Kilian Q. Weinberger",
        venue="36th International Conference on Machine Learning (ICML)",
        year=2019,
        url="https://arxiv.org/abs/1905.07121",
        bibtex=(
            "@inproceedings{guo2019simple,\n"
            "  title={Simple Black-box Adversarial Attacks},\n"
            "  author={Guo, Chuan and Gardner, Jacob R. and You, Yurong and Wilson, Andrew Gordon and Weinberger, Kilian Q.},\n"
            "  booktitle={Proceedings of the 36th International Conference on Machine Learning},\n"
            "  pages={2484--2493},\n"
            "  year={2019},\n"
            "  volume={97},\n"
            "  series={Proceedings of Machine Learning Research},\n"
            "  publisher={PMLR},\n"
            "  url={https://arxiv.org/abs/1905.07121},\n"
            "}"
        ),
        ieee='C. Guo, J. R. Gardner, Y. You, A. G. Wilson, and K. Q. Weinberger, "Simple black-box adversarial attacks," in Proc. 36th Int. Conf. Mach. Learn. (ICML), PMLR 97, 2019, pp. 2484-2493.',
        apa="Guo, C., Gardner, J. R., You, Y., Wilson, A. G., & Weinberger, K. Q. (2019). Simple black-box adversarial attacks. Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97, 2484-2493.",
    ),
)


def _simba_score(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x_in: torch.Tensor,
    y: torch.Tensor,
    targeted: bool,
    loss_fn: LossFn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return continuous attack score (higher = stronger attack) and fooled mask."""
    logits = model_or_fn(x_in)
    preds = logits.argmax(dim=1)
    fooled = (preds == y) if targeted else (preds != y)

    # Combine softmax confidence drop with logit objective for numerical sensitivity even at high confidence.
    probs = F.softmax(logits, dim=1)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    target_prob = probs[batch_idx, y]
    prob_obj = target_prob if targeted else (1.0 - target_prob)
    margin_obj = -loss_fn(logits, y) if targeted else loss_fn(logits, y)
    score = prob_obj + 1e-3 * torch.tanh(margin_obj)
    return score, fooled


def simba(
    model_or_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.03,
    step_size: float | None = None,
    max_queries: int = 600,
    norm: str = "linf",
    loss: str | LossFn | None = "margin",
    targeted: bool = False,
    clip_bounds: tuple[float, float] | None = None,
    seed: int | None = None,
) -> AttackResult:
    """Run Simple Black-box Attack (SimBA) probing random orthogonal directions."""
    if max_queries < 1:
        raise ValueError(f"max_queries must be >= 1, got {max_queries}.")

    t0 = time.perf_counter()
    x_clean = x.detach()
    batch_size = x_clean.shape[0]
    eps_budget = resolve_eps_budget(x_clean, eps=eps, norm=norm)

    if (eps_budget == 0).all():
        return AttackResult(
            x_adv=x_clean.clone(),
            perturbation=torch.zeros_like(x_clean),
            eps=eps,
            norm=norm,
            steps_run=0,
            attack_id=META.id,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            extra={"queries": 0, "steps_accepted": 0},
        )

    is_cplx = x_clean.is_complex()
    elem_dim = x_clean[0].numel()
    eff_dim = elem_dim * 2 if is_cplx else elem_dim
    real_dtype = x_clean.real.dtype if is_cplx else x_clean.dtype

    if step_size is None:
        if norm == "linf":
            alpha_vec = eps_budget.reshape(batch_size, 1) * 0.5
        else:
            alpha_vec = eps_budget.reshape(batch_size, 1) / math.sqrt(max(min(eff_dim, 64), 1)) * 1.5
    else:
        alpha_vec = torch.full((batch_size, 1), float(step_size), device=x_clean.device, dtype=real_dtype)

    loss_fn = resolve_loss_fn(loss, targeted=targeted)
    generator = make_generator(seed, x_clean.device)

    delta_flat = torch.zeros((batch_size, eff_dim), device=x_clean.device, dtype=real_dtype)

    def _unpack(flat: torch.Tensor) -> torch.Tensor:
        if is_cplx:
            return torch.complex(
                flat[:, :elem_dim].reshape_as(x_clean.real),
                flat[:, elem_dim:].reshape_as(x_clean.imag),
            ).to(x_clean.dtype)
        return flat.reshape_as(x_clean)

    def _pack(t: torch.Tensor) -> torch.Tensor:
        if is_cplx:
            return torch.cat([t.real.reshape(batch_size, -1), t.imag.reshape(batch_size, -1)], dim=1)
        return t.reshape(batch_size, -1)

    with eval_mode(model_or_fn), torch.no_grad():
        cur_score, fooled = _simba_score(model_or_fn, x_clean, y, targeted=targeted, loss_fn=loss_fn)
        queries = 1
        iter_steps = 0
        accepted = 0

        perm_cycle = torch.randperm(eff_dim, device=x_clean.device, generator=generator)
        ptr = 0

        while queries < max_queries and not fooled.all():
            active = ~fooled
            active_idx = active.nonzero(as_tuple=False).squeeze(1)
            if active_idx.numel() == 0:
                break

            if ptr >= eff_dim:
                perm_cycle = torch.randperm(eff_dim, device=x_clean.device, generator=generator)
                ptr = 0
            coord = int(perm_cycle[ptr].item())
            ptr += 1
            iter_steps += 1

            # [AI-Prompt Hook]: Replace Cartesian basis `q[:, coord] = 1` with orthogonal 2D-DCT or DFT
            # basis vectors for SimBA-DCT (Guo et al., 2019, Section 3.2) / spectral SimBA.
            step_delta = torch.zeros_like(delta_flat)
            step_delta[active_idx, coord] = alpha_vec[active_idx, 0]

            cand_plus = project_perturbation(
                _unpack(delta_flat + step_delta),
                x_clean,
                eps_budget,
                norm=norm,
                clip_bounds=clip_bounds,
            )
            score_p, fooled_p = _simba_score(
                model_or_fn,
                x_clean[active_idx] + cand_plus[active_idx],
                y[active_idx],
                targeted=targeted,
                loss_fn=loss_fn,
            )
            queries += 1

            improved_p = score_p > cur_score[active_idx]
            if improved_p.any():
                accepted += int(improved_p.sum().item())
                acc_rows = active_idx[improved_p]
                delta_flat[acc_rows] = _pack(cand_plus)[acc_rows]
                cur_score[acc_rows] = score_p[improved_p]
                fooled[acc_rows] = fooled_p[improved_p]

            still_idx = active_idx[~improved_p]
            if still_idx.numel() > 0 and queries < max_queries:
                cand_minus = project_perturbation(
                    _unpack(delta_flat - step_delta),
                    x_clean,
                    eps_budget,
                    norm=norm,
                    clip_bounds=clip_bounds,
                )
                score_m, fooled_m = _simba_score(
                    model_or_fn,
                    x_clean[still_idx] + cand_minus[still_idx],
                    y[still_idx],
                    targeted=targeted,
                    loss_fn=loss_fn,
                )
                queries += 1

                improved_m = score_m > cur_score[still_idx]
                if improved_m.any():
                    accepted += int(improved_m.sum().item())
                    acc_rows = still_idx[improved_m]
                    delta_flat[acc_rows] = _pack(cand_minus)[acc_rows]
                    cur_score[acc_rows] = score_m[improved_m]
                    fooled[acc_rows] = fooled_m[improved_m]

    delta = project_perturbation(
        _unpack(delta_flat), x_clean, eps_budget, norm=norm, clip_bounds=clip_bounds
    )
    return AttackResult(
        x_adv=x_clean + delta,
        perturbation=delta,
        eps=eps,
        norm=norm,
        steps_run=iter_steps,
        attack_id=META.id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={
            "queries": queries,
            "steps_accepted": accepted,
            "fooling_rate": float(fooled.float().mean().item()),
        },
    )
