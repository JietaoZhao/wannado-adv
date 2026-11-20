"""Production runtime adapters (ONNX / TorchScript) & Quantization-Shift Differential Red-Teaming.

Addresses real-world deployment robustness where FP32 models that pass training-time checks
collapse after FP16/INT8 quantization or export to ONNX Runtime / TorchScript.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from wannado.adv.attacks.blackbox_score.square_andriushchenko2020 import square_attack
from wannado.adv.attacks.first_order.apgd_ce_croce2020 import apgd_ce
from wannado.adv.core.losses import predict_labels, unwrap_model_outputs
from wannado.adv.core.utils import eval_mode, infer_model_device, materialize_eval_tensors


class OnnxModelAdapter:
    """Wrap an ONNX model path or ``onnxruntime.InferenceSession`` as a ``wannado-adv`` victim callable.

    Requires optional ``onnxruntime`` only when instantiated (zero mandatory core dependency).
    Exposes ``self.model_path`` so ``compute_model_fingerprint`` hashes the exact binary artifact.
    """

    def __init__(
        self,
        model_path_or_session: str | Path | Any,
        *,
        input_name: str | None = None,
        output_index: int = 0,
        providers: Sequence[str] | None = None,
    ) -> None:
        self.model_path: Path | None = None
        if isinstance(model_path_or_session, (str, Path)):
            try:
                import onnxruntime as ort  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "OnnxModelAdapter requires optional `onnxruntime` (`pip install onnxruntime`)."
                ) from exc
            self.model_path = Path(model_path_or_session).resolve()
            if not self.model_path.is_file():
                raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")
            prov = list(providers) if providers is not None else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(self.model_path), providers=prov)
        else:
            self.session = model_path_or_session
            maybe_path = getattr(model_path_or_session, "_model_path", None)
            if maybe_path:
                self.model_path = Path(str(maybe_path)).resolve()

        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError("ONNX session exposes zero inputs.")
        self.input_name = input_name or str(inputs[0].name)
        self.output_index = int(output_index)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        if np.issubdtype(x_np.dtype, np.floating):
            x_np = x_np.astype(np.float32, copy=False)
        outputs = self.session.run(None, {self.input_name: x_np})
        out_arr = np.asarray(outputs[self.output_index])
        if np.issubdtype(out_arr.dtype, np.floating):
            return torch.from_numpy(out_arr.astype(np.float32, copy=False)).to(device=x.device)
        return torch.from_numpy(out_arr).to(device=x.device)


def wrap_onnx_model(
    model_path_or_session: str | Path | Any,
    *,
    input_name: str | None = None,
    output_index: int = 0,
    providers: Sequence[str] | None = None,
) -> OnnxModelAdapter:
    """Convenience constructor wrapping an ``.onnx`` artifact or ``onnxruntime.InferenceSession``."""
    return OnnxModelAdapter(
        model_path_or_session,
        input_name=input_name,
        output_index=output_index,
        providers=providers,
    )


class TorchScriptModelAdapter(nn.Module):
    """Wrap a TorchScript ``.pt`` archive or ``torch.jit.ScriptModule`` cleanly for red-teaming."""

    def __init__(
        self,
        path_or_module: str | Path | torch.jit.ScriptModule,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.model_path: Path | None = None
        if isinstance(path_or_module, (str, Path)):
            p = Path(path_or_module).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"TorchScript archive not found: {p}")
            self.model_path = p
            map_loc = torch.device(device) if device is not None else "cpu"
            self.script_module: torch.jit.ScriptModule = torch.jit.load(str(p), map_location=map_loc)
        else:
            self.script_module = path_or_module
            if device is not None:
                self.script_module = self.script_module.to(torch.device(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return unwrap_model_outputs(self.script_module(x))


def wrap_torchscript_model(
    path_or_module: str | Path | torch.jit.ScriptModule,
    *,
    device: str | torch.device | None = None,
) -> TorchScriptModelAdapter:
    """Convenience constructor wrapping a TorchScript module or serialized archive."""
    return TorchScriptModelAdapter(path_or_module, device=device)


def _quantize_tensor_symmetric(t: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = float((1 << (int(bits) - 1)) - 1)
    if t.ndim >= 2:
        dims = tuple(range(1, t.ndim))
        max_abs = t.detach().abs().amax(dim=dims, keepdim=True).clamp(min=1e-8)
    else:
        max_abs = t.detach().abs().amax().clamp(min=1e-8)
    scale = max_abs / qmax
    q = torch.round(t / scale).clamp(-qmax, qmax)
    return (q * scale).to(dtype=t.dtype)


class _SimulatedQuantizedWrapper(nn.Module):
    """Pure-PyTorch simulated quantized deployment wrapper (weight/activation INT4/INT8 or FP16)."""

    def __init__(self, base_model: nn.Module, *, bits: int = 8, mode: str = "weight_int") -> None:
        super().__init__()
        self.mode = str(mode)
        self.bits = int(bits)
        cloned = copy.deepcopy(base_model)
        cloned.eval()
        if self.mode in ("weight_int", "weight_and_act_int"):
            with torch.no_grad():
                for param in cloned.parameters():
                    if param.is_floating_point() and param.numel() > 1:
                        param.copy_(_quantize_tensor_symmetric(param, bits=self.bits))
        elif self.mode == "fp16":
            cloned = cloned.half()
        else:
            raise ValueError(f"Unsupported simulated quantization mode '{mode}'. Use 'weight_int', 'weight_and_act_int', or 'fp16'.")
        self.model = cloned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "fp16":
            return unwrap_model_outputs(self.model(x.half())).to(dtype=x.dtype)
        if self.mode == "weight_and_act_int":
            x_q = _quantize_tensor_symmetric(x, bits=self.bits)
            return unwrap_model_outputs(self.model(x_q))
        return unwrap_model_outputs(self.model(x))


def make_simulated_quantized_model(
    model: nn.Module,
    *,
    bits: int = 8,
    mode: str = "weight_int",
) -> nn.Module:
    """Create a pure-PyTorch quantized deployment copy (INT8/INT4 weight/activation or FP16) without external engines."""
    return _SimulatedQuantizedWrapper(model, bits=bits, mode=mode)


@dataclass
class QuantizationShiftReport:
    """Differential FP32-vs-Quantized deployment robustness audit report."""

    passed: bool
    num_samples: int
    norm: str
    eps: float
    fp32_clean_acc: float
    quant_clean_acc: float
    clean_drop_pp: float
    fp32_robust_acc: float
    quant_transfer_robust_acc: float
    quant_direct_robust_acc: float
    quant_worst_case_robust_acc: float
    robustness_drop_pp: float
    clean_prediction_flip_rate: float
    adv_prediction_flip_rate: float
    transfer_asr_fp32_to_quant: float
    red_flags: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_report(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"[wannado-adv Quantization-Shift Audit: {verdict}] (N={self.num_samples}, norm={self.norm}, eps={self.eps:.4g})",
            f"  Clean Acc  : FP32={self.fp32_clean_acc * 100:.2f}% -> Quant={self.quant_clean_acc * 100:.2f}% (drop={self.clean_drop_pp * 100:+.2f}pp)",
            (
                f"  Robust Acc : FP32={self.fp32_robust_acc * 100:.2f}% -> Quant(worst)={self.quant_worst_case_robust_acc * 100:.2f}% "
                f"(transfer={self.quant_transfer_robust_acc * 100:.2f}%, direct={self.quant_direct_robust_acc * 100:.2f}%, drop={self.robustness_drop_pp * 100:+.2f}pp)"
            ),
            f"  Flip Rates : clean_flip={self.clean_prediction_flip_rate * 100:.2f}%, adv_flip={self.adv_prediction_flip_rate * 100:.2f}%",
        ]
        if self.red_flags:
            lines.append("  Red Flags:")
            lines.extend(f"    - {rf}" for rf in self.red_flags)
        if self.recommendations:
            lines.append("  Recommendations:")
            lines.extend(f"    - {rec}" for rec in self.recommendations)
        return "\n".join(lines)


def audit_quantization_shift(
    fp32_model: Callable[[torch.Tensor], torch.Tensor],
    quantized_model: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor | Iterable[tuple[torch.Tensor, torch.Tensor]],
    y: torch.Tensor | None = None,
    *,
    eps: float = 8.0 / 255.0,
    norm: str = "linf",
    clip_bounds: tuple[Any, Any] | None = (0.0, 1.0),
    attack_steps: int = 15,
    square_queries: int = 150,
    max_samples: int = 64,
    max_robust_drop: float = 0.05,
    max_clean_drop: float = 0.03,
    seed: int = 42,
) -> QuantizationShiftReport:
    """Compare adversarial robustness of a reference FP32 model vs its quantized/exported runtime."""
    t0 = time.perf_counter()
    x_mat, y_mat = materialize_eval_tensors(x, y, max_samples=max_samples)
    dev = infer_model_device(fp32_model, x_mat.device)
    x_t = x_mat.to(dev)
    y_t = y_mat.to(dev).reshape(-1).to(torch.long)
    n = int(x_t.shape[0])
    if n < 2:
        raise ValueError("audit_quantization_shift requires at least 2 evaluation samples.")

    with eval_mode(fp32_model), eval_mode(quantized_model):
        with torch.no_grad():
            p_fp32_clean = predict_labels(fp32_model(x_t))
            p_quant_clean = predict_labels(quantized_model(x_t))

        fp32_clean_ok = p_fp32_clean == y_t
        quant_clean_ok = p_quant_clean == y_t
        fp32_clean_acc = float(fp32_clean_ok.float().mean().item())
        quant_clean_acc = float(quant_clean_ok.float().mean().item())
        clean_flip = float((p_fp32_clean != p_quant_clean).float().mean().item())

        # 1. White-box APGD-CE on FP32 model + transfer onto quantized runtime
        res_fp32 = apgd_ce(
            fp32_model,
            x_t,
            y_t,
            eps=eps,
            norm=norm,
            steps=attack_steps,
            clip_bounds=clip_bounds,
            seed=seed,
        )
        with torch.no_grad():
            p_fp32_adv = predict_labels(fp32_model(res_fp32.x_adv))
            p_quant_transfer = predict_labels(quantized_model(res_fp32.x_adv))

        fp32_rob_ok = p_fp32_adv == y_t
        quant_transfer_ok = p_quant_transfer == y_t
        fp32_robust_acc = float(fp32_rob_ok.float().mean().item())
        quant_transfer_acc = float(quant_transfer_ok.float().mean().item())
        adv_flip = float((p_fp32_adv != p_quant_transfer).float().mean().item())

        fp32_fooled_quant_clean = quant_clean_ok & (~quant_transfer_ok)
        n_quant_clean = int(quant_clean_ok.sum().item())
        transfer_asr = float(fp32_fooled_quant_clean.sum().item() / max(n_quant_clean, 1))

        # 2. Direct black-box Square Attack on quantized runtime (catches non-differentiable quant instabilities)
        if norm in ("linf", "l2"):
            res_q_direct = square_attack(
                quantized_model,
                x_t,
                y_t,
                eps=eps,
                norm=norm,
                max_queries=square_queries,
                clip_bounds=clip_bounds,
                seed=seed + 17,
            )
            with torch.no_grad():
                p_quant_direct = predict_labels(quantized_model(res_q_direct.x_adv))
            quant_direct_ok = p_quant_direct == y_t
        else:
            quant_direct_ok = quant_transfer_ok

        quant_direct_acc = float(quant_direct_ok.float().mean().item())
        quant_worst_ok = quant_transfer_ok & quant_direct_ok
        quant_worst_acc = float(quant_worst_ok.float().mean().item())

    clean_drop = fp32_clean_acc - quant_clean_acc
    robust_drop = fp32_robust_acc - quant_worst_acc

    red_flags: list[str] = []
    recommendations: list[str] = []
    if clean_drop > float(max_clean_drop) + 1e-9:
        red_flags.append(f"quant:clean_accuracy_drop({clean_drop:.4f}>{max_clean_drop:.4f})")
        recommendations.append(
            "Quantization caused excessive clean accuracy degradation; calibrate activation ranges or keep sensitive layers in FP16."
        )
    if robust_drop > float(max_robust_drop) + 1e-9:
        red_flags.append(f"quant:robustness_collapse({robust_drop:.4f}>{max_robust_drop:.4f})")
        recommendations.append(
            "Quantized deployment model lost significant adversarial robustness relative to FP32; apply Quantization-Aware Adversarial Training (QAAT) or mixed-precision protection."
        )

    return QuantizationShiftReport(
        passed=len(red_flags) == 0,
        num_samples=n,
        norm=norm,
        eps=float(eps),
        fp32_clean_acc=round(fp32_clean_acc, 4),
        quant_clean_acc=round(quant_clean_acc, 4),
        clean_drop_pp=round(clean_drop, 4),
        fp32_robust_acc=round(fp32_robust_acc, 4),
        quant_transfer_robust_acc=round(quant_transfer_acc, 4),
        quant_direct_robust_acc=round(quant_direct_acc, 4),
        quant_worst_case_robust_acc=round(quant_worst_acc, 4),
        robustness_drop_pp=round(robust_drop, 4),
        clean_prediction_flip_rate=round(clean_flip, 4),
        adv_prediction_flip_rate=round(adv_flip, 4),
        transfer_asr_fp32_to_quant=round(transfer_asr, 4),
        red_flags=red_flags,
        recommendations=recommendations,
        elapsed_seconds=round(time.perf_counter() - t0, 4),
    )


__all__ = [
    "OnnxModelAdapter",
    "QuantizationShiftReport",
    "TorchScriptModelAdapter",
    "audit_quantization_shift",
    "make_simulated_quantized_model",
    "wrap_onnx_model",
    "wrap_torchscript_model",
]
