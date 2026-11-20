"""Physical-layer RF / 6G AI-RAN adversarial threat primitives (Complex64 IQ domain).

Extends ``wannado-adv``'s native ``torch.complex64`` + ``jsr_db`` moat with hardware-realistic
over-the-air (OTA) wireless transmitter and channel constraints:
  - FFT-domain spectral mask projection (band-limited / ACLR-compliant jamming)
  - Peak-to-Average Power Ratio (PAPR) projection (PA hardware linearity constraint)
  - Differentiable Over-the-Air (OTA) baseband channel EoT transform (carrier phase + Rayleigh fading + AWGN)
"""

from __future__ import annotations

import math

import torch

from wannado.adv.core.projections import resolve_eps_budget, safe_l2_norm


def _ensure_complex_iq(signal: torch.Tensor) -> torch.Tensor:
    if signal.is_complex():
        return signal
    if signal.ndim >= 2 and signal.shape[1] == 2:
        # Support 2-channel real/imag representation (B, 2, L) -> complex (B, L)
        return torch.complex(signal[:, 0].contiguous(), signal[:, 1].contiguous())
    raise ValueError(
        f"Expected complex64/complex128 IQ tensor or (B, 2, L) I/Q tensor, got dtype={signal.dtype}, shape={tuple(signal.shape)}."
    )


def project_spectral_mask(
    delta_iq: torch.Tensor,
    passband_mask: torch.Tensor,
    *,
    x_clean: torch.Tensor | None = None,
    eps: float | None = None,
    norm: str = "jsr_db",
    fft_dim: int = -1,
) -> torch.Tensor:
    """Project complex RF jamming perturbation onto an FFT passband spectral mask + optional power budget.

    Args:
        delta_iq: Complex perturbation tensor ``(..., L)``.
        passband_mask: Real/boolean frequency-domain mask of shape ``(L,)`` or broadcastable to ``delta_iq``,
            with ``1`` on permitted in-band subcarriers/bins and ``0`` in guard/adjacent bands.
        x_clean: Optional clean signal for exact ``jsr_db`` / ``l2`` budget re-projection after masking.
        eps: Optional power budget (e.g., ``-10.0`` for ``norm="jsr_db"``).
        norm: Budget geometry (``"jsr_db"``, ``"l2"``, or ``"relative_l2"``).
        fft_dim: Time-axis dimension along which the FFT is computed.
    """
    if not delta_iq.is_complex():
        raise ValueError(f"project_spectral_mask requires a complex IQ tensor, got dtype={delta_iq.dtype}.")

    spec = torch.fft.fft(delta_iq, dim=fft_dim)
    mask = passband_mask.to(device=delta_iq.device, dtype=spec.real.dtype)
    while mask.ndim < spec.ndim:
        mask = mask.unsqueeze(0)
    masked_spec = spec * mask
    proj = torch.fft.ifft(masked_spec, dim=fft_dim).to(dtype=delta_iq.dtype)

    if eps is not None:
        ref = x_clean if x_clean is not None else delta_iq
        budget = resolve_eps_budget(ref, eps, norm)
        b = proj.shape[0]
        cur_l2 = safe_l2_norm(proj.reshape(b, -1), dim=1).view((b,) + (1,) * (proj.ndim - 1))
        scale = torch.minimum(torch.ones_like(cur_l2), budget / cur_l2.clamp(min=1e-12))
        proj = proj * scale.to(proj.dtype)
    return proj


def measure_out_of_band_ratio(
    delta_iq: torch.Tensor,
    passband_mask: torch.Tensor,
    *,
    fft_dim: int = -1,
) -> float:
    """Return fraction of perturbation spectral energy leaking outside ``passband_mask``."""
    spec = torch.fft.fft(delta_iq, dim=fft_dim)
    power = spec.real.pow(2) + spec.imag.pow(2)
    mask = passband_mask.to(device=delta_iq.device, dtype=power.dtype)
    while mask.ndim < power.ndim:
        mask = mask.unsqueeze(0)
    oob = (power * (1.0 - mask)).sum()
    tot = power.sum().clamp(min=1e-12)
    return float((oob / tot).item())


def measure_papr_db(signal_iq: torch.Tensor) -> torch.Tensor:
    """Compute per-sample Peak-to-Average Power Ratio (PAPR in dB) for complex IQ signals."""
    z = _ensure_complex_iq(signal_iq)
    b = z.shape[0]
    inst_p = (z.real.pow(2) + z.imag.pow(2)).reshape(b, -1)
    peak = inst_p.amax(dim=1)
    avg = inst_p.mean(dim=1).clamp(min=1e-12)
    return 10.0 * torch.log10((peak / avg).clamp(min=1.0))


def project_papr_constraint(
    delta_iq: torch.Tensor,
    max_papr_db: float,
    *,
    iters: int = 6,
    passband_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project complex RF jamming waveform to satisfy ``PAPR <= max_papr_db`` via iterative clipping."""
    if not delta_iq.is_complex():
        raise ValueError(f"project_papr_constraint requires a complex IQ tensor, got dtype={delta_iq.dtype}.")
    max_ratio = math.pow(10.0, float(max_papr_db) / 10.0)
    if max_ratio < 1.0:
        raise ValueError(f"max_papr_db must be >= 0.0 dB, got {max_papr_db}.")

    out = delta_iq.clone()
    b = out.shape[0]
    for _ in range(max(1, int(iters))):
        mag = out.abs()
        inst_p = (out.real.pow(2) + out.imag.pow(2)).reshape(b, -1)
        avg_p = inst_p.mean(dim=1).clamp(min=1e-12)
        max_amp = (avg_p * max_ratio).sqrt().view((b,) + (1,) * (out.ndim - 1))
        clipped_mag = torch.minimum(mag, max_amp)
        out = torch.sgn(out) * clipped_mag.to(out.dtype)
        if passband_mask is not None:
            out = project_spectral_mask(out, passband_mask)
    return out


def over_the_air_eot_transform(
    x_iq: torch.Tensor,
    *,
    snr_db: float | None = 20.0,
    max_phase_rad: float = math.pi,
    rayleigh_fading: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Differentiable Over-the-Air (OTA) baseband channel transform for ``eot_pgd`` / ``bpda_eot``.

    Applies random per-sample carrier phase offset, optional flat Rayleigh fading ``h ~ CN(0, 1)``,
    and calibrated complex AWGN while preserving autograd flow wrt ``x_iq``.
    """
    if not x_iq.is_complex():
        raise ValueError(f"over_the_air_eot_transform requires complex64/complex128 tensor, got {x_iq.dtype}.")

    b = x_iq.shape[0]
    real_dtype = torch.float32 if x_iq.dtype == torch.complex64 else torch.float64
    view = (b,) + (1,) * (x_iq.ndim - 1)
    y = x_iq

    if rayleigh_fading:
        hr = torch.randn(view, device=x_iq.device, dtype=real_dtype, generator=generator) / math.sqrt(2.0)
        hi = torch.randn(view, device=x_iq.device, dtype=real_dtype, generator=generator) / math.sqrt(2.0)
        h = torch.complex(hr, hi).to(x_iq.dtype)
        y = y * h
    elif max_phase_rad > 0.0:
        phi = (torch.rand(view, device=x_iq.device, dtype=real_dtype, generator=generator) * 2.0 - 1.0) * float(max_phase_rad)
        rot = torch.complex(torch.cos(phi), torch.sin(phi)).to(x_iq.dtype)
        y = y * rot

    if snr_db is not None:
        sig_p = (y.detach().real.pow(2) + y.detach().imag.pow(2)).reshape(b, -1).mean(dim=1).clamp(min=1e-12)
        noise_p = sig_p / math.pow(10.0, float(snr_db) / 10.0)
        sigma = (noise_p / 2.0).sqrt().view(view)
        nr = torch.randn(x_iq.shape, device=x_iq.device, dtype=real_dtype, generator=generator) * sigma
        ni = torch.randn(x_iq.shape, device=x_iq.device, dtype=real_dtype, generator=generator) * sigma
        y = y + torch.complex(nr, ni).to(x_iq.dtype)
    return y


__all__ = [
    "measure_out_of_band_ratio",
    "measure_papr_db",
    "over_the_air_eot_transform",
    "project_papr_constraint",
    "project_spectral_mask",
]
