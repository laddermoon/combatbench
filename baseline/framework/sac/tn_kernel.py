"""N-SAC-01: numerically stable truncated-normal kernel on (-1, 1).

SAC-owned kernel (D11).  The previous PPO-side kernel computed
``Z = Φ(b) - Φ(a)`` with fixed clamps, which collapses for large σ
(uniform limit) and narrow supports.  This kernel instead:

- works in **erf space**: ``E(x) = erf((x - μ) / (σ√2))`` is a monotone
  bijection of the support onto ``(E_lo, E_hi)``;
- computes ``Z = (E_hi - E_lo) / 2`` with a sign-aware formulation
  (``erfc`` when both bounds share a sign) to avoid cancellation;
- samples by interpolating in erf space then applying ``erfinv``, so no
  division by a possibly-tiny ``Z`` is needed;
- evaluates ``log_prob`` with a stably-computed ``log Z``;
- counts every protection/clamp event instead of hiding them, and
  raises on non-finite parameters or degenerate inputs.

All probability-mass internals run in float64 regardless of the input
dtype; outputs are cast back.  A distribution whose erf-space bounds are
indistinguishable in float64 is *reported* via ``KERNEL_STATS`` rather
than silently widened.
"""
from __future__ import annotations

import math
import threading
from typing import Dict, Tuple

import torch

_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_LOG_SQRT_2PI = math.log(_SQRT_2PI)

# float64 erf saturates to ±1 at |x| ≳ 6.16; erfinv domain guard.
_ERFINV_MAX = float(torch.erfinv(torch.tensor(0.9999999999999999, dtype=torch.float64)))


class TNKernelStats:
    """Process-wide protection counters for the TN kernel.

    Counters are monotonic within a process; tests and diagnostics read a
    snapshot via ``snapshot()`` or reset via ``reset()``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {
            "erfinv_endpoint_clamp": 0,
            "u_endpoint_clamp": 0,
            "dtype_endpoint_clamp": 0,
            "narrow_support_unresolvable": 0,
            "log_z_fallback_logspace": 0,
        }

    def bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + int(n)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def reset(self) -> None:
        with self._lock:
            for k in self._counts:
                self._counts[k] = 0


KERNEL_STATS = TNKernelStats()


def _check_params(mu: torch.Tensor, sigma: torch.Tensor) -> None:
    if not torch.isfinite(mu).all():
        raise ValueError("tn_kernel: non-finite mu")
    if not torch.isfinite(sigma).all():
        raise ValueError("tn_kernel: non-finite sigma")
    if not (sigma > 0).all():
        raise ValueError("tn_kernel: sigma must be > 0")
    if not ((mu > -1.0) & (mu < 1.0)).all():
        raise ValueError("tn_kernel: mu must lie strictly inside (-1, 1)")


def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float64) if not t.requires_grad else t.to(torch.float64)


def erf_bounds(mu: torch.Tensor, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return erf-space bounds ``(E_lo, E_hi)`` of the support (-1, 1).

    ``E_lo = erf((-1-μ)/(σ√2))``, ``E_hi = erf((1-μ)/(σ√2))``.
    Differentiable w.r.t. mu/sigma; internally computed in float64.
    """
    _check_params(mu, sigma)
    mu64, s64 = mu.to(torch.float64), sigma.to(torch.float64)
    e_lo = torch.erf((-1.0 - mu64) / (s64 * _SQRT2))
    e_hi = torch.erf((1.0 - mu64) / (s64 * _SQRT2))
    if (e_lo == e_hi).any():
        KERNEL_STATS.bump(
            "narrow_support_unresolvable",
            int((e_lo == e_hi).sum().item()),
        )
    return e_lo.to(mu.dtype), e_hi.to(mu.dtype)


def log_normalizer(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Return ``log Z`` where ``Z = Φ(b) - Φ(a)`` over the support.

    Uses the sign-aware stable form:
      - a ≤ 0 ≤ b:  Z = (erf(b̂) - erf(â)) / 2  (terms have opposite sign)
      - both > 0:   Z = (erfc(â) - erfc(b̂)) / 2
      - both < 0:   Z = (erfc(-b̂) - erfc(-â)) / 2
    Differentiable w.r.t. mu/sigma.
    """
    _check_params(mu, sigma)
    mu64, s64 = mu.to(torch.float64), sigma.to(torch.float64)
    a = (-1.0 - mu64) / (s64 * _SQRT2)
    b = (1.0 - mu64) / (s64 * _SQRT2)

    z_mid = 0.5 * (torch.erf(b) - torch.erf(a))
    z_pos = 0.5 * (torch.erfc(a) - torch.erfc(b))
    z_neg = 0.5 * (torch.erfc(-b) - torch.erfc(-a))

    z = torch.where(
        (a <= 0.0) & (b >= 0.0),
        z_mid,
        torch.where(a > 0.0, z_pos, z_neg),
    )
    # Residual risk: Z may still underflow to 0 in float64 for extreme
    # narrow supports — report via the narrow-support counter.
    unresolvable = z <= 0.0
    if unresolvable.any():
        KERNEL_STATS.bump(
            "narrow_support_unresolvable", int(unresolvable.sum().item())
        )
        KERNEL_STATS.bump("log_z_fallback_logspace", int(unresolvable.sum().item()))
        z = torch.where(unresolvable, torch.full_like(z, torch.finfo(torch.float64).tiny), z)
    return torch.log(z).to(mu.dtype)


def normalizer(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Return ``Z`` (probability mass of the untruncated Normal inside (-1,1))."""
    return torch.exp(log_normalizer(mu, sigma))


def sample_from_uniform(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Inverse-CDF sample: ``u`` uniform on (0,1) → action in (-1,1).

    Interpolates in erf space: ``E = (1-u)·E_lo + u·E_hi`` then
    ``a = μ + σ√2·erfinv(E)``.  Differentiable w.r.t. mu/sigma (u is a
    constant noise input, no grad).  Endpoint hits of ``u`` or saturated
    erfinv inputs are counted, never silently absorbed.
    """
    _check_params(mu, sigma)
    if not torch.isfinite(u).all():
        raise ValueError("tn_kernel: non-finite u")
    if ((u < 0.0) | (u > 1.0)).any():
        raise ValueError("tn_kernel: u must lie in [0, 1]")
    u64 = u.to(torch.float64)
    endpoint = ((u64 == 0.0) | (u64 == 1.0)).sum().item()
    if endpoint:
        KERNEL_STATS.bump("u_endpoint_clamp", int(endpoint))
        u64 = u64.clamp(
            torch.finfo(torch.float64).eps, 1.0 - torch.finfo(torch.float64).eps
        )

    mu64, s64 = mu.to(torch.float64), sigma.to(torch.float64)
    e_lo, e_hi = erf_bounds(mu64, s64)
    e = (1.0 - u64) * e_lo + u64 * e_hi

    sat = (e.abs() >= 1.0).sum().item()
    if sat:
        KERNEL_STATS.bump("erfinv_endpoint_clamp", int(sat))
        e = e.clamp(-1.0 + torch.finfo(torch.float64).eps,
                    1.0 - torch.finfo(torch.float64).eps)
    action = mu64 + s64 * _SQRT2 * torch.erfinv(e)
    # Guard against a last-ulp overshoot outside the open support.
    action = action.clamp(
        -1.0 + torch.finfo(torch.float64).eps,
        1.0 - torch.finfo(torch.float64).eps,
    )
    out = action.to(mu.dtype)
    # The fp64 bound 1-eps64 rounds to exactly 1.0 in fp32; clamp again in
    # the output dtype so samples stay strictly inside (-1, 1) for the
    # downstream log_prob domain check.
    bound = 1.0 - torch.finfo(mu.dtype).eps
    over = ((out <= -bound) | (out >= bound)).sum().item()
    if over:
        KERNEL_STATS.bump("dtype_endpoint_clamp", int(over))
        out = out.clamp(-bound, bound)
    return out


def log_prob(
    action: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Joint per-dim log-density of the truncated Normal at ``action``.

    ``log p(a) = -0.5 ((a-μ)/σ)² - log(σ√(2π)) - log Z`` for a ∈ (-1,1).
    Differentiable w.r.t. mu/sigma.
    """
    _check_params(mu, sigma)
    if not torch.isfinite(action).all():
        raise ValueError("tn_kernel: non-finite action")
    if ((action <= -1.0) | (action >= 1.0)).any():
        raise ValueError("tn_kernel: action must lie strictly inside (-1, 1)")
    a64 = action.to(torch.float64)
    mu64, s64 = mu.to(torch.float64), sigma.to(torch.float64)
    log_z = log_normalizer(mu64, s64)
    lp = (
        -0.5 * ((a64 - mu64) / s64) ** 2
        - torch.log(s64)
        - _LOG_SQRT_2PI
        - log_z
    )
    return lp.to(mu.dtype)


def cdf(
    action: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """CDF of the truncated Normal at ``action`` ∈ (-1,1)."""
    _check_params(mu, sigma)
    a64 = action.to(torch.float64)
    mu64, s64 = mu.to(torch.float64), sigma.to(torch.float64)
    e_lo, e_hi = erf_bounds(mu64, s64)
    e_a = torch.erf((a64 - mu64) / (s64 * _SQRT2))
    width = (e_hi - e_lo).clamp_min(torch.finfo(torch.float64).tiny)
    return ((e_a - e_lo) / width).clamp(0.0, 1.0).to(mu.dtype)


def _std_bounds(mu64: torch.Tensor, s64: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    a = (-1.0 - mu64) / s64
    b = (1.0 - mu64) / s64
    return a, b


def _narrow_moments(
    mu64: torch.Tensor, s64: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Second-order moments for the near-uniform (σ → ∞) regime.

    Expands the density ∝ exp(-t²/2σ²) ≈ 1 - t²/2σ² in t = x - μ over
    x ∈ (-1, 1) and integrates the quadratic density analytically.
    Error is O((width/σ)²) relative — used when the standardized width
    2/σ is small enough that the closed form suffers cancellation.
    """
    lo = -1.0 - mu64
    hi = 1.0 - mu64
    w = hi - lo
    s2 = s64 * s64
    s0 = w - (hi**3 - lo**3) / (6.0 * s2)
    s1 = w * (lo + hi) / 2.0 - (hi**4 - lo**4) / (8.0 * s2)
    s2m = (hi**3 - lo**3) / 3.0 - (hi**5 - lo**5) / (10.0 * s2)
    et = s1 / s0
    var = s2m / s0 - et * et
    return mu64 + et, var.clamp_min(0.0)


_NARROW_STD_WIDTH = 1e-3


def analytic_mean(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """E[a] of the truncated Normal (float64 internal)."""
    _check_params(mu, sigma)
    mu64, s64 = mu.to(torch.float64), sigma.to(torch.float64)
    a, b = _std_bounds(mu64, s64)
    z = torch.exp(log_normalizer(mu64, s64))
    pdf_a = torch.exp(-0.5 * a * a) / _SQRT_2PI
    pdf_b = torch.exp(-0.5 * b * b) / _SQRT_2PI
    mean = mu64 + s64 * (pdf_a - pdf_b) / z.clamp_min(torch.finfo(torch.float64).tiny)
    narrow = (b - a) < _NARROW_STD_WIDTH
    if narrow.any():
        m_narrow, _ = _narrow_moments(mu64, s64)
        mean = torch.where(narrow, m_narrow, mean)
    return mean.to(mu.dtype)


def analytic_var(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Var[a] of the truncated Normal (float64 internal)."""
    _check_params(mu, sigma)
    mu64, s64 = mu.to(torch.float64), sigma.to(torch.float64)
    a, b = _std_bounds(mu64, s64)
    z = torch.exp(log_normalizer(mu64, s64)).clamp_min(torch.finfo(torch.float64).tiny)
    pdf_a = torch.exp(-0.5 * a * a) / _SQRT_2PI
    pdf_b = torch.exp(-0.5 * b * b) / _SQRT_2PI
    mean = analytic_mean(mu64, s64)
    var = s64 * s64 * (
        1.0
        + (a * pdf_a - b * pdf_b) / z
        - ((pdf_a - pdf_b) / z) ** 2
    )
    narrow = (b - a) < _NARROW_STD_WIDTH
    if narrow.any():
        _, v_narrow = _narrow_moments(mu64, s64)
        var = torch.where(narrow, v_narrow, var)
    return var.clamp_min(0.0).to(mu.dtype)


def analytic_entropy(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Differential entropy of the truncated Normal (per-dim, nats).

    ``H = log(σ√(2πe) Z) + (a φ(a) - b φ(b)) / (2Z)`` computed in float64.
    For large σ the ``log(σZ)`` term cancels cleanly because Z→erf form.
    """
    _check_params(mu, sigma)
    mu64, s64 = mu.to(torch.float64), sigma.to(torch.float64)
    a, b = _std_bounds(mu64, s64)
    log_z = log_normalizer(mu64, s64)
    z = torch.exp(log_z).clamp_min(torch.finfo(torch.float64).tiny)
    pdf_a = torch.exp(-0.5 * a * a) / _SQRT_2PI
    pdf_b = torch.exp(-0.5 * b * b) / _SQRT_2PI
    h = (
        torch.log(s64)
        + 0.5 * math.log(2.0 * math.pi * math.e)
        + log_z
        + (a * pdf_a - b * pdf_b) / (2.0 * z)
    )
    return h.to(mu.dtype)


__all__ = [
    "KERNEL_STATS",
    "TNKernelStats",
    "analytic_entropy",
    "analytic_mean",
    "analytic_var",
    "cdf",
    "erf_bounds",
    "log_normalizer",
    "log_prob",
    "normalizer",
    "sample_from_uniform",
]
