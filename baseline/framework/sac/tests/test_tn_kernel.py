"""Permanent tests for the N-SAC-01 truncated-normal kernel."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from baseline.framework.sac import tn_kernel as K


def _quad(fn, lo: float, hi: float, n: int = 4096) -> float:
    """Gauss-Legendre quadrature of fn over [lo, hi]."""
    x, w = np.polynomial.legendre.leggauss(n)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    pts = mid + half * x
    return float(half * np.sum(w * fn(pts)))


def _density_np(mu: float, sigma: float):
    mu_t = torch.tensor([mu], dtype=torch.float64)
    s_t = torch.tensor([sigma], dtype=torch.float64)
    log_z = float(K.log_normalizer(mu_t, s_t))

    def f(xs: np.ndarray) -> np.ndarray:
        lp = -0.5 * ((xs - mu) / sigma) ** 2 - math.log(sigma) - math.log(
            2 * math.pi
        ) / 2 - log_z
        return np.exp(lp)

    return f


CASES = [
    (0.0, 0.05),
    (0.999, 0.05),
    (-0.999, 2.0),
    (0.2, 1e-4),
    (0.4, 1e4),
    (0.4, math.e**20),
]


def test_density_normalizes_by_independent_integration():
    for mu, sigma in CASES:
        f = _density_np(mu, sigma)
        # Integrate only over the region holding real mass.
        lo = max(-1.0 + 1e-12, mu - 12 * sigma)
        hi = min(1.0 - 1e-12, mu + 12 * sigma)
        if hi - lo < 1e-10:
            # Numerically unresolvable narrow support: check CDF mass
            # instead of a quadrature integral that cannot see the spike.
            mu_t = torch.tensor([mu], dtype=torch.float64)
            s_t = torch.tensor([sigma], dtype=torch.float64)
            z = float(K.normalizer(mu_t, s_t))
            assert np.isfinite(z) and z >= 0.0
            continue
        mass = _quad(f, lo, hi)
        assert mass == pytest.approx(1.0, abs=2e-4), (mu, sigma, mass)


def test_uniform_limit_large_sigma():
    # σ=e^20: the TN should approximate Uniform[-1,1] — mean≈0, var≈1/3,
    # density ≈ 1/2 everywhere.
    mu_t = torch.tensor([0.4], dtype=torch.float64)
    s_t = torch.tensor([math.e**20], dtype=torch.float64)
    xs = torch.linspace(-0.9, 0.9, 41, dtype=torch.float64)
    lp = K.log_prob(xs, mu_t, s_t)
    assert torch.allclose(lp, torch.full_like(lp, -math.log(2.0)), atol=1e-3)
    # Uniform limit on (-1,1): mean → 0, var → 1/3 regardless of μ.
    assert float(K.analytic_mean(mu_t, s_t)) == pytest.approx(0.0, abs=1e-6)
    var = float(K.analytic_var(mu_t, s_t))
    assert var == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_inverse_cdf_is_consistent_with_cdf():
    for mu, sigma in [(0.0, 0.3), (-0.4, 1.5), (0.9, 0.1), (0.4, 1e3)]:
        mu_t = torch.tensor([mu], dtype=torch.float64)
        s_t = torch.tensor([sigma], dtype=torch.float64)
        u = torch.linspace(0.05, 0.95, 19, dtype=torch.float64)
        a = K.sample_from_uniform(mu_t, s_t, u)
        assert ((a > -1.0) & (a < 1.0)).all()
        assert (a[1:] > a[:-1]).all(), "quantiles must be strictly increasing"
        np.testing.assert_allclose(
            K.cdf(a, mu_t, s_t).numpy(), u.numpy(), atol=1e-8,
        )


def test_sampling_moments_match_analytic():
    torch.manual_seed(0)
    for mu, sigma in [(0.0, 0.2), (0.6, 0.8), (-0.8, 0.15)]:
        mu_t = torch.tensor([mu], dtype=torch.float64)
        s_t = torch.tensor([sigma], dtype=torch.float64)
        u = (torch.rand(1_000_000, dtype=torch.float64) * 0.998 + 0.001)
        a = K.sample_from_uniform(mu_t, s_t, u)
        assert abs(float(a.mean()) - float(K.analytic_mean(mu_t, s_t))) < 2e-3
        assert abs(float(a.var()) - float(K.analytic_var(mu_t, s_t))) < 2e-3


def test_fixed_noise_gradcheck_mu_and_log_sigma():
    # A4.10 representative points; objective mean[a^2 + 0.03*logp].
    torch.set_default_dtype(torch.float64)
    try:
        for mu_val, s_val in [
            (0.0, 0.05),
            (0.999, 0.05),
            (-0.999, 2.0),
            (0.2, 1e-4),
            (0.4, 1e4),
        ]:
            mu = torch.tensor([mu_val], dtype=torch.float64, requires_grad=True)
            log_s = torch.tensor([math.log(s_val)], dtype=torch.float64,
                                 requires_grad=True)
            u = torch.tensor([0.05, 0.2, 0.5, 0.8, 0.95], dtype=torch.float64)

            def fn(m, ls):
                s = ls.exp()
                a = K.sample_from_uniform(m, s, u)
                lp = K.log_prob(a, m, s)
                return (a**2 + 0.03 * lp).mean()

            assert torch.autograd.gradcheck(
                fn, (mu, log_s), eps=1e-6, atol=2e-6, rtol=2e-4
            ), (mu_val, s_val)
    finally:
        torch.set_default_dtype(torch.float32)


def test_endpoint_protection_is_counted_not_silent():
    K.KERNEL_STATS.reset()
    mu_t = torch.tensor([0.9999], dtype=torch.float64)
    s_t = torch.tensor([0.05], dtype=torch.float64)
    # u=1.0 is an endpoint hit: must be counted, and the action must
    # still land strictly inside the open support.
    a = K.sample_from_uniform(mu_t, s_t, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.isfinite(a).all()
    assert ((a > -1.0) & (a < 1.0)).all()
    snap = K.KERNEL_STATS.snapshot()
    assert snap["u_endpoint_clamp"] >= 2


def test_saturated_erf_bounds_still_sample_strictly_inside():
    # erf(arg≈7) saturates to exactly 1.0 in float64 — the erf-space
    # interpolation must still produce actions strictly inside (-1,1).
    mu_t = torch.tensor([0.9999999], dtype=torch.float64)
    s_t = torch.tensor([1e-8], dtype=torch.float64)
    e_lo, e_hi = K.erf_bounds(mu_t, s_t)
    assert float(e_hi) == 1.0
    u = torch.linspace(0.001, 0.999, 33, dtype=torch.float64)
    a = K.sample_from_uniform(mu_t.expand_as(u), s_t.expand_as(u), u)
    assert torch.isfinite(a).all()
    assert ((a > -1.0) & (a < 1.0)).all()
    assert (a[1:] > a[:-1]).all()


def test_invalid_inputs_fail_loudly():
    mu = torch.tensor([0.0])
    sig = torch.tensor([0.5])
    with pytest.raises(ValueError, match="non-finite mu"):
        K.log_prob(torch.zeros(1), torch.tensor([float("nan")]), sig)
    with pytest.raises(ValueError, match="sigma must be > 0"):
        K.log_prob(torch.zeros(1), mu, torch.tensor([0.0]))
    with pytest.raises(ValueError, match="inside"):
        K.log_prob(torch.zeros(1), torch.tensor([1.5]), sig)
    with pytest.raises(ValueError, match="inside"):
        K.log_prob(torch.tensor([1.0]), mu, sig)
    with pytest.raises(ValueError, match="non-finite u"):
        K.sample_from_uniform(mu, sig, torch.tensor([float("inf")]))


def test_float32_path_and_entropy():
    mu = torch.tensor([0.0, -0.3], dtype=torch.float32)
    s = torch.tensor([0.2, 1.5], dtype=torch.float32)
    u = torch.rand(64, 2)
    a = K.sample_from_uniform(mu.expand(64, 2), s.expand(64, 2), u)
    assert a.dtype == torch.float32
    assert torch.isfinite(a).all()
    h = K.analytic_entropy(mu, s)
    assert torch.isfinite(h).all()
    # Sanity: wider σ → larger entropy at μ=0.
    assert float(h[0]) < float(
        K.analytic_entropy(torch.tensor([0.0]), torch.tensor([1.0]))[0]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_path():
    dev = torch.device("cuda")
    mu = torch.zeros(8, device=dev)
    s = torch.full((8,), 0.5, device=dev)
    u = torch.rand(8, device=dev)
    a = K.sample_from_uniform(mu, s, u)
    assert a.device.type == "cuda"
    assert torch.isfinite(a).all()
    assert ((a > -1.0) & (a < 1.0)).all()


def test_fp32_boundary_overshoot_stays_inside_support():
    """μ near +1 with small σ and u≈1 overshoots in fp64; after cast to
    fp32 the fp64 bound rounds to exactly 1.0 — the output-dtype clamp must
    keep samples strictly inside (-1,1) so log_prob accepts them."""
    K.KERNEL_STATS.reset()
    # Saturated mean head + tiny σ: mass piles on the boundary and fp64
    # samples clamp to 1-eps64, which rounds to exactly 1.0 in fp32.
    mu = torch.full((2048,), 0.999, dtype=torch.float32)
    sig = torch.full((2048,), 0.001, dtype=torch.float32)
    u = torch.linspace(0.0, 1.0, 2048)
    a = K.sample_from_uniform(mu, sig, u)
    assert a.dtype == torch.float32
    assert ((a > -1.0) & (a < 1.0)).all(), f"boundary sample: {a.max()}"
    K.log_prob(a, mu, sig)
    assert K.KERNEL_STATS.snapshot()["dtype_endpoint_clamp"] > 0
    # Mirror at the lower boundary.
    a_lo = K.sample_from_uniform(-mu, sig, u)
    assert ((a_lo > -1.0) & (a_lo < 1.0)).all()
    K.log_prob(a_lo, -mu, sig)
