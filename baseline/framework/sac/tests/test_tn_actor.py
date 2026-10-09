"""Contract tests for the eight-cell TN actor family (P4-ACTOR-1)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from baseline.framework.sac.collection import SACBehaviorSpec
from baseline.framework.sac.tn_actor import (
    ARCH_SPECS,
    INIT_STD,
    SIGMA_MAX,
    SIGMA_MIN,
    TNActor,
    TNRuntimePolicy,
    _bounded_geometry,
)


OBS_DIM, ACT_DIM = 7, 3
SINGLE_ARCHS = ["s00", "s01", "s10", "s11"]


def _obs(batch: int = 8, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, OBS_DIM, generator=gen)


@pytest.mark.parametrize("arch", SINGLE_ARCHS)
def test_distribution_shapes_and_bounds(arch):
    actor = TNActor(OBS_DIM, ACT_DIM, arch=arch, hidden_dim=32, seed=1)
    dist = actor.distribution(_obs())
    assert dist["mu"].shape == (8, 1, ACT_DIM)
    assert dist["sigma"].shape == (8, 1, ACT_DIM)
    assert dist["logits"].shape == (8, 1)
    assert torch.all(dist["mu"].abs() <= 1.0)
    assert torch.all(dist["sigma"] > 0.0)
    if arch.endswith("1"):
        assert torch.all(dist["sigma"] >= SIGMA_MIN)
        assert torch.all(dist["sigma"] <= SIGMA_MAX)


@pytest.mark.parametrize("arch", SINGLE_ARCHS)
def test_init_sigma_matches_spec(arch):
    actor = TNActor(OBS_DIM, ACT_DIM, arch=arch, hidden_dim=32, seed=1)
    sigma = actor.distribution(_obs(batch=4))["sigma"]
    assert torch.allclose(
        sigma, torch.full_like(sigma, INIT_STD), atol=1e-5
    ), f"{arch} init σ should equal init_std={INIT_STD}"


@pytest.mark.parametrize("arch", SINGLE_ARCHS)
def test_expectation_samples_contract(arch):
    actor = TNActor(OBS_DIM, ACT_DIM, arch=arch, hidden_dim=32, seed=2)
    obs = _obs()
    M = 4
    u = torch.rand(8, 1, M, ACT_DIM, generator=torch.Generator().manual_seed(3))
    actions, logp, weights = actor.expectation_samples(obs, u)
    assert actions.shape == (8, 1, M, ACT_DIM)
    assert logp.shape == (8, 1, M)
    assert weights.shape == (8, 1, M)
    assert torch.all(actions.abs() < 1.0)
    assert torch.isfinite(logp).all()
    # K=1 → weights are exactly 1/M; sum over candidates equals 1.
    assert torch.allclose(weights.sum(dim=(1, 2)), torch.ones(8))


@pytest.mark.parametrize("arch", SINGLE_ARCHS)
def test_expectation_samples_gradients(arch):
    actor = TNActor(OBS_DIM, ACT_DIM, arch=arch, hidden_dim=32, seed=4)
    obs = _obs()
    u = torch.full((8, 1, 2, ACT_DIM), 0.5)
    actions, logp, weights = actor.expectation_samples(obs, u)
    loss = (actions.pow(2).sum() + logp.sum() + weights.sum())
    loss.backward()
    grads = [
        p.grad for p in actor.parameters() if p.grad is not None
    ]
    assert grads, "no parameter received gradients"
    assert any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("arch", SINGLE_ARCHS)
def test_sample_log_prob_consistency(arch):
    actor = TNActor(OBS_DIM, ACT_DIM, arch=arch, hidden_dim=32, seed=5)
    obs = _obs()
    action, logp = actor.sample_action(obs)
    assert action.shape == (8, ACT_DIM)
    rescored = actor.log_prob_of(obs, action)
    assert torch.allclose(logp, rescored, atol=1e-5)


def test_sample_action_reproducible_with_same_seed():
    a1 = TNActor(OBS_DIM, ACT_DIM, arch="s01", hidden_dim=32, seed=9)
    a2 = TNActor(OBS_DIM, ACT_DIM, arch="s01", hidden_dim=32, seed=9)
    a2.load_state_dict(a1.state_dict())
    obs = _obs()
    act1, _ = a1.sample_action(obs)
    act2, _ = a2.sample_action(obs)
    assert torch.equal(act1, act2)


def test_rng_state_round_trip():
    actor = TNActor(OBS_DIM, ACT_DIM, arch="s00", hidden_dim=32, seed=7)
    obs = _obs()
    actor.sample_action(obs)
    state = actor.rng_state()
    next_a, _ = actor.sample_action(obs)
    actor.set_rng_state(state)
    next_b, _ = actor.sample_action(obs)
    assert torch.equal(next_a, next_b)


@pytest.mark.parametrize("arch", SINGLE_ARCHS)
def test_deterministic_action_is_mu(arch):
    actor = TNActor(OBS_DIM, ACT_DIM, arch=arch, hidden_dim=32, seed=6)
    obs = _obs()
    det = actor.deterministic_action(obs)
    assert torch.allclose(det, actor.distribution(obs)["mu"][:, 0])


def test_explore_unbounded_scales_sigma():
    actor = TNActor(OBS_DIM, ACT_DIM, arch="s00", hidden_dim=32, seed=8)
    obs = _obs()
    spec = SACBehaviorSpec(mode="stochastic", explore_factor=0.5)
    base_sigma = actor.distribution(obs)["sigma"][:, 0]
    dist_e = actor._distribution_with_e(obs, 0.5)
    expected = base_sigma * (3.0 ** 0.5)
    assert torch.allclose(dist_e["sigma"][:, 0], expected, rtol=1e-5)
    _, info = actor.sample_behavior(obs, spec)
    assert abs(info["sigma_eff_mean"] - float(expected.mean())) < 1e-4


def test_explore_bounded_shift():
    actor = TNActor(OBS_DIM, ACT_DIM, arch="s01", hidden_dim=32, seed=8)
    obs = _obs()
    dist = actor._distribution_with_e(obs, 0.5)
    r_min, delta_r, _p0, _v = _bounded_geometry(
        actor.sigma_min, actor.sigma_max, actor.init_std,
    )
    # Expected: σ = exp(r_min + Δr·sigmoid(v_init + α·0.5))
    v_e = actor._v_init + actor.explore_alpha * 0.5
    expected = math.exp(r_min + delta_r * (1.0 / (1.0 + math.exp(-v_e))))
    assert torch.allclose(
        dist["sigma"][:, 0], torch.full((8, ACT_DIM), expected), rtol=1e-4
    )


def test_explore_out_of_range_rejected():
    actor = TNActor(OBS_DIM, ACT_DIM, arch="s01", hidden_dim=32, seed=8)
    spec = SACBehaviorSpec(mode="stochastic", explore_factor=1.5)
    with pytest.raises(ValueError, match="explore_factor"):
        actor.sample_behavior(_obs(), spec)


def test_deterministic_mode_rejects_e():
    actor = TNActor(OBS_DIM, ACT_DIM, arch="s01", hidden_dim=32, seed=8)
    spec = SACBehaviorSpec(mode="deterministic", explore_factor=0.3)
    with pytest.raises(ValueError, match="deterministic"):
        actor.sample_behavior(_obs(), spec)


def test_uncertainty_peak_and_l2_finite():
    actor = TNActor(OBS_DIM, ACT_DIM, arch="s11", hidden_dim=32, seed=8)
    obs = _obs()
    for kind in ("peak", "l2"):
        u = actor.uncertainty(obs, kind)
        assert u.shape == (8,)
        assert torch.isfinite(u).all() and (u > 0).all()
    with pytest.raises(ValueError, match="uncertainty"):
        actor.uncertainty(obs, "bogus")


def test_uncertainty_differentiable():
    actor = TNActor(OBS_DIM, ACT_DIM, arch="s00", hidden_dim=32, seed=8)
    u = actor.uncertainty(_obs(), "l2")
    u.sum().backward()
    grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert grads


@pytest.mark.parametrize("arch", SINGLE_ARCHS)
def test_export_round_trip(arch, tmp_path):
    actor = TNActor(OBS_DIM, ACT_DIM, arch=arch, hidden_dim=32, seed=11)
    bp = actor.to_blueprint(str(tmp_path / "pol"), stochastic=False)
    runtime = bp.build()
    obs_np = np.random.RandomState(0).randn(OBS_DIM).astype(np.float32)
    action, extra = runtime.act(obs_np, want_extra=True)
    det = actor.deterministic_action(
        torch.as_tensor(obs_np).unsqueeze(0)
    ).squeeze(0).detach().numpy()
    assert np.allclose(action, det, atol=1e-6)
    assert extra["explore_factor"] == 0.0


def test_export_stochastic_with_e(tmp_path):
    actor = TNActor(OBS_DIM, ACT_DIM, arch="s01", hidden_dim=32, seed=12)
    bp = actor.to_blueprint(
        str(tmp_path / "pol"), stochastic=True, explore_factor=0.4,
    )
    runtime = bp.build()
    obs_np = np.zeros(OBS_DIM, dtype=np.float32)
    action, extra = runtime.act(obs_np, want_extra=True)
    assert np.all(np.abs(action) < 1.0)
    assert extra["explore_factor"] == 0.4


def test_deterministic_export_rejects_e(tmp_path):
    actor = TNActor(OBS_DIM, ACT_DIM, arch="s01", hidden_dim=32, seed=12)
    with pytest.raises(ValueError, match="deterministic"):
        actor.to_blueprint(str(tmp_path / "p"), stochastic=False,
                           explore_factor=0.2)


def test_unknown_arch_rejected():
    with pytest.raises(ValueError, match="unknown SAC actor arch"):
        TNActor(OBS_DIM, ACT_DIM, arch="s99", hidden_dim=32)


@pytest.mark.parametrize("arch", ["m00", "m01", "m10", "m11"])
def test_mixture_archs_rejected_until_mix1(arch):
    with pytest.raises(ValueError, match="P4-MIX-1"):
        TNActor(OBS_DIM, ACT_DIM, arch=arch, hidden_dim=32)


def test_state_payload_round_trip():
    actor = TNActor(OBS_DIM, ACT_DIM, arch="s10", hidden_dim=32, seed=13)
    payload = actor.state_payload()
    assert payload["policy_arch"] == "tn_s10"
    assert payload["kernel_version"] == "tn_kernel_v1"
    restored = TNActor(OBS_DIM, ACT_DIM, arch="s10", hidden_dim=32, seed=0)
    restored.load_state_dict(payload["state_dict"])
    restored.set_rng_state(payload["rng_state"])
    obs = _obs()
    a, _ = actor.sample_action(obs)
    b, _ = restored.sample_action(obs)
    assert torch.equal(a, b)


def test_runtime_policy_rejects_bad_payload(tmp_path):
    actor = TNActor(OBS_DIM, ACT_DIM, arch="s00", hidden_dim=32, seed=14)
    actor.export_policy_artifacts(tmp_path / "pol")
    payload = torch.load(tmp_path / "pol" / "model.pt", weights_only=False)
    payload["policy_arch"] = "bogus"
    torch.save(payload, tmp_path / "pol" / "model.pt")
    with pytest.raises(ValueError, match="unsupported TN runtime payload"):
        TNRuntimePolicy(str(tmp_path / "pol" / "model.pt"))


def test_build_actor_dispatch():
    from baseline.experiments_sac.exp_sac_balance import SacBalance
    exp = SacBalance(actor_arch="s01", actor_hidden_dim=32)
    actor = exp.build_actor(torch.device("cpu"))
    assert isinstance(actor, TNActor)
    exp2 = SacBalance(actor_arch="legacy_tanh", actor_hidden_dim=32)
    from baseline.framework.sac.s01_actor import S01Actor
    assert isinstance(exp2.build_actor(torch.device("cpu")), S01Actor)
    with pytest.raises(ValueError, match="unsupported SAC actor_arch"):
        SacBalance(actor_arch="bogus").build_actor(torch.device("cpu"))
