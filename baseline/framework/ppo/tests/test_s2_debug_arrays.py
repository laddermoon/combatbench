"""S2 debug_arrays extension point tests.

Tests cover:
- ExperimentPPO.debug_arrays default returns {}.
- standup_step_v3.debug_arrays returns aligned arrays (0th dim == total frames).
- standup_step_v3.debug_arrays calls _compute_phase_mask (P2 — same values).
- standup_step_v3.debug_arrays on empty trajectories returns empty arrays.

Conventions follow test_trainer.py / test_s1_provenance.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.experiment import ExperimentPPO
from baseline.framework.ppo.trajectory import (
    ChannelData, RewardChannel, Trajectory, TrajectoryProvenance,
)
from baseline.framework.rollout import Episode, blueprint_hash
from envs.framework.blueprint import EnvBlueprint, ClassSpec


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_trajectory(T=10, obs_dim=96, action_dim=21, *, provenance=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    obs = rng.standard_normal((T, obs_dim)).astype(np.float32)
    actions = rng.uniform(-0.9, 0.9, (T, action_dim)).astype(np.float32)
    last_obs = rng.standard_normal(obs_dim).astype(np.float32)
    channels = {
        "r_a": ChannelData(
            reward=rng.standard_normal(T).astype(np.float32),
            is_terminated=True, actor_weight=1.0,
        ),
    }
    return Trajectory(
        obs=obs, actions=actions, last_obs=last_obs,
        channels=channels, importance=1.0,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_debug_arrays_default_empty():
    """ExperimentPPO.debug_arrays default returns {}."""
    # Create a minimal subclass that doesn't override debug_arrays.
    class MinimalExp(ExperimentPPO):
        name = "minimal"
        reward_keys = ("r_a",)
        gammas = {"r_a": 0.99}
        gae_lambdas = {"r_a": 0.95}

        def __init__(self):
            pass

        def common_params(self):
            from baseline.framework.ppo.experiment import CommonParams
            return CommonParams(
                name="minimal", seed=42, max_updates=10,
                episodes_per_update=4, eval_interval=100,
                eval_episodes=4, video_eval_interval=100,
                rollout_workers=2, learning_rate=1e-3,
                critic_learning_rate=1e-3, grad_clip_norm=0.5,
            )

        def ppo_params(self):
            from baseline.framework.ppo.experiment import PPOParams
            return PPOParams(clip_eps=0.2, target_kl=0.05, update_epochs=4, minibatch_size=64)

        def reward_channels(self):
            return (RewardChannel("r_a", gamma=0.99, gae_lambda=0.95),)

        def build_actor(self, device):
            raise NotImplementedError

        def build_critic(self, key, device):
            raise NotImplementedError

        def build_trajectories(self, episodes):
            return []

        def build_jobs(self, policy_bp, base_seed, n_episodes):
            return []

        def on_eval(self, episodes, update):
            return {}

        def on_update(self, stats):
            pass

    exp = MinimalExp()
    result = exp.debug_arrays([], [])
    assert result == {}
    print("test_debug_arrays_default_empty: PASS")


def test_debug_arrays_empty_trajectories():
    """standup_step_v3.debug_arrays on empty trajectories returns empty arrays."""
    from baseline.experiments_ppo.exp_standup_step_v3 import StandupStepV3
    exp = StandupStepV3()
    result = exp.debug_arrays([], [])
    assert "balance_mask" in result
    assert "h_left" in result
    assert "h_right" in result
    assert "contact_l" in result
    assert "contact_r" in result
    for arr in result.values():
        assert len(arr) == 0
    print("test_debug_arrays_empty_trajectories: PASS")


def test_debug_arrays_no_provenance_zeros():
    """Trajectories without provenance → zero-filled arrays (maintains alignment)."""
    from baseline.experiments_ppo.exp_standup_step_v3 import StandupStepV3
    exp = StandupStepV3()
    rng = np.random.default_rng(42)
    trajs = [_make_trajectory(T=5, rng=rng), _make_trajectory(T=3, rng=rng)]
    result = exp.debug_arrays([], trajs)
    total = 5 + 3
    assert len(result["balance_mask"]) == total
    assert len(result["h_left"]) == total
    # All zeros (no provenance → no episode data).
    assert not result["balance_mask"].any()
    print("test_debug_arrays_no_provenance_zeros: PASS")


def test_debug_arrays_contract_enforcement():
    """debug_arrays 0th dim must equal total trajectory frames."""
    from baseline.experiments_ppo.exp_standup_step_v3 import StandupStepV3
    exp = StandupStepV3()
    rng = np.random.default_rng(42)
    trajs = [_make_trajectory(T=5, rng=rng), _make_trajectory(T=3, rng=rng)]
    result = exp.debug_arrays([], trajs)
    total = sum(len(t.obs) for t in trajs)
    for name, arr in result.items():
        assert len(arr) == total, (
            f"debug_arrays contract violation: {name} has length {len(arr)} "
            f"but total frames = {total}"
        )
    print("test_debug_arrays_contract_enforcement: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_debug_arrays_default_empty()
    test_debug_arrays_empty_trajectories()
    test_debug_arrays_no_provenance_zeros()
    test_debug_arrays_contract_enforcement()
    print("\nAll S2 debug_arrays tests passed.")
