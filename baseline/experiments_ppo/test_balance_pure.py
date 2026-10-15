"""Tests for the simplified balance recovery experiment (balance_pure)."""
from __future__ import annotations

import numpy as np

from baseline.experiments_ppo import get_ppo_experiment


def test_balance_pure_registered_and_channels():
    exp = get_ppo_experiment("balance_pure")
    assert exp.obs_dim == 96
    assert [c.name for c in exp.reward_channels()] == [
        "r_potential", "r_fall", "r_cross",
    ]


def test_balance_pure_mbs_registered():
    exp = get_ppo_experiment("balance_pure_mbs")
    assert exp.obs_dim == 96
    assert "mixture" in exp.actor_blueprint


def test_balance_pure_env_obs_is_96():
    """The canonical contract: bare simulator, no gait-clock dims."""
    exp = get_ppo_experiment("balance_pure")
    env = exp._env_pb().materialize(max_steps=50).build()
    env.reset(seed=0, options={
        "initial_distance": 2.0,
        "impulse_params": {
            r: {"force": 40.0, "duration_min": 1, "duration_max": 10,
                "body": "torso", "seed": 0}
            for r in ("robot_a", "robot_b")
        },
    })
    for _ in range(10):
        if env.is_episode_over():
            break
        env.step(np.zeros(21, dtype=np.float32),
                 np.zeros(21, dtype=np.float32))
    obs, _ = env.get_observation()
    assert obs.shape == (96,)


def test_balance_pure_trajectory_two_channels():
    """Stub episode → trajectories carry only r_potential + r_fall."""
    exp = get_ppo_experiment("balance_pure")
    T = 30
    episode = type("Ep", (), {})()
    episode.num_frames = T
    episode.agent_frame_boundary = {"robot_a": T, "robot_b": T}
    episode.observations = {
        rid: np.zeros((T, 96), dtype=np.float32)
        for rid in ("robot_a", "robot_b")
    }
    episode.actions = {
        rid: np.zeros((T, 21), dtype=np.float32)
        for rid in ("robot_a", "robot_b")
    }
    episode.final_observation = {
        rid: np.zeros(96, dtype=np.float32)
        for rid in ("robot_a", "robot_b")
    }
    episode.observer_outputs = {
        f"standing_balance_{s}": {"potential": np.full(T, 0.9)}
        for s in ("a", "b")
    }
    episode.episode_metrics = {}
    episode.rollout_metrics = {}
    episode.sampling_contexts = {}

    trajs = exp.build_trajectories([episode])
    assert len(trajs) == 2
    for t in trajs:
        assert set(t.channels.keys()) == {
            "r_potential", "r_fall", "r_cross",
        }
        ch = t.channels["r_potential"]
        np.testing.assert_allclose(ch.reward, 0.9 * 0.01)
        np.testing.assert_allclose(ch.actor_weight, 3.0)
        np.testing.assert_allclose(
            t.channels["r_fall"].actor_weight, 2.0)
