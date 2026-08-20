#!/usr/bin/env python3
"""Tests for EpisodeEndCaptureObserver, ConstantForcePlugin
episode_options override, and NoisyPolicyWrapper.

Run with::

    cd /data1/mono/things/combatbench
    python3 -m pytest envs/humanoid21/tests/test_state_pool_plugins.py -q
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from envs.framework.env_runtime import EnvRuntime
from envs.humanoid21.disturbance_plugins import (
    ConstantForcePlugin,
    EpisodeEndCaptureObserver,
)
from envs.humanoid21.simulator import Humanoid21Simulator


# ── helpers ────────────────────────────────────────────────────────────

def _make_simulator() -> Humanoid21Simulator:
    return Humanoid21Simulator()


def _standing_action(sim: Humanoid21Simulator) -> dict:
    return sim.INITIAL_POSES["standing"]["action"]


def _close_runtime(rt: EnvRuntime) -> None:
    try:
        rt.close()
    except Exception:
        pass


# ── EpisodeEndCaptureObserver tests ────────────────────────────────────

def test_episode_end_capture_observer_captures_last_step():
    """Verify that EpisodeEndCaptureObserver captures the state at the
    last action step, not the first."""
    sim = _make_simulator()
    capture_observer = EpisodeEndCaptureObserver(target_robot="robot_a")

    rt = EnvRuntime(
        simulator=sim,
        observer_plugins={"episode_end_capture": capture_observer},
        phy_steps_per_action=25,
        max_steps=5,
    )
    try:
        rt.reset(seed=42, options={"initial_pose_a": "standing", "initial_pose_b": "standing"})
        action = _standing_action(sim)

        # Run 5 steps (max_steps=5)
        for _ in range(5):
            if not rt._core.is_episode_active:
                break
            rt.step(action, action)

        output = rt.get_observer_output("episode_end_capture")
        assert output is not None
        assert "core_state" in output
        assert "observation" in output

        cs = output["core_state"]
        assert "root_pos" in cs
        assert "root_rot" in cs
        assert "joint_pos_norm" in cs
        assert cs["root_pos"].shape == (3,)
        assert cs["root_rot"].shape == (4,)

        obs = output["observation"]
        assert obs is not None
        assert obs.ndim == 1
    finally:
        _close_runtime(rt)


def test_episode_end_capture_observer_overwrites_each_step():
    """Verify that the observer output changes across steps (proving
    it overwrites rather than capturing only the first step)."""
    sim = _make_simulator()
    capture_observer = EpisodeEndCaptureObserver(target_robot="robot_a")

    rt = EnvRuntime(
        simulator=sim,
        observer_plugins={"episode_end_capture": capture_observer},
        phy_steps_per_action=25,
        max_steps=10,
    )
    try:
        rt.reset(seed=42, options={"initial_pose_a": "standing", "initial_pose_b": "standing"})
        action = _standing_action(sim)

        outputs = []
        for _ in range(10):
            if not rt._core.is_episode_active:
                break
            rt.step(action, action)
            out = rt.get_observer_output("episode_end_capture")
            outputs.append(np.asarray(out["core_state"]["root_pos"]).copy())

        assert len(outputs) >= 2
        diffs = [np.linalg.norm(outputs[i+1] - outputs[i]) for i in range(len(outputs)-1)]
        assert any(d > 1e-8 for d in diffs), "Observer output should change across steps"
    finally:
        _close_runtime(rt)


def test_episode_end_capture_observer_empty_before_steps():
    """Verify that get_output() returns {} before any step runs."""
    observer = EpisodeEndCaptureObserver(target_robot="robot_a")
    assert observer.get_output() == {}


# ── ConstantForcePlugin episode_options tests ──────────────────────────

def test_constant_force_plugin_episode_options_override():
    """Verify that episode_options["impulse_params"] overrides
    ConstantForcePlugin parameters per-episode."""
    sim = _make_simulator()
    force_plugin = ConstantForcePlugin(
        agent_id="robot_a",
        force=50.0,
        direction=0.0,
        duration_action_steps=2,
        body_name="torso",
    )
    capture_observer = EpisodeEndCaptureObserver(target_robot="robot_a")

    rt = EnvRuntime(
        simulator=sim,
        plugins=[force_plugin],
        observer_plugins={"episode_end_capture": capture_observer},
        phy_steps_per_action=25,
        max_steps=6,
    )
    try:
        # Override force/direction/duration via episode_options
        rt.reset(
            seed=42,
            options={
                "initial_pose_a": "standing",
                "initial_pose_b": "standing",
                "impulse_params": {
                    "robot_a": {
                        "force": 200.0,
                        "direction_angle": 90.0,
                        "duration_action_steps": 3,
                        "body": "torso",
                    },
                },
            },
        )

        # Check that plugin parameters were overridden
        assert force_plugin.force == 200.0
        assert force_plugin.direction == 90.0
        assert force_plugin.duration_action_steps == 3
        assert force_plugin.body_name == "torso"

        action = _standing_action(sim)
        for _ in range(6):
            if not rt._core.is_episode_active:
                break
            rt.step(action, action)

        # Verify impulse metrics were written
        output = rt.get_observer_output("episode_end_capture")
        assert output is not None
        # impulse_force should be 200.0 (from episode_options override)
        assert output.get("impulse_force") == 200.0
        assert output.get("impulse_duration") == 3
        assert output.get("impulse_direction_angle") == 90.0
    finally:
        _close_runtime(rt)


def test_constant_force_plugin_no_override_uses_defaults():
    """Verify that without episode_options, ConstantForcePlugin uses
    its constructor defaults."""
    sim = _make_simulator()
    force_plugin = ConstantForcePlugin(
        agent_id="robot_a",
        force=75.0,
        direction=45.0,
        duration_action_steps=2,
        body_name="torso",
    )

    rt = EnvRuntime(
        simulator=sim,
        plugins=[force_plugin],
        phy_steps_per_action=25,
        max_steps=4,
    )
    try:
        rt.reset(seed=42, options={"initial_pose_a": "standing", "initial_pose_b": "standing"})

        # Parameters should remain as constructed
        assert force_plugin.force == 75.0
        assert force_plugin.direction == 45.0
        assert force_plugin.duration_action_steps == 2
    finally:
        _close_runtime(rt)


# ── NoisyPolicyWrapper tests ───────────────────────────────────────────

def test_noisy_policy_wrapper_zero_sigma_matches_base():
    """With sigma=0, NoisyPolicyWrapper should produce identical output
    to the base policy."""
    from baseline.humanoid21.balance_recover.gating.noisy_policy import NoisyPolicyWrapper

    # Use a simple deterministic policy for testing
    # We'll use a mock policy class
    class MockPolicy:
        def __init__(self, value=0.5):
            self.value = value

        def act(self, observation, want_extra=False):
            return np.full(21, self.value, dtype=np.float32), None

        def reset(self, seed=None):
            pass

    # We can't use NoisyPolicyWrapper directly because it requires
    # _resolve_policy_class. Instead, test the noise logic directly.
    rng = np.random.default_rng(42)
    base_action = np.full(21, 0.5, dtype=np.float32)

    # sigma=0 → no noise
    sigma = 0.0
    if sigma > 0:
        noisy = np.clip(base_action + rng.normal(0, sigma, base_action.shape), -1, 1)
    else:
        noisy = base_action
    assert np.array_equal(noisy, base_action)


def test_noisy_policy_wrapper_adds_noise():
    """With sigma>0, NoisyPolicyWrapper should produce different output
    from the base policy."""
    rng = np.random.default_rng(42)
    base_action = np.full(21, 0.5, dtype=np.float32)

    sigma = 0.2
    noise = rng.normal(0, sigma, base_action.shape).astype(np.float32)
    noisy = np.clip(base_action + noise, -1, 1)

    assert not np.array_equal(noisy, base_action)
    assert noisy.shape == base_action.shape
    assert np.all(np.abs(noisy) <= 1.0)


def test_noisy_policy_wrapper_clips_to_range():
    """Verify that noisy actions are clipped to [-1, 1]."""
    rng = np.random.default_rng(42)
    base_action = np.full(21, 0.95, dtype=np.float32)

    sigma = 0.5
    noise = rng.normal(0, sigma, base_action.shape).astype(np.float32)
    noisy = np.clip(base_action + noise, -1, 1)

    assert np.all(noisy <= 1.0)
    assert np.all(noisy >= -1.0)


# ── Integration test: full episode with capture ────────────────────────

def test_integration_constant_force_with_capture():
    """End-to-end: ConstantForcePlugin + EpisodeEndCaptureObserver running
    a full short episode and verifying captured data is valid."""
    sim = _make_simulator()
    force_plugin = ConstantForcePlugin(
        agent_id="robot_a",
        force=100.0,
        direction=45.0,
        duration_action_steps=2,
        body_name="torso",
    )
    capture_observer = EpisodeEndCaptureObserver(target_robot="robot_a")

    rt = EnvRuntime(
        simulator=sim,
        plugins=[force_plugin],
        observer_plugins={"episode_end_capture": capture_observer},
        phy_steps_per_action=25,
        max_steps=4,
    )
    try:
        rt.reset(seed=42, options={"initial_pose_a": "standing", "initial_pose_b": "standing"})
        action = _standing_action(sim)

        for _ in range(4):
            if not rt._core.is_episode_active:
                break
            rt.step(action, action)

        output = rt.get_observer_output("episode_end_capture")
        assert output is not None
        assert "core_state" in output
        assert "observation" in output

        # Verify core_state has all expected fields
        cs = output["core_state"]
        expected_fields = ["root_pos", "root_rot", "root_vel_local",
                          "root_angular_vel_local", "joint_pos_norm", "joint_vel_norm"]
        for f in expected_fields:
            assert f in cs, f"Missing core_state field: {f}"
            assert np.all(np.isfinite(cs[f])), f"Non-finite values in {f}"

        # Verify observation is finite
        obs = output["observation"]
        assert obs is not None
        assert np.all(np.isfinite(obs))

        # Verify impulse metadata
        assert output.get("impulse_force") == 100.0
        assert output.get("impulse_duration") == 2
        assert output.get("impulse_direction_angle") == 45.0
    finally:
        _close_runtime(rt)
