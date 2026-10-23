"""Tests for the observation-only standing MPC baseline."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.humanoid21.simulator import Humanoid21Simulator

from .observation_codec import decode_self_observation, standing_action
from .standing_mpc import FeedbackGains, StandingMPCPolicy


def test_decode_self_observation_recovers_visible_core_fields():
    sim = Humanoid21Simulator()
    sim.reset(seed=0)
    decoded = decode_self_observation(sim.get_observation()["robot_a"])
    core = sim.get_core_state()["robot_a"]

    np.testing.assert_allclose(
        decoded.core_state["joint_pos_norm"], core["joint_pos_norm"], atol=1e-6
    )
    np.testing.assert_allclose(
        decoded.core_state["joint_vel_norm"], core["joint_vel_norm"], atol=1e-6
    )
    np.testing.assert_allclose(
        decoded.core_state["root_vel_local"], core["root_vel_local"], atol=1e-6
    )
    np.testing.assert_allclose(
        decoded.core_state["root_angular_vel_local"],
        core["root_angular_vel_local"],
        atol=1e-6,
    )
    assert decoded.height == core["root_pos"][2]
    sim.close()


def test_decoded_state_is_global_yaw_equivalent():
    sim = Humanoid21Simulator()
    sim.reset(seed=0)
    decoded = decode_self_observation(sim.get_observation()["robot_a"])
    core = sim.get_core_state()["robot_a"]

    # The canonical reset happens to have zero yaw for robot_a, so the chosen
    # gauge recovers the absolute state exactly.  Other yaws are intentionally
    # allowed to differ by a global Z rotation.
    np.testing.assert_allclose(
        decoded.core_state["root_pos"], core["root_pos"], atol=1e-6
    )
    assert abs(float(np.dot(decoded.core_state["root_rot"], core["root_rot"]))) > 0.999999
    sim.close()


def test_policy_returns_valid_action():
    sim = Humanoid21Simulator()
    sim.reset(seed=0)
    policy = StandingMPCPolicy(
        horizon_control_steps=4,
        replan_interval=2,
        gain_candidates=(FeedbackGains(4.0, 0.5),),
    )
    action, extra = policy.act(sim.get_observation()["robot_a"], want_extra=True)
    assert action.shape == (21,)
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0)
    assert extra["event"] == "replan"
    policy.close()
    sim.close()


def test_mpc_holds_standing_for_two_seconds():
    sim = Humanoid21Simulator()
    sim.reset(seed=0)
    policy = StandingMPCPolicy(
        horizon_control_steps=20,
        replan_interval=5,
        gain_candidates=(
            FeedbackGains(4.0, 0.5),
            FeedbackGains(6.0, 1.0),
        ),
    )
    opponent_action = standing_action()
    min_height = float("inf")
    min_uprightness = float("inf")
    for _ in range(40):
        action, _ = policy.act(sim.get_observation()["robot_a"])
        sim.set_action({"robot_a": action, "robot_b": opponent_action})
        for _ in range(25):
            sim.physical_step()
        core = sim.get_core_state()["robot_a"]
        quat = core["root_rot"]
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        min_height = min(min_height, float(core["root_pos"][2]))
        min_uprightness = min(min_uprightness, float(rot[2, 2]))

    assert min_height > 1.20
    assert min_uprightness > 0.95
    policy.close()
    sim.close()
