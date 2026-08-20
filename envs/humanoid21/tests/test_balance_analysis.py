#!/usr/bin/env python3
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
from envs.humanoid21.observer_plugins import Humanoid21BalanceAnalysisObserver
from envs.humanoid21.simulator import Humanoid21Simulator


def _make_runtime_with_balance_observer():
    simulator = Humanoid21Simulator()
    observer = Humanoid21BalanceAnalysisObserver("robot_a")
    runtime = EnvRuntime(simulator=simulator, observer_plugins={"balance": observer})
    return simulator, runtime, observer


def _close_runtime(runtime: EnvRuntime) -> None:
    try:
        runtime.close()
    except Exception:
        pass


def _robot_total_mass(simulator: Humanoid21Simulator, robot_id: str) -> float:
    cache = simulator._robot(robot_id)
    body_ids = [int(body_id) for body_id in cache["body_ids"]]
    return float(np.sum(simulator.model.body_mass[body_ids]))


def _assert_ground_frame_self_consistency(balance_output: dict) -> None:
    assert balance_output["ground_support_frame_defined"]

    support_axis = np.asarray(balance_output["support_axis_unit_ground"], dtype=np.float64)
    support_lateral = np.asarray(balance_output["support_lateral_unit_ground"], dtype=np.float64)
    left_ground = np.asarray(balance_output["left_ankle_support_ground_projection"], dtype=np.float64)
    center_ground = np.asarray(balance_output["center_of_mass_ground_projection"], dtype=np.float64)
    velocity_ground = np.asarray(balance_output["center_of_mass_velocity_ground_projection"], dtype=np.float64)

    support_coordinate = float(balance_output["support_axis_projection_coordinate"])
    support_lateral_distance = float(balance_output["support_lateral_signed_distance"])
    velocity_along_support_axis = float(balance_output["center_of_mass_velocity_along_support_axis"])
    velocity_along_support_lateral_axis = float(balance_output["center_of_mass_velocity_along_support_lateral_axis"])

    reconstructed_center = left_ground + support_coordinate * support_axis + support_lateral_distance * support_lateral
    reconstructed_velocity = (
        velocity_along_support_axis * support_axis
        + velocity_along_support_lateral_axis * support_lateral
    )

    assert np.allclose(np.linalg.norm(support_axis), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(support_lateral), 1.0, atol=1e-6)
    assert np.allclose(np.dot(support_axis, support_lateral), 0.0, atol=1e-6)
    assert np.allclose(center_ground, reconstructed_center, atol=1e-6)
    assert np.allclose(velocity_ground, reconstructed_velocity, atol=1e-6)


def test_balance_geometry_projection_metrics() -> None:
    observer = Humanoid21BalanceAnalysisObserver("robot_a")
    result = observer._analyze_support_geometry(
        center_of_mass=np.array([0.5, 0.2, 1.15], dtype=np.float64),
        center_of_mass_velocity=np.array([0.3, -0.4, 2.0], dtype=np.float64),
        left_support=np.array([0.0, 0.0, 0.12], dtype=np.float64),
        right_support=np.array([1.0, 0.0, 0.08], dtype=np.float64),
        robot_forward_ground=np.array([0.0, 1.0], dtype=np.float64),
    )

    assert result["ground_support_frame_defined"]
    assert np.allclose(result["left_ankle_support_ground_projection"], [0.0, 0.0], atol=1e-6)
    assert np.allclose(result["right_ankle_support_ground_projection"], [1.0, 0.0], atol=1e-6)
    assert np.allclose(result["center_of_mass_ground_projection"], [0.5, 0.2], atol=1e-6)
    assert np.allclose(result["center_of_mass_velocity_ground_projection"], [0.3, -0.4], atol=1e-6)
    assert np.isclose(result["support_span"], 1.0, atol=1e-6)
    assert np.isclose(result["support_axis_projection_coordinate"], 0.5, atol=1e-6)
    assert np.isclose(result["support_segment_parameter"], 0.5, atol=1e-6)
    assert np.isclose(result["support_lateral_signed_distance"], 0.2, atol=1e-6)
    assert np.isclose(result["center_of_mass_velocity_along_support_axis"], 0.3, atol=1e-6)
    assert np.isclose(result["center_of_mass_velocity_along_support_lateral_axis"], -0.4, atol=1e-6)
    assert result["is_projected_between_support_points"]

    _assert_ground_frame_self_consistency(result)


def test_balance_geometry_degenerate_support_frame() -> None:
    observer = Humanoid21BalanceAnalysisObserver("robot_a")
    result = observer._analyze_support_geometry(
        center_of_mass=np.array([0.2, 0.1, 1.0], dtype=np.float64),
        center_of_mass_velocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        left_support=np.array([0.3, -0.2, 0.05], dtype=np.float64),
        right_support=np.array([0.3, -0.2, 0.25], dtype=np.float64),
        robot_forward_ground=np.array([1.0, 0.0], dtype=np.float64),
    )

    assert not result["ground_support_frame_defined"]
    assert not result["is_projected_between_support_points"]
    assert np.isclose(result["support_span"], 0.0, atol=1e-8)
    assert np.isnan(result["support_axis_projection_coordinate"])
    assert np.isnan(result["support_segment_parameter"])
    assert np.isnan(result["support_lateral_signed_distance"])
    assert np.isnan(result["center_of_mass_velocity_along_support_axis"])
    assert np.isnan(result["center_of_mass_velocity_along_support_lateral_axis"])


def test_balance_observer_initial_velocity_reflects_reset_state() -> None:
    simulator, runtime, _ = _make_runtime_with_balance_observer()
    try:
        runtime.reset(seed=42, options={"initial_pose_a": "standing", "initial_pose_b": "standing"})
        core_state = simulator.get_core_state()

        updated_state = {
            "robot_a": {
                **core_state["robot_a"],
                "root_vel_local": np.array([1.25, 0.0, 0.0], dtype=np.float32),
                "root_angular_vel_local": np.zeros(3, dtype=np.float32),
                "joint_vel_norm": np.zeros(21, dtype=np.float32),
            },
            "robot_b": core_state["robot_b"],
        }
        simulator.set_core_state(updated_state)
        runtime.refresh_observers(force=True)

        balance_output = runtime.get_observer_output("balance")
        center_of_mass_velocity = np.asarray(balance_output["center_of_mass_velocity"], dtype=np.float64)

        assert np.isclose(center_of_mass_velocity[0], 1.25, atol=0.08)
        assert np.isclose(center_of_mass_velocity[1], 0.0, atol=0.08)
        assert np.isclose(center_of_mass_velocity[2], 0.0, atol=0.08)
    finally:
        _close_runtime(runtime)


def test_balance_observer_static_standing_sanity() -> None:
    simulator, runtime, _ = _make_runtime_with_balance_observer()
    try:
        runtime.reset(seed=123, options={"initial_pose_a": "standing", "initial_pose_b": "standing"})
        standing_action = simulator.INITIAL_POSES["standing"]["action"]

        for _ in range(300):
            runtime.step(standing_action, standing_action)

        total_vertical_support_history = []
        last_balance_output = None
        for _ in range(80):
            runtime.step(standing_action, standing_action)
            balance_output = runtime.get_observer_output("balance")
            last_balance_output = balance_output
            total_vertical_support_history.append(
                float(balance_output["left_ankle_support_force"][2] + balance_output["right_ankle_support_force"][2])
            )

        assert last_balance_output is not None
        _assert_ground_frame_self_consistency(last_balance_output)

        center_of_mass = np.asarray(last_balance_output["center_of_mass"], dtype=np.float64)
        left_support_point = np.asarray(last_balance_output["left_ankle_support_point"], dtype=np.float64)
        right_support_point = np.asarray(last_balance_output["right_ankle_support_point"], dtype=np.float64)
        left_support_force = np.asarray(last_balance_output["left_ankle_support_force"], dtype=np.float64)
        right_support_force = np.asarray(last_balance_output["right_ankle_support_force"], dtype=np.float64)

        assert center_of_mass.shape == (3,)
        assert left_support_point.shape == (3,)
        assert right_support_point.shape == (3,)
        assert np.all(np.isfinite(center_of_mass))
        assert np.all(np.isfinite(left_support_point))
        assert np.all(np.isfinite(right_support_point))
        assert float(last_balance_output["support_span"]) > 0.05

        mean_total_vertical_support = float(np.mean(total_vertical_support_history))
        robot_weight = _robot_total_mass(simulator, "robot_a") * abs(float(simulator.model.opt.gravity[2]))

        assert left_support_force[2] >= 0.0
        assert right_support_force[2] >= 0.0
        assert mean_total_vertical_support > 0.0
        assert np.isclose(mean_total_vertical_support, robot_weight, rtol=0.35, atol=80.0)
    finally:
        _close_runtime(runtime)
