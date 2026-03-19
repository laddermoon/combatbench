from __future__ import annotations

import argparse
import os

import numpy as np

from ...core.humanoid_robot import HumanoidRobot
from ...envs.combat_gym import CombatGymEnv
from .selfplay_env import (
    STAND_ACTION_SCALE_MULTIPLIER,
    STAND_PD_KD,
    STAND_PD_KP,
    STAND_POLICY_ACTION_SCALE,
    STAND_REFERENCE_POSE,
)


def configure_runtime() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def build_stand_action_scale() -> np.ndarray:
    scale = np.zeros(HumanoidRobot.ACTION_DIM, dtype=np.float32)
    joint_names = tuple(HumanoidRobot.CONTROLLED_JOINTS)
    joint_name_to_index = {joint_name: idx for idx, joint_name in enumerate(joint_names)}
    for joint_name, joint_scale in STAND_POLICY_ACTION_SCALE.items():
        scale[joint_name_to_index[joint_name]] = float(joint_scale) * STAND_ACTION_SCALE_MULTIPLIER
    return scale


def configure_fight_baseline(env: CombatGymEnv) -> None:
    stand_pose = {"robot_a": dict(STAND_REFERENCE_POSE), "robot_b": dict(STAND_REFERENCE_POSE)}
    action_scale = build_stand_action_scale()
    env.set_robot_joint_positions(stand_pose)
    env.set_controller_reference_positions(stand_pose)
    env.set_controller_gains(kp=STAND_PD_KP, kd=STAND_PD_KD)
    env.set_controller_action_scale({"robot_a": action_scale, "robot_b": action_scale})


def validate_single_observation(env: CombatGymEnv, obs: dict[str, np.ndarray], atol: float) -> None:
    slices = HumanoidRobot.OBSERVATION_SLICES
    for robot_id, opponent_id in (("robot_a", "robot_b"), ("robot_b", "robot_a")):
        robot = env.robot_a if robot_id == "robot_a" else env.robot_b
        opponent = env.robot_b if opponent_id == "robot_b" else env.robot_a
        robot_obs = obs[f"{robot_id}_obs"]
        torso_state = robot.get_torso_state()
        opponent_torso = opponent.get_torso_state()
        rot_matrix = robot._get_rotation_matrix(torso_state["orientation"])

        expected_rel_pos = opponent_torso["position"] - torso_state["position"]
        expected_rel_vel = opponent_torso["linear_velocity"] - torso_state["linear_velocity"]
        expected_orientation = opponent_torso["orientation"]

        if not np.allclose(robot_obs[slices["opponent_relative_position"]], expected_rel_pos, atol=atol):
            raise AssertionError(f"{robot_id} opponent_relative_position mismatch")
        if not np.allclose(robot_obs[slices["opponent_relative_velocity"]], expected_rel_vel, atol=atol):
            raise AssertionError(f"{robot_id} opponent_relative_velocity mismatch")
        if not np.allclose(robot_obs[slices["opponent_orientation"]], expected_orientation, atol=atol):
            raise AssertionError(f"{robot_id} opponent_orientation mismatch")

        opponent_keypoints = opponent.get_keypoint_positions()
        expected_keypoint_positions = []
        for key in ["head", "right_hand", "left_hand", "right_elbow", "left_elbow", "right_knee", "left_knee", "right_foot", "left_foot"]:
            rel_pos = opponent_keypoints[key] - torso_state["position"]
            expected_keypoint_positions.append(rot_matrix.T @ rel_pos)
        expected_keypoint_positions = np.concatenate(expected_keypoint_positions)
        if not np.allclose(robot_obs[slices["opponent_keypoint_positions"]], expected_keypoint_positions, atol=atol):
            raise AssertionError(f"{robot_id} opponent_keypoint_positions mismatch")

        opponent_keyvels = opponent.get_keypoint_velocities()
        expected_keypoint_velocities = []
        for key in ["head", "right_hand", "left_hand", "right_elbow", "left_elbow", "right_knee", "left_knee", "right_foot", "left_foot"]:
            rel_vel = opponent_keyvels[key] - torso_state["linear_velocity"]
            expected_keypoint_velocities.append(rot_matrix.T @ rel_vel)
        expected_keypoint_velocities = np.concatenate(expected_keypoint_velocities)
        if not np.allclose(robot_obs[slices["opponent_keypoint_velocities"]], expected_keypoint_velocities, atol=atol):
            raise AssertionError(f"{robot_id} opponent_keypoint_velocities mismatch")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate fight damage logic and opponent observations")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--control-frequency", type=int, default=20)
    parser.add_argument("--initial-distance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-check-steps", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser


def main() -> None:
    configure_runtime()
    parser = build_arg_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    env = CombatGymEnv(
        render_mode=None,
        match_duration=args.duration,
        control_frequency=args.control_frequency,
        initial_distance=args.initial_distance,
    )
    obs, info = env.reset(seed=args.seed)
    configure_fight_baseline(env)
    obs = env._get_obs()
    info = env._build_info()
    validate_single_observation(env, obs, atol=args.atol)
    print("observation_check=passed step=0")

    previous_scores = dict(info["scores"])
    damage_events = []
    max_steps = int(args.duration * args.control_frequency)
    for step in range(1, max_steps + 1):
        action = {
            "robot_a": np.random.uniform(-1.0, 1.0, HumanoidRobot.ACTION_DIM).astype(np.float32),
            "robot_b": np.random.uniform(-1.0, 1.0, HumanoidRobot.ACTION_DIM).astype(np.float32),
        }
        obs, _, terminated, truncated, info = env.step(action)
        if step <= args.obs_check_steps:
            validate_single_observation(env, obs, atol=args.atol)
            print(f"observation_check=passed step={step}")

        for robot_id in ("robot_a", "robot_b"):
            hit_records = info["hit_records"][robot_id]
            if not hit_records:
                continue
            damage_sum = sum(float(record["damage"]) for record in hit_records)
            actual_delta = float(info["scores"][robot_id] - previous_scores[robot_id])
            expected_delta = max(damage_sum, -float(previous_scores[robot_id]))
            if not np.isclose(actual_delta, expected_delta):
                raise AssertionError(
                    f"Damage accounting mismatch for {robot_id}: score_delta={actual_delta}, expected_delta={expected_delta}, damage_sum={damage_sum}"
                )
            print(f"damage_check=passed step={step} defender={robot_id} score_delta={actual_delta} hits={hit_records}")
            damage_events.append((step, robot_id, actual_delta, hit_records))

        previous_scores = dict(info["scores"])
        if terminated or truncated:
            break

    final_scores = dict(info["scores"])
    print(f"damage_events={len(damage_events)}")
    print(f"final_scores={final_scores}")
    print(f"end_reason={info.get('end_reason')}")
    env.close()
    if final_scores["robot_a"] >= 100 and final_scores["robot_b"] >= 100:
        raise SystemExit("No damage observed within the configured rollout")


if __name__ == "__main__":
    main()
