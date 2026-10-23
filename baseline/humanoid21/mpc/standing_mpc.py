"""Observation-only simulation-feedback MPC for Humanoid21 standing.

The policy receives only the 96-dimensional observation.  Each replanning call
reconstructs the ego robot in a canonical yaw gauge, rolls out candidate
low-level feedback gains in an independent simulator, and commits to the best
gain set for a short interval.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

# Keep the policy importable both as a repository module and through a
# file:-based PolicyBlueprint.
for parent in Path(__file__).resolve().parents:
    if (parent / "envs" / "framework" / "policy.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from envs.framework.policy import Policy
from envs.humanoid21.simulator import Humanoid21Simulator

from .observation_codec import (
    DecodedSelfObservation,
    decode_self_observation,
    standing_action,
)


# Controlled-joint indices in Humanoid21Meta.CONTROLLED_JOINTS order.
ANKLE_Y_JOINTS = (7, 14)
ANKLE_X_JOINTS = (8, 15)


@dataclass(frozen=True)
class FeedbackGains:
    """Low-level standing-balance gains evaluated by the MPC rollout."""

    pitch_angle: float
    pitch_rate: float
    roll_angle: float = 0.0
    roll_rate: float = 0.0
    roll_sign: float = 0.0


@dataclass(frozen=True)
class StandingMPCConfig:
    horizon_control_steps: int = 40
    physics_steps_per_action: int = 25
    replan_interval: int = 10
    initial_distance: float = 2.0
    opponent_root_x: float = 3.0
    target_height: float = 1.282
    min_height: float = 0.85


DEFAULT_GAIN_CANDIDATES: Tuple[FeedbackGains, ...] = (
    # Weak/no feedback looks attractive over a short horizon because it has a
    # smaller action deviation, but it allows the slow forward-pitch failure
    # mode.  The default library therefore only includes gains that can hold
    # the canonical posture in a two-second branch rollout.
    FeedbackGains(4.0, 0.2),
    FeedbackGains(4.0, 0.5),
    FeedbackGains(4.0, 1.0),
    FeedbackGains(6.0, 0.5),
    FeedbackGains(6.0, 1.0),
    FeedbackGains(8.0, 1.0),
)


class StandingMPCPolicy(Policy):
    """Observation-only predictive baseline for canonical standing.

    The opponent is intentionally fixed to the standing PD target inside the
    branch simulator.  This first baseline solves self-motion only; opponent
    intent is outside the scope of the current planning contract.
    """

    def __init__(
        self,
        robot_id: str = "robot_a",
        horizon_control_steps: int = 40,
        replan_interval: int = 10,
        seed: int = 0,
        gain_candidates: Optional[Iterable[FeedbackGains]] = None,
        **overrides: Any,
    ):
        if robot_id not in ("robot_a", "robot_b"):
            raise ValueError(f"unsupported robot_id {robot_id!r}")
        self.robot_id = robot_id
        self.opponent_id = "robot_b" if robot_id == "robot_a" else "robot_a"
        cfg = dict(
            horizon_control_steps=horizon_control_steps,
            replan_interval=replan_interval,
        )
        cfg.update(overrides)
        self.config = StandingMPCConfig(**cfg)
        self.gain_candidates = tuple(gain_candidates or DEFAULT_GAIN_CANDIDATES)
        self.nominal_action = standing_action()
        self._seed = int(seed)
        self._probe: Optional[Humanoid21Simulator] = None
        self._template_data: Optional[mujoco.MjData] = None
        self._active_gains = self.gain_candidates[0]
        self._steps_since_replan = self.config.replan_interval
        self._last_action = self.nominal_action.copy()
        self._last_extra: Dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = int(seed)
        self._active_gains = self.gain_candidates[0]
        self._steps_since_replan = self.config.replan_interval
        self._last_action = self.nominal_action.copy()
        self._last_extra = {"event": "reset"}
        if self._probe is not None:
            self._probe.reset(seed=self._seed)

    def close(self) -> None:
        if self._probe is not None:
            self._probe.close()
            self._probe = None
            self._template_data = None

    def act(
        self,
        observation: Any,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        decoded = decode_self_observation(np.asarray(observation, dtype=np.float32))
        if self._steps_since_replan >= self.config.replan_interval:
            self._replan(decoded)
            self._steps_since_replan = 0
        else:
            self._last_extra = {
                "event": "reuse_gains",
                "active_gains": self._active_gains.__dict__.copy(),
                "steps_since_replan": self._steps_since_replan,
            }
        self._steps_since_replan += 1
        action = self._feedback_action(decoded, self._active_gains)
        self._last_action = action.copy()
        return action, (dict(self._last_extra) if want_extra else None)

    def _ensure_probe(self) -> Humanoid21Simulator:
        if self._probe is None:
            self._probe = Humanoid21Simulator(
                initial_distance=self.config.initial_distance,
                initial_pose_a="standing",
                initial_pose_b="standing",
            )
            self._probe.reset(seed=self._seed)
        return self._probe

    def _replan(self, decoded: DecodedSelfObservation) -> None:
        probe = self._ensure_probe()
        # Standardize the unobservable simulator context.  This is deliberately
        # not a privileged MjData clone: only Observation-derived ego state is
        # injected.  The opponent is moved away and held at standing target.
        probe.reset(seed=self._seed)
        probe_state = probe.get_core_state()
        opponent_state = probe_state[self.opponent_id]
        opponent_state["root_pos"] = opponent_state["root_pos"].copy()
        opponent_state["root_pos"][0] = self.config.opponent_root_x
        opponent_state["root_pos"][1] = 0.0
        probe.set_core_state({
            self.robot_id: decoded.core_state,
            self.opponent_id: opponent_state,
        })

        if self._template_data is None:
            self._template_data = mujoco.MjData(probe.model)
        mujoco.mj_copyData(self._template_data, probe.model, probe.data)
        template_targets = {
            rid: target.copy() for rid, target in probe._target_pos_norm.items()
        }

        t0 = time.perf_counter()
        scores = []
        for gains in self.gain_candidates:
            score, diagnostics = self._evaluate_gains(
                gains, template_targets=template_targets
            )
            scores.append((score, gains, diagnostics))
        scores.sort(key=lambda item: item[0])
        best_score, best_gains, best_diag = scores[0]
        self._active_gains = best_gains
        self._last_extra = {
            "event": "replan",
            "active_gains": best_gains.__dict__.copy(),
            "best_score": float(best_score),
            "scores": [
                {"score": float(score), "gains": gains.__dict__.copy()}
                for score, gains, _ in scores
            ],
            "diagnostics": best_diag,
            "plan_wall_time_s": float(time.perf_counter() - t0),
        }

    def _evaluate_gains(
        self,
        gains: FeedbackGains,
        *,
        template_targets: Dict[str, np.ndarray],
    ) -> Tuple[float, Dict[str, float]]:
        probe = self._ensure_probe()
        assert self._template_data is not None
        mujoco.mj_copyData(probe.data, probe.model, self._template_data)
        probe._target_pos_norm = {
            rid: target.copy() for rid, target in template_targets.items()
        }
        probe._data_cache.clear()

        cost = 0.0
        min_height = float("inf")
        max_abs_action_delta = 0.0
        discount = 1.0
        for _ in range(self.config.horizon_control_steps):
            decoded = self._read_probe_observation()
            action = self._feedback_action(decoded, gains)
            max_abs_action_delta = max(
                max_abs_action_delta,
                float(np.max(np.abs(action - self.nominal_action))),
            )
            probe.set_action({
                self.robot_id: action.astype(np.float32),
                self.opponent_id: self.nominal_action,
            })
            for _ in range(self.config.physics_steps_per_action):
                probe.physical_step()
            cost += discount * self._state_cost(probe)
            min_height = min(min_height, self._root_position(probe)[2])
            discount *= 0.995
            if self._root_position(probe)[2] < self.config.min_height:
                break

        if min_height < self.config.min_height:
            cost += 10_000.0 * (self.config.min_height - min_height)
        return cost, {
            "min_height": float(min_height),
            "max_abs_action_delta": float(max_abs_action_delta),
        }

    def _read_probe_observation(self) -> DecodedSelfObservation:
        """Read only Observation-equivalent ego quantities from the branch."""
        probe = self._ensure_probe()
        cache = probe._robot(self.robot_id)
        root_qpos = cache["root_qpos_adr"]
        root_qvel = cache["root_qvel_adr"]
        qpos_indices = cache["qpos_indices"]
        qvel_indices = cache["qvel_indices"]
        norm = probe._norm_params[self.robot_id]

        root_pos = probe.data.qpos[root_qpos:root_qpos + 3].copy()
        root_quat = probe.data.qpos[root_qpos + 3:root_qpos + 7].copy()
        rot = R.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
        rot_inv = rot.inv()
        projected_gravity = -rot.as_matrix()[2, :]
        linear_vel = rot_inv.apply(probe.data.qvel[root_qvel:root_qvel + 3])
        angular_vel = probe.data.qvel[root_qvel + 3:root_qvel + 6].copy()
        joint_pos_norm = (
            (probe.data.qpos[qpos_indices] - norm["reference"]) / norm["scale"]
        )
        joint_vel_norm = probe.data.qvel[qvel_indices] / norm["scale"]
        arena_center_local = rot_inv.apply(-root_pos)
        feet_forces = probe._get_feet_forces(self.robot_id)
        return DecodedSelfObservation(
            core_state={
                "root_pos": root_pos.astype(np.float32),
                "root_rot": root_quat.astype(np.float32),
                "root_vel_local": linear_vel.astype(np.float32),
                "root_angular_vel_local": angular_vel.astype(np.float32),
                "joint_pos_norm": joint_pos_norm.astype(np.float32),
                "joint_vel_norm": joint_vel_norm.astype(np.float32),
            },
            projected_gravity=projected_gravity,
            height=float(root_pos[2]),
            linear_vel_local=linear_vel,
            angular_vel_local=angular_vel,
            feet_forces=np.asarray(feet_forces),
            arena_center_local=arena_center_local,
        )

    def _feedback_action(
        self,
        decoded: DecodedSelfObservation,
        gains: FeedbackGains,
    ) -> np.ndarray:
        action = self.nominal_action.astype(np.float64).copy()
        pitch = gains.pitch_angle * decoded.projected_gravity[0]
        pitch += gains.pitch_rate * decoded.angular_vel_local[1]
        action[list(ANKLE_Y_JOINTS)] += pitch
        roll = gains.roll_sign * (
            gains.roll_angle * decoded.projected_gravity[1]
            + gains.roll_rate * decoded.angular_vel_local[0]
        )
        action[list(ANKLE_X_JOINTS)] += roll
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _root_position(self, sim: Humanoid21Simulator) -> np.ndarray:
        cache = sim._robot(self.robot_id)
        adr = cache["root_qpos_adr"]
        return sim.data.qpos[adr:adr + 3]

    def _state_cost(self, sim: Humanoid21Simulator) -> float:
        cache = sim._robot(self.robot_id)
        norm = sim._norm_params[self.robot_id]
        root_qpos = cache["root_qpos_adr"]
        root_qvel = cache["root_qvel_adr"]
        qpos_indices = cache["qpos_indices"]
        qvel_indices = cache["qvel_indices"]

        root_pos = sim.data.qpos[root_qpos:root_qpos + 3]
        root_quat = sim.data.qpos[root_qpos + 3:root_qpos + 7]
        rot = R.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
        uprightness = rot.as_matrix()[2, 2]
        joint_pos_norm = (
            (sim.data.qpos[qpos_indices] - norm["reference"]) / norm["scale"]
        )
        joint_vel_norm = sim.data.qvel[qvel_indices] / norm["scale"]
        root_lin_vel = sim.data.qvel[root_qvel:root_qvel + 3]
        root_ang_vel = sim.data.qvel[root_qvel + 3:root_qvel + 6]
        action_delta = (
            sim._target_pos_norm[self.robot_id] - self.nominal_action
        )

        return float(
            20.0 * (root_pos[2] - self.config.target_height) ** 2
            + 80.0 * (1.0 - uprightness)
            + 0.25 * np.sum((joint_pos_norm - self.nominal_action) ** 2)
            + 0.03 * np.sum(joint_vel_norm ** 2)
            + 0.40 * np.sum(root_lin_vel ** 2)
            + 0.10 * np.sum(root_ang_vel ** 2)
            + 0.05 * np.sum(action_delta ** 2)
        )
