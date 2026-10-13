"""Kinematic move plugin for a fixed-pose moving target.

Companion to ``RandomPoseInitPlugin``: captures whatever pose the
target robot is in at the first action step of the episode (joint
positions, root orientation incl. any lean, root height), then freezes
that pose for the whole episode while the root XY roams the arena —
the target is a mannequin sliding around.

Differences vs ``RandomMovePlugin``:

* Pose is CAPTURED (from the pose-init plugin's output), not forced to
  the canonical standing joint vector.
* Per-episode speed is sampled from ``[speed_min, speed_max]`` via the
  seeded RNG instead of being a fixed constructor value.
* Heading is NOT locked facing the learner: a target yaw is re-sampled
  whenever a new waypoint is picked and the root yaws toward it at
  ``yaw_rate`` rad/s — slow random heading drift, so the learner sees
  the target front/side/back across an episode.  Any lean in the
  captured root rotation is preserved (the yaw drift is applied as a
  world-frame Z rotation on top of it).
* ``min_avoid_distance`` defaults lower (0.6 m) — the whole point of
  this task is that the learner can REACH the target.

Same kinematic contract as RandomMovePlugin: position is teleported
each action step and velocities are zeroed — contacts inside an action
step are physically real, but the target does not get knocked back.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.framework import SimContext

from baseline.humanoid21.plugins.random_move import RandomMovePlugin


class PoseMovePlugin(RandomMovePlugin):
    """Frozen-pose moving target with slow random heading drift."""

    def __init__(
        self,
        target_robot: str = "robot_b",
        speed_min: float = 0.3,
        speed_max: float = 0.9,
        min_avoid_distance: float = 0.6,
        yaw_rate: float = 0.5,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            target_robot=target_robot,
            speed=speed_max,
            min_avoid_distance=min_avoid_distance,
            random_seed=random_seed,
        )
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.yaw_rate = float(yaw_rate)

        # Per-episode state (re-sampled / captured in on_pre_episode
        # and lazily on the first action step).
        self._ep_speed = self.speed_max
        self._pose: Optional[Dict[str, Any]] = None
        self._yaw_cur = 0.0
        self._yaw_target = 0.0

    @property
    def name(self) -> str:
        return f"{self.target_robot}_pose_move"

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robot": self.target_robot,
            "speed_min": self.speed_min,
            "speed_max": self.speed_max,
            "min_avoid_distance": self.min_avoid_distance,
            "yaw_rate": self.yaw_rate,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "PoseMovePlugin":
        return cls(**config)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._ep_speed = float(
            self._rng.uniform(self.speed_min, self.speed_max))
        self._step_distance = self._ep_speed * self.ACTION_DT
        self._pose = None            # captured on first action step
        self._yaw_cur = 0.0
        self._yaw_target = 0.0
        self._waypoint = None
        self._steps_on_current_waypoint = 0
        ctx.metrics[f"{self.target_robot}_move_speed"] = self._ep_speed

    def _capture_pose(self, ctx: SimContext) -> None:
        core_state = ctx.accessor.get_core_state()
        target = core_state.get(self.target_robot)
        if target is None:
            return
        self._pose = {
            "joint_pos_norm": np.asarray(
                target["joint_pos_norm"], dtype=np.float32).copy(),
            "root_rot": np.asarray(
                target["root_rot"], dtype=np.float32).copy(),
            "root_z": float(np.asarray(target["root_pos"])[2]),
        }

    # ------------------------------------------------------------------
    # Per-step kinematic control
    # ------------------------------------------------------------------

    def on_pre_action_step(self, ctx: SimContext) -> None:
        core_state = ctx.accessor.get_core_state()
        if self.trained_robot not in core_state or self.target_robot not in core_state:
            return
        if self._pose is None:
            self._capture_pose(ctx)
            if self._pose is None:
                return

        opp_pos_3d = np.asarray(
            core_state[self.target_robot]["root_pos"], dtype=np.float32)
        opp_pos = opp_pos_3d[:2]
        trained_pos = np.asarray(
            core_state[self.trained_robot]["root_pos"][:2],
            dtype=np.float32)

        if self._waypoint is None:
            self._pick_new_waypoint(trained_pos)
            self._yaw_target = float(self._rng.uniform(-np.pi, np.pi))

        self._steps_on_current_waypoint += 1

        dist_to_wp = np.linalg.norm(self._waypoint - opp_pos)
        if dist_to_wp < 0.2 or self._steps_on_current_waypoint > self._max_steps_per_waypoint:
            self._pick_new_waypoint(trained_pos)
            self._yaw_target = float(self._rng.uniform(-np.pi, np.pi))
            dist_to_wp = np.linalg.norm(self._waypoint - opp_pos)

        if dist_to_wp > 1e-5:
            move_dir = (self._waypoint - opp_pos) / dist_to_wp
        else:
            move_dir = np.zeros(2, dtype=np.float32)

        step_dist = self._step_distance
        proposed_pos = opp_pos + step_dist * move_dir

        dist_to_trained = np.linalg.norm(proposed_pos - trained_pos)
        if dist_to_trained < self.min_avoid_distance:
            repulse_dir = opp_pos - trained_pos
            repulse_dist = np.linalg.norm(repulse_dir)
            if repulse_dist > 1e-5:
                repulse_dir = repulse_dir / repulse_dist
            else:
                repulse_dir = np.array([1.0, 0.0], dtype=np.float32)
            proposed_pos = opp_pos + step_dist * repulse_dir
            self._pick_new_waypoint(trained_pos)
            self._yaw_target = float(self._rng.uniform(-np.pi, np.pi))

        if np.linalg.norm(proposed_pos) > self.ARENA_RADIUS:
            proposed_pos = (
                proposed_pos / np.linalg.norm(proposed_pos)
            ) * self.ARENA_RADIUS
            self._pick_new_waypoint(trained_pos)

        # Slow heading drift: slew current yaw offset toward the
        # waypoint-coupled random target at yaw_rate rad/s.  Shortest-
        # arc steering keeps drift continuous, never snapping.
        d = self._yaw_target - self._yaw_cur
        d = (d + np.pi) % (2.0 * np.pi) - np.pi
        dyaw_max = self.yaw_rate * self.ACTION_DT
        self._yaw_cur += float(np.clip(d, -dyaw_max, dyaw_max))

        root_rot = self._compose_yaw(self._pose["root_rot"], self._yaw_cur)

        new_state = {
            self.target_robot: {
                "root_pos": np.array(
                    [proposed_pos[0], proposed_pos[1], self._pose["root_z"]],
                    dtype=np.float32),
                "root_rot": root_rot,
                "root_vel_local": np.zeros(3, dtype=np.float32),
                "root_angular_vel_local": np.zeros(3, dtype=np.float32),
                "joint_pos_norm": self._pose["joint_pos_norm"].copy(),
                "joint_vel_norm": np.zeros(21, dtype=np.float32),
            }
        }
        ctx.mutator.set_core_state(new_state)
        ctx.mutator.set_action(
            {self.target_robot: self._pose["joint_pos_norm"].copy()})

    @staticmethod
    def _compose_yaw(root_rot: np.ndarray, yaw: float) -> np.ndarray:
        """Apply a world-frame yaw on top of a captured (w,x,y,z) quat."""
        rot_init = R.from_quat(
            [root_rot[1], root_rot[2], root_rot[3], root_rot[0]])
        rot_new = R.from_euler("z", yaw) * rot_init
        x, y, z, w = rot_new.as_quat()
        return np.array([w, x, y, z], dtype=np.float32)
