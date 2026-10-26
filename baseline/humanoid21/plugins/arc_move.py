"""Arc-segmented opponent mover (AttackPrep stage-3 target motion).

Kinematically drives the opponent robot, replacing RandomMovePlugin's
"random waypoint + fixed per-episode speed + avoid repulsion" model,
which biased follow training into pure forward chasing.

Motion model (three independent stochastic processes):

  * Path — segmented by ARC LENGTH: each segment lasts
    L ~ U(seg_len_min, seg_len_max) metres; inside a segment the heading
    evolves through tangent-joined small arcs (piecewise-constant
    curvature kappa ~ U(-kappa_max, +kappa_max), |kappa| <= 2 => turn
    radius >= 0.5 m).  At segment boundaries the heading is resampled
    arbitrarily — sharp direction reversals included, no smoothing.
    Near the arena boundary the segment ends early and the next heading
    is sampled inside the inward-pointing cone, so the mover can never
    hit the wall.

  * Speed — segmented by TIME: each segment lasts T ~ U(speed_seg_tmin,
    speed_seg_tmax) seconds with a fresh target v* ~ U(v_min, v_max).
    The actual speed ramps toward the target at |dv/dt| <= accel_max,
    so large target changes are bridged by a smooth transition.

  * Facing — re-sampled together with each speed segment: with
    probability face_learner_prob the torso tracks the learner;
    otherwise it holds a random fixed yaw for the whole segment.

The target may legitimately move TOWARD the learner — only a hard
``avoid_min`` contact radius is enforced (the step is clamped at that
circle instead of repelled), so close-range band-distance training is
possible.  The reported root velocity / yaw rate reflect the actual
kinematic displacement so the learner's 96-dim obs stays truthful.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.framework import BasePlugin, SimContext


class ArcMovePlugin(BasePlugin):
    """Arc-segment wandering opponent with time-segmented speed."""

    _ROBOT_IDS = ("robot_a", "robot_b")

    ARENA_RADIUS: float = 3.0
    ACTION_DT: float = 0.05  # s (20 Hz control)

    # Standing joint targets (joint_pos_norm space, from
    # INITIAL_POSES['standing']['action']); shared with RandomMovePlugin.
    STANDING_JOINT_POS: np.ndarray = np.array([
        -0.0000, 0.4286, -0.0000,
        0.5000, 0.2632, 0.7647, 0.9753, -0.0000, -0.0000,
        0.5000, 0.2632, 0.7647, 0.9753, -0.0000, -0.0000,
        0.1724, 0.1724, 0.3333, 0.1724, 0.1724, 0.3333,
    ], dtype=np.float32)

    def __init__(
        self,
        target_robot: str = "robot_b",
        v_max: float = 1.0,
        v_min: float = 0.0,
        speed_seg_tmin: float = 1.0,
        speed_seg_tmax: float = 3.0,
        accel_max: float = 0.8,
        seg_len_min: float = 0.5,
        seg_len_max: float = 2.0,
        kappa_max: float = 2.0,
        face_learner_prob: float = 0.5,
        boundary_margin: float = 0.4,
        avoid_min: float = 0.4,
        random_seed: Optional[int] = None,
    ) -> None:
        self.target_robot = str(target_robot)
        self.trained_robot = self._other_robot(target_robot)
        self.v_max = float(v_max)
        self.v_min = float(v_min)
        self.speed_seg_tmin = float(speed_seg_tmin)
        self.speed_seg_tmax = float(speed_seg_tmax)
        self.accel_max = float(accel_max)
        self.seg_len_min = float(seg_len_min)
        self.seg_len_max = float(seg_len_max)
        self.kappa_max = float(kappa_max)
        self.face_learner_prob = float(face_learner_prob)
        self.boundary_margin = float(boundary_margin)
        self.avoid_min = float(avoid_min)

        self._rng = np.random.RandomState(random_seed)

        # --- Motion state (reset per episode) ---
        self._v = 0.0                    # current speed (ramped)
        self._v_target = 0.0             # current segment's target speed
        self._speed_steps_left = 0       # action steps left in speed seg
        self._heading = 0.0              # motion direction (world yaw)
        self._seg_left = 0.0             # arc length left in path segment
        self._sub_left = 0.0             # arc length left in sub-arc
        self._kappa = 0.0                # curvature of current sub-arc
        self._face_learner = True        # facing mode of current segment
        self._fixed_yaw = 0.0            # yaw held when not facing learner
        self._facing = 0.0               # yaw written last step

    # ------------------------------------------------------------------
    @staticmethod
    def _other_robot(robot_id: str) -> str:
        if robot_id == "robot_a":
            return "robot_b"
        if robot_id == "robot_b":
            return "robot_a"
        raise ValueError(f"Unknown robot_id {robot_id!r}")

    def set_episode_seed(self, seed: int) -> None:
        self._rng = np.random.RandomState(int(seed))

    @property
    def name(self) -> str:
        return f"{self.target_robot}_arc_move"

    @property
    def require_mutator(self) -> bool:
        return True

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robot": self.target_robot,
            "v_max": self.v_max,
            "v_min": self.v_min,
            "speed_seg_tmin": self.speed_seg_tmin,
            "speed_seg_tmax": self.speed_seg_tmax,
            "accel_max": self.accel_max,
            "seg_len_min": self.seg_len_min,
            "seg_len_max": self.seg_len_max,
            "kappa_max": self.kappa_max,
            "face_learner_prob": self.face_learner_prob,
            "boundary_margin": self.boundary_margin,
            "avoid_min": self.avoid_min,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "ArcMovePlugin":
        return cls(**config)

    # ------------------------------------------------------------------
    # Segment sampling helpers
    # ------------------------------------------------------------------

    def _start_speed_segment(self) -> None:
        """New time-domain segment: speed target + facing mode."""
        t = self._rng.uniform(self.speed_seg_tmin, self.speed_seg_tmax)
        self._speed_steps_left = max(1, int(round(t / self.ACTION_DT)))
        self._v_target = float(
            self._rng.uniform(self.v_min, self.v_max))
        self._face_learner = bool(
            self._rng.uniform() < self.face_learner_prob)
        if not self._face_learner:
            self._fixed_yaw = float(self._rng.uniform(-np.pi, np.pi))

    def _start_path_segment(self, pos: np.ndarray) -> None:
        """New arc-length segment: resample heading (inward cone near wall)."""
        self._seg_left = float(
            self._rng.uniform(self.seg_len_min, self.seg_len_max))
        r = float(np.linalg.norm(pos))
        if r > self.ARENA_RADIUS - self.boundary_margin:
            # At/inside the margin ring: heading must point inward —
            # ±90° cone around the radially inward direction.
            inward = float(np.arctan2(-pos[1], -pos[0]))
            self._heading = float(
                inward + self._rng.uniform(-0.5 * np.pi, 0.5 * np.pi))
        else:
            self._heading = float(self._rng.uniform(-np.pi, np.pi))
        self._sub_left = 0.0  # force a fresh sub-arc too

    def _start_sub_arc(self) -> None:
        """New tangent-joined sub-arc inside the current path segment."""
        self._kappa = float(
            self._rng.uniform(-self.kappa_max, self.kappa_max))
        # Sub-arc length: a fraction of the remaining segment so a
        # segment typically contains 2-4 arcs.
        self._sub_left = min(
            self._seg_left,
            float(self._rng.uniform(0.2, 0.6)))

    # ------------------------------------------------------------------
    # Episode hooks
    # ------------------------------------------------------------------

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._v = 0.0
        self._v_target = 0.0
        self._speed_steps_left = 0
        self._seg_left = 0.0
        self._sub_left = 0.0
        self._kappa = 0.0
        core_state = ctx.accessor.get_core_state()
        if self.target_robot in core_state and self.trained_robot in core_state:
            opp_pos = np.asarray(
                core_state[self.target_robot]["root_pos"], dtype=np.float64)
            trained_pos = np.asarray(
                core_state[self.trained_robot]["root_pos"], dtype=np.float64)
            to_trained = trained_pos[:2] - opp_pos[:2]
            self._facing = float(np.arctan2(to_trained[1], to_trained[0]))

    def on_pre_action_step(self, ctx: SimContext) -> None:
        core_state = ctx.accessor.get_core_state()
        if self.trained_robot not in core_state or self.target_robot not in core_state:
            return

        opp_pos_3d = np.asarray(
            core_state[self.target_robot]["root_pos"], dtype=np.float64)
        opp_pos = opp_pos_3d[:2]
        trained_pos = np.asarray(
            core_state[self.trained_robot]["root_pos"], dtype=np.float64)[:2]

        # 1. Speed profile: roll a new time segment if due, then ramp.
        if self._speed_steps_left <= 0:
            self._start_speed_segment()
        self._speed_steps_left -= 1
        dv = np.clip(
            self._v_target - self._v,
            -self.accel_max * self.ACTION_DT,
            self.accel_max * self.ACTION_DT,
        )
        self._v += float(dv)

        # 2. Path: roll a new arc segment if due, then integrate the
        #    current sub-arc's curvature over this step's arc length.
        ds = self._v * self.ACTION_DT
        if self._seg_left <= 0.0:
            self._start_path_segment(opp_pos)
        if self._sub_left <= 0.0:
            self._start_sub_arc()
        self._heading += self._kappa * ds
        self._seg_left -= ds
        self._sub_left -= ds

        # 3. Proposed position.
        move_dir = np.array(
            [np.cos(self._heading), np.sin(self._heading)])
        proposed = opp_pos + ds * move_dir

        # 4. Arena boundary: clamp into the margin ring and end the
        #    segment — next step resamples inside the inward cone.
        r_prop = float(np.linalg.norm(proposed))
        r_lim = self.ARENA_RADIUS - self.boundary_margin
        if r_prop > r_lim:
            if r_prop > 1e-6:
                proposed = proposed * (r_lim / r_prop)
            self._seg_left = 0.0

        # 5. Learner contact radius: clamp the step at avoid_min instead
        #    of repelling — the target is allowed to approach.
        dvec = proposed - trained_pos
        d = float(np.linalg.norm(dvec))
        if d < self.avoid_min and ds > 0.0:
            if d > 1e-6:
                proposed = trained_pos + dvec * (self.avoid_min / d)
            else:
                proposed = opp_pos

        # 6. Facing: track learner or hold the segment's fixed yaw.
        prev_facing = self._facing
        if self._face_learner:
            to_trained = trained_pos - proposed
            if np.linalg.norm(to_trained) > 1e-6:
                self._facing = float(
                    np.arctan2(to_trained[1], to_trained[0]))
        else:
            self._facing = self._fixed_yaw

        # 7. Write kinematic state.  root_vel_local / angular_vel_local
        #    carry the TRUE motion so the learner's obs is consistent
        #    with the visible trajectory.
        rot = R.from_euler("z", self._facing)
        q = rot.as_quat()  # xyzw
        root_rot = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)

        world_vel = np.array(
            [(proposed[0] - opp_pos[0]) / self.ACTION_DT,
             (proposed[1] - opp_pos[1]) / self.ACTION_DT, 0.0])
        loc_vel = R.from_euler("z", -self._facing).apply(world_vel)
        yaw_rate = np.arctan2(
            np.sin(self._facing - prev_facing),
            np.cos(self._facing - prev_facing),
        ) / self.ACTION_DT

        new_state = {
            self.target_robot: {
                "root_pos": np.array(
                    [proposed[0], proposed[1], opp_pos_3d[2]],
                    dtype=np.float32),
                "root_rot": root_rot,
                "root_vel_local": loc_vel.astype(np.float32),
                "root_angular_vel_local": np.array(
                    [0.0, 0.0, yaw_rate], dtype=np.float32),
                "joint_pos_norm": self.STANDING_JOINT_POS.copy(),
                "joint_vel_norm": np.zeros(21, dtype=np.float32),
            }
        }
        ctx.mutator.set_core_state(new_state)
        # Override PD targets so the opponent's own policy action cannot
        # fight the scripted pose during physics sub-steps.
        ctx.mutator.set_action(
            {self.target_robot: self.STANDING_JOINT_POS.copy()})

        ctx.metrics[f"{self.target_robot}_move_speed"] = self._v
