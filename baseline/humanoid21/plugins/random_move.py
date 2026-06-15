"""Plugin to control the opponent robot to move randomly on the ground.

Used during training the Chaser (Approach) policy. The opponent robot acts as a moving point,
roaming around the arena smoothly while maintaining a safe distance from the trained robot
to avoid collisions, and staying perfectly upright in a standing posture.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.framework import BasePlugin, SimContext


class RandomMovePlugin(BasePlugin):
    """Controls the target robot to roam randomly within the arena, facing the trained robot."""

    _ROBOT_IDS = ("robot_a", "robot_b")

    # Static priors — arena geometry and control frequency are fixed.
    ARENA_RADIUS: float = 3.0
    ACTION_DT: float = 0.05  # s (control frequency 20Hz -> 0.05s)

    # Standing joint targets (joint_pos_norm space, from INITIAL_POSES['standing']['action']).
    # joint_pos_norm=0 is the joint-range midpoint (squat), not standing.
    STANDING_JOINT_POS: np.ndarray = np.array([
        -0.0000, 0.4286, -0.0000,
        0.5000, 0.2632, 0.7647, 0.9753, -0.0000, -0.0000,
        0.5000, 0.2632, 0.7647, 0.9753, -0.0000, -0.0000,
        0.1724, 0.1724, 0.3333, 0.1724, 0.1724, 0.3333,
    ], dtype=np.float32)

    def __init__(
        self,
        target_robot: str = "robot_b",
        speed: float = 0.6,               # m/s
        min_avoid_distance: float = 1.2,   # m (minimum distance to keep from trained robot)
        random_seed: Optional[int] = None,
    ) -> None:
        self.target_robot = str(target_robot)
        self.trained_robot = self._other_robot(target_robot)
        self.speed = float(speed)
        self.min_avoid_distance = float(min_avoid_distance)

        # Precompute per-step displacement (m/s -> m/step) so the step logic
        # doesn't need to multiply by ACTION_DT every call.
        self._step_distance: float = self.speed * self.ACTION_DT

        self._rng = np.random.RandomState(random_seed)
        self._waypoint: Optional[np.ndarray] = None  # 2D waypoint [x, y]
        self._steps_on_current_waypoint = 0
        self._max_steps_per_waypoint = 100            # Force change waypoint if stuck

    @staticmethod
    def _other_robot(robot_id: str) -> str:
        if robot_id == "robot_a":
            return "robot_b"
        if robot_id == "robot_b":
            return "robot_a"
        raise ValueError(f"Unknown robot_id {robot_id!r}; expected 'robot_a' or 'robot_b'")

    def set_episode_seed(self, seed: int) -> None:
        """Rebuild the RNG immediately for reproducibility."""
        self._rng = np.random.RandomState(int(seed))

    @property
    def name(self) -> str:
        return f"{self.target_robot}_random_move"

    @property
    def require_mutator(self) -> bool:
        return True

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robot": self.target_robot,
            "speed": self.speed,
            "min_avoid_distance": self.min_avoid_distance,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "RandomMovePlugin":
        return cls(**config)

    def _pick_new_waypoint(self, trained_pos: np.ndarray) -> None:
        """Sample a new random waypoint inside the arena radius, biased away from the trained robot."""
        for _ in range(20):  # Try up to 20 times to find a good waypoint
            angle = self._rng.uniform(0, 2 * np.pi)
            r = self._rng.uniform(0.5, self.ARENA_RADIUS * 0.9)
            wp = np.array([r * np.cos(angle), r * np.sin(angle)])
            
            # Avoid picking a waypoint that is directly on top of or too close to robot_a
            if np.linalg.norm(wp - trained_pos) > self.min_avoid_distance * 1.5:
                self._waypoint = wp
                self._steps_on_current_waypoint = 0
                return
                
        # Fallback to a simple mirrored or safe waypoint
        self._waypoint = -trained_pos * 0.8
        self._steps_on_current_waypoint = 0

    def on_pre_episode(self, ctx: SimContext) -> None:
        """Initialize waypoint at the start of the episode."""
        core_state = ctx.accessor.get_core_state()
        if self.trained_robot not in core_state or self.target_robot not in core_state:
            return
            
        trained_pos = np.asarray(core_state[self.trained_robot]["root_pos"][:2], dtype=np.float32)
        self._pick_new_waypoint(trained_pos)

    def on_pre_action_step(self, ctx: SimContext) -> None:
        """Update target robot position, orientation, and force standard standing pose."""
        core_state = ctx.accessor.get_core_state()
        if self.trained_robot not in core_state or self.target_robot not in core_state:
            return

        opp_pos_3d = np.asarray(core_state[self.target_robot]["root_pos"], dtype=np.float32)
        opp_pos = opp_pos_3d[:2]
        trained_pos = np.asarray(core_state[self.trained_robot]["root_pos"][:2], dtype=np.float32)

        # 1. Initialize waypoint if not done yet
        if self._waypoint is None:
            self._pick_new_waypoint(trained_pos)

        # 2. Increment step counters
        self._steps_on_current_waypoint += 1
        
        # 3. Check if we reached the waypoint or timed out
        dist_to_wp = np.linalg.norm(self._waypoint - opp_pos)
        if dist_to_wp < 0.2 or self._steps_on_current_waypoint > self._max_steps_per_waypoint:
            self._pick_new_waypoint(trained_pos)
            dist_to_wp = np.linalg.norm(self._waypoint - opp_pos)

        # 4. Calculate proposed move direction
        if dist_to_wp > 1e-5:
            move_dir = (self._waypoint - opp_pos) / dist_to_wp
        else:
            move_dir = np.zeros(2, dtype=np.float32)

        # 5. Proposed next position
        step_dist = self._step_distance
        proposed_pos = opp_pos + step_dist * move_dir

        # 6. Apply collision avoidance (repel from trained robot)
        dist_to_trained = np.linalg.norm(proposed_pos - trained_pos)
        if dist_to_trained < self.min_avoid_distance:
            # Pick a repulsive direction pointing directly away from the trained robot
            repulse_dir = opp_pos - trained_pos
            repulse_dist = np.linalg.norm(repulse_dir)
            if repulse_dist > 1e-5:
                repulse_dir = repulse_dir / repulse_dist
            else:
                repulse_dir = np.array([1.0, 0.0], dtype=np.float32)

            # Override movement to push away and pick a new waypoint
            move_dir = repulse_dir
            proposed_pos = opp_pos + step_dist * move_dir
            self._pick_new_waypoint(trained_pos)

        # 7. Apply arena boundaries (circle of arena_radius)
        if np.linalg.norm(proposed_pos) > self.ARENA_RADIUS:
            proposed_pos = (proposed_pos / np.linalg.norm(proposed_pos)) * self.ARENA_RADIUS
            self._pick_new_waypoint(trained_pos)

        # 8. Calculate heading rotation (always face the trained robot)
        vector_to_trained = trained_pos - proposed_pos
        heading_angle = np.arctan2(vector_to_trained[1], vector_to_trained[0])
        
        # Convert heading angle to quaternion [w, x, y, z]
        rot = R.from_euler('z', heading_angle)
        quat_xyzw = rot.as_quat()  # [x, y, z, w]
        root_rot = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

        # 9. Formulate new state with standard standing posture
        standing_z = opp_pos_3d[2]  # Keep current Z position to adapt to ground shape
        
        # Override target robot core state (position, rotation, velocities, joints).
        new_state = {
            self.target_robot: {
                "root_pos": np.array([proposed_pos[0], proposed_pos[1], standing_z], dtype=np.float32),
                "root_rot": root_rot,
                "root_vel_local": np.zeros(3, dtype=np.float32),
                "root_angular_vel_local": np.zeros(3, dtype=np.float32),
                "joint_pos_norm": self.STANDING_JOINT_POS.copy(),
                "joint_vel_norm": np.zeros(21, dtype=np.float32),  # Rigid stillness
            }
        }
        ctx.mutator.set_core_state(new_state)

        # Override PD targets so the external policy's action is completely
        # ignored during physics sub-steps.  Without this, the random policy's
        # joint targets would fight the standing pose and cause drift.
        ctx.mutator.set_action({self.target_robot: self.STANDING_JOINT_POS.copy()})
