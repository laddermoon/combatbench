from typing import Any, Dict, Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


class BaseConstraint:
    name = "base"

    def reset(self, env: Any) -> None:
        pass

    def begin_step(self, env: Any) -> None:
        pass

    def apply(self, env: Any) -> Optional[Dict[str, Any]]:
        return None


class NonFallOrientationClamp(BaseConstraint):
    name = "non_fall_orientation_clamp"

    def __init__(self, pitch_limit_deg: float = 5.0, roll_limit_deg: float = 5.0):
        self.pitch_limit_deg = float(pitch_limit_deg)
        self.roll_limit_deg = float(roll_limit_deg)
        self.enabled = True
        self.step_clamp_counts = {"robot_a": 0, "robot_b": 0}
        self.episode_clamp_counts = {"robot_a": 0, "robot_b": 0}

    def reset(self, env: Any) -> None:
        self.step_clamp_counts = {"robot_a": 0, "robot_b": 0}
        self.episode_clamp_counts = {"robot_a": 0, "robot_b": 0}

    def begin_step(self, env: Any) -> None:
        self.step_clamp_counts = {"robot_a": 0, "robot_b": 0}

    def _clamp_robot(self, env: Any, robot_id: str) -> bool:
        root_joint = env.get_root_joint_cache().get(robot_id)
        if root_joint is None:
            return False
        qpos_adr = root_joint["qpos_adr"]
        orientation_wxyz = np.asarray(env.physics.data.qpos[qpos_adr + 3:qpos_adr + 7], dtype=np.float64)
        if np.linalg.norm(orientation_wxyz) < 1e-8:
            return False
        orientation_xyzw = np.array(
            [orientation_wxyz[1], orientation_wxyz[2], orientation_wxyz[3], orientation_wxyz[0]],
            dtype=np.float64,
        )
        rotation = R.from_quat(orientation_xyzw)
        roll, pitch, yaw = rotation.as_euler("xyz", degrees=True)
        clamped_roll = float(np.clip(roll, -self.roll_limit_deg, self.roll_limit_deg))
        clamped_pitch = float(np.clip(pitch, -self.pitch_limit_deg, self.pitch_limit_deg))
        if np.isclose(roll, clamped_roll) and np.isclose(pitch, clamped_pitch):
            return False
        clamped_rotation = R.from_euler("xyz", [clamped_roll, clamped_pitch, yaw], degrees=True)
        clamped_xyzw = clamped_rotation.as_quat()
        clamped_wxyz = np.array(
            [clamped_xyzw[3], clamped_xyzw[0], clamped_xyzw[1], clamped_xyzw[2]],
            dtype=np.float64,
        )
        env.physics.data.qpos[qpos_adr + 3:qpos_adr + 7] = clamped_wxyz
        qvel_adr = root_joint["qvel_adr"]
        env.physics.data.qvel[qvel_adr:qvel_adr + 2] = 0.0
        return True

    def apply(self, env: Any) -> Dict[str, Any]:
        changed = False
        clamped = {"robot_a": False, "robot_b": False}
        for robot_id in ("robot_a", "robot_b"):
            was_clamped = self._clamp_robot(env, robot_id)
            clamped[robot_id] = was_clamped
            if was_clamped:
                self.step_clamp_counts[robot_id] += 1
                self.episode_clamp_counts[robot_id] += 1
            changed = changed or was_clamped
        if changed:
            mujoco.mj_forward(env.physics.model, env.physics.data)
        return {
            "name": self.name,
            "enabled": self.enabled,
            "pitch_limit_deg": self.pitch_limit_deg,
            "roll_limit_deg": self.roll_limit_deg,
            "clamped": clamped,
            "changed": changed,
            "current_step": self.step_clamp_counts.copy(),
            "episode": self.episode_clamp_counts.copy(),
        }
