"""Tool: Observation -> local P5 via FK (shared obs decoder).

The 96-dim policy obs carries no self keypoints — the local P5 must
be reconstructed by forward kinematics:

    q_rad = obs[0:21] * scale + ref            (ranges from FK model)
    root quat = minimal rot g_body -> -z       (tilt, arbitrary yaw)
    FK -> landmarks -> torso yaw -> local P5
"""
from __future__ import annotations

import mujoco
import numpy as np



def _level_rot(g_body: np.ndarray) -> np.ndarray:
    """Minimal rotation taking body-frame gravity to [0,0,-1]."""
    g = np.asarray(g_body, np.float64)
    g = g / (np.linalg.norm(g) + 1e-12)
    target = np.array([0.0, 0.0, -1.0])
    axis = np.cross(g, target)
    n = np.linalg.norm(axis)
    if n < 1e-9:                                     # (anti-)parallel
        return np.eye(3) if g[2] < 0 else np.diag([1.0, -1.0, -1.0])
    axis /= n
    ang = float(np.arccos(np.clip(np.dot(g, target), -1.0, 1.0)))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K



def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1) * 2
        return np.array([s / 4, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1) * 2
    q = np.empty(4)
    q[i + 1] = s / 4
    q[0] = (R[k, j] - R[j, k]) / s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


class ObsDecoder:
    """96-dim policy obs -> local P5 via FK (lazy Humanoid21FK)."""

    def __init__(self, robot_id: str = "robot_a"):
        self.robot_id = robot_id
        self._fk = None

    def fk(self):
        if self._fk is None:
            from point5.humanoid21.fk import Humanoid21FK
            self._fk = Humanoid21FK()
        return self._fk

    def _pose_leveled(self, obs: np.ndarray):
        """Pose the FK model from obs (tilt-leveled root).

        Returns (landmark positions in the leveled frame, torso yaw deg,
        R_tilt)."""
        fk, rid = self.fk(), self.robot_id
        rng = fk.joint_ranges(rid).astype(np.float64)
        ref, sc = (rng[:, 0] + rng[:, 1]) / 2, (rng[:, 1] - rng[:, 0]) / 2
        R_tilt = _level_rot(obs[42:45])
        fk.set_pose(rid, np.array([0.0, 0.0, obs[45]]),
                    _quat_wxyz(R_tilt), obs[0:21] * sc + ref)
        lm = fk.landmark_positions(rid)
        poses = fk.body_pose(rid)
        torso = next(v for k, v in poses.items() if "torso" in k)
        fwd = torso[1][:, 0]
        yaw = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
        return lm, yaw, R_tilt

    @staticmethod
    def _yaw_rot(yaw_deg):
        ang = np.radians(yaw_deg)
        return np.array([[np.cos(ang), np.sin(ang), 0],
                         [-np.sin(ang), np.cos(ang), 0], [0, 0, 1]])

    def decode_p5_local(self, obs: np.ndarray):
        """obs -> (local_p5 (5,3), R_local(3,3): body->P5 frame)."""
        from point5.contracts import ankle_mid_xy, to_local
        lm, yaw, R_tilt = self._pose_leveled(obs)
        local = to_local(lm, ankle_mid_xy(lm), yaw)
        return local, self._yaw_rot(yaw) @ R_tilt

    def decode_support_local(self, obs: np.ndarray):
        """obs -> (feet_xy {'right','left'} (2,), com (3,)) local frame.

        Direct support-state decode — one FK pass, no P5 framing:
        foot landmark xy + exact mass-weighted COM.
        """
        from point5.contracts import ankle_mid_xy
        lm, yaw, _ = self._pose_leveled(obs)
        Rz = self._yaw_rot(yaw)
        mid = np.array([*ankle_mid_xy(lm), 0.0])
        feet_xy = {}
        for side, fi in (("right", 3), ("left", 4)):
            feet_xy[side] = (Rz @ (lm[fi] - mid))[:2]
        r = self.fk()._robots[self.robot_id]
        ids = np.asarray(r["body_ids_sorted"], int)
        ms = np.asarray(self.fk().model.body_mass)[ids]
        xip = np.asarray(self.fk().data.xipos)[ids]
        com_lev = (ms[:, None] * xip).sum(0) / ms.sum()
        return feet_xy, Rz @ (com_lev - mid)
