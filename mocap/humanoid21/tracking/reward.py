"""DeepMimic 式运动模仿奖励 observer（M2/L2 跟踪训练）。

每控制步输出 dict，供 ``extract_per_step_field`` 抽取：
  reward / pose / vel / kp / height / kp_err_m / q_rmse_deg / frame

reward = 0.45·r_pose + 0.10·r_vel + 0.35·r_kp + 0.10·r_height
  r_pose   = exp(-2 · mean((q_norm − ref_q_norm)²))
  r_vel    = exp(-0.02 · mean((qv_norm − ref_qv_norm)²))
  r_kp     = exp(-40 · mean(||p_self − p_ref||²))   关键点: torso/head/双手/双足
  r_height = exp(-20 · (root_z − ref_root_z)²)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext
from mocap.humanoid21.robot_kin import CONTROLLED_JOINTS, joint_limits

KP_SUBSET = ('torso', 'head', 'foot_left', 'foot_right',
             'hand_left', 'hand_right')

_HALF = np.array([
    0.5 * (joint_limits(n)[1] - joint_limits(n)[0])
    for n in CONTROLLED_JOINTS
])

W_POSE, W_VEL, W_KP, W_HEIGHT = 0.45, 0.10, 0.35, 0.10


class MotionTrackRewarder(BaseObserverPlugin):
    def __init__(self, agent_id: str = 'robot_a') -> None:
        self.agent_id = str(agent_id)
        self._output: Dict[str, float] = {'reward': 0.0}

    def _compute(self, ctx: ReadOnlySimContext) -> Dict[str, float]:
        ds = ctx.accessor.get_derived_state()
        mt = ds['motion_track']
        core = ctx.accessor.get_core_state()[self.agent_id]

        q_err = np.asarray(core['joint_pos_norm'], dtype=np.float64) - \
            np.asarray(mt['ref_q_norm'], dtype=np.float64)
        qv_err = np.asarray(core['joint_vel_norm'], dtype=np.float64) - \
            np.asarray(mt['ref_qv_norm'], dtype=np.float64)
        r_pose = float(np.exp(-2.0 * np.mean(q_err ** 2)))
        r_vel = float(np.exp(-0.02 * np.mean(qv_err ** 2)))
        q_rmse_deg = float(np.degrees(np.sqrt(np.mean((q_err * _HALF) ** 2))))

        kn = mt['ref_keypoint_names']
        self_kp = mt['self_keypoints']
        sq = []
        for n in KP_SUBSET:
            if n in kn and n in self_kp:
                j = kn.index(n)
                sq.append(float(np.sum(
                    (np.asarray(self_kp[n], dtype=np.float64)
                     - np.asarray(mt['ref_keypoints'][j], dtype=np.float64)) ** 2)))
        kp_mse = float(np.mean(sq)) if sq else 0.0
        r_kp = float(np.exp(-40.0 * kp_mse))
        kp_err_m = float(np.sqrt(kp_mse))

        z_err = float(core['root_pos'][2] - mt['ref_root_pos'][2])
        r_height = float(np.exp(-20.0 * z_err ** 2))

        reward = (W_POSE * r_pose + W_VEL * r_vel
                  + W_KP * r_kp + W_HEIGHT * r_height)
        return {
            'reward': reward, 'pose': r_pose, 'vel': r_vel,
            'kp': r_kp, 'height': r_height,
            'kp_err_m': kp_err_m, 'q_rmse_deg': q_rmse_deg,
            'frame': float(mt['frame']), 'phase': float(mt['phase']),
        }

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._compute(ctx)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._compute(ctx)

    def get_output(self) -> Dict[str, float]:
        return dict(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {'agent_id': self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> 'MotionTrackRewarder':
        return cls(**config)
