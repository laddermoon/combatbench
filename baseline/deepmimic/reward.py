"""DeepMimic 式运动模仿奖励 observer。

每控制步输出 dict，供 ``extract_per_step_field`` 抽取：

  reward = w_pose·r_pose + w_vel·r_vel + w_kp·r_kp
           + w_height·r_height + w_root·r_root

  r_pose   = exp(-2   · mean((q_norm  − ref_q_norm)²))       21 hinge 均匀权重
  r_vel    = exp(-0.02· mean((qv_norm − ref_qv_norm)²))
  r_kp     = exp(-40  · mean(骨盆系关键点误差²))   torso/head/双手/双足
  r_height = exp(-20  · (root_z − ref_root_z)²)
  r_root   = 0.5·exp(-3·dxy²) + 0.5·exp(-1·dyaw²)

与论文配方的差异（DESIGN.md §5）：
- 关节项用归一化角 MSE 而非四元数测地距——hinge 关节下两者等价（单调），
  差异只剩均匀 vs per-joint 权重
- r_kp 在骨盆系而非世界系——有意偏差：世界系 kp 对平移滞后过罚会导致
  策略退化；全局轨迹约束由 r_root 兜底
- r_root 是论文没有的新增项：骨盆系 kp + 关节 pose 对根 yaw/xy 不变，
  缺它会出现"原地打拳"退化
- 无显式 COM 项；r_height 覆盖其 z 分量
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.framework import BaseObserverPlugin, ReadOnlySimContext
from mocap.humanoid21.robot_kin import CONTROLLED_JOINTS, joint_limits

KP_SUBSET = ('torso', 'head', 'foot_left', 'foot_right',
             'hand_left', 'hand_right')

_HALF = np.array([
    0.5 * (joint_limits(n)[1] - joint_limits(n)[0])
    for n in CONTROLLED_JOINTS
])


class DeepMimicRewarder(BaseObserverPlugin):
    def __init__(self, agent_id: str = 'robot_a',
                 w_pose: float = 0.35, w_vel: float = 0.10,
                 w_kp: float = 0.25, w_height: float = 0.10,
                 w_root: float = 0.20) -> None:
        self.agent_id = str(agent_id)
        self.w = (float(w_pose), float(w_vel), float(w_kp),
                  float(w_height), float(w_root))
        self._output: Dict[str, float] = {'reward': 0.0}

    def _compute(self, ctx: ReadOnlySimContext) -> Dict[str, float]:
        w_pose, w_vel, w_kp, w_height, w_root = self.w
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
        sq_world, sq_rel = [], []
        ref_kp = np.asarray(mt['ref_keypoints'], dtype=np.float64)
        if 'pelvis' in kn and 'pelvis' in self_kp:
            pi = kn.index('pelvis')
            p_ref_origin = ref_kp[pi]
            p_self_origin = np.asarray(self_kp['pelvis'], dtype=np.float64)
            q_ref = np.asarray(mt['ref_root_quat'], dtype=np.float64)
            R_ref = R.from_quat([q_ref[1], q_ref[2], q_ref[3], q_ref[0]])
            q_self = np.asarray(core['root_rot'], dtype=np.float64)
            R_self = R.from_quat([q_self[1], q_self[2], q_self[3], q_self[0]])
        else:
            p_ref_origin = p_self_origin = None
        for n in KP_SUBSET:
            if n in kn and n in self_kp:
                j = kn.index(n)
                p_self = np.asarray(self_kp[n], dtype=np.float64)
                p_ref = ref_kp[j]
                sq_world.append(float(np.sum((p_self - p_ref) ** 2)))
                if p_ref_origin is not None:
                    e_self = R_self.inv().as_matrix() @ (p_self - p_self_origin)
                    e_ref = R_ref.inv().as_matrix() @ (p_ref - p_ref_origin)
                    sq_rel.append(float(np.sum((e_self - e_ref) ** 2)))
        kp_mse_world = float(np.mean(sq_world)) if sq_world else 0.0
        kp_mse = float(np.mean(sq_rel)) if sq_rel else kp_mse_world
        r_kp = float(np.exp(-40.0 * kp_mse))
        kp_err_m = float(np.sqrt(kp_mse_world))

        z_err = float(core['root_pos'][2] - mt['ref_root_pos'][2])
        r_height = float(np.exp(-20.0 * z_err ** 2))

        self_pos = np.asarray(core['root_pos'], dtype=np.float64)
        ref_pos = np.asarray(mt['ref_root_pos'], dtype=np.float64)
        dxy = float(np.linalg.norm(self_pos[:2] - ref_pos[:2]))
        q_self = np.asarray(core['root_rot'], dtype=np.float64)
        yaw_s = float(R.from_quat([q_self[1], q_self[2], q_self[3], q_self[0]]
                                 ).as_euler('xyz')[2])
        q_ref = np.asarray(mt['ref_root_quat'], dtype=np.float64)
        yaw_r = float(R.from_quat([q_ref[1], q_ref[2], q_ref[3], q_ref[0]]
                                 ).as_euler('xyz')[2])
        dyaw = float((yaw_s - yaw_r + np.pi) % (2 * np.pi) - np.pi)
        r_root = 0.5 * float(np.exp(-3.0 * dxy ** 2)) \
            + 0.5 * float(np.exp(-1.0 * dyaw ** 2))

        reward = (w_pose * r_pose + w_vel * r_vel
                  + w_kp * r_kp + w_height * r_height + w_root * r_root)
        return {
            'reward': reward, 'pose': r_pose, 'vel': r_vel,
            'kp': r_kp, 'height': r_height, 'root': r_root,
            'kp_err_m': kp_err_m, 'kp_err_rel_m': float(np.sqrt(kp_mse)),
            'root_dxy_m': dxy, 'root_dyaw_deg': float(np.degrees(dyaw)),
            'q_rmse_deg': q_rmse_deg,
            'frame': float(mt['frame']), 'phase': float(mt['phase']),
        }

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._compute(ctx)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._compute(ctx)

    def get_output(self) -> Dict[str, float]:
        return dict(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {'agent_id': self.agent_id,
                'w_pose': self.w[0], 'w_vel': self.w[1], 'w_kp': self.w[2],
                'w_height': self.w[3], 'w_root': self.w[4]}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> 'DeepMimicRewarder':
        return cls(**config)
