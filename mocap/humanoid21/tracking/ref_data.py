"""参考轨迹加载与 20Hz 控制率重采样（M2 跟踪训练侧）。

消费 ``clips/cmu_*.npz``（120Hz, joint_space=humanoid21_v2）。
``frame k``（控制帧，20Hz）对应 npz 帧 ``k * REF_STRIDE``。

约定：
- 关节角归一化与 CONTROLSPEC 一致：norm = (q - mid)/half，
  mid/half 由 robot_spec.json 的 joint limits 推导（与仿真器
  jnt_ranges 同源）。
- root_quat 为 wxyz；root_linvel / root_angvel 为世界系。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from mocap.humanoid21.robot_kin import CONTROLLED_JOINTS, joint_limits

REF_STRIDE = 6            # 120Hz -> 20Hz
CTRL_FPS = 20.0
DEFAULT_CLIPS_DIR = Path('/data1/dev/combatbench_mocap/clips')


class RefClip:
    def __init__(self, clip_id: str, clips_dir=DEFAULT_CLIPS_DIR):
        clips_dir = Path(clips_dir)
        path = clips_dir / f'{clip_id}.npz'
        if not path.exists():
            path = clips_dir / f'cmu_{clip_id}.npz'
        if not path.exists():
            raise FileNotFoundError(f'clip not found: {clip_id} under {clips_dir}')
        z = np.load(path, allow_pickle=True)

        names = [str(n) for n in z['joint_names']]
        if names != list(CONTROLLED_JOINTS):
            idx = [names.index(n) for n in CONTROLLED_JOINTS]
            q = z['joint_angles'][:, idx]
            qv = z['joint_velocities'][:, idx]
        else:
            q = z['joint_angles']
            qv = z['joint_velocities']

        self.clip_id = clip_id
        self.joint_names = list(CONTROLLED_JOINTS)

        lo = np.array([joint_limits(n)[0] for n in CONTROLLED_JOINTS])
        hi = np.array([joint_limits(n)[1] for n in CONTROLLED_JOINTS])
        self.mid = 0.5 * (lo + hi)
        self.half = 0.5 * (hi - lo)

        # 控制率下采样
        self.q_rad = np.asarray(q, dtype=np.float64)[::REF_STRIDE]
        self.qv_rad = np.asarray(qv, dtype=np.float64)[::REF_STRIDE]
        self.q_norm = (self.q_rad - self.mid) / self.half
        self.qv_norm = self.qv_rad / self.half
        self.root_pos = np.asarray(z['root_pos'], dtype=np.float64)[::REF_STRIDE]
        self.root_quat = np.asarray(z['root_quat'], dtype=np.float64)[::REF_STRIDE]
        self.root_linvel = np.asarray(z['root_linvel'], dtype=np.float64)[::REF_STRIDE]
        self.root_angvel = np.asarray(z['root_angvel'], dtype=np.float64)[::REF_STRIDE]
        self.keypoints = np.asarray(z['keypoints'], dtype=np.float64)[::REF_STRIDE]
        self.keypoint_names = [str(n) for n in z['keypoint_names']]

        self.n_ctrl = self.q_norm.shape[0]
        self.duration_s = self.n_ctrl / CTRL_FPS
        # 片段引导区可能含 IK 分支瞬态或源数据毛刺（导出时标记）。
        # RSI 应从该控制帧之后采样。
        lead = int(z['lead_stable_frame']) if 'lead_stable_frame' in z else 0
        self.min_start_ctrl = int(np.ceil(lead / REF_STRIDE))

    def at(self, k: int) -> Dict[str, np.ndarray]:
        k = int(np.clip(k, 0, self.n_ctrl - 1))
        return {
            'frame': k,
            'phase': k / max(1, self.n_ctrl - 1),
            'q_norm': self.q_norm[k],
            'qv_norm': self.qv_norm[k],
            'q_rad': self.q_rad[k],
            'qv_rad': self.qv_rad[k],
            'root_pos': self.root_pos[k],
            'root_quat': self.root_quat[k],
            'root_linvel': self.root_linvel[k],
            'root_angvel': self.root_angvel[k],
            'keypoints': self.keypoints[k],
        }
