"""CoreRefView — CoreClip 的运动跟踪视图。

``CoreClip.state(k)`` 的六字段与 ``set_core_state`` 原生同构，RSI 落位
不需要任何转换；本类只补跟踪侧需要的派生量：

- 世界系根速度（CoreClip 存机体系，obs/奖励内部用世界系做 dyaw/dxy）
- 反归一化关节角/速度（rad —— 提供给需要绝对量的调试/评估路径）
- keypoints（``keypoint_pos`` 重导出后随 npz 提供）
- RSI 合法帧集合（``meta.rsi_bad_frames`` 屏蔽）

语义约定（与旧 RefClip 对齐）：
- ``phase = k / (n_ctrl - 1)``，线性非循环
- ``root_quat`` 为 wxyz；世界系速度 = ``R(root_quat) @ *_local``
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from mocap.humanoid21.clip.core_clip import CoreClip
from mocap.humanoid21.robot_kin import CONTROLLED_JOINTS, joint_limits

DATA_CORE = Path(__file__).resolve().parents[2] / 'mocap' / 'humanoid21' \
    / 'data' / 'core'

_MID = np.array([0.5 * (joint_limits(n)[0] + joint_limits(n)[1])
                 for n in CONTROLLED_JOINTS], dtype=np.float64)
_HALF = np.array([0.5 * (joint_limits(n)[1] - joint_limits(n)[0])
                  for n in CONTROLLED_JOINTS], dtype=np.float64)


class CoreRefView:
    """CoreClip 的跟踪侧查询接口。"""

    def __init__(self, clip_id_or_path: Union[str, Path],
                 clips_dir: Optional[Union[str, Path]] = None) -> None:
        p = Path(clip_id_or_path)
        if not p.suffix:
            root = Path(clips_dir) if clips_dir else DATA_CORE
            p = root / f'{clip_id_or_path}.npz'
        if not p.exists():
            raise FileNotFoundError(f'clip not found: {p}')
        self.path = p
        self.clip = CoreClip.load(p)
        self.clip_id = str(self.clip.meta.get('clip_id', p.stem))
        self.meta = self.clip.meta

        self.n_ctrl = self.clip.T
        self.duration_s = self.n_ctrl / 20.0
        self.keypoint_names = list(self.meta.get('keypoint_names') or [])
        if self.clip.keypoint_pos is None:
            raise ValueError(
                f'{self.clip_id}: npz 缺 keypoint_pos（需重导出 '
                'phc2core 或 fk_keypoints 补字段）')

        bad = set(int(k) for k in self.meta.get('rsi_bad_frames', []))
        self.rsi_frames = np.array(
            [k for k in range(self.n_ctrl) if k not in bad], dtype=np.int64)
        if self.rsi_frames.size == 0:
            self.rsi_frames = np.arange(self.n_ctrl, dtype=np.int64)

    # ------------------------------------------------------------------
    @staticmethod
    def _rot(quat_wxyz: np.ndarray) -> R:
        q = np.asarray(quat_wxyz, dtype=np.float64)
        return R.from_quat([q[1], q[2], q[3], q[0]])

    def core_state(self, k: int) -> Dict[str, np.ndarray]:
        """frame k 的 CoreState 字典 — 直接喂 sim.set_core_state。"""
        return self.clip.state(int(np.clip(k, 0, self.n_ctrl - 1)))

    def at(self, k: int) -> Dict[str, np.ndarray]:
        """frame k 的跟踪参考量（标量/向量均为 float64 视角）。"""
        k = int(np.clip(k, 0, self.n_ctrl - 1))
        st = self.core_state(k)
        rot = self._rot(st['root_rot'])
        qn = np.asarray(st['joint_pos_norm'], dtype=np.float64)
        qvn = np.asarray(st['joint_vel_norm'], dtype=np.float64)
        ra = (None if self.clip.ref_action is None
              else np.asarray(self.clip.ref_action[k], dtype=np.float64))
        return {
            'frame': k,
            'phase': k / max(1, self.n_ctrl - 1),
            'q_norm': qn,
            'qv_norm': qvn,
            'q_rad': qn * _HALF + _MID,
            'qv_rad': qvn * _HALF,
            'root_pos': np.asarray(st['root_pos'], dtype=np.float64),
            'root_quat': np.asarray(st['root_rot'], dtype=np.float64),
            'root_vel_local': np.asarray(st['root_vel_local'],
                                         dtype=np.float64),
            'root_angvel_local': np.asarray(st['root_angular_vel_local'],
                                            dtype=np.float64),
            'root_linvel_world': rot.apply(st['root_vel_local']),
            'root_angvel_world': rot.apply(st['root_angular_vel_local']),
            'ref_action': ra,
            'keypoints': np.asarray(self.clip.keypoint_pos[k],
                                    dtype=np.float64),
        }

    # ------------------------------------------------------------------
    def clip_stats(self) -> Dict[str, Any]:
        """供参数建议用的片段统计。"""
        rp = np.asarray(self.clip.states['root_pos'], dtype=np.float64)
        k0, k1 = int(self.rsi_frames[0]), int(self.rsi_frames[-1])
        d_xy = rp[:, :2] - rp[k0:k0 + 1, :2]
        return {
            'clip_id': self.clip_id,
            'n_ctrl': self.n_ctrl,
            'duration_s': self.duration_s,
            'labels': list(self.meta.get('labels', [])),
            'quality': self.meta.get('phc_quality_level'),
            'issues': list(self.meta.get('phc_issues', [])),
            'net_dxy': float(np.linalg.norm(rp[k1, :2] - rp[k0, :2])),
            'max_dxy': float(np.abs(d_xy).max()),
            'n_rsi_bad': self.n_ctrl - len(self.rsi_frames),
        }
