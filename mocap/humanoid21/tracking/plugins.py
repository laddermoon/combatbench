"""M2 跟踪环境的世界插件：摔倒 / 漂移 / 片段结束终止。

只读 accessor，不需要 mutator。判定来源：
- ``motion_track.ended``：参考片段相位走完 → 'track_clip_end'
- 根高度 root_pos[2] < fall_height → 'track_fall'
- 头低于骨盆（解剖翻转，等同 M1 检查语义）→ 'track_fall'
- 根 xy/yaw 与参考分离超阈值 → 'track_drift'（DeepMimic 式早停：
  软奖励下"原地/滞后"局部最优靠 survive 存活，漂移终止把
  r_root 软惩罚升级为硬约束；阈值默认关闭，蓝图显式开启）
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.framework import BasePlugin, SimContext, TerminationReason


class MotionTrackGuardPlugin(BasePlugin):
    def __init__(self, track_robot: str = 'robot_a',
                 fall_height: float = 0.55,
                 drift_dxy: Optional[float] = None,
                 drift_dyaw_deg: Optional[float] = None) -> None:
        self.track_robot = track_robot
        self.fall_height = float(fall_height)
        self.drift_dxy = drift_dxy
        self.drift_dyaw = (np.radians(drift_dyaw_deg)
                           if drift_dyaw_deg is not None else None)

    @property
    def name(self) -> str:
        return 'motion_track_guard'

    def to_blueprint(self) -> Dict[str, Any]:
        return {'track_robot': self.track_robot,
                'fall_height': self.fall_height,
                'drift_dxy': self.drift_dxy,
                'drift_dyaw_deg': (np.degrees(self.drift_dyaw)
                                   if self.drift_dyaw is not None else None)}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> 'MotionTrackGuardPlugin':
        return cls(**config)

    @staticmethod
    def _yaw(quat_wxyz: np.ndarray) -> float:
        q = quat_wxyz
        return float(R.from_quat([q[1], q[2], q[3], q[0]]
                                 ).as_euler('xyz')[2])

    def on_post_action_step(self, ctx: SimContext) -> None:
        ds = ctx.accessor.get_derived_state()
        mt = ds.get('motion_track')
        if mt is None:
            return
        if mt['ended']:
            ctx.request_termination('track_clip_end')
            return
        core = ctx.accessor.get_core_state()[self.track_robot]
        if float(core['root_pos'][2]) < self.fall_height:
            ctx.request_termination('track_fall')
            return
        self_kp = mt['self_keypoints']
        if 'head' in self_kp and 'pelvis' in self_kp:
            if float(self_kp['head'][2]) < float(self_kp['pelvis'][2]):
                ctx.request_termination('track_fall')
                return
        if self.drift_dxy is not None or self.drift_dyaw is not None:
            self_pos = np.asarray(core['root_pos'], dtype=np.float64)
            ref_pos = np.asarray(mt['ref_root_pos'], dtype=np.float64)
            if self.drift_dxy is not None and float(
                    np.linalg.norm(self_pos[:2] - ref_pos[:2])) \
                    > self.drift_dxy:
                ctx.request_termination('track_drift')
                return
            if self.drift_dyaw is not None:
                dyaw = (self._yaw(np.asarray(core['root_rot']))
                        - self._yaw(np.asarray(mt['ref_root_quat']))
                        + np.pi) % (2 * np.pi) - np.pi
                if abs(float(dyaw)) > self.drift_dyaw:
                    ctx.request_termination('track_drift')
