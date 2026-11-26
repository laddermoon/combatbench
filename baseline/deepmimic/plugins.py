"""DeepMimic 跟踪环境的世界插件：摔倒 / 漂移 / 片段结束终止。

只读 accessor，不需要 mutator。判定顺序（``on_post_action_step``）：

1. ``motion_track.ended`` → 'track_clip_end'
2. fall 判定（``fall_mode``）：
   - ``height``（默认）：root_z < fall_height 或 head_z < pelvis_z
     （解剖翻转代理；getup/lie/crawl 类 clip 会误伤，改用 contact/none）
   - ``contact``：contacts SoA 中 track_robot 的**非足 body** 与环境
     （aff=0，地面/墙）发生接触 → 更接近论文 "undesired contact" 语义
   - ``none``：关闭摔倒判定
3. drift 判定（阈值非 null 时，DeepMimic 式早停）：
   ``|self−ref|_xy > drift_dxy`` 或 ``|dyaw| > drift_dyaw_deg``
   → 'track_drift'
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.framework import BasePlugin, SimContext

from .sim import _yaw

_FALL_MODES = ('height', 'contact', 'none')


class DeepMimicGuardPlugin(BasePlugin):
    def __init__(self, track_robot: str = 'robot_a',
                 fall_mode: str = 'height',
                 fall_height: float = 0.55,
                 drift_dxy: Optional[float] = None,
                 drift_dyaw_deg: Optional[float] = None) -> None:
        if fall_mode not in _FALL_MODES:
            raise ValueError(f'fall_mode must be one of {_FALL_MODES}, '
                             f'got {fall_mode!r}')
        self.track_robot = track_robot
        self.fall_mode = fall_mode
        self.fall_height = float(fall_height)
        self.drift_dxy = drift_dxy
        self.drift_dyaw = (np.radians(drift_dyaw_deg)
                           if drift_dyaw_deg is not None else None)

    @property
    def name(self) -> str:
        return 'deepmimic_guard'

    def to_blueprint(self) -> Dict[str, Any]:
        return {'track_robot': self.track_robot,
                'fall_mode': self.fall_mode,
                'fall_height': self.fall_height,
                'drift_dxy': self.drift_dxy,
                'drift_dyaw_deg': (np.degrees(self.drift_dyaw)
                                 if self.drift_dyaw is not None else None)}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> 'DeepMimicGuardPlugin':
        return cls(**config)

    # ------------------------------------------------------------------
    def _height_fall(self, ctx: SimContext, mt: Dict[str, Any]) -> bool:
        core = ctx.accessor.get_core_state()[self.track_robot]
        if float(core['root_pos'][2]) < self.fall_height:
            return True
        kp = mt['self_keypoints']
        if 'head' in kp and 'pelvis' in kp:
            if float(kp['head'][2]) < float(kp['pelvis'][2]):
                return True
        return False

    def _contact_fall(self, ctx: SimContext, mt: Dict[str, Any]) -> bool:
        """track_robot 的非足 body 与环境（地面/墙）接触。"""
        ds = ctx.accessor.get_derived_state(['contacts'])
        cts = ds.get('contacts')
        if cts is None or cts['ncon'] == 0:
            return False
        aff = mt['track_robot_aff']
        nonfoot = mt['nonfoot_body_ids']
        b1, b2 = cts['body1'], cts['body2']
        a1, a2 = cts['aff1'], cts['aff2']
        bad1 = (a1 == aff) & np.isin(b1, list(nonfoot)) & (a2 == 0)
        bad2 = (a2 == aff) & np.isin(b2, list(nonfoot)) & (a1 == 0)
        return bool(np.any(bad1 | bad2))

    # ------------------------------------------------------------------
    def on_post_action_step(self, ctx: SimContext) -> None:
        ds = ctx.accessor.get_derived_state()
        mt = ds.get('motion_track')
        if mt is None:
            return
        if mt['ended']:
            ctx.request_termination('track_clip_end')
            return
        if self.fall_mode == 'height' and self._height_fall(ctx, mt):
            ctx.request_termination('track_fall')
            return
        if self.fall_mode == 'contact' and self._contact_fall(ctx, mt):
            ctx.request_termination('track_fall')
            return
        if self.drift_dxy is not None or self.drift_dyaw is not None:
            core = ctx.accessor.get_core_state()[self.track_robot]
            self_pos = np.asarray(core['root_pos'], dtype=np.float64)
            ref_pos = np.asarray(mt['ref_root_pos'], dtype=np.float64)
            if (self.drift_dxy is not None and float(
                    np.linalg.norm(self_pos[:2] - ref_pos[:2]))
                    > self.drift_dxy):
                ctx.request_termination('track_drift')
                return
            if self.drift_dyaw is not None:
                dyaw = (_yaw(np.asarray(core['root_rot']))
                        - _yaw(np.asarray(mt['ref_root_quat']))
                        + np.pi) % (2 * np.pi) - np.pi
                if abs(float(dyaw)) > self.drift_dyaw:
                    ctx.request_termination('track_drift')
