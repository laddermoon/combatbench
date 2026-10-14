"""M2 跟踪环境的世界插件：摔倒 / 片段结束终止。

只读 accessor，不需要 mutator。判定来源：
- ``motion_track.ended``：参考片段相位走完 → 'track_clip_end'
- 根高度 root_pos[2] < fall_height → 'track_fall'
- 头低于骨盆（解剖翻转，等同 M1 检查语义）→ 'track_fall'
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BasePlugin, SimContext, TerminationReason


class MotionTrackGuardPlugin(BasePlugin):
    def __init__(self, track_robot: str = 'robot_a',
                 fall_height: float = 0.55) -> None:
        self.track_robot = track_robot
        self.fall_height = float(fall_height)

    @property
    def name(self) -> str:
        return 'motion_track_guard'

    def to_blueprint(self) -> Dict[str, Any]:
        return {'track_robot': self.track_robot,
                'fall_height': self.fall_height}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> 'MotionTrackGuardPlugin':
        return cls(**config)

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
