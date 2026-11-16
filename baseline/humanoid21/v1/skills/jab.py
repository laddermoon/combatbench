"""直拳技能（jab_l / jab_r）：指定手快速到达打击点。

结果定义（v0，全部公式为待调初值）：
- score_posture：clip 内出拳手到达起步系打击点的最优接近度（到位判定
  的主体——没有它"站着不动"也能拿高分）+ 末帧躯干姿态；
- score_com：末帧支撑质量（允许比 step 更松的前倾）；
- score_step：不鼓励迈新步——双脚最大起步系位移越小越好。

前序 = 全部四技能。
"""

from __future__ import annotations

import numpy as np

from .base import Clip, Skill, end_support_score

_READY_PITCH = 0.29


class JabSkill(Skill):
    family = "jab"
    side = 1                     # +1 左手, −1 右手
    step_budget = 40
    punch_x = 0.45               # 起步系打击点 x（前, m）
    punch_y = 0.10               # 起步系打击点 |y|（侧向按 side 取号）
    punch_z = 1.30               # 打击点高度（world z, m）
    reach_tol = 0.08             # 到位容差 σ (m)
    foot_drift_free = 0.10       # 脚位移免费区 (m)
    foot_drift_full = 0.40       # 脚位移满分区上限
    predecessors = ("step_l", "step_r", "jab_l", "jab_r")

    def score_com(self, clip: Clip) -> float:
        return end_support_score(clip, ori_tol=0.35, vel_tol=0.7)

    def score_step(self, clip: Clip) -> float:
        d = 0.0
        for ank in (clip.ankle_l, clip.ankle_r):
            ego_xy = clip.ego(ank, frame=0)
            d = max(d, float(np.linalg.norm(
                ego_xy - ego_xy[0], axis=1).max()))
        span = self.foot_drift_full - self.foot_drift_free
        return float(np.clip(
            1.0 - (d - self.foot_drift_free) / span, 0.0, 1.0))

    def score_posture(self, clip: Clip) -> float:
        hand = clip.hand_l if self.side > 0 else clip.hand_r
        ego_xy = clip.ego(hand, frame=0)
        tgt_xy = np.array([self.punch_x, self.side * self.punch_y])
        err = np.sqrt(((ego_xy - tgt_xy[None, :]) ** 2).sum(-1)
                      + (hand[:, 2] - self.punch_z) ** 2)
        reach = float(np.exp(-0.5 * (err.min() / self.reach_tol) ** 2))
        p, r = float(clip.root_pry[-1, 0]), float(clip.root_pry[-1, 1])
        end = float(np.exp(
            -0.5 * ((p - _READY_PITCH) ** 2 + r ** 2) / 0.25 ** 2))
        return 0.7 * reach + 0.3 * end


jab_l = JabSkill()
jab_l.name = "jab_l"
jab_l.side = +1
jab_l.instruction = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])

jab_r = JabSkill()
jab_r.name = "jab_r"
jab_r.side = -1
jab_r.instruction = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 0.0])
