"""前迈技能（step_l / step_r）：摆动脚前迈并落地承重。

结果定义（v0，全部公式为待调初值）：
- score_step：摆动脚确实有效离地过（峰值踝高 ≥ lift_min），且末帧落点
  接近"起步系下起脚点 + 前向 step_len"的目标；
- score_com：末帧支撑质量（com 距双踝中点、躯干直立、速度低）；
- score_posture：末帧躯干 pitch/roll 接近准备姿态 + 过程 roll 不失控。

前序 = 全部四技能（含自身：step→step 即连续行走）。
"""

from __future__ import annotations

import numpy as np

from .base import Clip, Skill, end_support_score

_READY_PITCH = 0.29     # ready 姿态前倾角（rad）


class StepSkill(Skill):
    family = "step"
    side = 1                     # +1 左脚, −1 右脚
    step_budget = 40
    step_len = 0.28              # 目标步幅（前向位移, m）
    lift_min = 0.04              # 有效离地最小峰值踝高 (m)
    land_tol = 0.08              # 落点容差 σ (m)
    predecessors = ("step_l", "step_r", "jab_l", "jab_r")

    def _swing(self, clip: Clip) -> np.ndarray:
        return clip.ankle_l if self.side > 0 else clip.ankle_r

    def score_com(self, clip: Clip) -> float:
        return end_support_score(clip, ori_tol=0.25, vel_tol=0.5)

    def score_step(self, clip: Clip) -> float:
        sw = self._swing(clip)
        lifted = float(sw[:, 2].max()) >= self.lift_min
        start_ego = clip.ego(sw[0:1], frame=0)[0]
        end_ego = clip.ego(sw[-1:], frame=0)[0]
        err = float(np.linalg.norm(
            end_ego - (start_ego + np.array([self.step_len, 0.0]))))
        s = float(np.exp(-0.5 * (err / self.land_tol) ** 2))
        return s if lifted else 0.2 * s

    def score_posture(self, clip: Clip) -> float:
        p, r = float(clip.root_pry[-1, 0]), float(clip.root_pry[-1, 1])
        end = float(np.exp(
            -0.5 * ((p - _READY_PITCH) ** 2 + r ** 2) / 0.20 ** 2))
        roll_path = float(np.exp(
            -0.5 * (clip.root_pry[:, 1] / 0.40) ** 2).mean())
        return 0.6 * end + 0.4 * roll_path


step_l = StepSkill()
step_l.name = "step_l"
step_l.side = +1
step_l.instruction = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

step_r = StepSkill()
step_r.name = "step_r"
step_r.side = -1
step_r.instruction = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
