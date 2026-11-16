"""V1 技能契约基类（v0 最简版）。

``Skill`` = 无状态技能规约：描述"这个技能是什么"，不是控制器——
执行器是共享的技能条件策略。

第一阶段语义（2026-11-16 简化讨论，DECISION_LOG D3）：

- 技能按**完整执行**训练：从候选起点出发执行到固定步数或明确失败，
  不支持中途切换。某技能的候选起点 = 其前序技能终止状态集的并集。
- **不设计过程参考轨迹，只定义结果**（姿态设计问题→奖励引导问题）。
  技能契约只含三个整段评分函数：输入为一次完整执行的 clip，
  输出一个分数；实现方式不限。
- 评分是 clip 级单次输出，但可以使用 clip 内任意帧的信息。
  per-step 奖励分配、合格判定、尾状态入池规则由训练侧消费这些
  分数决定，不属于本契约。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Clip:
    """一次技能执行的逐帧特征记录（世界系）。

    由训练侧从 observer 每步输出组装成 (T,…) 数组；评分方法只读。
    ``ego()`` 把世界 xy 量变换到指定帧的机体系（x 前、y 左）；
    z 不变换（地面水平），高度直接用世界 z。
    """

    com: np.ndarray          # (T,3) world
    ankle_l: np.ndarray      # (T,3) 踝心
    ankle_r: np.ndarray      # (T,3)
    hand_l: np.ndarray       # (T,3)
    hand_r: np.ndarray       # (T,3)
    root_pos: np.ndarray     # (T,3) torso
    root_pry: np.ndarray     # (T,3) pitch/roll/yaw (rad)
    root_vel: np.ndarray     # (T,3) world
    contact: np.ndarray      # (T,2) float (left,right)

    def __len__(self) -> int:
        return int(self.com.shape[0])

    def head(self, n: int) -> "Clip":
        """前 n 帧子段（裁剪用）。"""
        return Clip(**{k: np.asarray(v)[:n] for k, v in vars(self).items()})

    def ego(self, p_world: np.ndarray, frame: int = 0) -> np.ndarray:
        """世界 (…,2) 点 → 第 frame 帧机体系 xy（x 前、y 左）。"""
        p = np.asarray(p_world, np.float64)[..., :2]
        d = p - self.root_pos[frame, :2]
        yaw = float(self.root_pry[frame, 2])
        c, s = np.cos(yaw), np.sin(yaw)
        return np.stack([c * d[..., 0] + s * d[..., 1],
                         -s * d[..., 0] + c * d[..., 1]], axis=-1)


def end_support_score(clip: Clip, ori_tol: float = 0.25,
                      vel_tol: float = 0.5) -> float:
    """末帧支撑质量（v0 通用件）：com 在双踝中点附近、躯干直立、
    速度低 → (0,1]。供各技能 score_com 复用或参考。"""
    mid = 0.5 * (clip.ankle_l[-1, :2] + clip.ankle_r[-1, :2])
    d = float(np.linalg.norm(clip.com[-1, :2] - mid))
    pitch = float(clip.root_pry[-1, 0])
    roll = float(clip.root_pry[-1, 1])
    v = float(np.linalg.norm(clip.root_vel[-1]))
    return float(np.exp(-0.5 * (d / 0.12) ** 2)
                 * np.exp(-0.5 * (pitch / ori_tol) ** 2)
                 * np.exp(-0.5 * (roll / ori_tol) ** 2)
                 * np.exp(-0.5 * (v / vel_tol) ** 2))


class Skill:
    """技能契约基类。

    类属性：
      name         技能名
      family       技能族（分组/采样用）
      instruction  6 维技能码（DECISION_LOG D1）
      predecessors 前序技能名列表——本技能候选起点 = 这些技能
                   终止状态集的并集（D3 简化一）
      step_budget  一次执行的固定时长上限（控制步数）

    子类按技能需要实现三个评分方法；不关心某维度的可保持默认 0.0。
    """

    name: str = ""
    family: str = ""
    instruction: np.ndarray = np.zeros(6, np.float64)
    predecessors: Tuple[str, ...] = ()
    step_budget: int = 40

    def trim_clip(self, clip: Clip) -> "Optional[int]":
        """判定本 clip 中"任务已完成"的最早裁剪点。

        返回应保留的帧数 k：clip[:k] 为有效执行段，奖励结算在
        第 k-1 帧，该帧的 core_state 快照即终止状态候选。
        返回 None = 未完成/不裁剪，训练侧使用完整 clip。

        语义上裁剪点即"技能任务完成"——对该技能的 MDP 是真终止
        （不是 timeout 截断），bootstrap 取 V=0。
        """
        return None

    def score_com(self, clip: Clip) -> float:
        """质心运动评价：质心运动状态与支撑稳定性。"""
        return 0.0

    def score_step(self, clip: Clip) -> float:
        """脚步评价：脚是否迈到指定位置。"""
        return 0.0

    def score_posture(self, clip: Clip) -> float:
        """姿态评价：终点姿态接近程度与运动过程表现。"""
        return 0.0
