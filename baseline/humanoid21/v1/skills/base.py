"""V1 技能契约基类（v0 最简版）。

``Skill`` = 无状态技能规约：描述"这个技能是什么"，不是控制器——
执行器是共享的技能条件策略。

第一阶段语义（2026-11-16 简化讨论）：

- 技能按**完整执行**训练：从候选起点出发执行到固定步数或明确失败，
  不支持中途切换。某技能的候选起点 = 其前序技能终止状态集的并集。
- **不设计过程参考轨迹，只定义结果**（姿态设计问题→奖励引导问题）。
  技能契约只含三个整段评分函数：输入为一次完整执行的 clip
  （RobotState 序列），输出一个分数；实现方式不限，不要求 CPose。
- 评分是 clip 级单次输出，但可以使用 clip 内任意帧的信息。
  per-step 奖励分配、合格判定、尾状态入池规则由训练侧消费这些
  分数决定，不属于本契约。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from point5.humanoid21.robot_state import RobotState

# 一次技能执行的完整轨迹：从起点状态到结束（固定步数/失败）的逐帧状态
Clip = Sequence[RobotState]


class Skill:
    """技能契约基类。

    类属性：
      name        技能名
      instruction 6 维技能码（DECISION_LOG D1）
      step_budget 一次执行的固定时长上限（控制步数）

    子类按技能需要实现三个评分方法；不关心某维度的可保持默认 0.0。
    """

    name: str = ""
    instruction: np.ndarray = np.zeros(6, np.float64)
    step_budget: int = 40

    def score_com(self, clip: Clip) -> float:
        """质心运动评价：质心运动状态与支撑稳定性。"""
        return 0.0

    def score_step(self, clip: Clip) -> float:
        """脚步评价：脚是否迈到指定位置。"""
        return 0.0

    def score_posture(self, clip: Clip) -> float:
        """姿态评价：终点姿态接近程度与运动过程表现。"""
        return 0.0
